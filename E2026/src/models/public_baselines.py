"""Aligned-50 adaptations of TFN, MulT, and MISA for the common Q2 protocol.

These are new implementations of the published core mechanisms. Public source
revisions and deviations are recorded in experiments/q2/public_baselines.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.unsqueeze(-1).to(x.dtype)
    return (x * weight).sum(1) / weight.sum(1).clamp_min(1)


class TFNBaseline(nn.Module):
    """Tensor outer-product fusion, with packed text LSTM and A/V subnets."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.text_rnn = nn.LSTM(768, 64, batch_first=True)
        self.text_out = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 32))
        self.audio_net = nn.Sequential(nn.BatchNorm1d(74), nn.Dropout(dropout),
                                       nn.Linear(74, 32), nn.ReLU(), nn.Linear(32, 32),
                                       nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.vision_net = nn.Sequential(nn.BatchNorm1d(35), nn.Dropout(dropout),
                                        nn.Linear(35, 32), nn.ReLU(), nn.Linear(32, 32),
                                        nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Dropout(dropout), nn.Linear(33 ** 3, 128),
                                    nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.cls_head = nn.Linear(128, 3)
        self.reg_head = nn.Linear(128, 1)

    def forward(self, batch: dict) -> dict:
        valid = batch["padding_mask"]
        lengths = valid.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(batch["text"], lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, (h, _) = self.text_rnn(packed)
        t = self.text_out(h[-1])
        a = self.audio_net(masked_mean(batch["audio"], valid))
        v = self.vision_net(masked_mean(batch["vision"], valid))
        ones = lambda x: torch.cat((x.new_ones(x.shape[0], 1), x), dim=1)
        fused = torch.einsum("bi,bj,bk->bijk", ones(t), ones(a), ones(v)).flatten(1)
        z = self.fusion(fused)
        return {"classification_logits": self.cls_head(z), "regression": self.reg_head(z).squeeze(-1)}


class CrossBlock(nn.Module):
    def __init__(self, dim: int = 30, heads: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, query, memory, valid):
        q = self.norm_q(query)
        m = self.norm_kv(memory)
        attended = self.attn(q, m, m, key_padding_mask=~valid, need_weights=False)[0]
        x = query + self.drop(attended)
        return x + self.drop(self.ff(self.norm_ff(x)))


class MulTBaseline(nn.Module):
    """Six directed pairwise cross-attention streams, then per-target memory."""

    def __init__(self, dim: int = 30, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.project = nn.ModuleDict({"text": nn.Conv1d(768, dim, 1, bias=False),
                                      "audio": nn.Conv1d(74, dim, 1, bias=False),
                                      "vision": nn.Conv1d(35, dim, 1, bias=False)})
        names = ("text", "audio", "vision")
        self.cross = nn.ModuleDict({f"{target}_{source}": nn.ModuleList(
            [CrossBlock(dim, 5, dropout) for _ in range(layers)])
            for target in names for source in names if source != target})
        self.memory = nn.ModuleDict({name: nn.ModuleList(
            [CrossBlock(2 * dim, 5, dropout) for _ in range(3)]) for name in names})
        self.fuse1 = nn.Linear(6 * dim, 6 * dim)
        self.fuse2 = nn.Linear(6 * dim, 6 * dim)
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(6 * dim, 3)
        self.reg_head = nn.Linear(6 * dim, 1)
        self.dim = dim

    @staticmethod
    def positions(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        index = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        rate = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) *
                         (-math.log(10000.0) / dim))
        position = torch.empty(length, dim, device=device, dtype=dtype)
        position[:, 0::2] = torch.sin(index * rate)
        position[:, 1::2] = torch.cos(index * rate)
        return position.unsqueeze(0)

    def forward(self, batch: dict) -> dict:
        valid = batch["padding_mask"]
        streams = {name: self.project[name](batch[name].transpose(1, 2)).transpose(1, 2)
                   for name in ("text", "audio", "vision")}
        for name, x in streams.items():
            streams[name] = x * math.sqrt(self.dim) + self.positions(x.shape[1], self.dim, x.device, x.dtype)
        final = []
        for target in ("text", "audio", "vision"):
            crossed = []
            for source in ("text", "audio", "vision"):
                if source == target:
                    continue
                x = streams[target]
                for block in self.cross[f"{target}_{source}"]:
                    x = block(x, streams[source], valid)
                crossed.append(x)
            x = torch.cat(crossed, -1)
            for block in self.memory[target]:
                x = block(x, x, valid)
            last = valid.sum(1).long() - 1
            final.append(x[torch.arange(x.shape[0], device=x.device), last])
        z = torch.cat(final, -1)
        z = z + self.drop(self.fuse2(F.relu(self.fuse1(z))))
        return {"classification_logits": self.cls_head(z), "regression": self.reg_head(z).squeeze(-1)}


class MISABaseline(nn.Module):
    """Shared/private factorization, reconstruction, orthogonality and CMD."""

    def __init__(self, hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        dims = {"text": 768, "audio": 74, "vision": 35}
        self.project = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                                                         nn.LayerNorm(hidden)) for m, d in dims.items()})
        self.private = nn.ModuleDict({m: nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
                                      for m in dims})
        self.shared = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.reconstruct = nn.ModuleDict({m: nn.Linear(hidden, hidden) for m in dims})
        enc = nn.TransformerEncoderLayer(hidden, nhead=2, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.fusion = nn.Sequential(nn.Linear(hidden * 6, hidden * 3), nn.Dropout(dropout), nn.Tanh())
        self.cls_head = nn.Linear(hidden * 3, 3)
        self.reg_head = nn.Linear(hidden * 3, 1)
        self.aux: dict[str, torch.Tensor] = {}

    def forward(self, batch: dict) -> dict:
        valid = batch["padding_mask"]
        names = ("text", "audio", "vision")
        originals = [self.project[m](masked_mean(batch[m], valid)) for m in names]
        private = [self.private[m](h) for m, h in zip(names, originals)]
        shared = [self.shared(h) for h in originals]
        reconstructed = [self.reconstruct[m](p + s) for m, p, s in zip(names, private, shared)]
        encoded = self.encoder(torch.stack(private + shared, dim=1)).flatten(1)
        z = self.fusion(encoded)
        self.aux = {"private": private, "shared": shared,
                    "originals": originals, "reconstructed": reconstructed}
        return {"classification_logits": self.cls_head(z), "regression": self.reg_head(z).squeeze(-1)}

    def auxiliary_loss(self) -> dict[str, torch.Tensor]:
        p, s = self.aux["private"], self.aux["shared"]
        # DifferenceLoss: normalized cross-correlation squared, as in MISA.
        diff = torch.stack([((F.normalize(x, dim=1) * F.normalize(y, dim=1)).sum(1) ** 2).mean()
                            for x, y in zip(p, s)]).mean()
        diff = diff + torch.stack([((F.normalize(p[i], dim=1) * F.normalize(p[j], dim=1))
                                    .sum(1) ** 2).mean() for i, j in ((0, 1), (0, 2), (1, 2))]).mean()
        rec = torch.stack([F.mse_loss(x, y) for x, y in zip(self.aux["reconstructed"],
                                                              self.aux["originals"])]).mean()
        # Central moment discrepancy, moments 1..5 averaged over modality pairs.
        def cmd(x, y):
            mx, my = x.mean(0), y.mean(0)
            value = torch.linalg.vector_norm(mx - my)
            xc, yc = x - mx, y - my
            for order in range(2, 6):
                value = value + torch.linalg.vector_norm((xc ** order).mean(0) - (yc ** order).mean(0))
            return value / x.shape[1]
        sim = torch.stack([cmd(s[i], s[j]) for i, j in ((0, 1), (0, 2), (1, 2))]).mean()
        return {"diff": diff, "recon": rec, "sim": sim}


MODELS = {"TFN": TFNBaseline, "MulT": MulTBaseline, "MISA": MISABaseline}

"""Architecture adaptations of published MSA models to aligned-50 features.

All classes consume the same Q2 batch and return the common two-task contract.
These implementations do not load external pretrained weights or sentiment data.
See each experiment's adaptation note for departures from the source paper.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .public_baselines import masked_mean


DIMS = {"text": 768, "audio": 74, "vision": 35}
NAMES = tuple(DIMS)


class Heads(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.cls = nn.Linear(dim, 3)
        self.reg = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> dict:
        return {"classification_logits": self.cls(x), "regression": self.reg(x).squeeze(-1)}


def valid_last(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    index = mask.sum(1).long().clamp_min(1) - 1
    return x[torch.arange(x.size(0), device=x.device), index]


class LMF(nn.Module):
    """Modality-specific encoders and low-rank tensor-fusion factors."""
    def __init__(self, rank: int = 8):
        super().__init__()
        self.text = nn.LSTM(768, 64, batch_first=True)
        self.t_out = nn.Linear(64, 32)
        self.a_net = nn.Sequential(nn.Linear(74, 32), nn.ReLU(), nn.Linear(32, 32))
        self.v_net = nn.Sequential(nn.Linear(35, 32), nn.ReLU(), nn.Linear(32, 32))
        self.factor = nn.ParameterDict({m: nn.Parameter(torch.empty(rank, 33, 128)) for m in NAMES})
        for p in self.factor.values():
            nn.init.xavier_normal_(p)
        self.rank_weight = nn.Parameter(torch.ones(rank) / rank)
        self.bias = nn.Parameter(torch.zeros(128))
        self.heads = Heads(128)

    def forward(self, batch: dict) -> dict:
        mask = batch["padding_mask"]
        packed = nn.utils.rnn.pack_padded_sequence(batch["text"], mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.text(packed)
        values = {"text": self.t_out(h[-1]), "audio": self.a_net(masked_mean(batch["audio"], mask)),
                  "vision": self.v_net(masked_mean(batch["vision"], mask))}
        factors = [torch.einsum("bd,rdo->bro", torch.cat((v.new_ones(v.size(0), 1), v), 1), self.factor[m])
                   for m, v in values.items()]
        fused = (factors[0] * factors[1] * factors[2] * self.rank_weight[None, :, None]).sum(1) + self.bias
        return self.heads(F.relu(fused))


class MFN(nn.Module):
    """Three recurrent cells and a gated, attended cross-view memory."""
    def __init__(self, hidden: int = 32, memory: int = 64):
        super().__init__()
        self.cell = nn.ModuleDict({m: nn.LSTMCell(d, hidden) for m, d in DIMS.items()})
        cdim = 6 * hidden
        self.attention = nn.Sequential(nn.Linear(cdim, 128), nn.ReLU(), nn.Linear(128, cdim))
        self.candidate = nn.Sequential(nn.Linear(cdim, 128), nn.ReLU(), nn.Linear(128, memory), nn.Tanh())
        self.gate_old = nn.Linear(cdim + memory, memory)
        self.gate_new = nn.Linear(cdim + memory, memory)
        self.fuse = nn.Sequential(nn.Linear(3 * hidden + memory, 128), nn.ReLU())
        self.heads = Heads(128)
        self.hidden, self.memory = hidden, memory

    def forward(self, batch: dict) -> dict:
        b = batch["text"].size(0)
        mask = batch["padding_mask"]
        state = {m: (batch["text"].new_zeros(b, self.hidden), batch["text"].new_zeros(b, self.hidden)) for m in NAMES}
        memory = batch["text"].new_zeros(b, self.memory)
        for t in range(batch["text"].size(1)):
            next_state = {m: self.cell[m](batch[m][:, t], state[m]) for m in NAMES}
            cstar = torch.cat([state[m][1] for m in NAMES] + [next_state[m][1] for m in NAMES], 1)
            attended = cstar * F.softmax(self.attention(cstar), dim=1)
            gate_input = torch.cat((attended, memory), 1)
            proposed = torch.sigmoid(self.gate_old(gate_input)) * memory + \
                torch.sigmoid(self.gate_new(gate_input)) * self.candidate(attended)
            active = mask[:, t:t + 1]
            memory = torch.where(active, proposed, memory)
            state = {m: tuple(torch.where(active, nxt, old) for nxt, old in zip(next_state[m], state[m])) for m in NAMES}
        z = self.fuse(torch.cat([state[m][0] for m in NAMES] + [memory], 1))
        return self.heads(z)


class _Unimodal(nn.Module):
    def __init__(self, dims: dict[str, int] = DIMS, hidden: int = 64):
        super().__init__()
        self.project = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU())
                                      for m, d in dims.items()})

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        mask = batch["padding_mask"]
        return {m: self.project[m](masked_mean(batch[m], mask)) for m in NAMES}


class SelfMM(nn.Module):
    """Fusion plus three self-supervised unimodal sentiment branches."""
    def __init__(self):
        super().__init__()
        self.enc = _Unimodal()
        self.fusion = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.2))
        self.heads = Heads(128)
        self.unimodal = nn.ModuleDict({m: nn.Linear(64, 1) for m in NAMES})
        self.aux: dict = {}

    def forward(self, batch: dict) -> dict:
        h = self.enc(batch)
        out = self.heads(self.fusion(torch.cat([h[m] for m in NAMES], 1)))
        self.aux = {m: self.unimodal[m](h[m]).squeeze(-1) for m in NAMES}
        return out

    def auxiliary_loss(self, batch: dict, prediction: dict) -> torch.Tensor:
        # Stop-gradient fused estimate provides adaptive unimodal pseudo targets.
        target = 0.5 * batch["reg_label"] + 0.5 * prediction["regression"].detach()
        return sum(F.smooth_l1_loss(v, target) for v in self.aux.values()) / 3


def _infonce(x: torch.Tensor, y: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    logits = F.normalize(x, dim=1) @ F.normalize(y, dim=1).T / temperature
    label = torch.arange(x.size(0), device=x.device)
    return 0.5 * (F.cross_entropy(logits, label) + F.cross_entropy(logits.T, label))


class MMIM(nn.Module):
    """Hierarchical unimodal-pair and fusion-to-unimodal MI objectives."""
    def __init__(self):
        super().__init__()
        self.enc = _Unimodal(hidden=128)
        self.fusion = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.2))
        self.heads = Heads(128)
        self.aux: dict = {}

    def forward(self, batch: dict) -> dict:
        h = self.enc(batch)
        z = self.fusion(torch.cat([h[m] for m in NAMES], 1))
        self.aux = {**h, "fusion": z}
        return self.heads(z)

    def auxiliary_loss(self, batch: dict, prediction: dict) -> torch.Tensor:
        h = self.aux
        pairs = (("text", "audio"), ("text", "vision"), ("audio", "vision"))
        return (sum(_infonce(h[a], h[b]) for a, b in pairs) +
                sum(_infonce(h["fusion"], h[m]) for m in NAMES)) / 6


class MAGBERT(nn.Module):
    """MAG shift on precomputed BERT token features, then a small encoder."""
    def __init__(self, hidden: int = 128, beta_shift: float = 1.0):
        super().__init__()
        self.text_proj = nn.Linear(768, hidden)
        self.audio_proj = nn.Linear(74, hidden)
        self.vision_proj = nn.Linear(35, hidden)
        self.gate_a = nn.Linear(2 * hidden, hidden)
        self.gate_v = nn.Linear(2 * hidden, hidden)
        self.shift_a = nn.Linear(hidden, hidden)
        self.shift_v = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.heads = Heads(hidden)
        self.beta_shift = beta_shift

    def forward(self, batch: dict) -> dict:
        t, a, v = self.text_proj(batch["text"]), self.audio_proj(batch["audio"]), self.vision_proj(batch["vision"])
        shift = F.relu(self.gate_a(torch.cat((t, a), -1))) * self.shift_a(a) + \
            F.relu(self.gate_v(torch.cat((t, v), -1))) * self.shift_v(v)
        scale = (self.beta_shift * t.norm(dim=-1) / shift.norm(dim=-1).clamp_min(1e-6)).clamp(max=1).unsqueeze(-1)
        x = self.norm(t + scale * shift)
        valid = batch["padding_mask"]
        encoded = self.encoder(x, src_key_padding_mask=~valid)
        return self.heads(masked_mean(encoded, valid))


class TFRNet(nn.Module):
    """Per-modality latent reconstruction under sampled missing blocks."""
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.project = nn.ModuleDict({m: nn.Linear(d, hidden) for m, d in DIMS.items()})
        layer = nn.TransformerEncoderLayer(3 * hidden, nhead=4, dim_feedforward=384,
                                           dropout=0.1, batch_first=True)
        self.context = nn.TransformerEncoder(layer, num_layers=1)
        self.reconstruct = nn.ModuleDict({m: nn.Linear(3 * hidden, hidden) for m in NAMES})
        self.fuse = nn.Sequential(nn.Linear(3 * hidden, 128), nn.ReLU())
        self.heads = Heads(128)
        self.aux: dict = {}

    def forward(self, batch: dict) -> dict:
        valid = batch["padding_mask"]
        observed = {m: self.project[m](batch[m]) for m in NAMES}
        context = self.context(torch.cat([observed[m] for m in NAMES], -1),
                               src_key_padding_mask=~valid)
        # Predictor sees only feature values and padding. All-zero vectors are
        # an observable trigger, not proof of synthetic missingness.
        missing = torch.stack([~(batch[m] == 0).all(-1) for m in NAMES], 1)
        h = {}
        targets = {}
        for i, m in enumerate(NAMES):
            rec = self.reconstruct[m](context)
            absent = valid & ~missing[:, i]
            h[m] = torch.where(absent.unsqueeze(-1), rec, observed[m])
            clean = batch.get("clean_features", {}).get(m, batch[m])
            targets[m] = (rec, self.project[m](clean).detach(), absent)
        z = self.fuse(torch.cat([masked_mean(h[m], valid) for m in NAMES], 1))
        self.aux = targets
        return self.heads(z)

    def auxiliary_loss(self, batch: dict, prediction: dict) -> torch.Tensor:
        terms = [F.smooth_l1_loss(rec[mask], target[mask]) for rec, target, mask in self.aux.values() if mask.any()]
        return sum(terms) / len(terms) if terms else prediction["regression"].sum() * 0


class MissModal(nn.Module):
    """Shared encoders with geometric, distribution and semantic alignment."""
    def __init__(self):
        super().__init__()
        self.enc = _Unimodal(hidden=128)
        self.fusion = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.2))
        self.heads = Heads(128)
        self.last_z: torch.Tensor | None = None

    def forward(self, batch: dict) -> dict:
        h = self.enc(batch)
        self.last_z = self.fusion(torch.cat([h[m] for m in NAMES], 1))
        return self.heads(self.last_z)

    def auxiliary_loss(self, batch: dict, prediction: dict) -> torch.Tensor:
        clean = batch.get("clean_features")
        if clean is None:
            return prediction["regression"].sum() * 0
        with torch.no_grad():
            clean_batch = {**batch, **clean}
            h = self.enc(clean_batch)
            teacher = self.fusion(torch.cat([h[m] for m in NAMES], 1))
            clean_pred = self.heads(teacher)
        geometric = _infonce(self.last_z, teacher)
        distribution = (F.mse_loss(self.last_z.mean(0), teacher.mean(0)) +
                        F.mse_loss(self.last_z.std(0), teacher.std(0)))
        semantic = F.kl_div(F.log_softmax(prediction["classification_logits"], -1),
                            F.softmax(clean_pred["classification_logits"], -1), reduction="batchmean")
        return 0.1 * geometric + 0.1 * distribution + 0.1 * semantic


class MMIN(nn.Module):
    """Missing-view latent imagination with reconstruction supervision."""
    def __init__(self):
        super().__init__()
        self.enc = _Unimodal(hidden=64)
        self.imagine = nn.ModuleDict({m: nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                                      nn.Linear(128, 64)) for m in NAMES})
        self.refine = nn.ModuleDict({m: nn.Sequential(nn.Linear(192, 128), nn.ReLU(),
                                                     nn.Linear(128, 64)) for m in NAMES})
        self.cycle = nn.Linear(192, 192)
        self.fuse = nn.Sequential(nn.Linear(192, 128), nn.ReLU())
        self.heads = Heads(128)
        self.aux: dict = {}

    def forward(self, batch: dict) -> dict:
        h = self.enc(batch)
        avail = torch.stack([(batch[m] != 0).any(dim=(1, 2)) for m in NAMES], 1)
        first_stage, targets = [], {}
        for i, m in enumerate(NAMES):
            others = [h[o] for o in NAMES if o != m]
            imagined = self.imagine[m](torch.cat(others, 1))
            first_stage.append(torch.where(avail[:, i:i + 1], h[m], imagined))
            clean = batch.get("clean_features", {}).get(m, batch[m])
            clean_h = self.enc.project[m](masked_mean(clean, batch["padding_mask"])).detach()
            targets[m] = (imagined, clean_h, ~avail[:, i])
        first_joint = torch.cat(first_stage, 1)
        effective = []
        for i, m in enumerate(NAMES):
            refined = first_stage[i] + self.refine[m](first_joint)
            effective.append(torch.where(avail[:, i:i + 1], first_stage[i], refined))
            _, clean_h, absent = targets[m]
            targets[m] = (refined, clean_h, absent)
        self.aux = {"targets": targets, "cycle_input": torch.cat(effective, 1),
                    "cycle_target": torch.cat([targets[m][1] for m in NAMES], 1)}
        return self.heads(self.fuse(torch.cat(effective, 1)))

    def auxiliary_loss(self, batch: dict, prediction: dict) -> torch.Tensor:
        losses = [F.smooth_l1_loss(imagined[mask], target[mask])
                  for imagined, target, mask in self.aux["targets"].values() if mask.any()]
        rec = sum(losses) / len(losses) if losses else prediction["regression"].sum() * 0
        cycle = F.smooth_l1_loss(self.cycle(self.aux["cycle_input"]), self.aux["cycle_target"])
        return rec + 0.1 * cycle


class M3S(LMF):
    """LMF prediction backbone; missing-view meta-sampling is in the trainer."""


EXTENDED_MODELS = {"LMF": LMF, "MFN": MFN, "Self-MM": SelfMM, "MMIM": MMIM,
                   "MAG-BERT": MAGBERT, "TFR-Net": TFRNet, "MissModal": MissModal,
                   "M3S": M3S, "MMIN": MMIN}

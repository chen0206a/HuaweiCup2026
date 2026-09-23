"""B4': task-trained reliability router around an exactly frozen B0 predictor."""
from __future__ import annotations

import torch
from torch import nn

from src.data.dataset import MODALITIES
from src.models.baseline import B0Baseline, masked_mean_pool

PREDICTOR_KEYS = frozenset((*MODALITIES, "padding_mask"))


def observable_reliability_stats(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return [zero_ratio(3), longest_zero_run_ratio(3), valid_length_ratio].

    Zeros are observed feature values, not an oracle missing label. Only true
    padding_mask positions count; availability/native_zero masks are not read.
    """
    if set(batch) != PREDICTOR_KEYS:
        raise ValueError("router observation must contain only text/audio/vision/padding_mask")
    pad = batch["padding_mask"]
    if pad.ndim != 2 or pad.dtype != torch.bool or not pad.any(dim=1).all():
        raise ValueError("padding_mask must be bool [B,T] with at least one valid position")
    batch_size, steps = pad.shape
    length = pad.sum(dim=1)
    zero_ratios = []
    run_ratios = []
    for modality in MODALITIES:
        features = batch[modality]
        if features.ndim != 3 or features.shape[:2] != (batch_size, steps):
            raise ValueError(f"{modality} must have shape [B,T,D]")
        is_zero = (features == 0).all(dim=-1) & pad
        zero_ratios.append(is_zero.sum(dim=1).float() / length)
        run = torch.zeros_like(length)
        longest = torch.zeros_like(length)
        for position in range(steps):
            run = torch.where(is_zero[:, position], run + 1, 0)
            longest = torch.maximum(longest, run)
        run_ratios.append(longest.float() / length)
    return torch.stack((*zero_ratios, *run_ratios, length.float() / steps), dim=-1)


class B4ReliabilityGate(B0Baseline):
    """Joint router; all B0 prediction parameters remain fixed and in eval mode."""

    def __init__(self, hidden_dim: int = 128, fusion_dim: int = 128,
                 dropout: float = 0.1, router_dim: int = 64,
                 router_dropout: float = 0.1) -> None:
        super().__init__(hidden_dim=hidden_dim, fusion_dim=fusion_dim, dropout=dropout)
        self.router = nn.Sequential(
            nn.Linear(3 * hidden_dim + 7, router_dim),
            nn.GELU(),
            nn.Dropout(router_dropout),
            nn.Linear(router_dim, 3),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.freeze_b0()

    def freeze_b0(self) -> None:
        for module in (self.modality_projection, self.fusion,
                       self.classification_head, self.regression_head):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.freeze_b0()
        self.router.train(mode)
        return self

    def encode_observation(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if set(batch) != PREDICTOR_KEYS:
            raise ValueError("router observation must contain only text/audio/vision/padding_mask")
        with torch.no_grad():
            pooled = [self.modality_projection[m](masked_mean_pool(batch[m], batch["padding_mask"]))
                      for m in MODALITIES]
            embeddings = torch.cat(pooled, dim=-1)
            stats = observable_reliability_stats(batch)
        return embeddings, stats

    def predict_encoded(self, embeddings: torch.Tensor, stats: torch.Tensor) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 2 or embeddings.shape[1] != 3 * self.modality_projection["text"][0].out_features:
            raise ValueError("encoded B0 embeddings must have shape [B,3*hidden_dim]")
        if stats.shape != (embeddings.shape[0], 7):
            raise ValueError("observable statistics must have shape [B,7]")
        logits = self.router(torch.cat((embeddings, stats), dim=-1))
        gates = 2.0 * torch.sigmoid(logits)
        gated = (embeddings.reshape(embeddings.shape[0], 3, -1) * gates.unsqueeze(-1))
        fused = self.fusion(gated.flatten(start_dim=1))
        return {"classification_logits": self.classification_head(fused),
                "regression": self.regression_head(fused).squeeze(-1),
                "gates": gates}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings, stats = self.encode_observation(batch)
        return self.predict_encoded(embeddings, stats)

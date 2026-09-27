"""B5-P: zero-initialized pooling residuals around the actual B0 path."""
from __future__ import annotations

import torch
from torch import nn

from src.data.dataset import EXPECTED_DIMS, MODALITIES
from src.models.baseline import B0Baseline, masked_mean_pool


def masked_max_pool(features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Max over real timesteps. Native all-zero vectors remain valid values."""
    if features.ndim != 3 or padding_mask.shape != features.shape[:2]:
        raise ValueError("expected features [B,T,D] and padding_mask [B,T]")
    if padding_mask.dtype != torch.bool or not padding_mask.any(dim=1).all():
        raise ValueError("every sample needs at least one valid timestep")
    return features.masked_fill(~padding_mask.unsqueeze(-1), -torch.inf).amax(dim=1)


def masked_attention_pool(features: torch.Tensor, padding_mask: torch.Tensor,
                          scorer: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-modality scalar attention over real timesteps only."""
    if features.ndim != 3 or padding_mask.shape != features.shape[:2]:
        raise ValueError("expected features [B,T,D] and padding_mask [B,T]")
    if padding_mask.dtype != torch.bool or not padding_mask.any(dim=1).all():
        raise ValueError("every sample needs at least one valid timestep")
    scores = scorer(features).squeeze(-1).masked_fill(~padding_mask, -torch.inf)
    weights = torch.softmax(scores, dim=1)
    pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
    return pooled, weights


class B5PoolingResidual(B0Baseline):
    """B0 projected means plus projected max/attention residuals.

    B0 first masked-means raw features, then applies its modality projection.
    Here the auxiliary raw pooled vector uses that *same* projection, so the
    residual has exactly the 128-dimensional B0 representation shape.
    Gamma starts at zero, giving exact B0 inference at initialization.
    """

    def __init__(self, mode: str, hidden_dim: int = 128, fusion_dim: int = 128,
                 dropout: float = 0.1) -> None:
        if mode not in {"mean_max", "mean_attention"}:
            raise ValueError("mode must be mean_max or mean_attention")
        super().__init__(hidden_dim=hidden_dim, fusion_dim=fusion_dim, dropout=dropout)
        self.mode = mode
        self.gamma = nn.ParameterDict({m: nn.Parameter(torch.zeros(())) for m in MODALITIES})
        self.attention_scorer = (nn.ModuleDict({m: nn.Linear(EXPECTED_DIMS[m], 1)
                                                for m in MODALITIES})
                                 if mode == "mean_attention" else nn.ModuleDict())

    def pool_diagnostics(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """P2 attention weights; padding positions have exactly zero mass."""
        if self.mode != "mean_attention":
            raise ValueError("attention diagnostics are only available for P2")
        return {m: masked_attention_pool(batch[m], batch["padding_mask"],
                                         self.attention_scorer[m])[1] for m in MODALITIES}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pad = batch["padding_mask"]
        mean_representations = {
            m: self.modality_projection[m](masked_mean_pool(batch[m], pad))
            for m in MODALITIES
        }
        pooled = []
        for modality in MODALITIES:
            if self.mode == "mean_max":
                auxiliary = masked_max_pool(batch[modality], pad)
            else:
                auxiliary, _ = masked_attention_pool(
                    batch[modality], pad, self.attention_scorer[modality])
            auxiliary_representation = self.modality_projection[modality](auxiliary)
            pooled.append(mean_representations[modality] +
                          self.gamma[modality] * auxiliary_representation)
        fused = self.fusion(torch.cat(pooled, dim=-1))
        return {"classification_logits": self.classification_head(fused),
                "regression": self.regression_head(fused).squeeze(-1)}

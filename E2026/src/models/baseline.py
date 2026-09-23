"""B0: modality-specific masked means, small MLP, classification + regression."""
from __future__ import annotations

import torch
from torch import nn

from src.data.dataset import EXPECTED_DIMS, MODALITIES


def masked_mean_pool(features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Average valid timesteps; native zero vectors remain included."""
    if features.ndim != 3 or padding_mask.shape != features.shape[:2]:
        raise ValueError("expected features [B,T,D] and padding_mask [B,T]")
    weights = padding_mask.to(dtype=features.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (features * weights).sum(dim=1) / denom


class B0Baseline(nn.Module):
    def __init__(self, hidden_dim: int = 128, fusion_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.modality_projection = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for name, dim in EXPECTED_DIMS.items()
        })
        joined = hidden_dim * len(MODALITIES)
        self.fusion = nn.Sequential(
            nn.Linear(joined, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classification_head = nn.Linear(fusion_dim, 3)
        self.regression_head = nn.Linear(fusion_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pooled = [
            self.modality_projection[m](masked_mean_pool(batch[m], batch["padding_mask"]))
            for m in MODALITIES
        ]
        fused = self.fusion(torch.cat(pooled, dim=-1))
        return {
            "classification_logits": self.classification_head(fused),
            "regression": self.regression_head(fused).squeeze(-1),
        }


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_reg: float = 1.0,
) -> dict[str, torch.Tensor]:
    ce = nn.functional.cross_entropy(outputs["classification_logits"], batch["cls_label"])
    smooth_l1 = nn.functional.smooth_l1_loss(outputs["regression"], batch["reg_label"])
    total = ce + float(lambda_reg) * smooth_l1
    return {"total": total, "cross_entropy": ce, "smooth_l1": smooth_l1}

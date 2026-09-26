"""Structural pooling variants for the Q2 LTARP component ablation.

The official A4 implementation is B5PoolingResidual from pooling_residual.py.
Other variants reuse the exact B0 projection, fusion, and prediction modules;
their differences are limited to how each modality's masked temporal pools are
combined before the original fusion layer.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.data.dataset import EXPECTED_DIMS, MODALITIES
from src.models.baseline import B0Baseline, masked_mean_pool
from src.models.pooling_residual import B5PoolingResidual, masked_attention_pool


VARIANTS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")
MODALITY_VARIANT = {"A5": "text", "A6": "audio", "A7": "vision"}


class LTARPComponentVariant(B0Baseline):
    """A1/A2/A3 and modality-specific A5/A6/A7 pooling variants."""

    def __init__(self, variant: str, hidden_dim: int = 128,
                 fusion_dim: int = 128, dropout: float = 0.1) -> None:
        if variant not in {"A1", "A2", "A3", "A5", "A6", "A7"}:
            raise ValueError(f"unsupported component variant: {variant}")
        super().__init__(hidden_dim=hidden_dim, fusion_dim=fusion_dim, dropout=dropout)
        self.variant = variant
        self.active_modalities = ((MODALITY_VARIANT[variant],) if variant in MODALITY_VARIANT
                                  else MODALITIES)
        self.attention_scorer = nn.ModuleDict({
            m: nn.Linear(EXPECTED_DIMS[m], 1) for m in self.active_modalities
        })
        if variant == "A2":
            self.concat_projection = nn.ModuleDict({
                m: nn.Linear(2 * hidden_dim, hidden_dim) for m in MODALITIES
            })
        if variant in {"A5", "A6", "A7"}:
            self.gamma = nn.ParameterDict({m: nn.Parameter(torch.zeros(()))
                                            for m in self.active_modalities})

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        padding = batch["padding_mask"]
        representations = []
        for modality in MODALITIES:
            mean = self.modality_projection[modality](
                masked_mean_pool(batch[modality], padding)
            )
            if modality not in self.active_modalities:
                representations.append(mean)
                continue
            attention_pool, _ = masked_attention_pool(
                batch[modality], padding, self.attention_scorer[modality]
            )
            attention = self.modality_projection[modality](attention_pool)
            if self.variant == "A1":
                representation = attention
            elif self.variant == "A2":
                representation = self.concat_projection[modality](
                    torch.cat((mean, attention), dim=-1)
                )
            elif self.variant in {"A3", "A4"}:
                representation = mean + attention
            else:  # A5/A6/A7: selected modality only, with learnable zero-init gamma.
                representation = mean + self.gamma[modality] * attention
            representations.append(representation)
        fused = self.fusion(torch.cat(representations, dim=-1))
        return {
            "classification_logits": self.classification_head(fused),
            "regression": self.regression_head(fused).squeeze(-1),
        }


def build_component_variant(variant: str, **model_kwargs) -> nn.Module:
    """Create an exact formal A0/A4 or one of the additional component variants."""
    if variant == "A0":
        return B0Baseline(**model_kwargs)
    if variant == "A4":
        return B5PoolingResidual("mean_attention", **model_kwargs)
    return LTARPComponentVariant(variant, **model_kwargs)


def freeze_b0_backbone(model: nn.Module) -> None:
    """Freeze the projection/fusion/heads as in formal LTARP and keep dropout off."""
    for name, parameter in model.named_parameters():
        if name.startswith(("modality_projection.", "fusion.",
                            "classification_head.", "regression_head.")):
            parameter.requires_grad_(False)
    keep_b0_eval(model)


def keep_b0_eval(model: nn.Module) -> None:
    """Call after model.train() to preserve formal LTARP's frozen B0 mode."""
    model.modality_projection.eval()
    model.fusion.eval()
    model.classification_head.eval()
    model.regression_head.eval()


def expected_missing_state_keys(variant: str) -> tuple[str, ...]:
    if variant == "A0":
        return ()
    if variant in {"A1", "A3"}:
        return ("attention_scorer.",)
    if variant == "A2":
        return ("attention_scorer.", "concat_projection.")
    if variant == "A4":
        return ("attention_scorer.", "gamma.")
    if variant in {"A5", "A6", "A7"}:
        return ("attention_scorer.", "gamma.")
    raise ValueError(variant)

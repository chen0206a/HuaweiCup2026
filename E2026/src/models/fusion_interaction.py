"""B5-F: low-rank pairwise interaction added to an exactly frozen B0 fusion."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.data.dataset import MODALITIES
from src.models.baseline import B0Baseline, masked_mean_pool

PREDICTOR_KEYS = frozenset((*MODALITIES, "padding_mask"))
PAIR_NAMES = ("TA", "TV", "AV")


class B5FusionInteraction(B0Baseline):
    def __init__(self, rank: int = 16, hidden_dim: int = 128,
                 fusion_dim: int = 128, dropout: float = 0.1) -> None:
        if rank != 16:
            raise ValueError("B5-F screening fixes rank=16")
        super().__init__(hidden_dim=hidden_dim, fusion_dim=fusion_dim, dropout=dropout)
        self.rank = rank
        self.interaction_projection = nn.ModuleDict({
            name: nn.Linear(hidden_dim, rank, bias=False) for name in MODALITIES
        })
        for projection in self.interaction_projection.values():
            nn.init.xavier_uniform_(projection.weight)
        self.interaction_output = nn.Linear(3 * rank, fusion_dim, bias=False)
        nn.init.zeros_(self.interaction_output.weight)
        self.freeze_b0()

    def freeze_b0(self) -> None:
        for module in (self.modality_projection, self.fusion,
                       self.classification_head, self.regression_head):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.freeze_b0()  # Dropout in the frozen B0 must remain disabled.
        self.interaction_projection.train(mode)
        self.interaction_output.train(mode)
        return self

    @torch.no_grad()
    def encode_b0(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the original projected means [B,3,128] and B0 fusion [B,128]."""
        if set(batch) != PREDICTOR_KEYS:
            raise ValueError("F1 predictor only accepts text/audio/vision/padding_mask")
        representations = [
            self.modality_projection[m](masked_mean_pool(batch[m], batch["padding_mask"]))
            for m in MODALITIES
        ]
        joined = torch.cat(representations, dim=-1)
        return torch.stack(representations, dim=1), self.fusion(joined)

    def interaction(self, representations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if representations.ndim != 3 or representations.shape[1:] != (3, self.interaction_projection["text"].in_features):
            raise ValueError("representations must have shape [B,3,hidden_dim]")
        u = [self.interaction_projection[m](representations[:, i, :])
             for i, m in enumerate(MODALITIES)]
        pairs = torch.cat((u[0] * u[1], u[0] * u[2], u[1] * u[2]), dim=-1)
        return self.interaction_output(pairs), pairs

    def pair_contributions(self, pairwise: torch.Tensor) -> dict[str, torch.Tensor]:
        if pairwise.ndim != 2 or pairwise.shape[1] != 3 * self.rank:
            raise ValueError("pairwise interaction must be [B,48]")
        weight = self.interaction_output.weight
        return {name: F.linear(pairwise[:, i * self.rank:(i + 1) * self.rank],
                               weight[:, i * self.rank:(i + 1) * self.rank])
                for i, name in enumerate(PAIR_NAMES)}

    def predict_encoded(self, representations: torch.Tensor, z_b0: torch.Tensor) -> dict[str, torch.Tensor]:
        if z_b0.ndim != 2 or z_b0.shape[0] != representations.shape[0] or z_b0.shape[1] != self.interaction_output.out_features:
            raise ValueError("frozen B0 fusion representation has wrong shape")
        delta_z, pairwise = self.interaction(representations)
        z = z_b0 + delta_z
        return {"classification_logits": self.classification_head(z),
                "regression": self.regression_head(z).squeeze(-1),
                "delta_z": delta_z, "z_b0": z_b0, "pairwise": pairwise}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        representations, z_b0 = self.encode_b0(batch)
        return self.predict_encoded(representations, z_b0)

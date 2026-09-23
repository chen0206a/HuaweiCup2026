"""B1: independent temporal Transformer encoders for each modality."""
from __future__ import annotations

import math

import torch
from torch import nn

from src.data.dataset import EXPECTED_DIMS, MODALITIES
from src.models.baseline import masked_mean_pool


class SinusoidalPositionEncoding(nn.Module):
    """Fixed position signal so self-attention can represent temporal order."""

    def __init__(self, hidden_dim: int, max_len: int = 50) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"sequence length {x.shape[1]} exceeds positional table")
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class B1TemporalEncoder(nn.Module):
    """Separate 2-layer temporal encoder per modality, with masked mean pooling.

    Padding is passed as a key-padding mask to every Transformer encoder. The
    mask excludes padding from valid queries' key/value sets; masked mean pooling
    excludes padded output positions from the fused representation. Native-zero
    feature vectors at valid positions remain included.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_dim: int = 128,
        feedforward_dim: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.input_projection = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim) for name, dim in EXPECTED_DIMS.items()
        })
        self.position = SinusoidalPositionEncoding(hidden_dim, max_len=50)
        self.encoders = nn.ModuleDict({
            name: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
            for name in MODALITIES
        })
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(MODALITIES), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classification_head = nn.Linear(fusion_dim, 3)
        self.regression_head = nn.Linear(fusion_dim, 1)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        padding_mask = batch["padding_mask"].bool()
        if not padding_mask.any(dim=1).all():
            raise ValueError("every sequence must contain at least one valid timestep")
        key_padding_mask = ~padding_mask
        encoded: dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            x = self.input_projection[modality](batch[modality])
            x = self.position(x)
            x = self.encoders[modality](x, src_key_padding_mask=key_padding_mask)
            # Padded queries never enter the pooled sequence representation.
            encoded[modality] = x
        return encoded

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encode(batch)
        pooled = [masked_mean_pool(encoded[m], batch["padding_mask"]) for m in MODALITIES]
        fused = self.fusion(torch.cat(pooled, dim=-1))
        return {
            "classification_logits": self.classification_head(fused),
            "regression": self.regression_head(fused).squeeze(-1),
        }

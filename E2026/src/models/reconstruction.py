"""B3: same-position cross-modal latent reconstruction on the B0 backbone."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.data.dataset import MODALITIES
from src.models.baseline import B0Baseline, masked_mean_pool


class B3LatentReconstruction(B0Baseline):
    """Keep B0 projections/fusion/heads; add independent light reconstructors.

    For B0's affine Linear, mean(Linear(x)) == Linear(mean(x)) on valid
    timesteps. LayerNorm, GELU and Dropout remain after pooling. Synthetic
    availability is never a predictor input; zero_trigger is observed from the
    current raw features and excludes padding. Native zeros may also trigger.
    """

    def __init__(self, hidden_dim: int = 128, fusion_dim: int = 128,
                 dropout: float = 0.1, recon_hidden_dim: int = 128) -> None:
        super().__init__(hidden_dim=hidden_dim, fusion_dim=fusion_dim, dropout=dropout)
        self.reconstructors = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(2 * hidden_dim, recon_hidden_dim),
                nn.GELU(),
                nn.Linear(recon_hidden_dim, hidden_dim),
            ) for target in MODALITIES
        })

    def timestep_latents(self, batch: dict) -> dict[str, torch.Tensor]:
        return {m: self.modality_projection[m][0](batch[m]) for m in MODALITIES}

    def full_targets(self, complete_batch: dict) -> dict[str, torch.Tensor]:
        """Detached B0 affine latents from the pre-augmentation full features."""
        return {m: latent.detach() for m, latent in self.timestep_latents(complete_batch).items()}

    def forward(self, batch: dict, *, reconstruction_enabled: bool = True,
                return_aux: bool = False) -> dict[str, torch.Tensor | dict]:
        pad = batch["padding_mask"].bool()
        if not pad.any(dim=1).all():
            raise ValueError("every sample needs a valid timestep")
        latents = self.timestep_latents(batch)
        zero_trigger = {m: pad & torch.all(batch[m] == 0, dim=-1) for m in MODALITIES}
        reconstructed: dict[str, torch.Tensor] = {}
        sources: dict[str, torch.Tensor] = {}
        effective = latents
        if reconstruction_enabled:
            # A zero source must not contribute its affine projection bias.
            sources = {m: latents[m].masked_fill(zero_trigger[m].unsqueeze(-1), 0.0)
                       for m in MODALITIES}
            effective = {}
            for target in MODALITIES:
                others = [m for m in MODALITIES if m != target]
                estimate = self.reconstructors[target](
                    torch.cat([sources[m] for m in others], dim=-1)
                ).masked_fill(~pad.unsqueeze(-1), 0.0)
                reconstructed[target] = estimate
                effective[target] = torch.where(zero_trigger[target].unsqueeze(-1),
                                                estimate, latents[target])
        pooled = [
            self.modality_projection[m][1:](masked_mean_pool(effective[m], pad))
            for m in MODALITIES
        ]
        fused = self.fusion(torch.cat(pooled, dim=-1))
        outputs: dict[str, torch.Tensor | dict] = {
            "classification_logits": self.classification_head(fused),
            "regression": self.regression_head(fused).squeeze(-1),
        }
        if return_aux:
            outputs["zero_trigger"] = zero_trigger
            outputs["reconstructed"] = reconstructed
            outputs["latents"] = latents
            outputs["source_latents"] = sources
            outputs["effective_latents"] = effective
        return outputs


def reconstruction_losses(model: B3LatentReconstruction, outputs: dict,
                          full_batch: dict, masked_batch: dict) -> dict[str, torch.Tensor | dict]:
    """Mean each modality's timestep/feature loss, then average three modalities.

    Missing supervision uses synthetic availability only here, never in the
    predictor. Targets are stop-gradient affine latents of original X_full.
    """
    if not outputs["reconstructed"]:
        raise ValueError("reconstruction must be enabled for its loss")
    targets = model.full_targets(full_batch)
    pad = masked_batch["padding_mask"].bool()
    availability = masked_batch["availability_mask"].bool()
    losses_rec, losses_id = {}, {}
    for index, modality in enumerate(MODALITIES):
        missing = pad & ~availability[:, index]
        observed = pad & availability[:, index]
        estimate = outputs["reconstructed"][modality]
        target = targets[modality]
        # Graph-connected zero permits batches with no mask for a modality.
        losses_rec[modality] = (F.smooth_l1_loss(estimate[missing], target[missing])
                                if missing.any() else estimate.sum() * 0.0)
        losses_id[modality] = (F.smooth_l1_loss(estimate[observed], target[observed])
                               if observed.any() else estimate.sum() * 0.0)
    return {"reconstruction": torch.stack(tuple(losses_rec.values())).mean(),
            "identity": torch.stack(tuple(losses_id.values())).mean(),
            "reconstruction_by_modality": losses_rec,
            "identity_by_modality": losses_id,
            "targets": targets}

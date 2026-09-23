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
                alpha: float = 1.0, return_aux: bool = False) -> dict[str, torch.Tensor | dict]:
        pad = batch["padding_mask"].bool()
        if not pad.any(dim=1).all():
            raise ValueError("every sample needs a valid timestep")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        latents = self.timestep_latents(batch)
        zero_trigger = {m: pad & torch.all(batch[m] == 0, dim=-1) for m in MODALITIES}
        reconstructed: dict[str, torch.Tensor] = {}
        sources: dict[str, torch.Tensor] = {}
        effective = latents
        if not reconstruction_enabled or float(alpha) == 0.0:
            # Use the exact original operation order for an exact B0 identity.
            pooled = [self.modality_projection[m](masked_mean_pool(batch[m], pad))
                      for m in MODALITIES]
        else:
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
                blended = (1.0 - float(alpha)) * latents[target] + float(alpha) * estimate
                effective[target] = torch.where(zero_trigger[target].unsqueeze(-1),
                                                blended, latents[target])
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

    def forward_alpha_grid(self, batch: dict, alphas: tuple[float, ...] | list[float]) -> dict[str, dict]:
        """Share latent and reconstructor forwards across a fixed alpha grid.

        The raw B0 path is retained verbatim for alpha=0, preserving exact
        checkpoint predictions. For alpha>0, H_base and H_hat are each formed
        once for this batch/scenario, then reused to predict every blend.
        """
        values = tuple(float(alpha) for alpha in alphas)
        if not values or any(not 0.0 <= alpha <= 1.0 for alpha in values):
            raise ValueError("alphas must be a non-empty sequence in [0,1]")
        if len(set(values)) != len(values):
            raise ValueError("alpha grid values must be unique")
        pad = batch["padding_mask"].bool()
        if not pad.any(dim=1).all():
            raise ValueError("every sample needs a valid timestep")
        latents = self.timestep_latents(batch)
        zero_trigger = {m: pad & torch.all(batch[m] == 0, dim=-1) for m in MODALITIES}
        reconstructed: dict[str, torch.Tensor] = {}
        if any(alpha > 0.0 for alpha in values):
            sources = {m: latents[m].masked_fill(zero_trigger[m].unsqueeze(-1), 0.0)
                       for m in MODALITIES}
            for target in MODALITIES:
                others = [m for m in MODALITIES if m != target]
                reconstructed[target] = self.reconstructors[target](
                    torch.cat([sources[m] for m in others], dim=-1)
                ).masked_fill(~pad.unsqueeze(-1), 0.0)

        outputs = {}
        for alpha in values:
            if alpha == 0.0:
                pooled = [self.modality_projection[m](masked_mean_pool(batch[m], pad))
                          for m in MODALITIES]
            else:
                pooled = []
                for modality in MODALITIES:
                    blended = (1.0 - alpha) * latents[modality] + alpha * reconstructed[modality]
                    effective = torch.where(zero_trigger[modality].unsqueeze(-1),
                                            blended, latents[modality])
                    pooled.append(self.modality_projection[modality][1:](
                        masked_mean_pool(effective, pad)))
            fused = self.fusion(torch.cat(pooled, dim=-1))
            outputs[str(alpha)] = {
                "classification_logits": self.classification_head(fused),
                "regression": self.regression_head(fused).squeeze(-1),
            }
        return outputs


class B31FrozenBackbone(B3LatentReconstruction):
    """B3.1: only reconstructors are trainable; B0 acts as a fixed function."""

    def __init__(self, hidden_dim: int = 128, fusion_dim: int = 128,
                 dropout: float = 0.1, recon_hidden_dim: int = 128) -> None:
        super().__init__(hidden_dim, fusion_dim, dropout, recon_hidden_dim)
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for module in (self.modality_projection, self.fusion,
                       self.classification_head, self.regression_head):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen B0 dropout disabled; only reconstructors enter train mode.
        for module in (self.modality_projection, self.fusion,
                       self.classification_head, self.regression_head):
            module.eval()
        self.reconstructors.train(mode)
        return self


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

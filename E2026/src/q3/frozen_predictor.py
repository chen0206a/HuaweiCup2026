"""Hash-checked, inference-only B5-P2 predictor; no training path."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from src.data.dataset import EXPECTED_DIMS, MODALITIES, TIME_STEPS
from src.models.pooling_residual import B5PoolingResidual


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predictor_inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Validate source semantics and convert features to the locked float32 interface."""
    pad = batch["padding_mask"]
    if not isinstance(pad, torch.Tensor) or pad.dtype != torch.bool or pad.ndim != 2:
        raise ValueError("padding_mask must be bool [B,50]")
    if pad.shape[1] != TIME_STEPS or not pad.any(dim=1).all():
        raise ValueError("padding_mask must have 50 steps and a nonempty prefix")
    if (pad[:, 1:] & ~pad[:, :-1]).any():
        raise ValueError("padding_mask must be a true prefix")
    result = {"padding_mask": pad.to(device)}
    for modality in MODALITIES:
        value = batch[modality]
        expected = (*pad.shape, EXPECTED_DIMS[modality])
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"{modality} must have shape {expected}")
        value = value.to(device=device, dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"{modality} contains nonfinite features")
        result[modality] = value
    return result


class FrozenP2Predictor:
    def __init__(self, checkpoint: str | Path, expected_sha256: str, seed: int,
                 device: str | torch.device = "cpu") -> None:
        self.checkpoint = Path(checkpoint)
        self.sha256 = sha256_file(self.checkpoint)
        if self.sha256 != expected_sha256:
            raise RuntimeError(f"checkpoint SHA256 mismatch: {self.sha256}")
        self.device = torch.device(device)
        # The verified local Q2 checkpoint contains config/state metadata, not just tensors.
        source = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if source.get("mode") != "mean_attention":
            raise RuntimeError("checkpoint is not P2 mean_attention")
        cfg = source.get("config", {})
        recorded_seed = source.get("seed", cfg.get("training", {}).get("seed"))
        if recorded_seed != seed:
            raise RuntimeError("checkpoint training seed mismatch")
        if cfg.get("preprocessing", {}).get("normalization") != "none":
            raise RuntimeError("checkpoint preprocessing is not normalization=none")
        model_cfg = cfg.get("model", {})
        self.model = B5PoolingResidual("mean_attention", **model_cfg).to(self.device)
        self.model.load_state_dict(source["model_state_dict"], strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
        if any(parameter.dtype != torch.float32 for parameter in self.model.parameters()):
            raise RuntimeError("predictor parameters are not float32")
        self.seed = seed

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = predictor_inputs(batch, self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(values)
        if output["classification_logits"].dtype != torch.float32 or output["regression"].dtype != torch.float32:
            raise RuntimeError("P2 output is not float32")
        if not all(torch.isfinite(value).all() for value in output.values()):
            raise RuntimeError("P2 produced nonfinite output")
        return output

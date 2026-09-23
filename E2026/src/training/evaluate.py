"""Prediction collection and full/vision_all_zero evaluation."""
from __future__ import annotations

import torch

from src.utils.metrics import compute_metrics


@torch.no_grad()
def evaluate_loader(
    model, loader, device: torch.device, *, lambda_reg: float = 1.0,
    class_weights: torch.Tensor | None = None,
) -> dict:
    model.eval()
    cls_true, cls_pred, reg_true, reg_pred, vision_zero = [], [], [], [], []
    total_loss = 0.0
    total_n = 0
    ce_sum = 0.0
    reg_sum = 0.0
    for batch in loader:
        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        outputs = model(moved)
        from src.models.baseline import multitask_loss
        losses = multitask_loss(
            outputs, moved, lambda_reg=lambda_reg, class_weights=class_weights
        )
        n = moved["cls_label"].shape[0]
        total_n += n
        total_loss += losses["total"].item() * n
        ce_sum += losses["cross_entropy"].item() * n
        reg_sum += losses["smooth_l1"].item() * n
        cls_true.extend(moved["cls_label"].cpu().tolist())
        cls_pred.extend(outputs["classification_logits"].argmax(dim=-1).cpu().tolist())
        reg_true.extend(moved["reg_label"].cpu().tolist())
        reg_pred.extend(outputs["regression"].cpu().tolist())
        vision_zero.extend(moved["vision_all_zero"].cpu().tolist())

    result = compute_metrics(cls_true, cls_pred, reg_true, reg_pred)
    result["loss"] = {
        "total": total_loss / max(total_n, 1),
        "cross_entropy": ce_sum / max(total_n, 1),
        "smooth_l1": reg_sum / max(total_n, 1),
    }
    subset = [i for i, is_zero in enumerate(vision_zero) if is_zero]
    result["vision_all_zero_count"] = len(subset)
    result["vision_all_zero_metrics"] = (
        compute_metrics(
            [cls_true[i] for i in subset], [cls_pred[i] for i in subset],
            [reg_true[i] for i in subset], [reg_pred[i] for i in subset],
        )
        if subset else None
    )
    return result

import torch
import torch.nn.functional as F

from src.models.baseline import B0Baseline, masked_mean_pool, multitask_loss


def test_masked_mean_includes_native_zero_and_ignores_padding():
    x = torch.tensor([[[2.0], [0.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(masked_mean_pool(x, mask), torch.tensor([[1.0]]))


def test_forward_backward_shapes():
    batch = {
        "text": torch.randn(2, 50, 768), "audio": torch.randn(2, 50, 74),
        "vision": torch.randn(2, 50, 35), "padding_mask": torch.ones(2, 50, dtype=torch.bool),
        "cls_label": torch.tensor([0, 2]), "reg_label": torch.tensor([-1.0, 1.0]),
    }
    model = B0Baseline()
    output = model(batch)
    assert output["classification_logits"].shape == (2, 3)
    assert output["regression"].shape == (2,)
    multitask_loss(output, batch)["total"].backward()
    assert all(param.grad is not None for param in model.parameters())


def test_weighted_ce_changes_only_classification_loss():
    outputs = {
        "classification_logits": torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]),
        "regression": torch.tensor([0.25, -0.25]),
    }
    batch = {"cls_label": torch.tensor([0, 1]), "reg_label": torch.tensor([0.0, 0.0])}
    weights = torch.tensor([0.5, 1.5, 1.0])
    plain = multitask_loss(outputs, batch, lambda_reg=1.0)
    weighted = multitask_loss(outputs, batch, lambda_reg=1.0, class_weights=weights)
    expected = F.cross_entropy(outputs["classification_logits"], batch["cls_label"], weight=weights)
    assert torch.allclose(weighted["cross_entropy"], expected)
    assert torch.equal(weighted["smooth_l1"], plain["smooth_l1"])

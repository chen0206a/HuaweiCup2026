import torch

from src.models.temporal import B1TemporalEncoder


def make_batch(batch_size=2):
    mask = torch.zeros(batch_size, 50, dtype=torch.bool)
    mask[:, :6] = True
    return {
        "text": torch.randn(batch_size, 50, 768),
        "audio": torch.randn(batch_size, 50, 74),
        "vision": torch.randn(batch_size, 50, 35),
        "padding_mask": mask,
    }


def test_padding_values_do_not_change_valid_temporal_predictions():
    torch.manual_seed(3)
    model = B1TemporalEncoder(dropout=0.0).eval()
    batch = make_batch()
    with torch.no_grad():
        baseline = model(batch)
        changed = {key: value.clone() for key, value in batch.items()}
        for modality in ("text", "audio", "vision"):
            changed[modality][:, 6:, :] = torch.randn_like(changed[modality][:, 6:, :]) * 1000
        altered = model(changed)
    assert torch.allclose(baseline["classification_logits"], altered["classification_logits"], atol=1e-6)
    assert torch.allclose(baseline["regression"], altered["regression"], atol=1e-6)


def test_temporal_forward_backward_and_parameterized_heads():
    model = B1TemporalEncoder(dropout=0.0)
    batch = make_batch()
    outputs = model(batch)
    assert outputs["classification_logits"].shape == (2, 3)
    assert outputs["regression"].shape == (2,)
    loss = outputs["classification_logits"].square().mean() + outputs["regression"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

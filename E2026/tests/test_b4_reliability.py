"""Focused B4' checks for observable statistics and frozen B0 identity."""
from __future__ import annotations

import pytest
import torch

from src.models.baseline import B0Baseline
from src.models.reliability_gate import B4ReliabilityGate, observable_reliability_stats


def small_observation(device: torch.device) -> dict:
    pad = torch.tensor([[True, True, True, True, False]], device=device)
    text = torch.ones(1, 5, 768, device=device)
    audio = torch.ones(1, 5, 74, device=device)
    vision = torch.ones(1, 5, 35, device=device)
    text[:, (1, 3), :] = 0       # two scattered native zeros
    audio[:, (1, 2), :] = 0      # one continuous two-step run
    vision[:, :4, :] = 0         # all valid positions are zero
    # The padded position is zero in every modality and must not count.
    text[:, 4, :] = audio[:, 4, :] = vision[:, 4, :] = 0
    return {"text": text, "audio": audio, "vision": vision, "padding_mask": pad}


def test_zero_ratios_runs_and_padding() -> None:
    batch = small_observation(torch.device("cpu"))
    stats = observable_reliability_stats(batch)
    expected = torch.tensor([[0.5, 0.5, 1.0, 0.25, 0.5, 1.0, 0.8]])
    torch.testing.assert_close(stats, expected, rtol=0, atol=0)
    for modality in ("text", "audio", "vision"):
        batch[modality][:, 4, :] = 7.0
    torch.testing.assert_close(observable_reliability_stats(batch), expected, rtol=0, atol=0)


def test_identity_router_only_input_and_cached_forward() -> None:
    torch.manual_seed(7)
    base = B0Baseline(hidden_dim=8, fusion_dim=8, dropout=0.1).eval()
    model = B4ReliabilityGate(hidden_dim=8, fusion_dim=8, dropout=0.1, router_dim=4).eval()
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("router.") for key in incompatible.missing_keys)
    batch = small_observation(torch.device("cpu"))
    expected = base(batch)
    actual = model(batch)
    assert torch.equal(actual["gates"], torch.ones_like(actual["gates"]))
    assert torch.equal(actual["classification_logits"], expected["classification_logits"])
    assert torch.equal(actual["regression"], expected["regression"])
    embeddings, stats = model.encode_observation(batch)
    cached = model.predict_encoded(embeddings, stats)
    assert all(torch.equal(actual[key], cached[key]) for key in actual)
    with pytest.raises(ValueError, match="only text/audio/vision/padding_mask"):
        model({**batch, "availability_mask": torch.ones(1, 3, 5, dtype=torch.bool)})


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_forward_backward_frozen_parameters_and_checkpoint(device_name: str, tmp_path) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    model = B4ReliabilityGate(hidden_dim=8, fusion_dim=8, router_dim=4).to(device)
    batch = small_observation(device)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()
              if not key.startswith("router.")}
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=0.001)
    model.train()
    outputs = model(batch)
    loss = outputs["classification_logits"].square().mean() + outputs["regression"].square().mean()
    loss.backward()
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("router."))
    assert any(p.grad is not None for p in model.router.parameters())
    optimizer.step()
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())
    model.eval()
    path = tmp_path / "router.pt"
    torch.save(model.state_dict(), path)
    restored = B4ReliabilityGate(hidden_dim=8, fusion_dim=8, router_dim=4).to(device).eval()
    restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    old, new = model(batch), restored(batch)
    assert all(torch.equal(old[key], new[key]) for key in old)

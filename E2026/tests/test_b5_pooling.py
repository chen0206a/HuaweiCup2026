"""B5-P identity, padding, short sequence, zero-content and device checks."""
from __future__ import annotations

import pytest
import torch

from src.models.baseline import B0Baseline, masked_mean_pool
from src.models.pooling_residual import (B5PoolingResidual, masked_attention_pool,
                                         masked_max_pool)


def observation(device: torch.device) -> dict[str, torch.Tensor]:
    mask = torch.tensor([[True, False, False, False],
                         [True, True, True, False]], device=device)
    batch = {"padding_mask": mask}
    for modality, dim in (("text", 768), ("audio", 74), ("vision", 35)):
        features = torch.randn(2, 4, dim, device=device)
        features[0, 1:, :] = 10_000  # padding must not enter any pooling.
        features[1, -1, :] = -10_000
        batch[modality] = features
    batch["vision"][0, 0, :] = 0  # all-zero valid vision sequence must survive.
    return batch


def test_masked_max_and_attention_exclude_padding() -> None:
    batch = observation(torch.device("cpu"))
    mask = batch["padding_mask"]
    maximum = masked_max_pool(batch["text"], mask)
    torch.testing.assert_close(maximum[0], batch["text"][0, 0], rtol=0, atol=0)
    torch.testing.assert_close(maximum[1], batch["text"][1, :3].amax(dim=0), rtol=0, atol=0)
    torch.testing.assert_close(masked_max_pool(batch["vision"], mask)[0],
                               torch.zeros(35), rtol=0, atol=0)
    scorer = torch.nn.Linear(768, 1)
    pooled, weights = masked_attention_pool(batch["text"], mask, scorer)
    assert torch.equal(weights[~mask], torch.zeros_like(weights[~mask]))
    torch.testing.assert_close(weights.sum(1), torch.ones(2), rtol=0, atol=1e-7)
    torch.testing.assert_close(pooled[0], batch["text"][0, 0], rtol=0, atol=0)
    assert torch.isfinite(pooled).all()


def test_p0_and_zero_gamma_identity() -> None:
    torch.manual_seed(42)
    source = B0Baseline(hidden_dim=8, fusion_dim=8).eval()
    batch = observation(torch.device("cpu"))
    # P0 is the unmodified B0 forward with the same state and same masked mean.
    for modality in ("text", "audio", "vision"):
        torch.testing.assert_close(masked_mean_pool(batch[modality], batch["padding_mask"])[0],
                                   batch[modality][0, 0], rtol=0, atol=0)
    expected = source(batch)
    for mode in ("mean_max", "mean_attention"):
        candidate = B5PoolingResidual(mode, hidden_dim=8, fusion_dim=8).eval()
        loaded = candidate.load_state_dict(source.state_dict(), strict=False)
        assert not loaded.unexpected_keys
        assert all(k.startswith(("gamma.", "attention_scorer.")) for k in loaded.missing_keys)
        actual = candidate(batch)
        assert all(torch.equal(actual[k], expected[k]) for k in expected)
        assert all(candidate.gamma[m].item() == 0 for m in ("text", "audio", "vision"))
        if mode == "mean_attention":
            for weights in candidate.pool_diagnostics(batch).values():
                assert torch.equal(weights[~batch["padding_mask"]],
                                   torch.zeros_like(weights[~batch["padding_mask"]]))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["mean_max", "mean_attention"])
def test_forward_backward_and_checkpoint(mode: str, device_name: str, tmp_path) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    batch = observation(device)
    model = B5PoolingResidual(mode, hidden_dim=8, fusion_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    output = model(batch)
    loss = output["classification_logits"].square().mean() + output["regression"].square().mean()
    loss.backward()
    assert all(torch.isfinite(g.grad).all() for g in model.gamma.values())
    optimizer.step()
    model.eval()
    expected = model(batch)
    checkpoint = tmp_path / f"{mode}_{device_name}.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = B5PoolingResidual(mode, hidden_dim=8, fusion_dim=8).to(device).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    actual = restored(batch)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)

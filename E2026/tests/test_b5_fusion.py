"""B5-F identity, frozen eval, gradients, shapes and checkpoint checks."""
from __future__ import annotations

import pytest
import torch

from src.models.baseline import B0Baseline
from src.models.fusion_interaction import B5FusionInteraction


def example(device: torch.device) -> dict[str, torch.Tensor]:
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]], device=device)
    return {"padding_mask": mask,
            "text": torch.randn(2, 4, 768, device=device),
            "audio": torch.randn(2, 4, 74, device=device),
            "vision": torch.randn(2, 4, 35, device=device)}


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_identity_gradients_frozen_dropout_and_roundtrip(device_name: str, tmp_path) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    device = torch.device(device_name)
    base = B0Baseline(hidden_dim=8, fusion_dim=8).to(device).eval()
    model = B5FusionInteraction(hidden_dim=8, fusion_dim=8).to(device).eval()
    mismatch = model.load_state_dict(base.state_dict(), strict=False)
    assert not mismatch.unexpected_keys
    assert all(k.startswith(("interaction_projection.", "interaction_output."))
               for k in mismatch.missing_keys)
    assert torch.count_nonzero(model.interaction_output.weight) == 0
    assert all(torch.count_nonzero(p.weight) > 0 for p in model.interaction_projection.values())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3 * 8 * 16 + 3 * 16 * 8
    batch = example(device)
    reference, new = base(batch), model(batch)
    assert all(torch.equal(reference[k], new[k]) for k in reference)
    assert new["delta_z"].shape == new["z_b0"].shape == (2, 8)
    assert new["pairwise"].shape == (2, 48)
    h, z = model.encode_b0(batch)
    cached = model.predict_encoded(h, z)
    assert all(torch.equal(new[k], cached[k]) for k in new)
    with pytest.raises(ValueError, match="only accepts"):
        model({**batch, "availability_mask": torch.ones(2, 3, 4, dtype=torch.bool, device=device)})
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()
             if not k.startswith(("interaction_projection.", "interaction_output."))}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    model.train()
    assert not model.modality_projection.training and not model.fusion.training
    assert all(not d.training for d in model.modules() if isinstance(d, torch.nn.Dropout))
    out = model(batch)
    loss = out["classification_logits"].square().mean() + out["regression"].square().mean()
    loss.backward()
    assert model.interaction_output.weight.grad is not None
    assert torch.count_nonzero(model.interaction_output.weight.grad) > 0
    assert all(torch.count_nonzero(p.weight.grad) == 0 for p in model.interaction_projection.values())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    loss = out["classification_logits"].square().mean() + out["regression"].square().mean()
    loss.backward()
    assert all(p.weight.grad is not None and torch.count_nonzero(p.weight.grad) > 0
               for p in model.interaction_projection.values())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in saved.items())
    model.eval()
    path = tmp_path / "f1.pt"
    torch.save(model.state_dict(), path)
    restored = B5FusionInteraction(hidden_dim=8, fusion_dim=8).to(device).eval()
    restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    old, loaded = model(batch), restored(batch)
    assert all(torch.equal(old[k], loaded[k]) for k in old)

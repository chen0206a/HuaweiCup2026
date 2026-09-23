from __future__ import annotations

import pytest
import torch

from src.data.block_mask import apply_blocks, predictor_inputs
from src.data.dataset import MODALITIES
from src.models.baseline import B0Baseline
from src.models.reconstruction import B3LatentReconstruction, reconstruction_losses


def batch(device="cpu"):
    torch.manual_seed(7)
    b, t = 3, 8
    pad = torch.arange(t).unsqueeze(0) < torch.tensor([8, 5, 2]).unsqueeze(1)
    data = {"id": ["a", "b", "c"],
            "text": torch.randn(b, t, 768, device=device),
            "audio": torch.randn(b, t, 74, device=device),
            "vision": torch.randn(b, t, 35, device=device),
            "padding_mask": pad.to(device),
            "availability_mask": torch.ones(b, 3, t, dtype=torch.bool, device=device),
            "native_zero_mask": torch.zeros(b, 3, t, dtype=torch.bool, device=device),
            "cls_label": torch.tensor([0, 1, 2], device=device),
            "reg_label": torch.tensor([-1., 0., 1.], device=device)}
    for modality in MODALITIES:
        data[modality][~data["padding_mask"]] = 0
    return data


def initialized_models(device="cpu"):
    torch.manual_seed(42)
    b0 = B0Baseline().to(device).eval()
    b3 = B3LatentReconstruction().to(device).eval()
    incompatible = b3.load_state_dict(b0.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("reconstructors.") for key in incompatible.missing_keys)
    return b0, b3


def test_b0_equivalence_and_clean_no_trigger_path():
    b0, b3 = initialized_models()
    source = batch()
    with torch.no_grad():
        original = b0(predictor_inputs(source))
        disabled = b3(predictor_inputs(source), reconstruction_enabled=False)
        enabled = b3(predictor_inputs(source), return_aux=True)
    for key in ("classification_logits", "regression"):
        torch.testing.assert_close(disabled[key], original[key], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(enabled[key], original[key], rtol=1e-5, atol=1e-6)
    assert not any(mask.any() for mask in enabled["zero_trigger"].values())


def test_trigger_only_replaces_zero_and_source_bias_removed():
    _, model = initialized_models()
    source = batch()
    source["text"][0, 2] = 0
    source["audio"][0, 2] = 0
    with torch.no_grad():
        aux = model(predictor_inputs(source), return_aux=True)
    assert aux["zero_trigger"]["text"][0, 2]
    assert aux["zero_trigger"]["audio"][0, 2]
    assert not aux["zero_trigger"]["vision"][0, 2]
    assert aux["source_latents"]["text"][0, 2].eq(0).all()
    assert aux["source_latents"]["audio"][0, 2].eq(0).all()
    for modality in MODALITIES:
        normal = ~aux["zero_trigger"][modality]
        torch.testing.assert_close(aux["effective_latents"][modality][normal],
                                   aux["latents"][modality][normal])
        assert aux["reconstructed"][modality][~source["padding_mask"]].eq(0).all()


def test_native_zero_triggers_but_availability_metadata_is_not_predictor_input():
    _, model = initialized_models()
    source = batch()
    source["vision"][1, 2] = 0
    source["native_zero_mask"][1, 2, 2] = True
    modified = dict(source)
    modified["availability_mask"] = source["availability_mask"].clone()
    modified["availability_mask"][1, 2, 2] = False
    with torch.no_grad():
        a = model(predictor_inputs(source), return_aux=True)
        b = model(predictor_inputs(modified), return_aux=True)
    assert a["zero_trigger"]["vision"][1, 2]
    for key in ("classification_logits", "regression"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_padding_excluded_even_if_padded_values_change():
    _, model = initialized_models()
    source = batch()
    modified = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in source.items()}
    for modality in MODALITIES:
        modified[modality][~source["padding_mask"]] = 12345
    with torch.no_grad():
        a = model(predictor_inputs(source), return_aux=True)
        b = model(predictor_inputs(modified), return_aux=True)
    for key in ("classification_logits", "regression"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=1e-6)
    for modality in MODALITIES:
        assert not b["zero_trigger"][modality][~source["padding_mask"]].any()
        assert b["reconstructed"][modality][~source["padding_mask"]].eq(0).all()


def test_stop_gradient_full_input_preservation_and_mask_separation():
    _, model = initialized_models()
    model.train()
    full = batch()
    original = {m: full[m].clone() for m in MODALITIES}
    masked, metadata = apply_blocks(full, [
        {"sample_index": 0, "modality": "text", "rho": .5, "location": "middle"},
        {"sample_index": 1, "modality": "vision", "rho": .2, "location": "late"},
    ])
    assert len(metadata) == 2
    for modality in MODALITIES:
        assert torch.equal(full[modality], original[modality])
        full[modality].requires_grad_()
    outputs = model(predictor_inputs(masked), return_aux=True)
    losses = reconstruction_losses(model, outputs, full, masked)
    assert all(not target.requires_grad for target in losses["targets"].values())
    (losses["reconstruction"] + .1 * losses["identity"]).backward()
    assert all(full[m].grad is None for m in MODALITIES)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.reconstructors.parameters())
    assert torch.equal(full["native_zero_mask"], masked["native_zero_mask"])
    assert torch.equal(full["padding_mask"], masked["padding_mask"])


def test_checkpoint_roundtrip(tmp_path):
    _, model = initialized_models()
    source = batch()
    path = tmp_path / "b3.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    restored = B3LatentReconstruction().eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    with torch.no_grad():
        a = model(predictor_inputs(source))
        b = restored(predictor_inputs(source))
    for key in ("classification_logits", "regression"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_forward_backward_device(device):
    _, model = initialized_models(device)
    model.train()
    full = batch(device)
    masked, _ = apply_blocks(full, [
        {"sample_index": 0, "modality": "vision", "rho": .5, "location": "early"},
    ])
    outputs = model(predictor_inputs(masked), return_aux=True)
    losses = reconstruction_losses(model, outputs, full, masked)
    total = (outputs["classification_logits"].square().mean()
             + outputs["regression"].square().mean()
             + losses["reconstruction"] + .1 * losses["identity"])
    total.backward()
    assert torch.isfinite(total)
    assert any(p.grad is not None for p in model.parameters())

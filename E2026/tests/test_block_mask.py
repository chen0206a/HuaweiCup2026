import random

import pytest
import torch

from src.data.block_mask import apply_blocks, augment_train_batch, block_length, block_start
from src.evaluation.missing_benchmark import evaluate_benchmark, mask_scenario, scenarios


def batch(lengths=(1, 5, 9)):
    b, t = len(lengths), 10
    pad = torch.zeros(b, t, dtype=torch.bool)
    for i, length in enumerate(lengths):
        pad[i, :length] = True
    features = {name: torch.ones(b, t, dim) for name, dim in
                (("text", 2), ("audio", 3), ("vision", 4))}
    features["vision"][1, 2] = 0  # original valid zero is content, not synthetic missing
    return {"id": [f"id{i}" for i in range(b)], **features,
            "padding_mask": pad, "availability_mask": torch.ones(b, 3, t, dtype=torch.bool),
            "native_zero_mask": torch.zeros(b, 3, t, dtype=torch.bool)}


@pytest.mark.parametrize("rho", (0.1, 0.5))
@pytest.mark.parametrize("length", (1, 2, 5, 9, 50))
def test_length_and_short_sequences(rho, length):
    expected = max(1, round(rho * length))
    assert block_length(length, rho) == expected
    assert 1 <= expected <= length


def test_locations():
    assert [block_start(9, 3, loc) for loc in ("early", "middle", "late")] == [0, 3, 6]
    assert [block_start(1, 1, loc) for loc in ("early", "middle", "late")] == [0, 0, 0]


def test_single_double_padding_native_zero_and_no_inplace():
    source = batch()
    before = {key: value.clone() for key, value in source.items() if isinstance(value, torch.Tensor)}
    masked, meta = apply_blocks(source, [
        {"sample_index": 0, "modality": "text", "rho": .5, "location": "late"},
        {"sample_index": 1, "modality": "audio", "rho": .5, "location": "middle"},
        {"sample_index": 1, "modality": "vision", "rho": .1, "location": "early"},
    ])
    assert len(meta) == 3 and meta[0]["block_length"] == 1
    assert masked["text"][0, 0].eq(0).all()
    assert masked["audio"][1, 1:3].eq(0).all()
    assert masked["vision"][1, 0].eq(0).all()
    assert masked["availability_mask"][1, 1, 1:3].eq(False).all()
    assert masked["availability_mask"][1, 2, 0].eq(False).all()
    assert masked["availability_mask"][1, 2, 2]  # native zero stays available
    for key, original in before.items():
        assert torch.equal(source[key], original), key
    assert torch.equal(masked["native_zero_mask"], source["native_zero_mask"])
    assert torch.equal(masked["padding_mask"], source["padding_mask"])
    for index, name in enumerate(("text", "audio", "vision")):
        hidden = ~masked["availability_mask"][:, index]
        assert not (hidden & ~source["padding_mask"]).any()
        assert masked[name][hidden].eq(0).all()


def test_benchmark_exactly_repeated_and_composition():
    suite = scenarios()
    assert len(suite) == 54 and len(set(suite)) == 54
    source = batch()
    for scene in suite:
        first, m1 = mask_scenario(source, scene)
        second, m2 = mask_scenario(source, scene)
        assert m1 == m2
        assert torch.equal(first["availability_mask"], second["availability_mask"])
        for name in ("text", "audio", "vision"):
            assert torch.equal(first[name], second[name])
        assert len(m1) == len(scene.modalities) * len(source["id"])
        for item in m1:
            assert item["end"] <= item["valid_length"]


def test_training_rng_reproducible():
    source = batch((2, 5, 9, 10))
    x, mx = augment_train_batch(source, random.Random(42))
    y, my = augment_train_batch(source, random.Random(42))
    assert mx == my
    assert torch.equal(x["availability_mask"], y["availability_mask"])
    with pytest.raises(ValueError):
        apply_blocks(source, [{"sample_index": 0, "modality": "text", "rho": .5, "start": 2}])


def test_benchmark_inference_is_deterministic_and_hides_bookkeeping():
    source = batch()
    source.update({"cls_label": torch.tensor([0, 1, 2]),
                   "reg_label": torch.tensor([-1.0, 0.0, 1.0]),
                   "vision_all_zero": torch.tensor([False, True, False])})

    class TinyModel(torch.nn.Module):
        def forward(self, inputs):
            assert set(inputs) == {"text", "audio", "vision", "padding_mask"}
            value = inputs["text"].sum((1, 2)) / 20
            return {"classification_logits": torch.stack((value, -value, value * 0), dim=1),
                    "regression": value}

    model = TinyModel()
    first = evaluate_benchmark(model, [source], torch.device("cpu"))
    second = evaluate_benchmark(model, [source], torch.device("cpu"))
    assert first == second
    assert len(first) == 55

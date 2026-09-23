import numpy as np
import pickle
import pytest
import torch

from src.data.dataset import Aligned50Dataset
from src.data.preprocess import TrainFeaturewiseScaler, build_datasets_and_loaders


def split(n=3):
    rng = np.random.default_rng(4)
    mask = np.zeros((n, 50), dtype=np.int64)
    mask[:, :4] = 1
    data = {
        "id": np.array([f"clip-{i}" for i in range(n)]),
        "text": rng.normal(size=(n, 50, 768)).astype(np.float32),
        "audio": rng.normal(size=(n, 50, 74)).astype(np.float64),
        "vision": rng.normal(size=(n, 50, 35)).astype(np.float64),
        "text_bert": np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], axis=1),
        "classification_labels": np.resize(np.array([0, 1, 2]), n),
        "regression_labels": np.resize(np.array([-1.0, 0.0, 1.0]), n),
    }
    for m in ("audio", "vision"):
        data[m][:, 4:, :] = 0
    # A valid zero vector is retained as native-zero and is not padding.
    data["audio"][0, 1, :] = 0
    data["vision"][1, :4, :] = 0
    return data


def test_schema_masks_and_dtype():
    ds = Aligned50Dataset(split(), "train")
    item = ds[1]
    assert item["text"].shape == (50, 768) and item["text"].dtype == torch.float32
    assert item["audio"].shape == (50, 74) and item["audio"].dtype == torch.float32
    assert item["vision"].shape == (50, 35) and item["vision"].dtype == torch.float32
    assert item["padding_mask"].shape == (50,) and item["padding_mask"].dtype == torch.bool
    assert item["availability_mask"].shape == (3, 50) and item["availability_mask"].all()
    assert item["native_zero_mask"].shape == (3, 50)
    assert item["native_zero_mask"][2, :4].all()
    assert not item["native_zero_mask"][2, 4:].any()
    assert item["vision_all_zero"].item()


def test_nonzero_feature_in_padding_suffix_is_rejected():
    data = split()
    data["vision"][0, 49, 0] = 1
    with pytest.raises(ValueError, match="padding suffix"):
        Aligned50Dataset(data, "train")


def test_train_scaler_excludes_padding_and_keeps_native_zero_in_statistics():
    data = split()
    # Constant-valued valid timesteps, while the padded suffix is deliberately huge.
    data["audio"][:, :4, :] = 2.0
    data["audio"][0, 1, :] = 0.0
    data["audio"][:, 4:, :] = 10000.0
    ds = Aligned50Dataset(data, "train", verify_padding_alignment=False)
    scaler = TrainFeaturewiseScaler.fit(ds)
    assert np.allclose(scaler.mean["audio"], (11 * 2.0) / 12)
    assert np.all(scaler.mean["audio"] < 3.0)


def test_include_test_false_does_not_construct_test_dataset_or_loader(tmp_path):
    source = {name: split(2) for name in ("train", "valid", "test")}
    path = tmp_path / "synthetic.pkl"
    with path.open("wb") as f:
        pickle.dump(source, f)
    datasets, loaders, _ = build_datasets_and_loaders(
        path, normalization="none", batch_size=2, include_test=False
    )
    assert set(datasets) == {"train", "valid"}
    assert set(loaders) == {"train", "valid"}

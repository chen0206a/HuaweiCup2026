import numpy as np
import pytest

from src.data.block_mask import apply_blocks
from src.data.dataset import Aligned50Dataset
from src.data.text_scaler import TextOnlyTrainScaler


def _split(name, valid_mask):
    n = len(valid_mask)
    text = np.zeros((n, 50, 768), dtype=np.float32)
    text[:, :, 0] = 0.0
    text[:, :, 1] = np.arange(50, dtype=np.float32)[None, :]
    audio = np.arange(n * 50 * 74, dtype=np.float32).reshape(n, 50, 74)
    vision = np.arange(n * 50 * 35, dtype=np.float32).reshape(n, 50, 35)
    bert = np.zeros((n, 3, 50), dtype=np.float32)
    bert[:, 1, :] = valid_mask
    return Aligned50Dataset({
        "id": [f"{name}-{i}" for i in range(n)],
        "text": text, "audio": audio, "vision": vision, "text_bert": bert,
        "classification_labels": np.zeros(n, dtype=np.int64),
        "regression_labels": np.zeros(n, dtype=np.float32),
    }, name, verify_padding_alignment=False)


def test_text_scaler_uses_train_valid_positions_and_preserves_other_modalities():
    mask = np.zeros((2, 50), dtype=np.float32)
    mask[0, :2] = 1
    mask[1, :1] = 1
    train = _split("train", mask)
    # Give the first text feature valid values [1,3,1] and large padded sentinels.
    train.features["text"][0, :2, 0] = [1.0, 3.0]
    train.features["text"][1, :1, 0] = [1.0]
    train.features["text"][0, 2:, 0] = 10000.0
    train.features["text"][1, 1:, 0] = 20000.0
    raw_audio = train.features["audio"].copy()
    raw_vision = train.features["vision"].copy()

    scaler = TextOnlyTrainScaler.fit(train, epsilon=1e-6, chunk_size=1)
    assert scaler.fitted_split == "train"
    assert scaler.valid_timestep_count == 3
    assert scaler.mean[0] == pytest.approx(5.0 / 3.0)
    assert scaler.raw_feature_std[0] == pytest.approx(np.sqrt(8.0 / 9.0))
    assert np.array_equal(scaler.transform("audio", raw_audio), raw_audio)
    assert np.array_equal(scaler.transform("vision", raw_vision), raw_vision)
    assert train.padding_mask.sum() == scaler.valid_timestep_count


def test_scaler_rejects_nontrain_split_and_valid_data_cannot_affect_fit():
    mask = np.zeros((1, 50), dtype=np.float32)
    mask[0, :2] = 1
    train_a = _split("train", mask)
    train_b = _split("train", mask)
    train_a.features["text"][0, :2, 0] = [2.0, 4.0]
    train_b.features["text"][0, :2, 0] = [2.0, 4.0]
    valid = _split("valid", mask)
    valid.features["text"][0, :2, 0] = [1e6, -1e6]
    scaler_a = TextOnlyTrainScaler.fit(train_a)
    scaler_b = TextOnlyTrainScaler.fit(train_b)
    assert np.array_equal(scaler_a.mean, scaler_b.mean)
    assert np.array_equal(scaler_a.std, scaler_b.std)
    with pytest.raises(ValueError, match="train"):
        TextOnlyTrainScaler.fit(valid)


def test_artificial_missing_is_overwritten_to_exact_zero_after_normalization():
    mask = np.zeros((1, 50), dtype=np.float32)
    mask[0, :6] = 1
    train = _split("train", mask)
    train.features["text"][0, :6, 0] = np.arange(6, dtype=np.float32)
    scaler = TextOnlyTrainScaler.fit(train)
    train.scaler = scaler
    sample = train[0]
    assert sample["text"][0, 0].item() != 0.0
    batch = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]) for k, v in sample.items()}
    masked, _ = apply_blocks(batch, [{"sample_index": 0, "modality": "text",
                                     "rho": 0.5, "start": 1}])
    assert np.all(masked["text"][0, 1:4].numpy() == 0.0)
    assert np.array_equal(masked["text"][0, 0].numpy(), sample["text"][0].numpy())

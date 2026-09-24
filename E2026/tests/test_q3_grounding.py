"""Q3-2 label-free input and evidence-withholding checks."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from src.q3.data_adapter import UnlabeledAligned50Dataset
from src.q3.evidence_grounding import ground_interval


def test_unlabeled_adapter_uses_text_bert_mask_not_zero_features(tmp_path):
    bert = np.zeros((3, 50), dtype=np.int64)
    bert[1, :3] = 1
    text = np.ones((50, 768), dtype=np.float32)
    audio = np.zeros((50, 74), dtype=np.float64)
    vision = np.zeros((50, 35), dtype=np.float64)
    # Native zero observations at valid positions remain valid.
    sample = {"id": "01", "raw_text": "example", "text_bert": bert,
              "text": text, "audio": audio, "vision": vision}
    path = tmp_path / "01.pkl"
    path.write_bytes(pickle.dumps(sample))
    dataset = UnlabeledAligned50Dataset([path])
    item = dataset[0]
    assert len(dataset) == 1
    assert item["padding_mask"].sum().item() == 3
    assert item["native_zero_mask"][1, :3].all()
    assert item["audio"].dtype == torch.float32
    assert item["text"][3:].ne(0).all()  # Padding is not inferred from text values.
    assert "cls_label" not in item and "reg_label" not in item


def test_grounding_withholds_unproved_local_text_and_times():
    result = ground_interval(sample_id="01", modality="audio", start_index=0,
                             end_index=3, valid_length=10, media_file="01.mp4")
    assert result["grounding_status"] == "unverified"
    assert result["media_file"] == "01.mp4"
    assert result["text_fragment"] is None
    assert result["audio_time_range"] is None
    assert result["video_frame_time"] is None


def test_approximate_grounding_requires_uncertainty_and_exact_interval():
    mapping = {"sample_id": "01", "modality": "vision", "start_index": 2,
               "end_index": 4, "grounding_status": "approximate",
               "mapping_method": "synthetic-reviewed-method",
               "source_metadata": "synthetic source", "assumptions": ["synthetic"],
               "evidence": {"video_frame_time": 1.2}}
    with pytest.raises(ValueError, match="uncertainty"):
        ground_interval(sample_id="01", modality="vision", start_index=2,
                        end_index=4, valid_length=10, media_file="01.mp4", mapping=mapping)
    mapping["estimated_uncertainty_seconds"] = 0.25
    result = ground_interval(sample_id="01", modality="vision", start_index=2,
                             end_index=4, valid_length=10, media_file="01.mp4", mapping=mapping)
    assert result["grounding_status"] == "approximate"
    assert result["video_frame_time"] == 1.2

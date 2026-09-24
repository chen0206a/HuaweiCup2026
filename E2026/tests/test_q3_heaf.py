"""Focused checks for HEAF's masking and exact three-player algebra."""
from __future__ import annotations

import numpy as np
import torch

from src.q3.coalitions import exact_shapley, masked_coalition, pair_interactions
from src.q3.temporal_occlusion import window_length


def test_excluded_modality_zeros_only_valid_rows_without_mutating_source():
    pad = torch.tensor([[True, True, False, False] + [False] * 46])
    batch = {
        "padding_mask": pad,
        "native_zero_mask": torch.tensor([[[True] + [False] * 49] * 3]),
        "text": torch.ones(1, 50, 768),
        "audio": torch.ones(1, 50, 74),
        "vision": torch.ones(1, 50, 35),
    }
    saved = {key: value.clone() for key, value in batch.items()}
    masked = masked_coalition(batch, ("text", "vision"))
    assert torch.all(masked["audio"][pad] == 0)
    assert torch.equal(masked["audio"][~pad], batch["audio"][~pad])
    assert torch.equal(masked["native_zero_mask"], batch["native_zero_mask"])
    assert all(torch.equal(batch[key], value) for key, value in saved.items())


def test_shapley_and_pair_interaction_for_known_three_player_game():
    # v(S)=2*T+3*A+5*V+7*T*A. Interaction T-A is 7 in both V contexts.
    values = np.array([[0, 2, 3, 5, 12, 7, 8, 17]], dtype=float)
    phi = exact_shapley(values)
    np.testing.assert_allclose(phi, [[5.5, 6.5, 5.0]])
    np.testing.assert_allclose(phi.sum(axis=1), values[:, -1] - values[:, 0])
    interaction = pair_interactions(values)
    np.testing.assert_allclose(interaction["text_audio"]["value"], [7])
    np.testing.assert_allclose(interaction["text_vision"]["value"], [0])


def test_window_length_is_based_on_valid_prefix_and_can_be_one():
    assert window_length(3, 0.10) == 1
    assert window_length(50, 0.30) == 15

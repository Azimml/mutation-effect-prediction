"""Tests for the CNN input encoding — the 9-channel ref/alt/mask tensor.

torch-gated: skipped if torch isn't installed (it's an optional [cnn] extra).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mutation_effect_prediction.cnn import (  # noqa: E402
    BASE_TO_INDEX,
    encode_sequence_pair,
    one_hot_encode,
)


def test_one_hot_shape_and_content():
    enc = one_hot_encode("ACGT")
    assert enc.shape == (4, 4)
    # each column is a one-hot over {A,C,G,T}
    assert np.array_equal(enc.sum(axis=0), np.ones(4))
    assert enc[BASE_TO_INDEX["A"], 0] == 1.0
    assert enc[BASE_TO_INDEX["T"], 3] == 1.0


def test_unknown_base_is_all_zero_column():
    enc = one_hot_encode("N")
    assert enc.shape == (4, 1)
    assert enc.sum() == 0.0


def test_sequence_pair_channel_layout():
    ref = "ACGTA"
    alt = "ACCTA"  # differs at the center position (index 2)
    t = encode_sequence_pair(ref, alt)
    # 4 ref + 4 alt + 1 mutation mask = 9 channels
    assert t.shape == (9, len(ref))
    assert t.dtype == torch.float32
    # channels 0-3 are the ref one-hot, 4-7 the alt one-hot
    assert torch.equal(t[0:4], torch.tensor(one_hot_encode(ref)))
    assert torch.equal(t[4:8], torch.tensor(one_hot_encode(alt)))


def test_mutation_mask_marks_center():
    ref = alt = "ACGTACG"  # length 7 -> center index 3
    t = encode_sequence_pair(ref, alt)
    mask = t[8]
    assert mask.sum() == 1.0
    assert mask[len(ref) // 2] == 1.0

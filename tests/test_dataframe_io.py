"""Tests for the input-validation helper used by the predict commands.

Pure pandas, no torch/pysam/network/data files required.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mutation_effect_prediction.dataframe_io import require_columns


def test_passes_when_all_columns_present():
    frame = pd.DataFrame({"ref_seq": ["ACGT"], "alt_seq": ["ACCT"]})
    # Should not raise.
    require_columns(frame, ("ref_seq", "alt_seq"), "example.csv")


def test_raises_listing_missing_columns():
    frame = pd.DataFrame({"ref_seq": ["ACGT"]})
    with pytest.raises(ValueError) as excinfo:
        require_columns(frame, ("ref_seq", "alt_seq"), "example.csv")
    message = str(excinfo.value)
    assert "alt_seq" in message
    # Present columns must not be reported as missing.
    assert "missing required column(s): alt_seq" in message
    # The offending source is named to make the error actionable.
    assert "example.csv" in message


def test_error_names_the_available_columns():
    frame = pd.DataFrame({"chrom": ["1"], "pos": [10]})
    with pytest.raises(ValueError) as excinfo:
        require_columns(frame, ("ref_seq",), "in.csv")
    message = str(excinfo.value)
    assert "chrom" in message and "pos" in message

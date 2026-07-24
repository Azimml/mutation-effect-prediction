"""Tests for the stratified subsetting used to build the 50k/250k datasets.

The subset must (a) hit the requested row count exactly, (b) never invent rows
that don't exist, and (c) roughly preserve the split x label composition. The
script lives under ``scripts/`` (not an installed package), so it is loaded by
file path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "create_dataset_subset.py"
_spec = importlib.util.spec_from_file_location("create_dataset_subset", _SCRIPT)
assert _spec is not None and _spec.loader is not None
create_dataset_subset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(create_dataset_subset)
stratified_subset = create_dataset_subset.stratified_subset


def _make_frame(n: int = 1000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        split = ("train", "val", "test")[i % 3]
        label = "pathogenic" if i % 5 == 0 else "benign"
        rows.append({"split": split, "label_name": label, "value": i})
    return pd.DataFrame(rows)


def test_hits_exact_target_row_count():
    frame = _make_frame(1000)
    subset = stratified_subset(frame, target_rows=200, seed=17)
    assert len(subset) == 200


def test_subset_rows_come_from_source():
    frame = _make_frame(1000)
    subset = stratified_subset(frame, target_rows=150, seed=17)
    assert set(subset["value"]).issubset(set(frame["value"]))
    # No duplicated rows introduced.
    assert subset["value"].is_unique


def test_deterministic_with_same_seed():
    frame = _make_frame(1000)
    a = stratified_subset(frame, target_rows=120, seed=7)
    b = stratified_subset(frame, target_rows=120, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_every_present_group_is_represented():
    frame = _make_frame(1000)
    subset = stratified_subset(frame, target_rows=300, seed=17)
    original_groups = set(map(tuple, frame[["split", "label_name"]].drop_duplicates().to_numpy()))
    subset_groups = set(map(tuple, subset[["split", "label_name"]].drop_duplicates().to_numpy()))
    assert subset_groups == original_groups

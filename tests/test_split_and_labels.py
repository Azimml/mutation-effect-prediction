"""Tests for the load-bearing pure functions: the leakage-aware split, the
ClinVar significance -> binary label mapping, k-merization, and the metric
helpers. No torch, no network, no data files required.

The split invariants here are the repository's central methodological claim,
so they get the most scrutiny.
"""
from __future__ import annotations

import numpy as np

from mutation_effect_prediction.baseline import sequence_to_kmers
from mutation_effect_prediction.evaluation import (
    compute_binary_metrics,
    select_best_f1_threshold,
)
from mutation_effect_prediction.preprocess import (
    assign_split,
    resolve_binary_label,
    split_tokens,
)
from mutation_effect_prediction.utils import stable_fraction

SPLIT_KW = dict(region_bin_size=1_000_000, validation_fraction=0.1, test_fraction=0.2, seed=13)


class TestStableFraction:
    def test_deterministic(self):
        assert stable_fraction("chr1:5") == stable_fraction("chr1:5")

    def test_in_unit_interval(self):
        for key in ("a", "chr1:0", "17:42", "X:999"):
            v = stable_fraction(key)
            assert 0.0 <= v < 1.0

    def test_different_keys_differ(self):
        assert stable_fraction("chr1:5") != stable_fraction("chr1:6")


class TestAssignSplit:
    def test_same_region_same_split(self):
        """The core leakage guard: two variants in the same 1 Mb region bin
        must always land in the same split, regardless of exact position."""
        # positions 10 and 999_999 are both in region bin 0 of chr1
        s1, g1 = assign_split("1", 10, **SPLIT_KW)
        s2, g2 = assign_split("1", 999_999, **SPLIT_KW)
        assert g1 == g2 == "1:0"
        assert s1 == s2

    def test_region_boundary_splits_bins(self):
        # position 1_000_001 crosses into region bin 1 (0-based via (pos-1)//size)
        _, g_lo = assign_split("1", 1_000_000, **SPLIT_KW)   # (1e6-1)//1e6 = 0
        _, g_hi = assign_split("1", 1_000_001, **SPLIT_KW)   # (1e6)//1e6   = 1
        assert g_lo == "1:0"
        assert g_hi == "1:1"

    def test_deterministic_across_calls(self):
        a = assign_split("7", 123_456, **SPLIT_KW)
        b = assign_split("7", 123_456, **SPLIT_KW)
        assert a == b

    def test_only_valid_split_names(self):
        for pos in range(1, 40_000_000, 250_000):
            split, _ = assign_split("2", pos, **SPLIT_KW)
            assert split in {"train", "val", "test"}

    def test_fractions_are_roughly_respected(self):
        # Assign many distinct regions and check the split proportions land
        # near the configured fractions (region-level, not row-level).
        splits = [assign_split("3", 1 + i * 1_000_000, **SPLIT_KW)[0] for i in range(3000)]
        n = len(splits)
        assert 0.15 < splits.count("test") / n < 0.25
        assert 0.05 < splits.count("val") / n < 0.15


class TestResolveBinaryLabel:
    def test_pathogenic_maps_to_1(self):
        assert resolve_binary_label("Pathogenic") == 1
        assert resolve_binary_label("Likely_pathogenic") == 1

    def test_benign_maps_to_0(self):
        assert resolve_binary_label("Benign") == 0
        assert resolve_binary_label("Likely_benign") == 0

    def test_uncertain_and_conflicting_excluded(self):
        assert resolve_binary_label("Uncertain_significance") is None
        assert resolve_binary_label("Conflicting_classifications_of_pathogenicity") is None

    def test_mixed_pathogenic_and_benign_is_ambiguous(self):
        # A record carrying both a pathogenic and a benign term is not a clean
        # binary label and must be dropped.
        assert resolve_binary_label("Pathogenic/Benign") is None

    def test_empty_or_none(self):
        assert resolve_binary_label(None) is None
        assert resolve_binary_label("") is None


class TestSplitTokens:
    def test_case_and_separator_normalization(self):
        assert set(split_tokens("Pathogenic/Likely_pathogenic")) == {
            "pathogenic",
            "likely_pathogenic",
        }

    def test_handles_tuple_input(self):
        toks = set(split_tokens(("Benign", "Likely_benign")))
        assert "benign" in toks and "likely_benign" in toks


class TestSequenceToKmers:
    def test_basic_windowing(self):
        assert sequence_to_kmers("ACGT", 3) == ["ACG", "CGT"]

    def test_kmer_count(self):
        seq = "A" * 20
        assert len(sequence_to_kmers(seq, 3)) == 20 - 3 + 1

    def test_short_sequence_returns_whole(self):
        assert sequence_to_kmers("AC", 3) == ["AC"]


class TestThresholdAndMetrics:
    def test_threshold_degenerate_single_class(self):
        y = np.zeros(10, dtype=int)
        assert select_best_f1_threshold(y, np.linspace(0, 1, 10)) == 0.5

    def test_threshold_separable_case(self):
        # Perfectly separable: a threshold between the classes should be chosen.
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        t = select_best_f1_threshold(y, s)
        assert 0.3 < t <= 0.7

    def test_perfect_scores_give_auroc_one(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_binary_metrics(y, s, threshold=0.5)
        assert m["auroc"] == 1.0
        assert m["f1"] == 1.0

    def test_metrics_single_class_auroc_nan(self):
        y = np.ones(4, dtype=int)
        m = compute_binary_metrics(y, np.array([0.6, 0.7, 0.8, 0.9]))
        assert np.isnan(m["auroc"])

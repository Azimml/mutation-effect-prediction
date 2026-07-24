"""Tests for the pure preprocessing helpers that don't need pysam.

Covers gene-symbol parsing, token normalization, and the reference-window
extraction (including the N-padding at chromosome edges). The window function
only calls ``get_reference_length`` and ``fetch`` on its reference argument, so
a tiny stub stands in for a real ``pysam.FastaFile``.
"""
from __future__ import annotations

from mutation_effect_prediction.preprocess import (
    fetch_sequence_window,
    normalize_token,
    parse_gene_symbol,
)


class FakeReference:
    """Minimal stand-in for pysam.FastaFile over a single sequence."""

    def __init__(self, sequence: str) -> None:
        self._sequence = sequence

    def get_reference_length(self, _chrom: str) -> int:
        return len(self._sequence)

    def fetch(self, _chrom: str, start: int, end: int) -> str:
        return self._sequence[start:end]


class TestParseGeneSymbol:
    def test_extracts_symbol_before_id(self):
        assert parse_gene_symbol("BRCA1:672") == "BRCA1"

    def test_takes_first_of_multiple_genes(self):
        assert parse_gene_symbol("BRCA1:672|NBR2:10230") == "BRCA1"

    def test_none_returns_empty_string(self):
        assert parse_gene_symbol(None) == ""


class TestNormalizeToken:
    def test_lowercases_and_replaces_separators(self):
        assert normalize_token("Likely-Pathogenic") == "likely_pathogenic"
        assert normalize_token("Reviewed by expert panel") == "reviewed_by_expert_panel"

    def test_strips_surrounding_whitespace(self):
        assert normalize_token("  Benign  ") == "benign"


class TestFetchSequenceWindow:
    def test_centered_window_has_expected_length(self):
        reference = FakeReference("ACGTACGTACGT")  # length 12
        window = fetch_sequence_window(reference, "1", position=6, window_radius=2)
        # radius 2 -> 2 + 1 + 2 = 5 bases
        assert len(window) == 5
        # position is 1-based; center base is the reference base at that locus
        assert window[2] == "ACGTACGTACGT"[6 - 1]

    def test_left_edge_is_n_padded(self):
        reference = FakeReference("ACGT")
        # position 1 with radius 3 runs off the left edge and must pad with N
        window = fetch_sequence_window(reference, "1", position=1, window_radius=3)
        assert len(window) == 7
        assert window.startswith("NNN")
        assert window[3] == "A"

    def test_right_edge_is_n_padded(self):
        reference = FakeReference("ACGT")  # length 4
        window = fetch_sequence_window(reference, "1", position=4, window_radius=3)
        assert len(window) == 7
        assert window.endswith("NNN")
        assert window[3] == "T"

    def test_output_is_uppercased(self):
        reference = FakeReference("acgtacgt")
        window = fetch_sequence_window(reference, "1", position=4, window_radius=1)
        assert window == window.upper()

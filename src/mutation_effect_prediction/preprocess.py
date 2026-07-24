from __future__ import annotations

import csv
import gzip
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pysam

from .constants import (
    BENIGN_TERMS,
    EXCLUDED_SIGNIFICANCE_TERMS,
    PATHOGENIC_TERMS,
    VALID_BASES,
)
from .utils import ensure_directory, stable_fraction, write_json

TOKEN_SPLIT_PATTERN = re.compile(r"[|/]")


@dataclass
class PreprocessingSummary:
    total_records: int = 0
    kept_records: int = 0
    skipped_non_canonical_chromosome: int = 0
    skipped_multiallelic: int = 0
    skipped_non_snv: int = 0
    skipped_invalid_bases: int = 0
    skipped_missing_label: int = 0
    skipped_review_status: int = 0
    skipped_reference_mismatch: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)


def preprocess_clinvar(
    clinvar_vcf_path: str | Path,
    reference_fasta_path: str | Path,
    output_csv_path: str | Path,
    summary_json_path: str | Path,
    window_radius: int,
    region_bin_size: int,
    validation_fraction: float,
    test_fraction: float,
    allowed_review_statuses: Iterable[str],
    allowed_chromosomes: Iterable[str],
    seed: int,
    max_records: int | None = None,
) -> dict[str, int | dict[str, int]]:
    allowed_review = {normalize_token(value) for value in allowed_review_statuses}
    allowed_chromosome_set = set(allowed_chromosomes)

    # Imported here rather than at module scope so the pure helpers in this
    # module (label mapping, region split, k-merization) can be imported and
    # tested without the heavy pysam C-extension installed.
    import pysam

    variant_file = pysam.VariantFile(str(clinvar_vcf_path))
    reference = pysam.FastaFile(str(reference_fasta_path))

    output_path = Path(output_csv_path)
    ensure_directory(output_path.parent)
    ensure_directory(Path(summary_json_path).parent)

    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    summary = PreprocessingSummary()

    fieldnames = [
        "variant_id",
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "label_name",
        "split",
        "group_id",
        "review_status",
        "clinical_significance",
        "variation_id",
        "gene_symbol",
        "ref_seq",
        "alt_seq",
    ]

    with gzip.open(output_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in variant_file:
            summary.total_records += 1

            if max_records is not None and summary.kept_records >= max_records:
                break

            chrom = str(record.chrom)
            if chrom not in allowed_chromosome_set:
                summary.skipped_non_canonical_chromosome += 1
                continue

            if not record.alts or len(record.alts) != 1:
                summary.skipped_multiallelic += 1
                continue

            ref = record.ref.upper()
            alt = record.alts[0].upper()
            if len(ref) != 1 or len(alt) != 1:
                summary.skipped_non_snv += 1
                continue

            if ref not in VALID_BASES or alt not in VALID_BASES:
                summary.skipped_invalid_bases += 1
                continue

            label = resolve_binary_label(record.info.get("CLNSIG"))
            if label is None:
                summary.skipped_missing_label += 1
                continue

            review_status_tokens = split_tokens(record.info.get("CLNREVSTAT"))
            if allowed_review and not any(token in allowed_review for token in review_status_tokens):
                summary.skipped_review_status += 1
                continue

            ref_seq = fetch_sequence_window(reference, chrom, record.pos, window_radius)
            if ref_seq[window_radius] != ref:
                summary.skipped_reference_mismatch += 1
                continue

            alt_seq = ref_seq[:window_radius] + alt + ref_seq[window_radius + 1 :]
            split, group_id = assign_split(
                chrom=chrom,
                position=record.pos,
                region_bin_size=region_bin_size,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                seed=seed,
            )

            label_name = "pathogenic" if label == 1 else "benign"
            variant_id = f"{chrom}:{record.pos}:{ref}>{alt}"
            row = {
                "variant_id": variant_id,
                "chrom": chrom,
                "pos": record.pos,
                "ref": ref,
                "alt": alt,
                "label": label,
                "label_name": label_name,
                "split": split,
                "group_id": group_id,
                "review_status": "|".join(review_status_tokens),
                "clinical_significance": "|".join(split_tokens(record.info.get("CLNSIG"))),
                "variation_id": record.id or "",
                "gene_symbol": parse_gene_symbol(record.info.get("GENEINFO")),
                "ref_seq": ref_seq,
                "alt_seq": alt_seq,
            }

            writer.writerow(row)
            summary.kept_records += 1
            label_counts[label_name] += 1
            split_counts[split] += 1

    summary.label_counts = dict(label_counts)
    summary.split_counts = dict(split_counts)
    payload = asdict(summary)
    write_json(payload, summary_json_path)
    return payload


def normalize_token(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    normalized = normalized.replace("__", "_")
    return normalized


def split_tokens(raw_value: object) -> list[str]:
    if raw_value is None:
        return []

    if isinstance(raw_value, tuple):
        parts = [str(item) for item in raw_value]
        values = [",".join(parts), *parts]
    elif isinstance(raw_value, list):
        parts = [str(item) for item in raw_value]
        values = [",".join(parts), *parts]
    else:
        values = [str(raw_value)]

    tokens: list[str] = []
    for value in values:
        for token in TOKEN_SPLIT_PATTERN.split(value):
            normalized = normalize_token(token)
            if normalized:
                tokens.append(normalized)
    return tokens


def resolve_binary_label(raw_significance: object) -> int | None:
    tokens = set(split_tokens(raw_significance))
    if not tokens:
        return None

    if tokens & EXCLUDED_SIGNIFICANCE_TERMS:
        return None

    if tokens <= PATHOGENIC_TERMS:
        return 1

    if tokens <= BENIGN_TERMS:
        return 0

    return None


def parse_gene_symbol(raw_geneinfo: object) -> str:
    if raw_geneinfo is None:
        return ""

    first_value = str(raw_geneinfo).split("|", maxsplit=1)[0]
    return first_value.split(":", maxsplit=1)[0]


def fetch_sequence_window(reference: pysam.FastaFile, chrom: str, position: int, window_radius: int) -> str:
    chromosome_length = reference.get_reference_length(chrom)
    center_index = position - 1
    start = center_index - window_radius
    end = center_index + window_radius + 1

    left_padding = max(0, -start)
    right_padding = max(0, end - chromosome_length)
    fetch_start = max(0, start)
    fetch_end = min(chromosome_length, end)

    sequence = reference.fetch(chrom, fetch_start, fetch_end).upper()
    return ("N" * left_padding) + sequence + ("N" * right_padding)


def assign_split(
    chrom: str,
    position: int,
    region_bin_size: int,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[str, str]:
    region_index = (position - 1) // region_bin_size
    group_id = f"{chrom}:{region_index}"
    fraction = stable_fraction(f"{seed}:{group_id}")

    if fraction < test_fraction:
        return "test", group_id

    if fraction < test_fraction + validation_fraction:
        return "val", group_id

    return "train", group_id

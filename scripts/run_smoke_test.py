from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pysam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mutation_effect_prediction.baseline import train_baseline_model
from mutation_effect_prediction.preprocess import preprocess_clinvar


def main() -> None:
    smoke_root = PROJECT_ROOT / "data" / "interim" / "smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True)

    fasta_path = smoke_root / "smoke.fa"
    vcf_path = smoke_root / "smoke.vcf"
    processed_path = smoke_root / "processed.csv.gz"
    summary_path = smoke_root / "summary.json"
    baseline_dir = smoke_root / "baseline"

    sequences = {
        "1": ("ACGT" * 250),
        "2": ("TGCA" * 250),
    }
    write_fasta(fasta_path, sequences)
    pysam.faidx(str(fasta_path))

    records = []
    variation_id = 1
    for chrom, sequence in sequences.items():
        for offset in range(60, 460, 20):
            ref = sequence[offset - 1]
            alt = next(base for base in "ACGT" if base != ref)
            label = "Pathogenic" if (variation_id % 2 == 0) else "Benign"
            records.append(
                {
                    "chrom": chrom,
                    "pos": offset,
                    "id": str(variation_id),
                    "ref": ref,
                    "alt": alt,
                    "label": label,
                    "gene": f"GENE{variation_id}",
                }
            )
            variation_id += 1

    write_vcf(vcf_path, records)
    compressed_vcf = str(vcf_path) + ".gz"
    pysam.tabix_compress(str(vcf_path), compressed_vcf, force=True)
    pysam.tabix_index(compressed_vcf, preset="vcf", force=True)

    preprocess_clinvar(
        clinvar_vcf_path=compressed_vcf,
        reference_fasta_path=fasta_path,
        output_csv_path=processed_path,
        summary_json_path=summary_path,
        window_radius=15,
        region_bin_size=40,
        validation_fraction=0.20,
        test_fraction=0.20,
        allowed_review_statuses=[
            "criteria_provided,_single_submitter",
            "criteria_provided,_multiple_submitters,_no_conflicts",
        ],
        allowed_chromosomes=["1", "2"],
        seed=17,
    )

    train_baseline_model(
        dataset_path=processed_path,
        output_dir=baseline_dir,
        kmer_size=3,
        max_features=5000,
        max_iter=1000,
        class_weight="balanced",
    )

    print(f"Smoke test artifacts written to {smoke_root}")


def write_fasta(path: Path, sequences: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chrom, sequence in sequences.items():
            handle.write(f">{chrom}\n")
            for index in range(0, len(sequence), 60):
                handle.write(sequence[index : index + 60] + "\n")


def write_vcf(path: Path, records: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write('##INFO=<ID=CLNSIG,Number=.,Type=String,Description="Clinical significance">\n')
        handle.write('##INFO=<ID=CLNREVSTAT,Number=.,Type=String,Description="Review status">\n')
        handle.write('##INFO=<ID=GENEINFO,Number=1,Type=String,Description="Gene symbol">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for record in records:
            info = (
                f"CLNSIG={record['label']};"
                "CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;"
                f"GENEINFO={record['gene']}:1234"
            )
            handle.write(
                f"{record['chrom']}\t{record['pos']}\t{record['id']}\t{record['ref']}\t"
                f"{record['alt']}\t.\t.\t{info}\n"
            )


if __name__ == "__main__":
    main()

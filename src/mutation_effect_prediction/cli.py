from __future__ import annotations

import argparse

from .config import load_config
from .utils import decompress_gzip, download_file, ensure_directory


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mutation effect prediction pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-data", help="Download ClinVar and reference FASTA.")
    download_parser.add_argument("--config", default="configs/default.toml")
    download_parser.add_argument("--raw-dir", default="data/raw")
    download_parser.add_argument("--skip-clinvar", action="store_true")
    download_parser.add_argument("--skip-reference", action="store_true")
    download_parser.set_defaults(func=handle_download)

    preprocess_parser = subparsers.add_parser("preprocess", help="Build filtered SNV dataset with sequence windows.")
    preprocess_parser.add_argument("--config", default="configs/default.toml")
    preprocess_parser.add_argument("--clinvar-vcf", default="data/raw/clinvar.vcf.gz")
    preprocess_parser.add_argument("--reference-fasta", default="data/raw/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    preprocess_parser.add_argument("--output-csv", default="data/processed/clinvar_snv_windows.csv.gz")
    preprocess_parser.add_argument("--summary-json", default="reports/preprocessing_summary.json")
    preprocess_parser.add_argument("--max-records", type=int, default=None)
    preprocess_parser.set_defaults(func=handle_preprocess)

    baseline_parser = subparsers.add_parser("train-baseline", help="Train k-mer logistic regression baseline.")
    baseline_parser.add_argument("--config", default="configs/default.toml")
    baseline_parser.add_argument("--dataset", default="data/processed/clinvar_snv_windows.csv.gz")
    baseline_parser.add_argument("--output-dir", default="models/baseline")
    baseline_parser.set_defaults(func=handle_train_baseline)

    cnn_parser = subparsers.add_parser("train-cnn", help="Train sequence-pair CNN.")
    cnn_parser.add_argument("--config", default="configs/default.toml")
    cnn_parser.add_argument("--dataset", default="data/processed/clinvar_snv_windows.csv.gz")
    cnn_parser.add_argument("--output-dir", default="models/cnn")
    cnn_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cnn_parser.add_argument("--resume-from", default=None)
    cnn_parser.set_defaults(func=handle_train_cnn)

    predict_parser = subparsers.add_parser("predict", help="Score sequences with a trained model.")
    predict_parser.add_argument("--model-type", choices=("baseline", "cnn"), required=True)
    predict_parser.add_argument("--model-path", required=True)
    predict_parser.add_argument("--input-csv", required=True)
    predict_parser.add_argument("--output-csv", required=True)
    predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict_parser.set_defaults(func=handle_predict)

    interpret_parser = subparsers.add_parser("interpret-cnn", help="Generate a saliency plot for one variant.")
    interpret_parser.add_argument("--checkpoint", required=True)
    interpret_parser.add_argument("--input-csv", required=True)
    interpret_parser.add_argument("--variant-id", required=True)
    interpret_parser.add_argument("--output-path", required=True)
    interpret_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    interpret_parser.set_defaults(func=handle_interpret_cnn)

    return parser


def handle_download(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    raw_dir = ensure_directory(args.raw_dir)
    data_config = config["data"]

    if not args.skip_clinvar:
        download_file(data_config["clinvar_vcf_url"], raw_dir / "clinvar.vcf.gz")
        download_file(data_config["clinvar_tbi_url"], raw_dir / "clinvar.vcf.gz.tbi")

    if not args.skip_reference:
        compressed_path = download_file(
            data_config["reference_fasta_url"],
            raw_dir / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
        )
        decompress_gzip(compressed_path, raw_dir / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")

        import pysam

        pysam.faidx(str(raw_dir / "Homo_sapiens.GRCh38.dna.primary_assembly.fa"))


def handle_preprocess(args: argparse.Namespace) -> None:
    from .preprocess import preprocess_clinvar

    config = load_config(args.config)
    preprocessing = config["preprocessing"]
    preprocess_clinvar(
        clinvar_vcf_path=args.clinvar_vcf,
        reference_fasta_path=args.reference_fasta,
        output_csv_path=args.output_csv,
        summary_json_path=args.summary_json,
        window_radius=preprocessing["window_radius"],
        region_bin_size=preprocessing["region_bin_size"],
        validation_fraction=preprocessing["validation_fraction"],
        test_fraction=preprocessing["test_fraction"],
        allowed_review_statuses=preprocessing["allowed_review_statuses"],
        allowed_chromosomes=preprocessing["allowed_chromosomes"],
        seed=preprocessing["seed"],
        max_records=args.max_records,
    )


def handle_train_baseline(args: argparse.Namespace) -> None:
    from .baseline import train_baseline_model

    config = load_config(args.config)
    baseline = config["baseline"]
    train_baseline_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        kmer_size=baseline["kmer_size"],
        max_features=baseline["max_features"],
        max_iter=baseline["max_iter"],
        class_weight=baseline["class_weight"],
    )


def handle_train_cnn(args: argparse.Namespace) -> None:
    from .cnn import train_cnn_model

    config = load_config(args.config)
    preprocessing = config["preprocessing"]
    cnn = config["cnn"]
    train_cnn_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        batch_size=cnn["batch_size"],
        epochs=cnn["epochs"],
        learning_rate=cnn["learning_rate"],
        weight_decay=cnn["weight_decay"],
        dropout=cnn["dropout"],
        hidden_dim=cnn["hidden_dim"],
        early_stopping_patience=cnn["early_stopping_patience"],
        seed=preprocessing["seed"],
        device_preference=args.device,
        resume_from_checkpoint=args.resume_from,
        max_temperature_c=cnn.get("max_temperature_c"),
        cooldown_temperature_c=cnn.get("cooldown_temperature_c"),
        temperature_check_interval_batches=cnn.get("temperature_check_interval_batches", 25),
        batch_sleep_seconds=cnn.get("batch_sleep_seconds", 0.0),
    )


def handle_predict(args: argparse.Namespace) -> None:
    if args.model_type == "baseline":
        from .baseline import predict_with_baseline

        predict_with_baseline(args.model_path, args.input_csv, args.output_csv)
        return

    from .cnn import predict_with_cnn

    predict_with_cnn(args.model_path, args.input_csv, args.output_csv, device_preference=args.device)


def handle_interpret_cnn(args: argparse.Namespace) -> None:
    from .cnn import save_saliency_plot

    save_saliency_plot(
        checkpoint_path=args.checkpoint,
        input_csv_path=args.input_csv,
        variant_id=args.variant_id,
        output_path=args.output_path,
        device_preference=args.device,
    )

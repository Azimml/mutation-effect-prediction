# Mutation Effect Prediction

[![CI](https://github.com/Azimml/mutation-effect-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Azimml/mutation-effect-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Leakage-aware `ClinVar` SNV pathogenicity prediction using official `ClinVar GRCh38` calls and `GRCh38` reference sequence context.

This repository is a reproducible genomics ML prototype, not a clinical tool. It focuses on a defensible first task: binary pathogenic vs benign classification for single-nucleotide variants using local sequence context.

## Headline Results

This repository already has a complete real-data story:

- full preprocessing on official `ClinVar GRCh38` produced `1,413,909` filtered binary-labeled SNVs from `4,398,837` raw records
- a `50k` kept-variant subset was used for stable model comparison
- the `k-mer + logistic regression` baseline was weak under the leakage-aware split
- the mutation-centered `CNN` materially improved discrimination on held-out loci
- a thermally-limited full-data CNN run on the full filtered dataset reached strong held-out performance under conservative local hardware constraints

`50k` subset metrics:

| Model | Test AUROC | Test Average Precision | Test F1 |
| --- | ---: | ---: | ---: |
| Logistic baseline | 0.477 | 0.082 | 0.126 |
| Sequence-pair CNN | 0.750 | 0.261 | 0.283 |

Full filtered dataset CNN metrics:

| Model | Test AUROC | Test Average Precision | Test F1 |
| --- | ---: | ---: | ---: |
| Full-data CNN | 0.883 | 0.609 | 0.570 |

Public repo contents:

- code for download, preprocessing, training, evaluation, prediction, and saliency interpretation
- lightweight report artifacts in `reports/`
- compact metric snapshots in `reports/model_metrics.json`
- full methodology and results writeup in `reports/results_summary.md`

Not committed to Git:

- raw ClinVar / FASTA downloads
- processed datasets
- model checkpoints and large local training artifacts

The README and report files preserve the key metrics so the project remains readable without shipping multi-gigabyte files.

## What This Repository Does

This project builds a real genomics-flavored ML workflow:

- downloads `ClinVar` short-variant labels from NCBI
- downloads the `GRCh38` primary assembly FASTA from Ensembl
- filters to binary, high-confidence `benign` vs `pathogenic` SNVs
- extracts fixed windows around each mutation from the reference genome
- creates deterministic region-based train/validation/test splits to reduce local leakage
- trains a `k-mer + logistic regression` baseline
- trains a sequence-pair `CNN` on reference and alternate windows
- generates metrics, ROC/PR/calibration plots, and simple CNN saliency visualizations

## Data Choices

The defaults are intentionally conservative:

- source labels: `ClinVar` VCF on `GRCh38`
- reference: `Homo_sapiens.GRCh38.dna.primary_assembly.fa`
- kept labels:
  - `Pathogenic`
  - `Likely_pathogenic`
  - `Benign`
  - `Likely_benign`
- excluded labels:
  - `Uncertain_significance`
  - `Conflicting_classifications_of_pathogenicity`
  - non-binary significance classes
- kept variants: single-alt SNVs only
- kept review statuses:
  - `criteria_provided,_single_submitter`
  - `criteria_provided,_multiple_submitters,_no_conflicts`
  - `reviewed_by_expert_panel`
  - `practice_guideline`

## Why The Split Is Leakage-Aware

Naive random row splits are too optimistic for sequence-window tasks because nearby loci can leak similar context across train and test.

This repository assigns variants by hashed genomic region bins:

- group key: `chromosome + floor(position / 1 Mb)`
- all variants in the same region stay in the same split
- split assignment is deterministic and reproducible

It is not perfect biology-aware splitting, but it is much more defensible than random row splitting for a first production-style project.

## Project Layout

```text
configs/
reports/
scripts/
src/mutation_effect_prediction/
```

## Environment

### Option 1: pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

If you want the CNN training path on CPU, install PyTorch separately:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you want to force CPU even when a CUDA build is installed:

```bash
mep train-cnn --device cpu
```

### Option 2: conda / mamba

```bash
mamba env create -f environment.yml
mamba activate mutation-effect-prediction
pip install -e .
```

The provided `environment.yml` is CPU-oriented by default.

## Safe Local Training

This project does not require a full-dataset laptop GPU run to be credible.

Recommended safe path on a laptop:

- use the validated `50k` subset result as the main benchmark
- use `configs/safe_local_gpu.toml` if you choose to run on a local GPU
- prefer staged scaling with subset files rather than a single full-speed full-data run
- monitor temperature and stop if the machine gets too hot for your comfort

Create a stratified subset from a processed dataset:

```bash
python scripts/create_dataset_subset.py \
  --input-csv data/processed/clinvar_snv_windows_full.csv.gz \
  --output-csv data/processed/clinvar_snv_windows_250k.csv.gz \
  --target-rows 250000
```

Run a conservative local GPU experiment:

```bash
mep train-cnn \
  --config configs/safe_local_gpu.toml \
  --dataset data/processed/clinvar_snv_windows_250k.csv.gz \
  --output-dir models/cnn_250k_safe \
  --device cuda
```

## End-to-End Usage

### 1. Download official data

This downloads:

- `ClinVar` VCF + tabix index
- `GRCh38` primary assembly FASTA, then decompresses and indexes it with `faidx`

```bash
mep download-data
```

Expected storage:

- `clinvar.vcf.gz`: about 200 MB
- `GRCh38` FASTA `.gz`: about 0.9 GB
- decompressed FASTA: multiple GB

### 2. Build processed SNV dataset

```bash
mep preprocess
```

Outputs:

- `data/processed/clinvar_snv_windows.csv.gz`
- `reports/preprocessing_summary.json`

Each processed row includes:

- `variant_id`
- `chrom`, `pos`, `ref`, `alt`
- `label`, `label_name`
- `split`, `group_id`
- `review_status`, `clinical_significance`
- `gene_symbol`
- `ref_seq`, `alt_seq`

### 3. Train the baseline

```bash
mep train-baseline
```

Outputs:

- `models/baseline/baseline_model.pkl`
- `models/baseline/metrics.json`
- `models/baseline/plots/*.png`

### 4. Train the CNN

```bash
mep train-cnn
```

Outputs:

- `models/cnn/best_checkpoint.pt`
- `models/cnn/metrics.json`
- `models/cnn/plots/*.png`
- `models/cnn/training_history.png`

The trained checkpoint stores the validation-selected classification threshold and uses it during `mep predict`.

### 5. Run inference on prepared sequence windows

Input CSV must contain `ref_seq` and `alt_seq`.

```bash
mep predict \
  --model-type baseline \
  --model-path models/baseline/baseline_model.pkl \
  --input-csv data/processed/clinvar_snv_windows.csv.gz \
  --output-csv reports/baseline_predictions.csv
```

```bash
mep predict \
  --model-type cnn \
  --model-path models/cnn/best_checkpoint.pt \
  --input-csv data/processed/clinvar_snv_windows.csv.gz \
  --output-csv reports/cnn_predictions.csv
```

### 6. Generate a CNN saliency plot

```bash
mep interpret-cnn \
  --checkpoint models/cnn/best_checkpoint.pt \
  --input-csv data/processed/clinvar_snv_windows.csv.gz \
  --variant-id "1:879317:C>T" \
  --output-path reports/saliency_example.png
```

## Smoke Test

The smoke test creates a tiny synthetic FASTA + VCF, preprocesses them, and trains the logistic baseline.

```bash
python scripts/run_smoke_test.py
```

Artifacts land in `data/interim/smoke/`.

## Development

Install the dev extras and run the same checks CI runs:

```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```

A `Makefile` wraps the common tasks:

```bash
make install-dev   # editable install with ruff + pytest
make lint          # ruff check src tests
make test          # pytest
make smoke         # end-to-end synthetic run
```

The pure-function tests need neither `torch`, `pysam`, the network, nor any
data files; the sequence-encoding tests are skipped automatically when `torch`
is not installed. Check the installed version with `mep --version`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and [CHANGELOG.md](CHANGELOG.md)
for release notes.

## Recommended Next Step

The strongest next step is not automatically a transformer. For this repo, the best sequence is:

- stabilize the `50k` and moderate-scale subset results
- add a small scaling curve such as `50k -> 100k -> 250k`
- only then add a transformer comparison with `DNABERT-2` or `Nucleotide Transformer`

That keeps the project honest and readable instead of turning it into a hardware story.

## Official Sources

- ClinVar downloads: <https://www.ncbi.nlm.nih.gov/clinvar/docs/downloads/>
- ClinVar GRCh38 VCF: <https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz>
- Ensembl human FASTA: <https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz>
- samtools faidx reference: <https://www.htslib.org/doc/1.12/samtools-faidx.html>
- VCF specification: <https://samtools.github.io/hts-specs/VCFv4.3.pdf>

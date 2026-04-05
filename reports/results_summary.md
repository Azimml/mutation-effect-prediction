# Results Summary

## Project Framing

This project targets a real genomics machine-learning task:

- input: a single-nucleotide variant plus local `GRCh38` sequence context
- label: binary `pathogenic` vs `benign` derived from filtered `ClinVar`
- scope: `SNVs` only, not indels or structural variants

The goal is not to claim clinical deployment. The goal is to demonstrate credible variant-effect modeling with real public genomics data, leakage-aware evaluation, and reproducible engineering.

## Data Pipeline

Official sources:

- `ClinVar` `GRCh38` VCF from NCBI
- `GRCh38` primary assembly FASTA from Ensembl

Filtering rules:

- keep only single-alt `SNVs`
- keep only canonical chromosomes
- keep only high-confidence review states
- map labels to binary:
  - positive: `Pathogenic`, `Likely_pathogenic`
  - negative: `Benign`, `Likely_benign`
- drop `VUS`, conflicting labels, and non-binary clinical significance classes

Sequence representation:

- `201 bp` window centered on the mutation
- reference sequence window
- alternate sequence window with the mutant base inserted

Splitting strategy:

- deterministic hashed region-based split
- grouping key: chromosome + `1 Mb` genomic bin
- all loci within the same region stay in the same split

This is stricter and more defensible than a naive random row split for sequence-window classification.

## Full Data Summary

From [preprocessing_summary_full.json](preprocessing_summary_full.json):

- total raw `ClinVar` records scanned: `4,398,837`
- kept filtered binary-labeled `SNVs`: `1,413,909`
- benign: `1,249,484`
- pathogenic: `164,425`
- split counts:
  - train: `1,000,502`
  - val: `132,189`
  - test: `281,218`

This full-data preprocessing pass is an important result by itself because it shows the project is not a toy CSV exercise.

## Stable Model Comparison

To keep model training stable and safe on local hardware, the first full model comparison was run on a `50k` kept-variant subset.

Subset composition from [preprocessing_summary_50k.json](preprocessing_summary_50k.json):

- kept records: `50,000`
- benign: `45,648`
- pathogenic: `4,352`
- train: `33,814`
- val: `5,342`
- test: `10,844`

### Baseline

Model:

- `3-mer` TF-IDF over reference and alternate sequence windows
- `LogisticRegression`

From [model_metrics.json](model_metrics.json):

- test AUROC: `0.477`
- test average precision: `0.082`
- test F1: `0.126`

Interpretation:

- the baseline is weak under the leakage-aware split
- that is useful, not embarrassing
- it suggests the split is nontrivial and simple local bag-of-k-mers is not enough

### CNN

Model:

- one-hot encoded reference window
- one-hot encoded alternate window
- explicit mutation-center mask
- 1D convolutional encoder with pooled and mutation-centered features

From [model_metrics.json](model_metrics.json):

- validation AUROC: `0.770`
- test AUROC: `0.750`
- test average precision: `0.261`
- test F1: `0.283`

The public repository keeps the metric snapshot and code, while larger checkpoints remain local-only artifacts.

### Direct Comparison

| Model | Test AUROC | Test AP | Test F1 |
| --- | ---: | ---: | ---: |
| Logistic baseline | 0.477 | 0.082 | 0.126 |
| CNN | 0.750 | 0.261 | 0.283 |

Key takeaway:

- the project already demonstrates meaningful learning signal from local sequence context
- the CNN is materially better than the baseline on held-out loci

## Interpretation

The repository supports saliency-style mutation interpretation through `mep interpret-cnn`, which highlights which nearby positions most influence the prediction around a chosen variant.

This is not a biological causal explanation, but it is a useful model-inspection layer and a stronger portfolio signal than raw probabilities alone.

Example artifact:

- [saliency_1_943995_C_T.png](saliency_1_943995_C_T.png)

## What This Project Honestly Claims

This project supports the following honest claim:

> A leakage-aware CNN trained on real `ClinVar`/`GRCh38` local sequence context can distinguish benign from pathogenic held-out `SNVs` substantially better than a simple k-mer logistic baseline.

This project does **not** claim:

- universal mutation-effect prediction across all variant types
- clinical-grade pathogenicity classification
- complete biological understanding from local sequence alone

## Full-Dataset CNN

The project was also run on the full filtered dataset under a thermally limited local-GPU setup, with repeated cooldown pauses to stay under a conservative laptop temperature ceiling.

Recovered metrics from [model_metrics.json](model_metrics.json):

- best epoch: `3`
- validation AUROC: `0.8857`
- test AUROC: `0.8829`
- test average precision: `0.6092`
- test F1: `0.5702`

Detailed metrics:

| Split | AUROC | Average Precision | F1 |
| --- | ---: | ---: | ---: |
| Train | 0.9457 | 0.7525 | 0.6859 |
| Val | 0.8857 | 0.5941 | 0.5604 |
| Test | 0.8829 | 0.6092 | 0.5702 |

The full-data result strengthens the project substantially because it shows the sequence model still generalizes under the leakage-aware split when scaled to the entire filtered dataset, not just the `50k` subset.

## Why The `50k` Result Still Matters

The `50k` comparison is still important because it cleanly shows the step from baseline to CNN under a controlled subset:

- the baseline fails under the honest split
- the CNN improves sharply even before full-data scaling

That makes the project readable. The full-data CNN is the stronger final result, but the `50k` comparison is the clearest modeling story.

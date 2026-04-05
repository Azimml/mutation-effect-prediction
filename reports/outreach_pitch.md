# Outreach Pitch

## Short Version

I built a genomics ML project that predicts `ClinVar`-style pathogenic vs benign `SNVs` from local `GRCh38` sequence context. The pipeline uses official `ClinVar GRCh38` and reference FASTA files, applies leakage-aware regional splitting, and compares a weak k-mer logistic baseline against a mutation-centered CNN.

On a real `50k` filtered subset, the CNN improved test AUROC from `0.477` to `0.750` and average precision from `0.082` to `0.261`. I then scaled the same approach to the full filtered dataset of `1.41M` SNVs and reached about `0.883` test AUROC and `0.609` average precision.

## What Makes It Strong

- uses real genomics formats: `VCF`, `FASTA`, indexed reference extraction
- avoids naive random row splitting
- filters `ClinVar` into a defensible binary task
- shows clear baseline-to-CNN improvement
- includes interpretation tooling and reproducible training/inference code

## One-Paragraph Intro Message

I’ve been building a small genomics ML project focused on variant-effect prediction from local DNA sequence context. I used official `ClinVar GRCh38` and `GRCh38` reference data, built a leakage-aware preprocessing pipeline for binary-labeled `SNVs`, and compared a k-mer logistic baseline with a mutation-centered CNN. On a held-out `50k` subset, the CNN improved test AUROC from `0.48` to `0.75`, and on the full filtered dataset it reached about `0.883` test AUROC. I’m interested in genomics/AI roles where careful data handling and biologically grounded modeling matter, and I’d value any feedback on the project or pointers to relevant teams.

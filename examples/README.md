# Examples

## `sample_windows.csv`

A tiny, self-contained input for the `mep predict` command. It has the two
columns prediction requires, `ref_seq` and `alt_seq`, plus some context columns
that are simply passed through to the output.

Each row is a `41 bp` window (radius 20) centered on the variant. The `ref_seq`
and `alt_seq` are identical except at the center base, which holds the
reference and alternate allele respectively. This mirrors the layout produced
by `mep preprocess`, just with synthetic flanking sequence so it can live in
the repository.

> These are illustrative synthetic windows, not real ClinVar loci. Use them to
> check that a trained model loads and scores end to end, not to judge accuracy.

### Scoring with a trained baseline

```bash
mep predict \
  --model-type baseline \
  --model-path models/baseline/baseline_model.pkl \
  --input-csv examples/sample_windows.csv \
  --output-csv examples/sample_predictions.csv
```

### Scoring with a trained CNN

```bash
mep predict \
  --model-type cnn \
  --model-path models/cnn/best_checkpoint.pt \
  --input-csv examples/sample_windows.csv \
  --output-csv examples/sample_predictions.csv
```

The output CSV is the input rows with two extra columns appended,
`predicted_probability` and `predicted_label`. If you have not trained a model
yet, see the "End-to-End Usage" section of the top-level `README.md`, or run
`python scripts/run_smoke_test.py` to produce a baseline model on synthetic
data.

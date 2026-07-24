# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `.editorconfig`, `.gitattributes`, and repository metadata files for a
  cleaner contributor experience.
- `CONTRIBUTING.md` describing the development setup and the checks CI runs.

## [0.1.0] - 2026

Initial public prototype.

### Added

- End-to-end pipeline: `download-data`, `preprocess`, `train-baseline`,
  `train-cnn`, `predict`, and `interpret-cnn` commands under the `mep` CLI.
- Leakage-aware, deterministic region-based train/validation/test split
  (chromosome + 1 Mb genomic bin grouping).
- ClinVar significance to binary label mapping with high-confidence review
  status and single-alt SNV filtering.
- `k-mer` + logistic-regression baseline and a sequence-pair 1D CNN with a
  mutation-center mask channel.
- Metrics, ROC/PR/calibration plots, training-history plot, and CNN saliency
  interpretation.
- Thermal-safe local GPU training path with configurable cooldown pauses.
- Test suite for the split, label mapping, sequence encoding, and metric
  helpers, plus a synthetic-data smoke test.
- GitHub Actions CI running ruff and pytest on Python 3.11 and 3.12.

### Security

- CNN checkpoints are loaded with `weights_only=True` to avoid unpickling
  arbitrary objects from untrusted files.

[Unreleased]: https://github.com/Azimml/mutation-effect-prediction/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Azimml/mutation-effect-prediction/releases/tag/v0.1.0

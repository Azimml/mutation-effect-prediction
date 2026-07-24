# Security Policy

This is a research prototype, not a clinical or production system. Even so, a
few security considerations are worth stating.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
["Report a vulnerability"](https://github.com/Azimml/mutation-effect-prediction/security/advisories/new)
advisory flow rather than opening a public issue. Include steps to reproduce
and the affected commit. You can expect an initial response within a few days.

## Handling of untrusted inputs

- **Model checkpoints.** CNN checkpoints are loaded with
  `torch.load(..., weights_only=True)`, which refuses to unpickle arbitrary
  Python objects. Do not disable this when loading checkpoints you did not
  produce yourself; a malicious `.pt` file can execute code on load otherwise.
- **Baseline model files.** `baseline_model.pkl` is a standard Python pickle.
  Only load baseline models from sources you trust.
- **Downloaded data.** `mep download-data` fetches files over HTTPS from NCBI
  and Ensembl. Prefer the official URLs in `configs/default.toml` and verify
  large downloads before processing.

## Scope

This project makes no clinical-grade guarantees. Predictions must not be used
for diagnosis or treatment decisions.

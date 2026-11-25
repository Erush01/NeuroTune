# ADR-001: Use CPU PyTorch for the exploratory N0 model

## Status

Accepted

## Date

2026-11-08

## Context

N0 requires a shared-weight feed-forward encoder, automatic differentiation,
and the contrastive margin loss. The current verified feature catalog has 939
tracks, so a GPU is unnecessary. The course's final-submission language remains
subject to instructor confirmation.

## Decision

Use the CPU-only PyTorch wheel for the Python N0 prototype. Pin it in
`requirements.txt` and train on CPU with fixed seeds and hyperparameters.

## Alternatives considered

- **Manual NumPy backpropagation:** rejected because it duplicates
  autodifferentiation and increases numerical/testing risk.
- **GPU-enabled PyTorch:** rejected because current catalog size does not
  justify the large CUDA runtime dependency.
- **MATLAB implementation:** remains an option if the instructor requires
  MATLAB for the submitted code; confirm before finalizing the course hand-in.

## Consequences

- The CPU environment is reproducible with the pinned requirements.
- Three 50-epoch runs complete on the current subset without GPU hardware.
- N0 results remain exploratory because the held-out graph has only 11
  observed positive edges; this choice does not resolve that data limitation.

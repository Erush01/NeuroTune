# D Modeli Validation Duyarlılığı Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evaluate one-at-a-time input-masking sensitivity of nine frozen D checkpoints using only their A–D CV validation galleries, without retraining or constructing test features/metrics.

**Architecture:** Reuse P0's validated fold input loader, MXM preflight, train-only development features, D ragged dataset and retrieval evaluator. Add a small pure masking/measurement module and a separate read-only-checkpoint runner with strict original-validation parity and isolated no-overwrite output. The original P0 and C namespaces are immutable.

**Tech Stack:** Python, unittest, NumPy, PyTorch, h5py, existing NeuroTune feature/dataset/retrieval APIs.

**Approved design:** `docs/plans/2026-11-17-neurotune-d-modality-sensitivity-design.md`. This workspace is not a Git repo; do not create commits/worktrees. Do not open the old C `experiment.json`, compute old or current test-cohort metrics, or retrain.

---

**Execution status:** All three tasks complete. Task 1 helper: 6 focused tests. Task 2 validation-only runner: 11 focused tests, original P0 parity and logical-versus-byte split-provenance regression. Task 3 real command: exit 0; report/CSV have 45 records and 9 verified checkpoint hashes. Turkish diagnostic report: `docs/NeuroTune_D_Modalite_Duyarliligi_2026-11-17.md`. Final verification: 440 unittest cases, compileall, and pip check all exit 0. Independent final review found no actionable issue; the previously inspected test cohort was not evaluated.

### Task 1 — Pure masking and behavior-change measurements

**Files:** Add `src/evaluate_d_modality_sensitivity.py`; add `tests/test_evaluate_d_modality_sensitivity.py`. Read-only references: `src/model_v1.py`, `src/data/neurotune_v1_dataset.py`, `src/evaluate_neurotune_v1.py`, `src/retrieve.py`.

1. Test first with a tiny D batch containing nonzero audio/BPM/genre/lyrics and padding. Assert `mask_one_modality(batch, "audio")` zeros audio only, leaves `mask`/`lengths`/IDs and other features unchanged, and never mutates the caller; repeat for other three modalities and reject invalid names. Check that the mask applies to all rows.
2. Run `.venv/bin/python -m unittest tests.test_evaluate_d_modality_sensitivity -v` and confirm the new test fails for the missing helper.
3. Implement copying only the selected tensor and replacing it with `torch.zeros_like` (or equivalent without touching nonselected tensors). Keep the boolean time mask identical; do not drop model-required D branches.
4. Test ranking/metric/embedding calculations on tiny deterministic vectors: self-exclusion, ID tie-break, `k=10` clamping, mean per-query Top-10 intersection divided by eligible list length, and per-track original/masked embedding cosine with finite outputs. Use existing ranking/evaluation functions rather than a new ranking implementation. Do not read test labels.
5. Run focused tests (exit 0); review metric definitions and invalid/nonfinite inputs before moving on.

### Task 2 — Isolated validation-only runner with provenance and hard parity

**Files:** Add `scripts/evaluate_neurotune_v1_d_sensitivity.py`; add `tests/test_neurotune_v1_d_sensitivity_cli.py`. Read-only references: `scripts/run_neurotune_v1_full_modal_ablation.py:63-85,148-185,279-375`, `src/data/neurotune_v1_abcd_cv_training.py:16-50,84-102,125-342`, `src/train_neurotune_v1.py:48-58`, `src/evaluate_neurotune_v1.py:77-111`.

1. Add a synthetic runner test with tiny fold/seed fixtures and sentinel test-track features. Assert it reads only P0's D validation records and checkpoint snapshots; uses `build_abcd_cv_development_features` with train+validation song IDs, then slices only validation IDs; rejects a second checkpoint variant, fold/seed/epoch/hash, transform or source mismatch; never creates test features, relevance, embeddings, or metrics.
2. Run focused test to RED. Implement fixed folds 0/1/2 × seeds 684/685/686 and fixed `original/audio/bpm/genre/lyrics` conditions. Keep reusable code small: pure helper for one checkpoint + validation gallery and a thin CLI orchestration layer. Checkpoint SHA-256 must cover the same immutable bytes deserialized; preflight source/manifest/provenance snapshot before evaluating.
3. Add integration test: full unmasked D validation metrics must match P0 `model_validation` record with a fixed small absolute tolerance (e.g. `1e-6`) and equal query/edge counts; injected mismatch must abort before publishing any artifact. Freeze model in `eval()`/`torch.no_grad()`; verify identical original embeddings/ranking before and after masking (no mutation).
4. Add tests for report schema: per-fold/seed/condition retrieval metrics, Top-10 overlap, cosine embedding drift, aggregate means, output/data/code/checkpoint hash provenance, `test_evaluation_deferred=true`, and isolation from the previously inspected test cohort. Explicitly distinguish 9 correlated fold-seed observations from 3 artist folds; no significance claim.
5. Add output collision and injected failure tests. Publish to a new `results/neurotune_v1_d_modality_sensitivity/` namespace only via staged no-overwrite publication, rollback solely new outputs, and refuse pre-existing output before heavy I/O. Run focused tests and request scope/quality review. Never reuse P0's training `_run` (it retrains).

### Task 3 — Real validation-only run, artifacts, and documentation

**Files:** Add short results section to `docs/NeuroTune_D_Modeli_Uygulama_ve_Validation_Sonuclari_2026-11-17.md` and/or a new focused sensitivity results Markdown; do not modify prior result files.

1. Before run, verify new output namespace absent and snapshot required P0 D checkpoint/report hashes. Run `.venv/bin/python -m scripts.evaluate_neurotune_v1_d_sensitivity` once. Do not score old test or use it for tuning.
2. Independently inspect new report: 9 checkpoint × 5 conditions = 45 records, source/checkpoint hashes, all original endpoint parities, finite drift/overlap and metric ranges. Confirm P0/C artifacts remain unchanged. Capture command exit status and concise evidence.
3. Explain in Turkish which input masks change validation behavior most, that zero masking is distribution-shifted and not a causal contribution estimate, and that no test data were evaluated. Do not modify model or choose a new checkpoint using these sensitivity scores.
4. Run `.venv/bin/python -m unittest discover -s tests -q`, `.venv/bin/python -m compileall -q src scripts tests`, `.venv/bin/python -m pip check` (capture each exact exit code); review final diff/artifacts and stop when checks pass. No Git commit is possible in this workspace.

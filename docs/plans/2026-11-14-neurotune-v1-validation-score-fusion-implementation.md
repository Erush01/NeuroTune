# NeuroTune v1 Validation Score-Fusion Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evaluate a small set of score-level blends between the existing three C checkpoints and the C-matched cosine baseline using only the A/B pilot's train and validation partitions.

**Architecture:** Add a separate analysis CLI and tests; do not modify the A/B trainer, checkpoints, split, or reports. Fit the existing transforms on the 5,632 train IDs, create features only for train+validation IDs, embed the 610 validation tracks with the three saved C checkpoints, and combine model and baseline cosine score matrices at five fixed weights. Choose one global weight by mean validation NDCG@10 across seeds, tie-breaking toward the smaller model weight, and write a separate provenance-rich report.

**Tech Stack:** Python 3, PyTorch, NumPy, existing NeuroTune v1 feature/model/retrieval modules, unittest.

---

## Task 1: Score-matrix fusion and retrieval tests

**Files:**
- Create: `scripts/evaluate_neurotune_v1_validation_fusion.py`
- Test: `tests/test_neurotune_v1_validation_fusion.py`

1. Add failing unit tests for a pure helper that fuses normalized model and baseline score matrices with `lambda * model + (1-lambda) * baseline`.
2. Cover invalid/non-finite/asymmetric matrices, invalid lambda, alpha endpoints, self-exclusion, full-gallery ranking, and deterministic tie-breaking by song ID.
3. Build query relevance from unique undirected validation edges and evaluate rankings with the existing `evaluate_rankings` helper at `k=10`.
4. Verify synthetic endpoint cases: lambda 0 reproduces baseline scores exactly; lambda 1 reproduces model scores exactly; intermediate weights are exact convex combinations.
5. Run `.venv/bin/python -m unittest tests.test_neurotune_v1_validation_fusion -v`; confirm the missing-helper failures, implement the smallest helper, and rerun until green.

## Task 2: Test-only validation runner and provenance checks

**Files:**
- Modify: `scripts/evaluate_neurotune_v1_validation_fusion.py`
- Modify: `tests/test_neurotune_v1_validation_fusion.py`

1. Add the production CLI with defaults for the existing A/B pilot inputs under `data/msd_hf_pilot/`, C checkpoints under `results/neurotune_v1_ab/checkpoints/`, the existing A/B report, and a new output namespace `results/neurotune_v1_validation_score_fusion/`.
2. Load the A/B split and `neurotune_v1_validation_edges.csv`; use only train and validation IDs/edges. Do not open the A/B test-edge file, construct test features, or read test metrics into the new report.
3. Read `train_ids` and `validation_ids` from `split["track_ids"]`; do not use `split["track_ids"]["test"]`. Call `build_neurotune_v1_features` with `train_ids=train_ids` and `song_ids=train_ids + validation_ids`, using the original split file's SHA-256. Slice validation rows by explicit ID lookup; verify no test ID is requested or present.
4. Build the fixed cosine baseline with modalities `("audio", "bpm", "genre")`. Load `C_seed_{684,685,686}.pt`, verify each checkpoint hash and transform manifest against the recomputed training-only fit, and produce normalized validation embeddings only.
5. Evaluate fixed weights `{0, .25, .5, .75, 1}` for each seed over the 610-track validation gallery and 185 validation edges (238 positive queries). Select one global weight by mean NDCG@10 across seeds; on ties select the smaller weight. Report all four metrics by seed/weight and unweighted means; do not retrain or compute any test metric.
6. Check the `lambda=0` endpoint against the existing retrieval evaluator applied to the same C-matched baseline embeddings; check `lambda=1` against each selected C checkpoint's recorded validation metrics within a fixed numeric tolerance. The saved A/B report's baseline includes lyrics and is not the C-matched comparator, so do not use its baseline metrics as the `lambda=0` expectation. Treat mismatch as an error.
7. Record split, cache, source, code, and C-checkpoint hashes, transform fit-ID hash, fixed weight grid, per-weight metrics, selection rule, and explicit post-hoc validation caveats. Refuse output collisions; stage and roll back the separate report namespace without touching existing artifacts.
8. Add synthetic runner tests for validation-only ID/edge use, fit-ID isolation, row ordering, endpoint parity, artifact hashes, output collision, and rollback on injected failure. Run `.venv/bin/python -m unittest tests.test_neurotune_v1_validation_fusion -v`.

## Task 3: Real validation-only ablation

**Files:**
- New output only: `results/neurotune_v1_validation_score_fusion/`

1. Confirm the output namespace and lock/staging paths are absent.
2. Run `.venv/bin/python -m scripts.evaluate_neurotune_v1_validation_fusion`.
3. Verify the report contains exactly three seeds × five weights, no test metrics or test IDs, query/edge counts of 238/185, finite metrics, matching baseline/C endpoints, and hashes matching current input files and checkpoints.
4. Recompute the selected lambda from recorded means and verify the documented tie-break. Preserve `results/neurotune_v1_ab/` and its checkpoints byte-for-byte.

## Task 4: Document and final verification

**Files:**
- Modify: `README.md` only if the validation result adds useful, carefully qualified information.
- Modify: this plan with execution results.

1. Document the chosen validation-only fusion weight, per-weight validation metrics, and the fact that this same validation set already selected the C checkpoints.
2. Do not describe validation gains as independent performance evidence; the existing A/B test set remains untouched and has already been inspected in prior work.
3. Run `.venv/bin/python -m unittest discover -s tests -q`.
4. Run `.venv/bin/python -m compileall -q src scripts tests` and `.venv/bin/python -m pip check`.
5. Verify existing A/B outputs and the official-10K CV report/checkpoints/caches are unchanged.

## Execution constraints

The workspace has no Git repository or worktree support; implement in the current directory, do not create commits, and do not modify or overwrite existing experiment artifacts. The A/B Hugging Face archives have unverified provenance/license, and this validation-only analysis is exploratory development work. A new independent artist-held-out cohort is required before making performance or generalization claims.

## Execution log

### Task 1 — complete (2026-11-14)

Created pure helpers for convex score-matrix fusion, full non-self gallery ranking with song-ID tie breaks, unique undirected validation relevance, and retrieval metric evaluation. Tests cover fusion endpoints/intermediate weights, invalid and complex matrices, invalid weights, ranking tie order, validation metrics, and malformed unhashable endpoints. Final focused suite: 6 tests passed. Spec and quality reviews passed; the two minor input-validation findings were fixed and re-reviewed.

### Task 2 — complete (2026-11-14)

Added the train/validation-only runner and synthetic coverage. It fits transforms on train IDs, requests only train+validation features, embeds the existing three C checkpoints once each from the same bytes that are hashed, and evaluates the 5-weight grid against a C-matched audio+BPM+genre cosine baseline. It compares fitted-transform fields separately from selected-view source hashes, checks split/cache/checkpoint provenance and both lambda endpoints, selects one weight by mean validation NDCG with lower-weight tie-break, and publishes a separate report without overwriting a concurrently created output directory. Synthetic tests cover exact IDs, view-specific hashes, rectangular baseline embeddings, endpoints, all 15 seed/weight cases, checkpoint byte identity, report contents, collision, and rollback. Final focused suite: 12 tests passed. Spec and quality reviews passed with no remaining findings.

### Task 3 — complete (2026-11-14)

The first real attempt stopped before publication because the cosine builder returns embeddings, not pairwise scores; after correcting that interface, the second attempt exposed that the stored A/B baseline includes lyrics and therefore is not a valid C-matched lambda=0 expectation. The design and plan were corrected to compare lambda=0 via the existing retrieval evaluator on the same C-matched embeddings. Both failed attempts published no report or altered prior outputs. The final command `.venv/bin/python -m scripts.evaluate_neurotune_v1_validation_fusion` exited 0 and wrote only `results/neurotune_v1_validation_score_fusion/experiment.json`. It records 5,632 train IDs, 610 validation tracks, 185 validation edges, and 238 positive queries. Mean metrics across three seeds by lambda (Recall@10/HitRate@10/MRR@10/NDCG@10): 0.00 = 0.188415/0.247899/0.098513/0.108865; 0.25 = 0.182432/0.239496/0.101899/0.110915; 0.50 = 0.193512/0.250700/0.098728/0.112779; 0.75 = 0.205183/0.261905/0.096629/0.114468; 1.00 = 0.235683/0.292717/0.102130/0.126446. Lambda 1.00 (C alone) was selected. Independent checks confirmed checkpoint endpoints, the selected-weight rule, input/code/checkpoint hashes, and absence of test metric fields. The same validation set had selected the C checkpoints and selected lambda, so this remains post-hoc development evidence, not an unbiased comparison.

### Task 4 — complete (2026-11-14)

Updated the README, design doc, and this execution plan with the C-matched comparison, all five mean validation results, selected lambda, and the validation-reuse/source-license caveats. Final verification after documentation: `.venv/bin/python -m unittest discover -s tests -q` — 322 tests passed; `.venv/bin/python -m compileall -q src scripts tests` passed; `.venv/bin/python -m pip check` reported no broken requirements. The report contains one new output file and its source, code, and checkpoint hashes were independently verified.

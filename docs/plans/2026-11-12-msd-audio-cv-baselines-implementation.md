# MSD Audio CV Baselines Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add reproducible random-ranking and mean-pooled audio-cosine references for the completed three-fold MSD audio-only CV evaluation.

**Architecture:** Implement a separate CLI evaluator and JSON report so the completed model report, checkpoints, caches, and pair trace remain immutable. Reuse the existing CV audit, fold-specific cache normalization, ranking, relevance, and metric contracts; validate source provenance before evaluating.

**Tech Stack:** Python, NumPy, h5py, PyTorch CPU only for cache loading where applicable, existing NeuroTune evaluation/retrieval helpers, `unittest`.

---

## Fixed protocol and constraints

- Use the already-approved 9,939-track three-fold CV audit and its fold-specific caches. Do not regenerate the model or alter its report/checkpoints.
- Evaluate only each held-out fold's internal observed Last.fm edges against its full test gallery; exclude each query from its gallery.
- Random baseline: deterministic uniform ordered top-10 samples without replacement for seeds 684/685/686, repeated per query and fold. This is equivalent to taking the first 10 entries from a full random gallery permutation, without materializing unused gallery candidates.
- Audio baseline: average normalized continuous audio channels 0–24 over time, L2-normalize the resulting track vectors, and rank by cosine. Exclude raw event-count channels 25–27.
- Use the existing Recall@10, HitRate@10, MRR@10, and NDCG@10 definitions. Record random results per fold/seed and cosine results per fold; avoid suggesting the repetitions are independent statistical samples.
- This comparison is post hoc because the existing CV results have already been seen. It is descriptive only and must not be used for model selection or tuning.
- Write only `results/msd_audio_only_cv_baselines.json`; preserve all current outputs. Reject an existing target and clean a newly created report on a caught failure.

## Task 1: Add deterministic ranking baseline helpers

**Files:** Create `scripts/evaluate_msd_audio_cv_baselines.py`; create `tests/test_msd_audio_cv_baselines.py`.

1. Add a failing test for a seeded random-top-k helper: same fold/seed/query inputs give identical rankings; the query is excluded; the top-k has the expected size and contains only unique non-query gallery IDs.
2. Run `.venv/bin/python -m unittest tests.test_msd_audio_cv_baselines -v`; confirm the missing helper causes the expected failure.
3. Implement the smallest helper using a local NumPy generator seeded from the fold and seed and sample only K=10 items without replacement; do not mutate global random state or materialize full gallery rankings.
4. Add a failing test for mean pooling: `(N, 1024, 28)` input becomes `(N, 25)` from channels 0–24, vectors are L2-normalized, and input is not mutated.
5. Run the focused test to confirm the expected failure, implement the helper, and rerun the focused suite.

## Task 2: Validate baseline inputs and provenance

**Files:** Modify `scripts/evaluate_msd_audio_cv_baselines.py`; extend `tests/test_msd_audio_cv_baselines.py`.

1. Add tests that reject a stale CV audit, changed Last.fm tree, mismatched cache `audit_sha256`, wrong fold split/normalization IDs, output collisions, and an existing report target before creating outputs.
2. Run the focused tests and confirm the provenance/output failures occur before report creation.
3. Reuse the audit's `_source_sha256` helper for Last.fm consistency; compare cache metadata's audit hash with the current audit file hash. Reuse `_validate_audit`, `load_audio_sequence_cache`, and the existing evaluation/retrieval helpers where their contracts fit.
4. Keep the completed experiment files read-only; add tests that sentinel model reports/checkpoints are unchanged.

## Task 3: Implement fold-by-fold baseline evaluation and report

**Files:** Modify `scripts/evaluate_msd_audio_cv_baselines.py`; extend `tests/test_msd_audio_cv_baselines.py`.

1. Add a synthetic end-to-end test with three small folds, distinct audio rows, within-fold positive edges, and two seeds. Assert fold-local IDs/relevance, finite metrics, deterministic random scores, pooled-cosine metrics, copied model reference metrics, and correct report structure/hashes.
2. Run `.venv/bin/python -m unittest tests.test_msd_audio_cv_baselines -v` red.
3. Implement the evaluator: validate audit/Last.fm/model-report/cache provenance; load only one fold cache at a time; construct relevance from that fold's positive edges; compute random top-k and cosine rankings; record per-fold queries/gallery/edge counts and metrics.
4. Copy model fold/seed metrics from the existing CV report after validating its audit, Last.fm, and cache hashes. Do not alter the model report, checkpoints, or trace.
5. Write a separate report at `results/msd_audio_only_cv_baselines.json` with source, code, cache, audit, model-report and runtime hashes; record the post-hoc exploratory caveat. Use deterministic report serialization and refuse to overwrite an existing output.
6. Run the focused synthetic suite green.

## Task 4: Run, document, and verify

**Files:** Modify `README.md` with the baseline command and measured results.

1. Run `.venv/bin/python -m scripts.evaluate_msd_audio_cv_baselines` once on the existing CV artifacts.
2. Verify all fold/query/gallery/edge counts match the original CV report; check random ranking reproducibility, query exclusion, finite metrics, input/cache hashes, and that the model report/checkpoints/caches/trace remain unchanged.
3. Add measured per-fold and aggregate baseline metrics to the README. Clearly label this comparison post hoc and exploratory; distinguish three-fold cosine results from nine random/model fold-seed observations.
4. Run final checks:

   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   .venv/bin/python -m compileall -q src scripts tests
   .venv/bin/python -m pip check
   ```

5. Report exact commands, statuses, baseline values, and any limitations. No significance or generalization claims.

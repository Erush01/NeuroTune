# MSD Audio-Only Last.fm Cross-Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Evaluate the fixed MSD-audio-only Siamese protocol with three artist-disjoint Last.fm test folds and fold-specific train-only normalization.

**Architecture:** Add deterministic artist-fold audit data, build one normalized audio cache per outer fold from raw MSD HDF5, and reuse the audio-only model with a no-validation fixed-epoch training path. A separate runner trains three seeds per fold, saves all nine checkpoints before any test evaluation, and reports fold/seed metrics without changing the completed single-split experiment.

**Tech Stack:** Python, NumPy, h5py, PyTorch CPU, existing MSD/Last.fm parsers, existing pair sampler and retrieval metrics, `unittest`.

---

## Fixed protocol and constraints

- Use the 9,939 duplicate-filtered tracks and current Last.fm subset (8,209 unique undirected catalog links). Do not use Taste Profile or the full Last.fm archives; the latter add no catalog links.
- Three outer folds: group by existing artist-key fallback (`artist_id` → `artist_name` → `track_id`), sort artist keys, shuffle with seed 684, distribute groups round-robin. Each fold is test once; the other two folds are train. Exclude all cross-fold edges.
- The preliminary fold assignment is 3,249/3,321/3,369 tracks, with 1,065/965/981 within-fold test links and 5,198 cross-fold links. Recompute and verify these values in the real audit.
- Each fold gets its own audio cache generated from raw MSD HDF5; fit normalization only on that fold's train IDs. Never reuse the 80/10/10 cache for CV.
- Keep the 32-D audio-only encoder, one uniformly sampled unobserved negative per positive, seeds 684/685/686, 50 epochs, batch size 64, Adam 0.001, margin 1.0, and K=10. No validation split or checkpoint selection; report train loss only. This measures the existing fixed-epoch protocol and does not fix its observed overfitting.
- Train 9 models total (3 folds × 3 seeds), save all checkpoints before any test embeddings/relevance, and evaluate only test-fold internal observed Last.fm links against the complete fold gallery.
- Output all CV files under new names. Keep the previous audio-only, M1, M2, N0, and B0 artifacts unchanged. This workspace has no Git repo; do not make worktrees or commits.
- Preflight all report, CSV, and checkpoint targets; refuse existing targets. On an exception/interrupt, remove only files newly created by that invocation and preserve the original exception. Reject an empty seed list before writing.
- Load and release one full fold cache at a time. Verify each cache's audit hash matches the current CV audit, and the current Last.fm tree hash matches the Last.fm source hash recorded in that audit.

## Task 1: Add deterministic artist-fold assignment

**Files:** Create `src/data/audio_only_cv.py`; test `tests/test_audio_only_cv.py`.

1. Write failing tests for artist grouping/fallback, deterministic seed-684 round-robin assignment, exact coverage (every eligible track in one fold), no artist split across folds, and positive/cross-fold edge counting.
2. Run `.venv/bin/python -m unittest tests.test_audio_only_cv -v`; confirm the module/function is missing.
3. Implement a pure `build_audio_only_cv_audit(tracks, edges, duplicate_groups=None, fold_count=3, seed=684)` that applies the same two duplicate policies as the single-split audit, then assigns eligible tracks by artist. Emit JSON-serializable fold IDs/counts, unique undirected edge counts per fold, cross-fold count, source/duplicate statistics, and the fold policy.
4. Run `.venv/bin/python -m unittest tests.test_audio_only_cv -v`.

## Task 2: Add and verify the CV audit CLI

**Files:** Create `scripts/audit_msd_audio_cv.py`; test `tests/test_audio_only_cv_cli.py`.

1. Write temp-source CLI tests for overrides, hashes, missing-source blocked output, fold assignment, edge counts, and deterministic results.
2. Run `.venv/bin/python -m unittest tests.test_audio_only_cv_cli -v` red.
3. Implement repo-root defaults: `data/processed/msd_tracks.csv`, `data/processed/lastfm_similar`, `downloads/msd_duplicates.txt`, output `results/msd_audio_only_cv_audit.json`, seed 684, fold count 3. Expose path/seed overrides and record chunked source SHA-256 values.
4. Run the focused suite, then `.venv/bin/python -m scripts.audit_msd_audio_cv`.
5. Assert 9,939 tracks are covered once; verify the real fold track/link counts and 5,198 excluded cross-fold links before proceeding.

## Task 3: Build three fold-specific normalized audio caches

**Files:** Create `scripts/preprocess_audio_cv.py`; test `tests/test_preprocess_audio_cv.py`; generate three new HDF5/metadata pairs.

1. Write synthetic HDF5 tests with three artist folds; assert each cache contains exactly its train+test IDs, has an empty validation list, and its normalization `train_ids` equal only that fold's train IDs.
2. Run `.venv/bin/python -m unittest tests.test_preprocess_audio_cv -v` red.
3. Implement a CLI that reads the master CV audit and calls `build_audio_sequence_cache` once per fold from the raw MSD HDF5 input. Pass the master CV audit path for audit hashing and `split_ids={"train": other_two_folds, "validation": [], "test": held_out_fold}`. Output:
   - `data/processed/msd_audio_only_cv_fold_0.h5` and `_meta.json`
   - `data/processed/msd_audio_only_cv_fold_1.h5` and `_meta.json`
   - `data/processed/msd_audio_only_cv_fold_2.h5` and `_meta.json`
4. Run the focused suite and `.venv/bin/python -m scripts.preprocess_audio_cv`.
5. Verify each cache is finite float32 `(9939, 1024, 28)`, has the same sorted full ID catalog, and records fold-specific train-only normalization; verify the existing single-split M1 and audio-only caches remain unchanged.

## Task 4: Support fixed-epoch training without validation data

**Files:** Modify `src/train_audio_only.py`; test `tests/test_train_audio_only.py`.

1. Add a failing test that calls `train_audio_only_seed(..., validation_pools=None)` and asserts finite train losses, `validation_loss is None`, and an empty validation pair trace. Keep existing validation-enabled behavior unchanged.
2. Run `.venv/bin/python -m unittest tests.test_train_audio_only -v` red.
3. Add the minimal optional no-validation branch. Preserve all audio/ID/pool checks, per-epoch random negatives, deterministic seeds, and existing 1:1 positive/negative policy.
4. Run `.venv/bin/python -m unittest tests.test_train_audio_only -v`.

## Task 5: Add the separate cross-validation runner

**Files:** Create `scripts/run_msd_audio_cv.py`; test `tests/test_msd_audio_cv_cli.py`.

1. Write a synthetic end-to-end test with three artist folds and distinct per-ID audio rows. Assert no artist/edge leakage, correct train-ID alignment, random-only negatives, nine fixed-seed checkpoints saved before any test relevance/encoding, finite metrics, and report/curve/pair/hash artifacts.
2. Run `.venv/bin/python -m unittest tests.test_msd_audio_cv_cli -v` red.
3. Implement runner defaults for the CV audit, fold caches/metadata, Last.fm subset, output, and checkpoint directory. Validate fold coverage/disjointness, the audited Last.fm hash, each fold cache's audit hash, exact split, and train-only normalization provenance. Load/release one full cache at a time. Preflight all outputs and reject existing files; clean newly created outputs on failed runs without masking the original error. Reject empty seeds. Train with other two folds only; do not create test embeddings/relevance until all nine checkpoints and the completed pair-trace hash are fixed.
4. Write:
   - `results/msd_audio_only_cv_experiment.json`
   - `results/msd_audio_only_cv_training_curves.csv`
   - `results/msd_audio_only_cv_sampled_pairs.csv`
   - `results/checkpoints/msd_audio_only_cv_fold_{0,1,2}_seed_{684,685,686}.pt`
5. Report per-fold/per-seed Recall@10, HitRate@10, MRR@10, and NDCG@10; include descriptive fold summaries, fixed configuration, train IDs, source/code/cache hashes, runtime versions, trace hash, and a caution that CV results are exploratory and the larger fold gallery is not directly comparable to the previous 1,033-track gallery.
6. Run `.venv/bin/python -m unittest tests.test_msd_audio_cv_cli -v`.

## Task 6: Document, run, and verify

**Files:** Modify `README.md` with a dedicated CV section; preserve the prior audio-only results and warnings.

1. Document the three audit/cache/run commands, fold-specific normalization, fixed protocol, outputs, cross-fold exclusion, and exploratory interpretation.
2. Run `.venv/bin/python -m scripts.run_msd_audio_cv` with 3 folds × 3 seeds × 50 epochs.
3. Verify the report has 3 folds/9 seeds, expected per-fold track/link counts, fold-specific train-only normalization, 450 curve rows, partition-local traces with only valid negatives, nine checkpoints, and matching source/code/cache/trace hashes. Confirm the previous single-split result files and caches remain unchanged. Verify an injected mid-run failure cleans newly created outputs and permits a fresh retry.
4. Run final checks:

   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   .venv/bin/python -m compileall -q src scripts tests
   .venv/bin/python -m pip check
   ```

5. Report exact commands, exit statuses, counts, and metrics with no significance or generalization claims.

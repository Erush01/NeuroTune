# NeuroTune v1 Official-10K Artist-Grouped CV Implementation Plan

> **For implementer:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the validation-selected NeuroTune v1 C model against a source-verified, artist-grouped 3-fold CV on the official 10K MSD subset's 939-track complete-modality catalog.

**Architecture:** Keep this path separate from the completed A/B pilot. Build a raw ragged 26-channel cache from official 10K HDF5 records, split artists into three outer folds, then edge-split the non-test positives into train and validation. Fit transforms on non-test tracks per fold, train variant C with known-positive-safe negative sampling, compare with a same-modality cosine baseline, and evaluate each held-out fold only after the nine fold/seed checkpoints are fixed.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, h5py, existing MSD/MXM/CD2C/Last.fm parsers, `unittest`.

---

## Fixed decisions

- Use only the official `data/msd_subset_raw/MillionSongSubset/` HDF5 subset and the 939 duplicate-filtered full-feature IDs recorded in `results/data_audit.json`. Do not read any A/B Hugging Face archive for this CV.
- Use `lastfm_train.zip` as the official positive-edge source. The expected common-catalog graph is 600 unique undirected edges. `lastfm_test.zip` is not used because its common-catalog edge count is zero.
- Outer split: sorted artist keys with `artist_id → artist_name → track_id` fallback, shuffle with seed 684, assign artists round-robin to three folds. Expected folds are 323/310/306 tracks and 63/78/79 within-fold test edges. Of the 600 edges, 220 are within-fold test edges and 380 cross-fold edges are excluded from every outer test set.
- For each held-out test fold, collect all 289/262/269 edges whose endpoints are in the other two folds. Shuffle with `random.Random(684 + fold_index)`, reserve `int(0.20 * edge_count)` validation edges, and use the remaining 232/210/216 edges for training. Validation edges are held out by edge, not by track; their endpoints may appear in training edges. Train/validation positive labels must be disjoint.
- Fit all transforms on all non-test tracks for that outer fold. No test fold track contributes to audio/BPM/IDF/genre fitting.
- Train only variant C (audio+BPM+genre), with Adam 0.001, batch 64, cosine triplet margin 0.2, 50 epochs, seeds 684/685/686. Select by validation NDCG@10; ties use lower validation triplet loss, then earlier epoch.
- Train and validation schedules may use only non-test IDs. Both must exclude every observed non-test positive neighbor from negative candidates, including validation positives. Each fold's test edges are excluded from its train/validation and negative pools; those labels may be training positives in other folds as standard CV.
- Compare against a deterministic equal-modality cosine baseline with the same C modalities (audio summary, BPM, genre). Keep the existing four-modality baseline behavior as the default elsewhere.
- This is artist-grouped CV on official train labels, not an independent external test. Labels may be training positives in other folds, as expected in k-fold CV. Keep results exploratory.
- Use new outputs only: `data/processed/neurotune_v1_official10k_audio_30s_26ch.h5`, `data/processed/neurotune_v1_official10k_audio_30s_26ch_meta.json`, `results/neurotune_v1_official10k_cv_audit.json`, and `results/neurotune_v1_official10k_cv/`. Refuse pre-existing targets; do not alter prior caches/reports/checkpoints.
- The workspace has no Git repository; work in place and do not create commits/worktrees.

## Task 1: Official 10K CV audit and edge split manifest

**Files:** Create `src/data/neurotune_v1_official10k_cv.py`, `scripts/audit_neurotune_v1_official10k_cv.py`, `tests/test_neurotune_v1_official10k_cv.py`, and `tests/test_audit_neurotune_v1_official10k_cv.py`.

1. Add pure-builder tests for deterministic artist folds, exact 20% edge validation allocation, disjoint train/validation/test positive edges, cross-fold exclusion, and order independence.
2. Run `.venv/bin/python -m unittest tests.test_neurotune_v1_official10k_cv -v`; confirm failures before implementation.
3. Load the 939 eligible IDs from the union of `results/data_audit.json` split IDs and validate that parsed `data/processed/msd_tracks.csv`, MXM counts, and CD2C genres cover every selected ID.
4. Reuse the existing size-limited `_lastfm_edges` archive reader from `scripts/audit_msd_hf_ab.py` with `lastfm_train.zip` and the 939 eligible IDs. Confirm its canonical edge set equals the known 600 within-catalog edges.
5. Assign artists using the approved seed/fallback policy. For each held-out fold, compute its internal test edges, non-test pool edges, deterministic 20% validation edge split using seed `684 + fold_index`, remaining train edges, and cross-to-test exclusions. Store sorted IDs and canonical unique undirected edges.
6. Add a CLI that validates inputs, hashes every source and relevant code file, reports missing inputs with unknown counts, refuses output collisions, and atomically writes `results/neurotune_v1_official10k_cv_audit.json`.
7. Run `.venv/bin/python -m unittest tests.test_neurotune_v1_official10k_cv tests.test_audit_neurotune_v1_official10k_cv -v` and the real audit CLI. Confirm 939 IDs, 600 source edges, fold tracks 323/310/306, train edges 232/210/216, validation edges 57/52/53, and test edges 63/78/79 before cache creation.

## Task 2: Raw official-10K ragged audio cache

**Files:** Create `src/data/preprocess_neurotune_v1_official10k.py`, `scripts/preprocess_neurotune_v1_official10k.py`, and `tests/test_preprocess_neurotune_v1_official10k.py`.

1. Add synthetic tests for ragged offset/ID round-trip, raw unnormalized values, deterministic ordering, source hashing, output collisions, and rollback on extraction/write failure.
2. Run `.venv/bin/python -m unittest tests.test_preprocess_neurotune_v1_official10k -v`; confirm the expected missing-module failures.
3. Resolve official subset paths from MSD IDs (`track_id[2] / track_id[3] / track_id[4] / track_id.h5`) and validate each path/ID using `preprocess_audio_window` before writing.
4. Write `features`, `offsets`, and `track_ids` with the same ragged HDF5 contract used by `src/features_neurotune_v1.py`. Preserve raw float32 26-channel values; do not fit normalization here.
5. Record `[0,30)` window/channel schema, 939 track count, total segment count, audit/source hashes, code hashes, and per-track source hashes in `data/processed/neurotune_v1_official10k_audio_30s_26ch_meta.json`. Stage HDF5 and metadata, refuse any existing target, and clean newly created files on any exception/interruption.
6. Run the focused suite and real cache builder. Verify all IDs, offsets, finite raw values, channel schema, metadata hashes, and preprocessing round trips for representative tracks.

## Task 3: Known-positive-aware triplet schedules

**Files:** Modify `src/triplets.py`; extend `tests/test_triplets.py`.

1. Add tests showing that a training or validation schedule can sample positives from one edge subset while excluding all known positives from a separate superset; reject exclusion edges outside the supplied ID set and positive edges absent from the known-positive set.
2. Run `.venv/bin/python -m unittest tests.test_triplets -v` and confirm the new API tests fail before implementation.
3. Add an optional `known_positive_edges` parameter to `sample_triplet_schedule`. Keep its default behavior identical by treating `positive_edges` as the known-positive set when the optional value is omitted. Build negative adjacency from all known edges but emit directed positive rows only from `positive_edges`.
4. Run the focused suite. Confirm the existing A/B v1 sampler tests remain unchanged and pass.

## Task 4: C-matched cosine baseline

**Files:** Modify `src/evaluate_neurotune_v1.py`; extend `tests/test_evaluate_neurotune_v1.py`.

1. Add a C-modality baseline test for audio+BPM+genre, including equal weighting, row normalization, zero modalities, and omission of lyrics.
2. Run `.venv/bin/python -m unittest tests.test_evaluate_neurotune_v1 -v` and confirm the C-specific baseline test fails before implementation.
3. Add an explicit modality selection argument or a dedicated helper for the three C modalities. Preserve the existing all-four-modality baseline as the default; continue to use batched ranking and shared retrieval metrics.
4. Run the focused suite and assert the old default baseline results/ordering remain unchanged.

## Task 5: CV-specific edge-validation trainer

**Files:** Create `src/train_neurotune_v1_cv.py` and `tests/test_train_neurotune_v1_cv.py`.

1. Add synthetic tests for edge-level train/validation labels sharing the same non-test track set, validation no-grad behavior, deterministic updates, selected epoch/tie-break, known-positive negative exclusions, and absence of test inputs in the function signature.
2. Run `.venv/bin/python -m unittest tests.test_train_neurotune_v1_cv -v`; confirm the intended missing-module failures.
3. Implement a fold/seed trainer over one non-test feature dataset, with `train_positive_edges`, `validation_positive_edges`, `known_positive_edges`, and separate train/validation schedules. Require train and validation labels to be disjoint and their union to equal the known non-test edge set. Verify each schedule's positive index pairs match its corresponding edge set and that no schedule negative is a known positive neighbor. The validation positive edges must not enter optimizer updates; validation schedules use a fixed fold-specific seed and fixed negatives across epochs.
4. Evaluate validation retrieval against the complete non-test gallery after every epoch. Select by maximum validation NDCG@10, then lower fixed validation triplet loss, then earliest epoch. Never accept test IDs, test edges, or test embeddings.
5. Save the selected model state, fold, seed, config, transform manifest, fit-ID hash, split/source/code/cache/schedule hashes, selected epoch, and validation metrics; refuse checkpoint overwrite.
6. Run focused tests for the CV trainer and existing v1 trainer/model modules.

## Task 6: Isolated CV runner and integration tests

**Files:** Create `scripts/run_neurotune_v1_official10k_cv.py` and `tests/test_run_neurotune_v1_official10k_cv.py`.

1. Add synthetic integration tests proving each fold's test IDs are absent from its train/validation transforms and schedules, train/validation edge labels are disjoint, known non-test positives are never sampled as negatives, and all nine checkpoints exist before test relevance/embeddings are created.
2. Add tests for audit/cache/hash mismatch, wrong catalog boundary, report/CSV/checkpoint collisions, atomic publication, and rollback on exception/interruption.
3. Run `.venv/bin/python -m unittest tests.test_run_neurotune_v1_official10k_cv -v` red.
4. Implement real-input preflight: exact 939 eligible IDs; audit source edge counts; cache IDs and raw normalization marker; fold/schedule counts; no prior output targets; and source/code/cache hashes.
5. For each fold, fit transforms using the other two folds' track IDs; build train/validation datasets over only those non-test tracks; create one train schedule per fold/seed and one fixed validation schedule per fold; train C for seeds 684/685/686.
6. Save all nine selected checkpoints and serialize training curves before constructing any test relevance or embeddings. Then evaluate C and the C-matched cosine baseline on the same held-out fold gallery and edges.
7. Publish only to `results/neurotune_v1_official10k_cv/`; report fold/seed metrics, fixed settings, source and artifact hashes, excluded-edge counts, caveats, and descriptive aggregate metrics. Remove staging files on failure/interruption without touching existing artifacts.
8. Run the focused runner suite and confirm the synthetic all-nine-checkpoint barrier.

## Task 7: Real official-10K CV run

**Files:** Only new audit/cache/CV artifacts; do not modify prior experiment outputs.

1. Confirm all new targets are absent before each generating command.
2. Run the official CV audit CLI; verify exact 939/600 catalog coverage and per-fold split counts.
3. Build and verify the separate raw 26-channel ragged cache from official 10K HDF5 files.
4. Run `.venv/bin/python -m scripts.run_neurotune_v1_official10k_cv` with fixed settings.
5. Recompute all report source/code/cache/split/schedule/checkpoint hashes. Verify 9 checkpoints, expected curves and schedules, finite metrics, validation-only epoch selection, and fold-local test isolation.

## Task 8: Final docs and verification

**Files:** Update `docs/plans/2026-11-14-neurotune-v1-official10k-cv-design.md`, this implementation plan, and `README.md` only as needed.

1. Document the source-verified 939-track scope, edge-level validation policy, per-fold counts, C-vs-cosine results, and the fact that CV uses previously used official train labels.
2. Run `.venv/bin/python -m unittest discover -s tests -q`.
3. Run `.venv/bin/python -m compileall -q src scripts tests`.
4. Run `.venv/bin/python -m pip check`.
5. Recheck that `results/neurotune_v1_ab/`, all earlier experiment outputs, and M1/M2/audio-only caches are unchanged.
6. Report results as exploratory CV robustness evidence only; do not call it an independent external test or make significance/generalization claims.

## Execution constraints

Work in the current directory because the workspace is not a Git repository. Do not create commits, worktrees, or delete unrelated outputs. Delegate only non-overlapping scopes; require focused test evidence and re-review each implementation lane before the next dependent task.

## Execution log

### Task 1 — complete (2026-11-14)

Created `src/data/neurotune_v1_official10k_cv.py`, `scripts/audit_neurotune_v1_official10k_cv.py`, and focused tests. The audit reads the 939 duplicate-filtered common-catalog IDs from `results/data_audit.json`, verifies MSD/MXM/CD2C coverage, reuses the size-limited official Last.fm train ZIP reader, and records source/code hashes. The real artifact is `results/neurotune_v1_official10k_cv_audit.json`: 939 tracks, 600 source edges, 380 cross-fold edges; fold tracks 323/310/306; test edges 63/78/79; non-test pool edges 289/262/269; train edges 232/210/216; validation edges 57/52/53. The public builder now validates IDs, metadata, edge shapes/endpoints, and seed values. Final focused suite: 9 tests passed. After the validation fix, the audit was regenerated to a temporary target, checked identical except for the updated builder code hash, then atomically replaced. Spec and quality reviews passed; the initial minor API-validation finding is closed.

### Task 2 — complete (2026-11-14)

Created `src/data/preprocess_neurotune_v1_official10k.py`, its CLI, and focused tests. The new raw cache at `data/processed/neurotune_v1_official10k_audio_30s_26ch.h5` contains 939 sorted tracks and 98,420 `[0,30)` segments in float32 `(T,26)` form; matching metadata records the current code hashes, audit hash, and all 939 source-HDF5 hashes. Hardening rejects noncanonical/path-traversing IDs and identical resolved output targets. Publication uses staged files plus no-clobber hard links, with rollback and a race regression test. Final focused suite: 9 tests passed. A regeneration to temporary paths exited 0 in 4.26 seconds with 70,904 KiB maximum RSS; feature/offset/ID datasets exactly matched the prior build, all 939 source hashes independently recomputed, and three sampled tracks round-tripped byte-exactly. The verified cache and current metadata were installed at their final paths. Spec and quality reviews passed; the initial memory concern was closed for this fixed scope by measured payload/RSS evidence, and the concurrent-writer finding was closed by no-clobber publication.

### Task 3 — complete (2026-11-14)

Extended `sample_triplet_schedule` with optional `known_positive_edges`. Scheduled positives still come only from `positive_edges`; negative exclusions use the full known-positive adjacency, and scheduled edges must be contained in that set. The omitted-argument behavior is fixed by a legacy expected-schedule regression. Added checks for withheld-positive exclusion, subset validation, and non-iterable track IDs. Final focused suite: 12 tests passed. Spec and quality reviews passed; the minor non-iterable-ID error inconsistency was fixed and re-reviewed.

### Task 4 — complete (2026-11-14)

Extended `build_neurotune_v1_cosine_baseline` with explicit modality selection while preserving the four-modality default. The CV baseline can now use only audio+BPM+genre and does not require or validate lyrics. Tests cover equal modality contribution, zero vectors, selected-feature validation, and default compatibility. Final focused suite: 11 tests passed. Spec and quality reviews passed; malformed and unhashable modality entries now consistently raise `ValueError`.

### Task 5 — complete (2026-11-14)

Created `src/train_neurotune_v1_cv.py` as a separate C-only trainer over one non-test dataset, with edge-disjoint train/validation positives. It validates that the edge sets partition known non-test positives, that schedule positive pairs match their assigned edge sets, and that no negative is a known positive neighbor. Validation loss/ranking run under no-grad; selection is NDCG@10, then lower validation loss, then earlier epoch. Checkpoints require split/source/code/cache SHA-256 provenance and record those hashes plus schedule/known-edge hashes. Publication stages the checkpoint and uses atomic no-clobber `os.link`. Final focused suite: 10 tests passed. Spec and quality reviews passed; the provenance requirement and concurrent overwrite race were fixed and re-reviewed.

### Task 6 — complete (2026-11-14)

Created `scripts/run_neurotune_v1_official10k_cv.py` and synthetic integration tests. The runner preflights the official 939-track audit and separate raw `[0,30)` T×26 cache, verifies input/cache/preprocessing hashes and fold counts, and fits C features only on each fold's non-test pool. It builds known-positive-safe train/validation schedules, trains the three fixed seeds, and requires all nine checkpoints, schedules, and curves before held-out feature construction and evaluation. It then compares C with the matched audio+BPM+genre cosine baseline and reports all four metric aggregates, provenance, and exploratory caveats. Output uses a sibling `O_EXCL` namespace lock, staged publication, and rollback; cache finiteness checks use bounded chunks. Final focused suite: 9 tests passed; actual `_preflight(ROOT)` passed for 939 tracks, 600 source edges, 3 folds, 98,420 segments, and 939 raw-HDF5 source hashes. Spec and quality reviews passed; the initial cache-metadata assumptions, output reservation race, and test-barrier coverage gaps were fixed and re-reviewed.

### Task 7 — complete (2026-11-14)

The already-generated official audit and raw cache were source/hash-preflighted rather than regenerated. Two initial production starts exposed runner handoff issues (checkpoint parent creation and the feature builder's requirement that its song view include all fit IDs); both failed stages and locks were rolled back, and all pre-existing artifacts remained unchanged. The corrected fixed run completed in 213.88 seconds. It produced the report, 450 training-curve rows, nine schedules, and nine checkpoints. Independent artifact checks confirmed hashes, fold-local schedules and test isolation, validation-based selected epochs, finite metrics, and the all-nine-checkpoint barrier. C's unweighted mean over nine fold-seed observations was Recall@10 0.2618, HitRate@10 0.3270, MRR@10 0.1202, NDCG@10 0.1440; the C-matched cosine mean over three folds was 0.2322, 0.2910, 0.1106, 0.1307. These are descriptive, differently sized observation sets over reused official train labels, not independent or inferential results. The pre-run comparison confirmed all 53 existing results files and 15 top-level processed files were byte-for-byte unchanged.

### Task 8 — complete (2026-11-14)

Updated the design document and README with the verified fold counts, C-versus-cosine metrics, artifact locations, and limitations on label reuse, unverified negatives, and inference. Final verification after the documentation update: `.venv/bin/python -m unittest discover -s tests -q` — 310 tests passed; `.venv/bin/python -m compileall -q src scripts tests` passed; `.venv/bin/python -m pip check` reported no broken requirements. The result/checkpoint/schedule hashes, selected epochs, test isolation, and prior-artifact snapshots were independently checked after the completed run.

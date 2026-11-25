# NeuroTune D4 Multi-window Audio Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and evaluate D4 as an isolated audio-input change against D0, using start/middle/end 30-second views on the unchanged 13,822-track artist-grouped cohort.

**Architecture:** Add a timestamp-aware three-window extractor and a new immutable cache; keep the current opening-window cache and D0–D3 outputs unchanged. Fit audio statistics on unique source segments from each fold's training tracks, represent the three windows with the existing shared CNN/projection and mean only valid 64D projected vectors before the unchanged D fusion. Reuse D0 folds, schedules, trainer seed/optimizer/epochs/checkpoint selection, and evaluate against D0 plus a recomputed D4-matched cosine baseline.

**Tech Stack:** Python 3.12, NumPy, h5py, PyTorch CPU, existing tar/HDF5 and NeuroTune v1 pipeline, `unittest`.

---

## Locked protocol and invariants

- Cohort: exactly the existing 13,822 A–D tracks and three artist-grouped folds; retain track IDs, labels, candidate pools, and edge sources unchanged.
- D4 isolates only multi-window audio. Do not add D1 mining, D2 dropout, or D3 combination. Do not initialize from D0/D1/D2 checkpoints.
- D0 trainer seeds 684/685/686, folds 0/1/2, 50 epochs, batch 64, Adam at 0.001, margin 0.2, D0 uniform triplet schedules, and validation NDCG@10 checkpoint selection remain unchanged.
- Window starts for duration `D`: `0`, `max(0,(D-30)/2)`, `max(0,D-30)`. Use half-open timestamp intervals; allow overlap. For `D < 30`, all three views reuse the available opening sequence. Empty individual views are invalid and excluded from mean pooling; zero valid views is a fatal preflight error. Do not drop tracks.
- Preserve source segment indices and timestamps in the D4 cache. Fit normalization on train IDs only, counting each `(track_id, source_segment_index)` at most once despite window overlap.
- Use the same CNN/projection weights for all windows. Average raw projected 64D vectors over valid windows and pass that mean to the unchanged fusion; retain only D0's final fused-vector L2 normalization.
- Recompute D4-matched cosine using the same three-window audio representation. Do not label the original first-window D0 cosine as matched to D4.
- Preserve D0's known-positive exclusion policy, including official/legacy test-source edges only as train-local negative exclusions. No independent-test features, embeddings, relevance, targets, or metrics.
- New cache, pilot, and final result namespaces must refuse existing targets. Never overwrite the current 30-second cache or any D0–D3 artifact.

## Task 1: Add timestamp-aware three-window extraction — complete

**Files:**
- Modify `src/data/preprocess_audio_window.py` without changing `preprocess_audio_window` behavior.
- Extend `tests/test_audio_window_preprocessing.py`.

**Steps:**
1. Add failing cases for start/middle/end placement, fractional center/end starts, half-open timestamp membership, duration clipping, overlap, `D < 30` duplication, empty-window validity, and returned source segment indices/timestamps.
2. Run `.venv/bin/python -m unittest discover -s tests -p 'test_audio_window_preprocessing.py' -q`; verify new cases fail while existing opening-window cases remain unchanged.
3. Add one new function, for example `preprocess_audio_windows(h5_path, expected_track_id, window_seconds=30.0)`, that reads the HDF5 arrays once and returns three ordered feature sequences plus window starts/validity and source segment indices/timestamps.
4. Re-run the focused test command; verify existing and new extraction tests pass.

**Completion evidence:** `.venv/bin/python -m unittest discover -s tests -p 'test_audio_window_preprocessing.py' -q` passed (18 tests). Spec-compliance and code-quality reviews approved the extractor and focused tests. The direct timestamp-order comparison also rejects descending unsigned timestamps.

## Task 2: Build a separate, provenance-checked D4 cache — complete

**Files:**
- Modify `src/data/preprocess_msd_hf_abcd.py` by adding a separate multi-window run path; preserve existing `run(...)` defaults and output schema.
- Create `scripts/preprocess_neurotune_v1_d4_audio_windows.py` with explicit CLI paths for the A/B/C/D tar archives, audit, catalog, and new outputs. Default cache outputs are `data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch.h5`, `abcd_audio_start_mid_end_30s_26ch_meta.json`, and `abcd_audio_start_mid_end_30s_26ch_rejections.json`.
- Extend `tests/test_preprocess_msd_hf_abcd.py` and add CLI coverage in a D4 preprocessing test file if needed.

**Steps:**
1. Add failing tiny-tar/HDF5 tests for three windows, window offsets, source segment indices/timestamps, `window_valid`, sorted exact track IDs, empty individual windows, short-track duplication, archive/catalog checksums, and existing-output refusal.
2. Run the focused preprocessing tests and verify the new cases fail before implementation.
3. Add a D4-only cache schema with `features` (`float32`, `N×26`), sorted `track_ids`, per-track `window_offsets` with four cumulative boundaries, per-row original `segment_indices` and `segment_starts`, per-track `window_starts`, `window_valid`, and `song_durations`. Empty windows have equal adjacent offsets and `window_valid=False`.
4. Reuse the existing A–D catalog/archive validation and SHA-256 checks; write cache metadata with the new window formula, channel contract, source/audit/catalog/code hashes, counts, and rejection reasons. Do not publish a cache if the expected 13,822 IDs are missing or any track has no valid window. Stage outputs and roll back caught publication errors; do not treat the three sequential file links as group-atomic under abrupt process termination, and require all three outputs plus complete metadata before consumption.
5. Keep the existing first-window cache, metadata, and rejection file byte-for-byte unchanged; test that the new CLI refuses a pre-existing D4 cache target.
6. Run `.venv/bin/python -m unittest discover -s tests -p 'test_preprocess_msd_hf_abcd.py' -q` plus the D4 preprocessing test file; expect all cases to pass.

**Completion evidence at implementation:** `.venv/bin/python -m unittest discover -s tests -p 'test_preprocess_msd_hf_abcd.py' -q` passed (20 tests), and `.venv/bin/python -m scripts.preprocess_neurotune_v1_d4_audio_windows --help` exited 0. Spec-compliance and code-quality reviews approved the D4 cache and CLI after adding the missing-output-parent regression. The production archive scan was completed later in Task 8; see its measured coverage, runtime, and RSS evidence below.

## Task 3: Add fold-local, train-only multi-window audio features — complete

**Files:**
- Modify `src/features_neurotune_v1.py` with a D4-specific cache reader/statistics path; preserve existing `fit_audio_stats` behavior for the original cache.
- Modify `src/data/neurotune_v1_abcd_cv_training.py` with a D4 multi-window feature builder that reuses the existing fold-local catalog/lyrics workflow.
- Extend `tests/test_features_neurotune_v1.py` and `tests/test_neurotune_v1_abcd_cv_training.py`.

**Steps:**
1. Add failing tests proving that overlapping copies of one source segment are elementwise-averaged into one statistics-only representative per track, only training IDs contribute, empty windows remain marked invalid, transformed per-window rows preserve their original track/window/source-index alignment, and invalid segment indices/timestamps (negative or beyond song duration) are rejected.
2. Run the focused feature and CV-training tests; verify the new cases fail before implementation.
3. Add a D4-specific reader that validates nonnegative int64 source segment indices, finite float64 timestamps within each track's `[0,song_duration]` range, positive durations, row/window alignment, and segment order inside each window. Add statistics that group by `(track_id, source_segment_index)`, elementwise-average that segment's window-specific feature rows for statistics only, and then compute the same channelwise mean/std used by D0. The per-window training inputs keep their original clipped-duration rows.
4. Add a D4 builder that applies these train-only statistics to each valid window and returns window features, validity, unchanged BPM/genre/lyrics features, and transform/source provenance.
5. Re-run `.venv/bin/python -m unittest discover -s tests -p 'test_features_neurotune_v1.py' -q` and `.venv/bin/python -m unittest discover -s tests -p 'test_neurotune_v1_abcd_cv_training.py' -q`.

**Completion evidence:** `.venv/bin/python -m unittest discover -s tests -p 'test_features_neurotune_v1.py' -q` passed (20 tests), and `.venv/bin/python -m unittest discover -s tests -p 'test_neurotune_v1_abcd_cv_training.py' -q` passed (26 tests). Spec-compliance and code-quality reviews approved the D4 reader/statistics/builder. The reader requires explicit track IDs, reads only selected slices, and fits overlap-averaged representatives online; the CV builder selects train+validation only and does not load test feature rows.

## Task 4: Support ragged multi-window batches without changing D0 batches — complete

**Files:**
- Modify `src/data/neurotune_v1_dataset.py` to accept either existing `audio` sequences or D4 `audio_windows` plus a validity mask.
- Extend `tests/test_neurotune_v1_dataset.py`.

**Steps:**
1. Add failing tests for a three-window track, an empty middle/end window, a short-track repeated view, mixed sequence lengths, and unchanged single-window D0 collation.
2. Run `.venv/bin/python -m unittest discover -s tests -p 'test_neurotune_v1_dataset.py' -q`; verify new D4 cases fail and existing single-window tests pass.
3. Add a D4 batch contract: `audio_windows` has shape `(B,3,Tmax,26)`, timestep `mask` has `(B,3,Tmax)`, and `window_mask` has `(B,3)`. Invalid windows use padded zeros and an all-false timestep/window mask; every track must have at least one valid window.
4. Preserve the existing single-window return keys and tensor shapes exactly; run the focused dataset tests again.

**Completion evidence:** `.venv/bin/python -m unittest discover -s tests -p 'test_neurotune_v1_dataset.py' -q` passed (15 tests). Spec-compliance and code-quality reviews approved the dataset/collator contract after adding explicit D4 malformed-row validation. D0 batch keys and shapes remain unchanged.

## Task 5: Add shared-encoder mean pooling for D4 — complete

**Files:**
- Modify `src/model_v1.py` with an explicit multi-window path while preserving the existing `audio` 3D path.
- Modify `src/train_neurotune_v1.py` only where batch-key routing is explicit (notably `_validation_rankings`).
- Extend `tests/test_model_v1.py` and `tests/test_train_neurotune_v1.py`.

**Steps:**
1. Add failing model tests for shared weights across windows, mean of valid raw 64D projected vectors, exclusion of invalid windows, rejection of zero valid windows, fixed 64D fusion input/output, gradient flow, and unchanged D0 single-window behavior.
2. Add a failing trainer test that trains and validates on the D4 collator output using the real `train_neurotune_v1` API; verify D0 and D3 single-window trainer tests remain unchanged.
3. Extend `NeuroTuneV1Encoder.forward` to accept D4 `audio_windows`, timestep mask, and `window_mask`; run the existing Conv1d/projection weights only for valid windows, average raw projected vectors over valid windows, and pass the result to the unchanged auxiliary fusion.
4. Extend `forward_triplets` accepted keys and trainer validation routing to pass the D4 window mask. Keep all existing modality dropout behavior untouched; D4 config sets dropout to zero.
5. Run `.venv/bin/python -m unittest discover -s tests -p 'test_model_v1.py' -q` and `.venv/bin/python -m unittest discover -s tests -p 'test_train_neurotune_v1.py' -q`; also run D3 regression tests.

**Completion evidence:** Model tests passed (15), trainer tests passed (20), and `test_neurotune_v1_d3_combined_cli.py` passed (14). Spec-compliance review approved. The quality review's three-window-count concern was rechecked and withdrawn: the existing shape validator enforces exactly three windows, and the two-window rejection test passes.

## Task 6: Add the D4-matched cosine baseline — complete

**Files:**
- Add a D4-specific multi-window cosine builder in `src/evaluate_neurotune_v1.py`; do not change existing D0 baseline semantics.
- Extend `tests/test_evaluate_neurotune_v1_cv.py`.

**Steps:**
1. Add failing tests for per-window mean summaries, mean across valid windows, invalid-window exclusion, equal-modality weighting, finite normalized embeddings, and unchanged existing D0 baseline outputs.
2. Run `.venv/bin/python -m unittest discover -s tests -p 'test_evaluate_neurotune_v1_cv.py' -q` and verify the new D4 cases fail before implementation.
3. Implement the new D4 builder: summarize each transformed window over its segment rows, average valid-window summaries, then use the current equal-modality cosine weighting and normalization.
4. Re-run the focused evaluator tests.

**Completion evidence:** `.venv/bin/python -m unittest discover -s tests -p 'test_evaluate_neurotune_v1_cv.py' -q` passed (12 tests). Spec-compliance and code-quality reviews approved the separate D4 builder; the existing D0 baseline remains unchanged.

## Task 7: Add the isolated D4 config and runner — complete

**Files:**
- Create `config/neurotune_v1_d4_multi_window_audio.json`.
- Create `scripts/run_neurotune_v1_d4_multi_window_audio.py`.
- Create `tests/test_neurotune_v1_d4_multi_window_audio_cli.py`.

**Steps:**
1. Add failing config tests for exact D0 folds/seeds/epochs/batch/optimizer/margin/schedule, window layout, pooling, normalization placement, reference namespaces, and output namespace; reject booleans, changed values, and extra fields.
2. Add failing runner tests for D0 report/checkpoint/schedule hash validation using the actual nine-D-row report shape in both pilot and full modes, exact track/fold coverage, fresh D4 initialization, unchanged trainer seeds, D0 schedule reuse, validation counts, D4-matched cosine, no test-derived artifacts, and output collision refusal. Include at least one successful synthetic `_run` execution for pilot and full scopes.
3. Run `.venv/bin/python -m unittest discover -s tests -p 'test_neurotune_v1_d4_multi_window_audio_cli.py' -q`; confirm new cases fail before implementation.
4. Implement the runner from the D0 full-modal runner's established loading, train/validation, validation selection, hash, and staged-publish patterns, but train only variant D with D4 features. Read `results/neurotune_v1_full_modal_ablation/experiment.json`; require `status == "complete"`, validate the full fixed set of D references and select the requested pilot/full `(fold, seed)` pairs without rejecting valid out-of-scope references. D0 checkpoints are `checkpoints/fold_{fold}_D_seed_{seed}.pt`; the shared schedules are `triplet_schedules/fold_{fold}_seed_{seed}.npz` with `train_ids`, `validation_ids`, `train_schedule`, and `validation_schedule`. Reuse those schedules byte-for-byte and never load D0 model weights.
5. Validate the D4 cache, metadata, and rejection sidecar before use: require complete metadata, zero rejections, the approved 30-second start/middle/end formula and raw float32/26-channel schema, HDF5/sidecar agreement, current preprocessing-code hashes, matching HDF5/metadata IDs, and exact equality of sorted D4 IDs with the original D0 cache IDs. Snapshot their hashes and source/code provenance. Check test IDs against the development IDs but pass only `{fold, train, validation}` into the feature builder; never provide test IDs or read their feature rows.
6. Use separate namespaces: `results/neurotune_v1_d4_multi_window_audio_pilot/` for the fixed resource pilot and `results/neurotune_v1_d4_multi_window_audio/` for the full run; refuse overwrites.
7. Pair D4 validation rows to D0 D rows by exact `(fold, seed)`. Store the D0 D cosine reference separately by fold from `cosine_validation`; recompute the D4-matched cosine from D4 features, using modalities `audio`, `bpm`, `lyrics`, and `genre`. Record that D4 audio transform hashes naturally differ from D0; compare/verify the unchanged non-audio transform components rather than requiring full transform-hash equality.
8. Persist 9 fold/seed records, 9 checkpoints, 9 verified schedule references/copies, 450 training-curve rows, D0 and D4-matched cosine comparisons, window-coverage diagnostics, runtime, and data/code/split/transform/reference/output hashes. Mark the experiment `validation_only_development` and explicitly record that no independent-test artifacts were created.
9. Add `--pilot` to run only fold 0/seed 684 for the fixed 50-epoch resource pilot; the default command runs the full fixed protocol. Both modes use distinct output namespaces and refuse existing targets.
10. Re-run the focused D4 runner test file; expect all config, reference, synthetic runner, pilot/full namespace, and no-overwrite tests to pass.

**Completion evidence:** The frozen config and runner validate all nine D0 D references before selecting pilot/full scope, hash-check but never load D0 checkpoints, verify shared schedule bytes, validate D4 cache/metadata/rejection sidecars and preprocessing code hashes, require exact D0/D4 track-ID equality, and pass only train/validation partitions to the feature builder. Synthetic pilot and full runner executions cover seeds, schedule copies, output counts, cosine records, and test-ID exclusion. Spec and code-quality reviews approved. The D4 CLI tests passed (8), model tests (15), trainer tests (20), cosine evaluator tests (12), `compileall -q src scripts tests`, and runner `--help`; no production extraction or training was run.

## Task 8: Preflight, resource pilot, and fixed full run — complete

**Files/artifacts:**
- New cache under `data/msd_hf_abcd_pilot/` with a D4-specific basename; never replace `abcd_audio_30s_26ch.h5`.
- Pilot and final results under their separate D4 namespaces.

**Steps:**
1. Run focused extraction, feature, dataset, model, trainer, baseline, and D4 runner tests.
2. Run `.venv/bin/python -m unittest discover -s tests -q`, `.venv/bin/python -m compileall -q src scripts tests`, `.venv/bin/python -m pip check`, `.venv/bin/python -m scripts.preprocess_neurotune_v1_d4_audio_windows --help`, and `.venv/bin/python -m scripts.run_neurotune_v1_d4_multi_window_audio --help`; do not extract or train if any fails.
3. Run `/usr/bin/time -v -o /tmp/opencode/neurotune_d4_cache_timing.txt .venv/bin/python -m scripts.preprocess_neurotune_v1_d4_audio_windows` once to create the separate multi-window cache. This full archive scan is also the coverage preflight: verify archive hashes, exact equality with the 13,822 D0 track IDs, per-window validity/coverage, segment/timestamp integrity, and non-overwrite protections; record elapsed time and maximum RSS. If extraction exceeds available memory headroom, stop before training and revise the writer to stream rather than retain and concatenate the entire feature set. Abort without changing the cohort if source coverage is incomplete. Before consuming the cache, verify all three outputs exist and metadata status is complete; inspect any partial outputs after abrupt interruption before cleanup.
4. Verify fold-local train-only transforms and their deduplicated source-segment accounting on the completed cache.
5. Run a synthetic end-to-end smoke using the real D4 model/trainer, including empty-window masking and mean pooling; all synthetic outputs stay in a temporary directory.
6. Run `.venv/bin/python -m scripts.run_neurotune_v1_d4_multi_window_audio --pilot` into its separate namespace without hyperparameter tuning; verify checkpoint/schedule hashes, runtime, memory, and unchanged D0 references.
7. If cache preflight and pilot pass, run `.venv/bin/python -m scripts.run_neurotune_v1_d4_multi_window_audio` once for folds 0/1/2 and seeds 684/685/686. Do not alter the locked protocol based on pilot validation scores.
8. Independently verify runner exit, 9 checkpoints, schedules identical to D0, 450 curves, D0 and D4-cosine comparisons, finite metrics/runtime, window coverage, every output/reference hash, unchanged D0–D3 artifacts, and absence of test-derived features/metrics.

**Preflight evidence (2026-11-22):** The 10 focused suites passed (163 tests total); the full suite passed (546 tests), `compileall -q src scripts tests`, `pip check`, and both CLI `--help` commands passed. At cache-extraction start, 11.74 GiB was available, the four source archives totaled about 30.5 GiB, and the original D0 feature matrix occupied about 0.14 GiB. The D4 cache extraction was launched with `/usr/bin/time -v -o /tmp/opencode/neurotune_d4_cache_timing.txt`; stdout/stderr are logged at `/tmp/opencode/neurotune_d4_cache_run.log`. Extraction and coverage validation are complete; measured results follow.

**Cache/transform evidence:** Extraction exited 0 in 5:11.40 with maximum RSS 1,096,616 KiB. It produced 13,822 tracks, 4,003,002 feature rows (315.18 MiB HDF5), zero rejections, and 13,822/13,813/13,813 valid start/middle/end windows; all tracks had at least two valid windows. D4 IDs exactly matched the D0 cache; window starts matched the approved formula; offsets, validity flags, segment order, timestamps, metadata IDs and segment count were consistent. For each fold, D4 features contained exactly train+validation IDs, fit-ID hashes matched train IDs, unique-source-segment row counts matched `audio_fit_rows`, non-audio transforms matched D0, and audio statistics were finite with positive standard deviations. The real model/trainer smoke passed (three D4 model tests and one D4 trainer test).

**Pilot evidence:** The fixed fold-0/seed-684 pilot exited 0 and published one checkpoint, one byte-identical D0 schedule copy, and 50 finite epoch rows. The selected epoch was 10; D4 validation NDCG@10 was 0.092061 (one fold/seed, validation-only and not evidence of generalization). The D4 cosine baseline and three separate D0 cosine references were recorded. Checkpoint/schedule/output hashes and every protected input/reference/code hash matched; no independent-test artifact was created. Wall time was 2:21:03 and peak RSS was 2,364,004 KiB. The full 3-fold × 3-seed run is the next step; the pilot score will not be used to tune the frozen protocol.

**Full-run evidence:** The fixed default command `.venv/bin/python -m scripts.run_neurotune_v1_d4_multi_window_audio` exited 0. It completed the nine fold/seed runs, 450 curve rows, nine checkpoints, nine exact D0 schedule copies, and three D4-matched cosine records in 15:58:25 wall time with peak RSS 2,959,092 KiB. All metrics were finite; checkpoint, schedule, transform, curve, and protected-input/reference/code hashes matched. D0–D3 output artifacts matched their own recorded manifests. No independent-test features, embeddings, relevance, or metrics were created.

**Task 8 comparison summary:** D4 validation means were Recall@10 0.127762, HitRate@10 0.205434, MRR@10 0.076902, and NDCG@10 0.075412. The paired D0 means were 0.119240/0.194019/0.072913/0.071530. D4-matched cosine means were 0.128981/0.219120/0.089028/0.080621. Fold-mean NDCG@10 differences D4−D0 were +0.004770/+0.000460/+0.006415; paired seed differences were positive for 8/9 observations. D4 remained below its matched cosine on overall means and in folds 1/2. These are descriptive validation-only findings, not evidence of superiority or generalization.

## Task 9: Update results documentation — complete

**Files:**
- Update `README.md` with D4 protocol, environment/run commands, results, window/extraction costs, D0/D4-cosine comparisons, limitations, and validation-only caveats.
- Update `NeuroTune_v1_Performans_Iyilestirme_Deney_Plani.md` with D4 completion and the same cautious interpretation.

**Steps:**
1. Compute all documented means, fold deltas, realized coverage, extraction size/time, training runtime, and selected-epoch summaries from saved artifacts with a script; do not copy values by hand from partial logs.
2. Add a D4 results section without changing D0–D3 scientific values or claiming superiority/generalization.
3. Verify every reported number against `experiment.json` and the CSV/NPZ files; search for stale D4-pending wording and confirm existing D0–D3 namespaces remain intact.

**Completion evidence:** `README.md` now records the isolated D4 protocol, cache construction and run commands, extraction/training resource costs, aggregate and fold-level validation/cosine comparisons, window coverage, provenance paths, and explicit validation-only caveats. `NeuroTune_v1_Performans_Iyilestirme_Deney_Plani.md` records the final D4 design, cache/train-only transform checks, results, limitations, and completed status. Numbers were computed from the saved D4 `experiment.json`, training curve, cache metadata, and timing files; output and D0–D3 artifact manifests were verified. No scientific or numerical values in prior D0–D3 sections were changed.

## Workspace constraint

At the time of this work (2026-11-22), the workspace had no Git metadata, so worktree isolation, `git diff`, commits, and merge steps were unavailable. Keep changes additive or narrowly scoped, preserve the current D0–D3 outputs, and record exact test/run evidence in the implementation plan and experiment artifacts.

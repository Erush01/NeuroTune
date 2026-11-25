# NeuroTune Full-Modal Validation Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add lyrics to the existing A–D fold feature path and produce an isolated, validation-only A/B/C/D ablation without evaluating the previously inspected test cohort.

**Architecture:** Reuse the existing `NeuroTuneV1Encoder`, including its D lyrics branch, and the existing train-fitted BPM/TF-IDF/genre/audio transformations. Keep the completed C run immutable and use its checkpoints, schedules, validation metrics, and curves read-only after verifying fold, seed, schedule, and active-transform parity. Train A/B/D on the same train/validation folds in a new namespace; do not create test features, embeddings, relevance, or metrics.

**Tech Stack:** Python, NumPy, h5py, PyTorch, CSV/JSON/NPZ, the existing `unittest` suite.

---

**Approved design:** `docs/plans/2026-11-16-neurotune-full-modal-ablation-design.md`  
**User plan:** `docs/NeuroTune_Sonraki_Adimlar_ve_Deney_Plani.md`  
**Important constraints:** Freeze `results/neurotune_v1_abcd_cv/`; do not score the old test cohort; do not use test-source links as training/validation positives; retain them only in the approved known-positive negative-exclusion set. Use a new result namespace. This workspace is not a Git repository, so worktree and commit steps are unavailable.

## Execution status

- [x] Task 1 — model dimension contract and documentation correction (spec/quality review passed; 9 focused tests passed).
- [x] Task 2 — candidate-filtered MXM parser and provenance/coverage preflight (spec/quality review passed; 25 focused tests passed).
- [x] Task 3 — train/validation-only multimodal fold features (spec/quality review passed; 38 focused tests passed).
- [x] Task 4 — variant-specific datasets in the shared trainer (spec/quality review passed; 10 focused tests passed).
- [x] Task 5 — isolated validation-only A/B/C/D runner (spec/quality review passed; 13 focused tests and compileall passed; read-only fold-0 memory preflight peaked at 881.29 MiB with 10,213 MiB available).
- [x] Task 6 — real validation run, documentation, and final verification (report `status: complete`; 38 artifacts and 56 hashes independently checked; checkpoint metadata 36/36; 423 tests, compileall, and `pip check` passed. Host did not return the run process exit code).

## Task 1: Correct and lock the A/B/C/D model-dimension contract

**Files:**
- Modify: `tests/test_model_v1.py`
- Modify: `docs/NeuroTune_durum_raporu_2026-11-16.md`
- Modify: `docs/NeuroTune_Sonraki_Adimlar_ve_Deney_Plani.md`
- Verify only unless needed: `src/model_v1.py`

**Step 1: Add failing architecture-width assertions.** Assert that A has no fusion layer, B's first fusion layer accepts 72 inputs, C accepts 88, and D accepts 120. Also assert that the active modality sets remain A=`()`, B=`("bpm",)`, C=`("bpm", "genre")`, D=`("bpm", "genre", "lyrics")`.

**Step 2: Run the focused test and confirm the new assertions fail or expose a mismatch.**

Run: `.venv/bin/python -m unittest tests.test_model_v1 -v`  
Expected: Existing tests pass; new contract assertions pass only if current code matches the inspected architecture. Do not change layer widths to satisfy stale documentation.

**Step 3: Correct documentation.** Record C as `88→128→64` and D as `120→128→64`; state that D's encoder already exists. Add a short code-audit decision to the user-provided next-steps document, including the user's decision to defer all current test evaluation until a new independent cohort.

**Step 4: Re-run the focused model tests.**

Run: `.venv/bin/python -m unittest tests.test_model_v1 -v`  
Expected: All model tests pass; existing encoder behavior and checkpoints remain unchanged.

## Task 2: Add candidate-filtered MXM parsing and coverage preflight

**Files:**
- Modify: `src/data/parse_mxm.py`
- Modify: `tests/test_parse_mxm.py`
- Modify: `src/data/neurotune_v1_abcd_cv_training.py`
- Modify: `tests/test_neurotune_v1_abcd_cv_training.py`

**Step 1: Write failing parser tests.** Cover an optional exact `track_ids` filter for official train/test text and normalized long-form extracts; preserve the existing unfiltered API. Verify train/test duplicate counts merge, non-candidate tracks are not retained, and an explicit empty MXM record remains distinguishable from an absent record.

**Step 2: Run parser tests and confirm the filtered entry point fails.**

Run: `.venv/bin/python -m unittest tests.test_parse_mxm -v`  
Expected: The new filtered-parser tests fail before implementation; existing parser behavior remains covered.

**Step 3: Implement minimal candidate-filtered parsing.** Parse by canonical MSD `track_id` only, skip storing non-catalog tracks, preserve strict count/vocabulary validation, and collect SHA-256 hashes for each consumed MXM source file. Do not silently replace missing catalog records with empty documents.

**Step 4: Add A–D lyric-coverage preflight tests.** Assert every candidate ID is present in the parsed MXM map, missing IDs fail before training/output creation, and present-but-empty documents are counted explicitly.

**Step 5: Implement the preflight helper.** Add a fold-training loader/helper that parses the verified MXM files for the 13,822 catalog IDs, returns only those sparse documents, and records raw-file and parsed-subset hashes plus present/empty counts.

**Step 6: Run the focused parser and data tests.**

Run: `.venv/bin/python -m unittest tests.test_parse_mxm tests.test_neurotune_v1_abcd_cv_training -v`  
Expected: Exact ID coverage and empty-document policy pass; existing A–D loader/C behavior is unchanged.

## Task 3: Build train/validation-only multimodal fold features

**Files:**
- Modify: `src/data/neurotune_v1_abcd_cv_training.py`
- Modify: `tests/test_neurotune_v1_abcd_cv_training.py`
- Reuse: `src/features_neurotune_v1.py`

**Step 1: Write failing fold-feature tests.** For a synthetic fold, assert that only train and validation IDs are requested, the returned feature keys are audio/BPM/genre/lyrics, lyrics use the existing 5,000-term representation, and changing validation/test lyric values cannot alter the train-fitted IDF or other transform parameters.

**Step 2: Run the focused test and confirm the development feature helper is missing.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv_training -v`  
Expected: The new helper tests fail while prior C-only tests pass.

**Step 3: Implement the isolated development feature builder.** Call the existing `build_neurotune_v1_features` with the filtered MXM mapping and `song_ids=sorted(train_ids + validation_ids)` only. Fit all learned transforms on train IDs. Keep the existing C-only `build_abcd_cv_fold_features` behavior unchanged.

**Step 4: Verify feature-memory bounds and exact alignment.** Assert no test ID appears in returned features, each modality is aligned to the requested IDs, and one fold is built and released at a time. Do not create a three-fold dense lyric matrix.

**Step 5: Re-run feature and data tests.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv_training tests.test_features_neurotune_v1 -v`  
Expected: Train-only IDF, exact lyric alignment, and unchanged audio/BPM/genre C transforms pass.

## Task 4: Support variant-specific datasets in the shared trainer

**Files:**
- Modify: `src/train_neurotune_v1.py`
- Modify: `tests/test_train_neurotune_v1.py`
- Reuse: `src/data/neurotune_v1_dataset.py`, `src/model_v1.py`

**Step 1: Add failing A/B trainer tests with only active modalities.** A uses audio only; B uses audio+BPM. Ensure neither dataset needs to carry unused lyrics/genre matrices. Keep current C/D tests and trainer call sites compatible.

**Step 2: Run focused trainer tests and confirm the unconditional genre requirement fails.**

Run: `.venv/bin/python -m unittest tests.test_train_neurotune_v1 -v`  
Expected: New A/B cases fail at current genre-width validation/model construction; current C/D cases pass.

**Step 3: Implement variant-aware validation/model construction.** Validate that train and validation datasets contain each active modality. Require matching genre widths for C/D only; use a harmless positive genre width for A/B, which have no genre branch. Do not pass inactive modalities to the encoder.

**Step 4: Run trainer, model, and dataset tests.**

Run: `.venv/bin/python -m unittest tests.test_train_neurotune_v1 tests.test_model_v1 tests.test_neurotune_v1_dataset -v`  
Expected: A/B/C/D training contracts pass, with old default behavior unchanged.

## Task 5: Add the isolated validation-only A/B/C/D runner

**Files:**
- Create: `scripts/run_neurotune_v1_full_modal_ablation.py`
- Create: `tests/test_neurotune_v1_full_modal_ablation_cli.py`
- Reuse: Tasks 1–4, `src/evaluate_neurotune_v1.py`, `src/train_neurotune_v1.py`
- Read-only references: `results/neurotune_v1_abcd_cv/checkpoints/`, `triplet_schedules/`, `training_curves.csv`

**Step 1: Write failing synthetic runner tests.** Cover fixed folds/seeds/settings, exact C checkpoint and schedule hash/parity checks, A/B/D variant-specific datasets, one fold at a time, matched-modality validation cosine baselines, 27 newly trained checkpoints, 9 shared schedules, and 1,800 unified curve rows (1,350 new A/B/D + 450 read-only C reference).

**Step 2: Add test-boundary tests.** Instrument feature construction/evaluation to prove that no test IDs or test edges are passed to feature builders, datasets, relevance creation, or evaluators. Assert the new runner uses `build_abcd_cv_development_features` and never calls the historical full-fold C helper `build_abcd_cv_fold_features`. Assert the runner never opens the old C `experiment.json`; C validation selection metrics come from checkpoints and C curves come from `training_curves.csv`. A malformed/sentinel old C report should not affect the synthetic run. Assert the new report has no test metric fields and marks test evaluation as deferred. The known-positive superset may be used only for negative exclusion.

**Step 3: Run focused tests and confirm the runner is absent.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_full_modal_ablation_cli -v`  
Expected: New runner import/integration tests fail.

**Step 4: Implement the runner.** Use fixed production settings: variants A/B/D newly trained, seeds 684/685/686, 50 epochs, batch 64, Adam 0.001, margin 0.2. Read C checkpoints and their validation selection metrics, and read only train/validation rows from `training_curves.csv`; never open the old C `experiment.json` or copy existing test metrics. Reuse the same fold/seed schedules after verifying IDs and hashes. Compute per-variant matched cosine validation references. Do not create test features, embeddings, relevance, or results.

**Step 5: Stage and publish safely.** Write to `results/neurotune_v1_full_modal_ablation/`; refuse any existing target, stage all outputs, publish without clobbering, and roll back only files created by this run. Preserve `results/neurotune_v1_abcd_cv/` byte-for-byte.

**Step 6: Re-run synthetic integration tests.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_full_modal_ablation_cli -v`  
Expected: All settings, C-reference parity, train/validation-only behavior, baseline matching, hashes, no-test boundary, and rollback tests pass.

## Task 6: Run the validation-only ablation, document, and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/NeuroTune_Sonraki_Adimlar_ve_Deney_Plani.md`
- Modify: `docs/NeuroTune_durum_raporu_2026-11-16.md`
- Update: `docs/plans/2026-11-16-neurotune-full-modal-ablation-design.md`
- Preserve: all existing `results/`, data, cache, and split artifacts

**Step 1: Run preflight and snapshot protected artifacts.** Verify the exact MXM candidate coverage and empty-record count; confirm the new result target is absent; snapshot the current C namespace and prior data/result files.

**Step 2: Run the fixed validation-only experiment.**

Run: `.venv/bin/python -m scripts.run_neurotune_v1_full_modal_ablation`  
Expected: Publish 27 newly trained A/B/D checkpoints, 9 schedules, 1,800 curve rows, validation-only model/baseline metrics, and the read-only C reference. No test metric is produced.

**Step 3: Independently verify output and test boundary.** Recompute MXM/source/code/split/transform/schedule/checkpoint hashes; verify 36 variant/fold/seed validation records, 12 matched-cosine fold/variant records, selected epochs, known-positive exclusions, absence of test metric fields, and unchanged prior artifact hashes.

**Step 4: Update documentation from measured results.** Correct C/D fusion dimensions, record that D existed before this work, report validation-only outcomes and missing-lyrics counts, and state that current test evaluation remains deferred until a new independent cohort exists. Do not make generalization or superiority claims from validation.

**Step 5: Run final verification.**

Run: `.venv/bin/python -m unittest discover -s tests -q`  
Expected: All tests pass.

Run: `.venv/bin/python -m compileall -q src scripts tests`  
Expected: Exit 0.

Run: `.venv/bin/python -m pip check`  
Expected: `No broken requirements found.`

## Execution notes

- Keep the existing C report/checkpoints/schedules immutable; C is a read-only reference for this validation phase.
- No evaluation of the previously inspected test cohort is approved in P0. A future independent artist-held-out cohort must be precommitted before final test evaluation.
- The project workspace is not a Git repository. Do not create a worktree or claim a commit; make small, test-first changes in the current workspace.

# NeuroTune A–D C-Model Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train the existing NeuroTune v1 C model across the verified three-fold A–D artist split and compare it with a C-matched cosine reference, reporting the two Last.fm test sources separately.

**Architecture:** Keep the existing encoder, trainer, triplet sampler, retrieval metrics, and baseline utilities. Add an isolated A–D CV input loader and runner that validates the cache/split provenance, fits features per fold on training IDs, and uses source-separated edge targets. Stage all C checkpoints and run artifacts in a fresh namespace before any held-out test embedding or relevance construction.

**Tech Stack:** Python, NumPy, h5py, PyTorch, existing `unittest` suite, CSV/JSON/NPZ/HDF5 artifacts.

---

**Approved design:** `docs/plans/2026-11-15-neurotune-abcd-c-cv-training-design.md`  
**Important project constraints:** Preserve all existing A/B, official 10K CV, and A–D cache/split artifacts. The entire 1,940-edge old A/B test-source file is excluded from new target files; known positives are exclusions only. Do not combine the two held-out label sources or tune on test metrics. The current workspace has no Git repository, so worktree and commit steps are unavailable.

## Task 1: Add an A–D CV training-input loader and preflight

**Files:**
- Create: `src/data/neurotune_v1_abcd_cv_training.py`
- Create: `tests/test_neurotune_v1_abcd_cv_training.py`
- Reference: `src/data/neurotune_v1_abcd_cv.py`, `data/msd_hf_abcd_pilot/cv/manifest.json`

**Step 1: Write failing loader tests.** Build a small temporary audit, catalog, HDF5 cache, cache metadata, manifest, known-positive CSV, and three sets of fold edge files. Cover valid loading, missing/duplicate IDs, stale input hashes, cache-sidecar HDF5 provenance mismatch, overlapping track partitions, old A/B artist in a test fold, and old A/B test-source edge in any target.

**Step 2: Run the focused tests and confirm the expected failures.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv_training -v`  
Expected: New loader import or validation assertions fail.

**Step 3: Implement the minimal input loader.** Read the A–D audit/catalog/cache metadata and CV artifacts. Verify the manifest's recorded input digests and source-edge counts against the files, validate the HDF5 audit/catalog/source/code attributes against its sidecar, check sorted cache IDs and exact cache-to-catalog coverage, and semantically validate the legacy A/B split and artist assignments. Require the known-positive CSV to match its manifest count/digest and the cache-valid union of both source edge files; require the full old A/B test-source set to appear there but never in a fold target. Recompute all four target edge sets per fold from the source files, partition IDs, and old-edge exclusion set; compare them with both target CSVs and manifest records. Keep train-source and official-test-source test labels separate.

**Step 4: Re-run the focused tests.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv_training -v`  
Expected: All loader and provenance tests pass; invalid inputs fail before output creation.

**Task 1 status — complete (2026-11-15):** Implemented `load_abcd_cv_training` and 15 focused tests. The loader verifies HDF5-to-sidecar provenance, source/input hashes and counts, cache/catalog coverage, the legacy split semantics, known-positive completeness/digest, the full A/B edge exclusion, and exact source-derived fold targets. The focused command passed all 15 tests. A real read-only load validated 13,822 tracks, 75,690 known positives, all 1,940 old A/B test-source exclusions, and the published fold/edge counts without changing artifacts.

## Task 2: Build fold-specific C inputs without reading lyrics

**Files:**
- Modify: `src/data/neurotune_v1_abcd_cv_training.py`
- Modify: `src/data/neurotune_v1_dataset.py`
- Test: `tests/test_neurotune_v1_abcd_cv_training.py`
- Test: `tests/test_neurotune_v1_dataset.py`
- Reuse: `src/features_neurotune_v1.py`, `src/data/neurotune_v1_dataset.py`, `src/model_v1.py`

**Step 1: Add failing tests for C-only fold features and batches.** Assert that audio statistics, BPM mean/std, and genre vocabulary are fitted only on fold training IDs; no MXM parsing occurs; and a C-mode `NeuroTuneV1Dataset` can hold only audio, BPM, and genre. Its collated batch must omit lyrics while the default dataset/collator still includes them for existing experiments.

**Step 2: Run the focused tests and confirm the C-only dataset path fails.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_dataset tests.test_neurotune_v1_abcd_cv_training -v`  
Expected: Dataset construction or collation fails because lyrics are currently mandatory.

**Step 3: Implement the fold C-feature helper and optional dataset modalities.** Reuse `build_neurotune_v1_features` with the fold's training IDs and complete fold IDs, passing an empty lyric mapping instead of reading MXM. Discard the unused zero lyric matrix. Add an explicit optional modality list to `NeuroTuneV1Dataset` with the existing all-auxiliary default preserved; make collation include exactly the available auxiliary keys. Build C datasets with audio, BPM, and genre only, then verify `NeuroTuneV1Encoder("C", ...)` receives no lyrics.

**Step 4: Re-run feature and compatibility tests.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_dataset tests.test_features_neurotune_v1 tests.test_model_v1 tests.test_neurotune_v1_abcd_cv_training -v`  
Expected: New C-only checks pass and existing default dataset/model behavior remains unchanged.

**Task 2 status — complete (2026-11-15):** Added train-fitted A–D fold features with an empty lyrics mapping, and a C-mode dataset/collator that excludes lyric values and preserves the existing all-modality default. The combined focused command passed 46 tests; spec and quality reviews found no remaining issues. A read-only real-data check built all three fold feature views over 13,822 tracks each, confirmed a 15-genre training vocabulary and distinct fold-specific fit-ID hashes, and verified collated C model keys are only audio, BPM, and genre. No split/cache outputs were changed.

## Task 3: Add fold-local known-positive-safe schedules

**Files:**
- Modify: `src/data/neurotune_v1_abcd_cv_training.py`
- Modify: `tests/test_neurotune_v1_abcd_cv_training.py`
- Reuse: `src/triplets.py`, `src/train_neurotune_v1.py`
- Preserve unchanged: `src/train_neurotune_v1_cv.py` and `tests/test_train_neurotune_v1_cv.py`

**Step 1: Write failing schedule-construction tests.** For training and validation candidate pools, verify that schedule targets use only their corresponding train-source positive edges, schedules are deterministic by seed, and no negative is a known positive internal to that candidate pool—including withheld or official-test-source positives. Verify the helper filters the full-catalog known-positive set down to the current candidate IDs before calling `sample_triplet_schedule`.

**Step 2: Run the focused tests and confirm they fail.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv_training -v`  
Expected: The new helper is missing or scheduled negatives violate the assertions.

**Step 3: Implement the minimal fold schedule helper.** For each candidate set independently, filter known-positive edges to those with both endpoints in that set, then call `sample_triplet_schedule` with the appropriate train-source positives. Reuse `train_neurotune_v1` for C-only model updates, 50 epochs, batch 64, Adam 0.001, margin 0.2, and validation-NDCG checkpoint selection. Do not reuse `train_neurotune_v1_cv`: that trainer expects edge-split train/validation labels in a shared non-test dataset, while these A–D train and validation track datasets are disjoint.

**Step 4: Re-run the focused tests.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv_training tests.test_train_neurotune_v1 tests.test_triplets -v`  
Expected: New schedule tests and all existing trainer/sampler tests pass.

**Task 3 status — complete (2026-11-15):** Added deterministic train and validation schedule creation with pool-specific known-positive filtering; the existing `train_neurotune_v1` remains the planned trainer and the edge-split `train_neurotune_v1_cv` remains unchanged. The focused command passed 37 tests, and spec/quality reviews found no actionable issues. A read-only real-data preflight generated and discarded seed-684 schedules for all three folds; training schedules had the expected `(50, 2 × train_edges, 3)` shape and were 57–59 MB each. No schedule artifacts were published.

## Task 4: Add source-separated C and cosine evaluation

**Files:**
- Create: `src/evaluate_neurotune_v1_cv.py`
- Create: `tests/test_evaluate_neurotune_v1_cv.py`
- Reuse: `src/evaluate_neurotune_v1.py`, `src/evaluate.py`, `src/retrieve.py`

**Step 1: Write failing evaluation tests.** Cover one fold's C rankings against train-source test edges and official-test-source edges independently; assert no combined-source score is produced. Cover the matched equal-modality cosine baseline, per-fold and per-seed metric records, query/edge counts, empty-source behavior, and deterministic ranking ties.

**Step 2: Run the focused tests and confirm they fail.**

Run: `.venv/bin/python -m unittest tests.test_evaluate_neurotune_v1_cv -v`  
Expected: The new source-separated evaluation helper is absent.

**Step 3: Implement the minimal evaluator.** Use the existing retrieval metrics for Recall@10, HitRate@10, MRR@10, and NDCG@10. Create separate relevance maps and metric records for the two held-out edge files. Compute one cosine result per fold using the same C modalities, train-fitted transforms, and held-out gallery. Do not merge source labels.

**Step 4: Re-run focused tests.**

Run: `.venv/bin/python -m unittest tests.test_evaluate_neurotune_v1_cv tests.test_evaluate_neurotune_v1 -v`  
Expected: New source separation and baseline parity tests pass; existing evaluation defaults remain unchanged.

**Task 4 status — complete (2026-11-15):** Added the fixed audio+BPM+genre cosine builder and source-separated retrieval evaluator. It rejects overlapping edge sources before ranking, supports one-shot ID iterables and all hashable ID types accepted by the existing evaluator, and reports zero metrics for an empty source. The focused command passed 18 tests; spec and quality reviews found no remaining issues. No real test embeddings or relevance were constructed.

## Task 5: Add the isolated three-fold C training runner

**Files:**
- Create: `scripts/run_neurotune_v1_abcd_cv.py`
- Create: `tests/test_neurotune_v1_abcd_cv_cli.py`
- Reuse: Tasks 1–4, `scripts/run_neurotune_v1.py` publication/barrier patterns

**Step 1: Write failing synthetic runner tests.** Cover exact production settings (C only, seeds 684/685/686, 50 epochs, batch 64), fold-specific feature fitting, source-specific train/validation/test routing, nine selected checkpoints, nine compressed schedule artifacts, 450 curve rows, C-matched cosine baseline, and input/code/checkpoint hashes. Instrument the synthetic run to prove all nine checkpoints, all schedules, and the curves exist before any test embeddings or relevance are constructed. Cover existing-output refusal, staged rollback, and preservation of prior namespaces.

**Step 2: Run the focused tests and confirm they fail.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv_cli -v`  
Expected: Runner import or integration assertions fail.

**Step 3: Implement the production runner.** Use fixed paths rooted at the repository and publish only to `results/neurotune_v1_abcd_cv/`. Process one fold at a time, build train-fitted C inputs, create one schedule pair per seed, run and save C checkpoints and curves, and release fold-local objects after persistence. Do not construct test embeddings or relevance until every checkpoint, schedule, and the complete curves file is staged. After that barrier, rebuild each fold's test view and report C and cosine metrics separately for both test-edge sources. Stage the complete result directory and publish without overwriting; roll back newly created outputs on failure.

**Step 4: Re-run the synthetic runner tests.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv_cli -v`  
Expected: All integration, barrier, provenance, and output-safety tests pass.

**Task 5 status — complete (2026-11-15):** Added the isolated production runner and synthetic integration tests. It fixes C/folds/seeds/epochs, uses train-only transforms and source-specific targets, writes nine checkpoints and nine `triplet_schedules/` artifacts plus 450 curve rows, and keeps test embeddings/relevance behind the full artifact barrier. Input/code hashes are captured before loading and verified before publication; checkpoints carry fold/seed/variant/source/transform/schedule provenance. Publication uses exclusive directory creation and no-clobber hard links with rollback, and the report retains separate train-source and official-test-source metrics plus a C-matched cosine reference. The 9 focused synthetic tests passed; final spec and quality reviews found no actionable issues. No real training has been run yet.

## Task 6: Run the real experiment, document, and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-11-15-neurotune-abcd-c-cv-training-design.md`
- Preserve: all prior `results/`, `data/msd_hf_pilot/`, official 10K CV, and A–D cache/split artifacts

**Step 1: Run focused checks and preflight.** Run Tasks 1–5 focused tests, inspect the exact output targets, confirm `results/neurotune_v1_abcd_cv/` does not exist, and snapshot hashes for the protected A/B and A–D preparation artifacts.

**Step 2: Run the fixed real C CV experiment.**

Run: `.venv/bin/python -m scripts.run_neurotune_v1_abcd_cv`  
Expected: Exit 0; publish nine checkpoints, nine schedules, 450 curve rows, and the source-separated experiment report only after the checkpoint barrier.

**Step 3: Independently validate the report and artifacts.** Recompute all source, data, code, transform, schedule, and checkpoint hashes; verify selected epochs use validation only; check all test metrics are separated by source; verify no old A/B test-source edge appears as a target; and confirm all prior protected artifact hashes match their snapshots. Confirm no output contains A/B/D model runs or combined-source metrics.

**Step 4: Update documentation from verified outputs.** Record run counts, settings, fold/source-specific metric counts, descriptive baseline comparison, selected epochs, and the same-source/incomplete-label caveats. Do not write a generalization or superiority claim.

**Step 5: Run final verification.**

Run: `.venv/bin/python -m unittest discover -s tests -q`  
Expected: All tests pass.

Run: `.venv/bin/python -m compileall -q src scripts tests`  
Expected: Exit 0.

Run: `.venv/bin/python -m pip check`  
Expected: `No broken requirements found.`

**Task 6 status — complete (2026-11-15):** The fixed runner completed in 18,165 seconds on CPU and published `results/neurotune_v1_abcd_cv/` with nine C checkpoints, nine `triplet_schedules/` files, 450 curve rows, and the report. Aggregate model and cosine results are documented by label source in the README and design record; they are descriptive only, with C higher on the held-out train-source labels and cosine higher on the official-test-source labels. Post-run checks matched all report input/code/manifest/cache/transform/schedule/checkpoint hashes, validated target and known-positive exclusion for all 43,749,162 scheduled triplets, and confirmed all 9,465 pre-run result/cache files unchanged. The final suite passed 396 tests; `compileall` succeeded and `pip check` reported no broken requirements. No Git repository was available, so no commit was created.

## Execution notes

- Follow the approved design document; do not expand to A/B/D or alter the evaluation scope without renewed approval.
- Use test-first changes and review each task before moving to the next.
- The workspace is not a Git repository; do not claim a commit or create a worktree. Keep all new results in the isolated namespace.

# NeuroTune A–D Audio Cache and Fresh Artist-CV Implementation Plan

> **For the implementation agent:** Follow this plan task-by-task. Preserve all existing A/B artifacts and do not train a model in this work unit.

**Goal:** Build a separate raw 30-second/26-channel cache for the A–D candidate catalog, report invalid tracks, and create deterministic artist-grouped CV folds whose test artists are absent from the old A/B catalog.

**Architecture:** Add a separate A–D cache builder that reuses the existing audio-window reader and audit provenance without changing A/B entry points or outputs. Build a separate split module and CLI from the cache-valid IDs; use only Last.fm train-source links for train/validation and keep the two test-edge sources separate. Publish all new artifacts with provenance, no-overwrite safeguards, and rollback cleanup.

**Tech Stack:** Python 3, NumPy, h5py, tarfile, CSV/JSON, unittest, existing `preprocess_audio_window` and A–D audit artifacts.

---

## Constraints

- Existing A/B audit, cache, split, results, and checkpoints stay byte-for-byte unchanged.
- The A–D audit catalog currently contains 13,822 candidates. The valid cache size and all fold counts must be recomputed after per-track validation.
- A/B-absent artist keys form the only eligible test cohort. The current estimate is 1,764 artist keys and 2,371 tracks.
- Assign eligible test artists by sorting, shuffling with seed 684, then assigning `artists[i::3]` to fold `i`.
- In each fold, split the remaining artists into validation (first floor 10% after shuffling with seed 684) and training (the rest). Old A/B test artists may be in either partition, but not in test.
- Train and validation labels come only from the A–D Last.fm train-edge file. Test labels from that file and from the official Last.fm test-edge file are written separately. Cross-partition links are excluded.
- Old A/B test-source edges must not appear as train, validation, or test targets. A full known-positive superset may contain them only as a future negative-sampling exclusion set, never as a training target or model input.
- No model training, checkpoint selection, or performance claims in this plan.
- No Git repository is present, so commits/worktrees are unavailable.

## Task 1: Distinguish track-data failures from unsafe HDF5 inputs

**Files:**
- Modify: `src/data/preprocess_audio_window.py`
- Test: `tests/test_audio_window_preprocessing.py`

**Step 1 — Add failing error-class tests.** Add tests asserting that expected per-track data problems (missing segment data, invalid alignment/non-finite values, or an empty 30-second window) raise a dedicated `ValueError` subclass such as `AudioWindowTrackError`. Add tests asserting that unsafe HDF5 structures (external/soft links, VDS, VLEN/object fields, or dimension/element limits) raise a different `ValueError` subclass such as `AudioWindowUnsafeError`. Existing callers must still be able to catch `ValueError`.

**Step 2 — Run the focused tests and confirm the new assertions fail.**

Run: `.venv/bin/python -m unittest tests.test_audio_window_preprocessing -v`  
Expected: New error-type assertions fail; existing preprocessing tests remain green.

**Step 3 — Add only the two error categories and classify existing validation sites.** Keep successful output and validation boundaries unchanged. Identity/provenance mismatches and unsafe HDF5 structures remain fatal; only explicitly track-local audio-data failures use `AudioWindowTrackError`.

**Step 4 — Run the focused audio-window suite.**

Run: `.venv/bin/python -m unittest tests.test_audio_window_preprocessing -v`  
Expected: All tests pass; both custom exceptions remain subclasses of `ValueError`.

**Task 1 status — complete (2026-11-15):** Added `AudioWindowTrackError` for track-local audio/data failures and `AudioWindowUnsafeError` for unsafe HDF5 structures and resource-bound violations. Both remain `ValueError` subclasses, so existing callers keep their error contract. Zero-length segment datasets now use the track-local category while over-limit counts remain unsafe. The focused command above passed all 13 tests (exit 0). Spec review found and closed the zero-length classification issue; quality review found no required changes. The optional caller-level compatibility test was not added because the existing A/B caller still catches `ValueError` and the subtype contract is directly tested.

## Task 2: Implement the separate A–D raw cache builder

**Files:**
- Create: `src/data/preprocess_msd_hf_abcd.py`
- Create: `scripts/preprocess_msd_hf_abcd.py`
- Test: `tests/test_preprocess_msd_hf_abcd.py`
- Reuse: `src/data/preprocess_audio_window.py`, `scripts/audit_msd_hf_ab.py`

**Step 1 — Add failing synthetic archive tests.** Build small A/B/C/D tar fixtures and a matching synthetic A–D audit/catalog. Cover sorted cache IDs, raw finite `(T,26)` float32 rows, exact metadata match, invalid-track skip/reason reporting, missing candidate member failure, duplicate IDs, source-hash mismatch, unsafe HDF5 abort, output collision refusal, and rollback when publication fails.

**Step 2 — Run the new cache tests to confirm they fail.**

Run: `.venv/bin/python -m unittest tests.test_preprocess_msd_hf_abcd -v`  
Expected: Fail because the A–D cache builder does not exist.

**Step 3 — Implement the cache core.** Validate the completed A–D audit, exact 30-second/26-channel contract, required source hashes, catalog IDs, and output paths before reading. Stream all four named archives, enforce tar/HDF5 bounds, verify member and metadata IDs, and reject duplicate/missing source members. Catch only `AudioWindowTrackError` as a track-level rejection; propagate unsafe-input, source, identity, and I/O errors. Recheck archive hashes after scanning.

**Step 4 — Stage and publish cache outputs.** Write `abcd_audio_30s_26ch.h5`, `abcd_audio_30s_26ch_meta.json`, and `abcd_audio_30s_26ch_rejections.json` under `data/msd_hf_abcd_pilot/`. Keep the HDF5 ragged and unnormalized; record valid IDs, offsets/lengths, rejected IDs/reason codes, all input hashes, and code hashes. Publish the three files without clobbering; clean all newly published targets on any failure.

**Step 5 — Implement repository-rooted CLI defaults and overrides.** The default inputs are the A–D audit/catalog, A/B/C/D archives, and source files already recorded in the audit. Output defaults use only the new `abcd_*` names. Add a strict CLI-routing test so parsed argument names match the core function signature.

**Step 6 — Run focused tests and review the cache lane.**

Run: `.venv/bin/python -m unittest tests.test_audio_window_preprocessing tests.test_preprocess_msd_hf_abcd -v`  
Expected: All tests pass; all old A/B cache tests remain green.

**Step 7 — Run the real cache build after target preflight.** Confirm all three A–D cache targets are absent, snapshot the six existing A/B artifact hashes, then run:

Run: `.venv/bin/python -m scripts.preprocess_msd_hf_abcd`  
Expected: Exit 0; print only accepted/rejected track and segment counts. If a target exists, stop without overwriting it.

**Step 8 — Verify the real cache.** Reopen the HDF5 and check sorted unique IDs, positive ragged offsets, finite `(N,26)` float32 values, channel/window metadata, audit/source/code hashes, and rejection counts. Confirm cache IDs equal the A–D catalog minus exactly the reported rejections, and confirm all six A/B artifact hashes are unchanged.

**Task 2 status — complete (2026-11-15):** Added the isolated cache core and CLI with synthetic coverage for invalid-track skip reporting, fatal unsafe HDF5, metadata identity, archive/catalog hashes, missing/duplicate members, output collisions, rollback, and CLI argument routing. The spec review's missing `AudioWindowUnsafeError` import, false-positive missing-member test, and absent catalog hash were fixed and re-reviewed. The quality review found no blocking issues; it noted that all accepted arrays are retained before concatenation. This remains unchanged for the fixed A–D scope; the peak was not measured, so future shard growth may require incremental writes. The focused command `.venv/bin/python -m unittest tests.test_audio_window_preprocessing tests.test_preprocess_msd_hf_abcd -v` passed 24 tests (exit 0).

The real cache build produced 13,822 valid tracks, 1,430,546 segments, zero rejected tracks, and a 96,590,414-byte HDF5 file. Independent checks verified catalog/cache ID equality, offsets, finite float32 values, raw `[0,30)` metadata, exact audit/catalog/source/code hashes, and the rejection report. All six A/B audit/cache artifact hashes remain unchanged. No model or split artifact was produced in this task.

## Task 3: Implement the artist-grouped A–D CV split core

**Files:**
- Create: `src/data/neurotune_v1_abcd_cv.py`
- Test: `tests/test_neurotune_v1_abcd_cv.py`
- Inputs: `data/msd_hf_abcd_pilot/abcd_catalog_audit.json`, `abcd_catalog_tracks.csv`, the A–D train/test edge CSVs, cache metadata/rejections, `data/msd_hf_pilot/ab_catalog_tracks.csv`, and `data/msd_hf_pilot/neurotune_v1_split.json`

**Step 1 — Add failing split tests.** Cover artist-key fallback, cache-valid ID filtering, A/B-absent test-artist selection, deterministic seed-684 round-robin folds, deterministic 10% validation artists per fold, and pairwise artist disjointness.

**Step 2 — Add failing edge-routing tests.** Verify train-source links route only to train/validation/test when both endpoints are in the matching partition; official-test-source links route only to their test-fold file; cross-partition links are counted and excluded; old A/B test-source edges are absent from all label partitions; and the known-positive superset is separate from training targets.

**Step 3 — Run the new split tests to confirm the intended failures.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv -v`  
Expected: Fail because the A–D CV builder does not exist.

**Step 4 — Implement deterministic fold construction and edge routing.** Build the A/B artist-key set from the old catalog. From cache-valid A–D IDs, choose test-eligible keys absent from that set, shuffle sorted keys with seed 684, and assign `keys[i::3]`. For each fold, shuffle sorted remaining artist keys with seed 684, assign floor 10% to validation, and use the rest for training. Use only train-source edges for train/validation labels; preserve train-source and official-test-source test edges in separate outputs. Record invalid-track edge exclusions and cross-partition exclusions by source.

**Step 5 — Add provenance and safe publication.** The manifest records audit/cache/catalog/edge/A/B-reference hashes, code hashes, seeds, artist/track IDs per partition, edge counts by source, exclusions, and the known-positive exclusion-set hash. Stage the manifest and all fold edge files under `data/msd_hf_abcd_pilot/cv/`; refuse all collisions and roll back partial publication.

**Step 6 — Run focused split tests and review the split lane.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv -v`  
Expected: All tests pass; no test fold contains an A/B artist key.

**Task 3 status — complete (2026-11-15):** Added the split core with complete cache-contract/provenance checks, exact legacy A/B split consistency validation, typed artist fallback, deterministic new-only test folds, train/validation/test source routing, full old A/B test-source edge exclusion, known-positive output/hash, and rollback/no-clobber publication. Review findings were closed, including the distinction that the protected A/B test-source set is the full 1,940-edge source file, not only the 215 edges internal to the old model test cohort. The focused command above passed all 10 tests (exit 0). The real split has not yet been published; Task 4 will exercise the CLI and verify actual fold manifests/edge outputs.

## Task 4: Add the split CLI and run the real A–D split audit

**Files:**
- Create: `scripts/audit_neurotune_v1_abcd_cv.py`
- Modify: `tests/test_neurotune_v1_abcd_cv.py`

**Step 1 — Add failing CLI tests.** Test repository-rooted defaults, explicit path overrides, strict argument routing, audit/cache/split hash checks, output collisions, and rollback.

**Step 2 — Implement the CLI.** Keep it separate from the existing A/B split CLI and default all outputs to the new A–D CV namespace.

**Step 3 — Run the focused split suite.**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv -v`  
Expected: All tests pass.

**Step 4 — Run the real split only after cache validation.** Confirm all CV output targets are absent, then run:

Run: `.venv/bin/python -m scripts.audit_neurotune_v1_abcd_cv`  
Expected: Exit 0; emit per-fold valid tracks, artists, train/validation/test edges by source, and exclusion counts.

**Step 5 — Verify the real split.** Recompute artist disjointness and edge routing from the published files. Check that each test fold uses only new-only artist keys, each cache ID occurs in exactly one track partition per fold, train/validation contain no official-test-source targets, test sources remain separate, A/B test-source edges do not appear as label targets, and report hashes match current files.

**Task 4 status — complete (2026-11-15):** Added the repository-rooted CLI and direct summary of fold/edge/exclusion counts. Focused verification `.venv/bin/python -m unittest tests.test_neurotune_v1_abcd_cv -v` passed all 14 tests (exit 0). The real CLI completed and published 14 files under `data/msd_hf_abcd_pilot/cv/`. Independent artifact checks verified deterministic artist assignment, full track partitioning, source-specific routing, no old A/B test-source edges in targets, the 75,690-edge known-positive set/hash, and all input/code hashes. Track counts (train/validation/test) are 11,640/1,377/805, 11,693/1,335/794, and 11,669/1,381/772. Train-source test edges are 183/221/235; official-test-source test edges are 50/30/45. All six protected A/B artifact hashes remain unchanged.

## Task 5: Document the validated cache/split and perform final verification

**Files:**
- Modify: `README.md`
- Update: `docs/plans/2026-11-15-neurotune-abcd-audio-cache-cv-design.md`

**Step 1 — Update documentation from the real artifacts only.** Record cache-valid and rejected-track counts, actual fold/edge counts, output paths, and the same-source CV limitation. State explicitly that no model was trained and no performance claim is made.

**Step 2 — Run the complete test suite.**

Run: `.venv/bin/python -m unittest discover -s tests -q`  
Expected: Exit 0 and all tests pass.

**Step 3 — Run compilation and dependency verification.**

Run: `.venv/bin/python -m compileall -q src scripts tests`  
Expected: Exit 0.

Run: `.venv/bin/python -m pip check`  
Expected: `No broken requirements found.`

**Step 4 — Verify protected artifacts one final time.** Compare the six A/B audit/cache hashes with the pre-task snapshot and confirm no A/B output path changed. Confirm the A–D output directory contains only the intended audit, cache, metadata/rejection, and CV artifacts. No model, checkpoint, or training-curve artifact should be created.

**Task 5 status — complete (2026-11-15):** Updated `README.md` and this design document from the real cache and split artifacts, including valid/rejected counts, fold/source edge counts, output paths, and the same-source evaluation limitation. Final verification `.venv/bin/python -m unittest discover -s tests -q` passed all 360 tests; `.venv/bin/python -m compileall -q src scripts tests` exited 0; `.venv/bin/python -m pip check` reported `No broken requirements found.` Final provenance validation confirmed the regenerated CV manifest matches current HDF5/sidecar/code hashes, all 13 edge CSVs are byte-identical to the prior deterministic split, the full 1,940-edge old A/B test-source set is excluded from targets, and all six protected A/B artifact hashes remain unchanged. No model, checkpoint, or training-curve artifacts were created.

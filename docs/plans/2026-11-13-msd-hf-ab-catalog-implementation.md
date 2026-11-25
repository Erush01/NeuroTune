# MSD HF A/B Multimodal Pilot Catalog Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible A/B-only catalog/cache for the 6,967 tracks with MSD audio, positive BPM, musiXmatch lyrics, and tagtraum genre.

**Architecture:** Keep all outputs in a new `data/msd_hf_pilot/` namespace. Stream the downloaded A/B tarballs to audit IDs, feature coverage, and duplicate metadata; save Last.fm train/test edges separately. Extract first-30-second temporal sequences as variable-length 26-channel raw features in a ragged HDF5 cache, with offsets and source/audit provenance; do not fit train-only normalization or train the model in this data-preparation phase.

**Tech Stack:** Python, tarfile, h5py, NumPy, CSV/JSON, existing MSD parsing and preprocessing conventions, `unittest`.

---

## Fixed decisions and safety constraints

- Only use downloaded `data/msd_hf_pilot/A.tar.gz` and `B.tar.gz`; do not download C–Z.
- Target all eligible A/B-cohort IDs: current audit found 6,967 all-input candidates, including 6,028 beyond the 10K subset. Recompute from sources rather than hard-coding these counts.
- Audio window is `[0, 30)` seconds. Each segment row has 26 columns: pitches 0–11, timbre 12–23, loudness 24, and clipped segment duration 25. Duration is `min(next_segment_start, 30, song_duration) - segment_start`; include only segments starting before 30 seconds. Reject malformed/non-finite/misaligned arrays, non-positive duration rows, and tracks with no segment start in the window (the temporal encoder requires at least one timestep).
- Preserve variable sequence lengths; do not pad or fit normalization in the on-disk cache. Store flattened `float32` features plus `offsets`; later training collate will pad and construct masks. Fit any learned scaling only after the train/validation/test artist split is fixed.
- Apply the established artist/song duplicate rule and official duplicate groups over the downloaded A/B audio cohort before feature filtering. Keep the lexicographically smallest track ID per duplicate group/key.
- Last.fm train and test archives remain separate. Never merge test edges into training labels.
- Never modify existing 10K, M1, M2, audio-only, or CV artifacts. Refuse to overwrite pilot outputs and clean newly-created outputs on caught errors.
- Third-party HF provenance/license/checksum are unverified; outputs remain a local research pilot.

## Task 1: First-30-second temporal feature extraction

**Files:** Create `src/data/preprocess_audio_window.py`; create `tests/test_audio_window_preprocessing.py`.

1. Add synthetic-HDF5 tests for channel ordering, window clipping at 30 seconds, duration calculation from segment starts, exclusion of segments starting at/after 30 seconds, output dtype/shape, and no input mutation.
2. Add invalid-input tests for missing arrays, misaligned/non-finite data, unordered starts, bad tempo/duration, segments with non-positive clipped durations, and empty windows.
3. Run `.venv/bin/python -m unittest tests.test_audio_window_preprocessing -v`; confirm expected failures before implementation.
4. Implement `preprocess_audio_window(h5_path, expected_track_id, window_seconds=30.0)` returning one `(T, 26)` finite `float32` array and canonical track ID/tempo metadata needed by the audit.
5. Rerun the focused tests green.

## Task 2: A/B feature catalog, duplicate policy, and edge audit

**Files:** Create `scripts/audit_msd_hf_ab.py`; create `tests/test_msd_hf_ab_audit.py`.

1. Add synthetic tar/HDF5 fixtures with artists, songs, duplicate groups, lyric IDs, genre IDs, and train/test Last.fm JSON. Test duplicate resolution, valid-input intersections, edge deduplication, self-edge removal, source split preservation, and input-order invariance.
2. Add validation tests for unsafe tar member paths/types, track-ID/path mismatches, missing sources, changed-source hashes, and pre-existing output targets. Ensure rejection happens before output creation.
3. Run `.venv/bin/python -m unittest tests.test_msd_hf_ab_audit -v` red.
4. Implement a streaming A/B audit that reads HDF5 scalar metadata for the full A/B cohort, resolves duplicates, then retains records with positive tempo and IDs in both local musiXmatch and CD2C sets. Parse Last.fm train/test source JSON separately and retain unique undirected edges whose endpoints are both eligible.
5. Atomically write `data/msd_hf_pilot/ab_catalog_audit.json`, `ab_catalog_tracks.csv`, `ab_lastfm_train_edges.csv`, and `ab_lastfm_test_edges.csv`, recording source/code hashes, counts, window contract, duplicate policy, and the third-party provenance caveat. Refuse existing output files.
6. Rerun focused audit tests green.

## Task 3: Ragged HDF5 audio cache

**Files:** Create `scripts/preprocess_msd_hf_ab.py`; create `tests/test_preprocess_msd_hf_ab.py`.

1. Add tests for sorted unique IDs, variable sequence lengths encoded by monotonic offsets, `(sum_T,26)` feature storage, exact track-ID alignment with audit CSV, window/channel metadata, and round-trip reconstruction.
2. Add tests for missing/duplicate source tracks, output collisions, stale audit/source hashes, existing targets, and cleanup/preservation on extraction/write failures.
3. Run `.venv/bin/python -m unittest tests.test_preprocess_msd_hf_ab -v` red.
4. Implement streaming extraction of eligible audit IDs from A/B tar files using `preprocess_audio_window`; build `track_ids`, `offsets`, and concatenated raw `features` arrays in sorted ID order. Do not persist padded rows or train-fitted statistics.
5. Write `data/msd_hf_pilot/ab_audio_30s_26ch.h5` and `ab_audio_30s_26ch_meta.json` atomically, with audit/source hashes, `window_seconds=30`, channel names, dtype, total segment count, and per-track sequence lengths. Reject existing files.
6. Rerun focused cache tests green.

## Task 4: Real A/B pilot, documentation, and verification

**Files:** Update `docs/plans/2026-11-12-msd-hf-ab-pilot.md`; do not alter the original proposal or completed experiment reports.

1. Run `.venv/bin/python -m scripts.audit_msd_hf_ab` and `.venv/bin/python -m scripts.preprocess_msd_hf_ab` once.
2. Verify cached IDs equal the audited catalog; reconstruct sample ragged rows and compare them with direct HDF5 preprocessing; validate all offsets, channel values, finite data, Last.fm split separation, and audit/cache hashes.
3. Verify all existing model reports, checkpoints, and prior caches retain their recorded hashes.
4. Record exact post-duplicate catalog and train/test edge counts in the pilot report. State explicitly that the A/B cohort is not an unbiased full-MSD sample and no model was trained.
5. Run final checks:

   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   .venv/bin/python -m compileall -q src scripts tests
   .venv/bin/python -m pip check
   ```

6. Report exact command statuses, catalog/cache counts, hashes, and limitations. No performance claims.

# M1 MSD Audio Preprocessing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and validate fixed-size temporal MSD inputs for the current 939-track common catalog, preserving the approved audio-first design and train-only normalization.

**Architecture:** Read the group-level temporal datasets from each selected MSD HDF5 file, interpolate continuous segment features onto a 1,024-point normalized-duration grid, and bin beat/bar/section events as count channels. Fit statistics for the 25 continuous channels only on the existing train IDs, apply them unchanged to every split, and save sequences plus provenance to a consolidated HDF5 cache and JSON metadata file.

**Tech Stack:** Python 3.12, h5py, NumPy, stdlib `csv`/`json`/`hashlib`, unittest.

---

## Fixed contract

- Current artist-disjoint split: 756 train, 94 validation, 89 test tracks.
- Input arrays are HDF5 group-level datasets under `analysis/`, not fields of `analysis/songs`:
  `segments_pitches (T,12)`, `segments_timbre (T,12)`,
  `segments_loudness_max (T,)`, `segments_start (T,)`, plus
  `beats_start`, `bars_start`, and `sections_start`.
- Output order is sorted `track_id`, independent of filesystem traversal order.
- Output sequence shape is `(1024, 28)`: pitch channels 0–11, timbre 12–23,
  loudness 24, beat counts 25, bar counts 26, section counts 27.
- Resample continuous features linearly at 1,024 equally spaced times from
  0 through each track's duration. Map each structural event time to the nearest
  grid bin and count collisions rather than collapsing them to a binary flag.
- Fit mean/std for continuous channels 0–24 from only the 756 train tracks and
  all of their grid steps. Apply to all 939 tracks; leave event-count channels
  unscaled. Use scale 1 for a zero-variance channel.
- Write `data/processed/audio_sequences.h5` with `track_ids` and float32
  `sequences (939,1024,28)`. Write
  `data/processed/audio_sequences_meta.json` with channel names, grid settings,
  train-only normalization, split counts, and source hashes. Do not change
  `data/processed/msd_tracks.csv`.

## Task 1: Validated single-track temporal extraction

**Files:**
- Create `src/data/preprocess_audio.py`
- Test `tests/test_audio_preprocessing.py`

**Steps:**
1. Write an HDF5 fixture with known irregular segment times and known pitch,
   timbre, loudness, beat, bar, and section values.
2. Run the focused test and confirm it fails because the temporal reader is
   absent.
3. Implement a minimal single-file reader that verifies the requested track ID,
   positive duration, required datasets, aligned segment lengths, finite values,
   strictly increasing segment times, and event times inside the duration.
4. Implement linear interpolation to 1,024 duration-spanning grid points and
   nearest-bin event counting; return float32 `(1024,28)` values and channel
   names.
5. Test exact interpolation values, event-bin counts, output shape/dtype,
   determinism, and each invalid-input error case.
6. Run `.venv/bin/python -m unittest tests.test_audio_preprocessing -v` and
   verify it passes.

## Task 2: Train-only normalization

**Files:**
- Modify `src/data/preprocess_audio.py`
- Test `tests/test_audio_preprocessing.py`

**Steps:**
1. Add a failing test where validation/test values are changed but fitted
   statistics must remain identical.
2. Fit per-channel mean/std over only IDs in the audit's train partition for
   channels 0–24; handle zero std with scale 1.
3. Apply those statistics to every selected track without modifying event
   counts, and return serializable fit metadata.
4. Test exact train-only means/stds, validation/test isolation, finite output,
   and zero-variance behavior.
5. Run the focused unittest module and verify it passes.

## Task 3: Consolidated HDF5 cache and metadata

**Files:**
- Modify `src/data/preprocess_audio.py`
- Test `tests/test_audio_preprocessing.py`

**Steps:**
1. Add a failing cache round-trip test using several synthetic track files and
   a temporary output path.
2. Build a deterministic track-ID-to-HDF5-path index from the input root; reject
   missing or duplicate selected IDs rather than silently dropping them.
3. Write sorted IDs and normalized float32 sequences to one HDF5 cache; write
   metadata JSON with split counts, channel order, grid definition, normalization
   statistics, and SHA-256 hashes for the audit and selected HDF5 inputs.
4. Test exact ID ordering, shape, dtype, metadata provenance, and round-trip
   sequence values.
5. Run the focused unittest module and verify it passes.

## Task 4: CLI, real 939-track cache, and docs

**Files:**
- Create `scripts/preprocess_audio.py`
- Modify `README.md`
- Generate `data/processed/audio_sequences.h5`
- Generate `data/processed/audio_sequences_meta.json`

**Steps:**
1. Add a synthetic CLI test that loads `results/data_audit.json`, resolves its
   939 IDs, writes cache/metadata under a temporary directory, and verifies
   train-only fit provenance. Confirm the test fails before CLI implementation.
2. Implement CLI defaults for `data/msd_subset_raw/MillionSongSubset`,
   `results/data_audit.json`, and the two processed outputs; provide path
   overrides for tests and alternate local layouts.
3. Run the synthetic CLI test and verify it passes.
4. Run `.venv/bin/python -m scripts.preprocess_audio` on the real subset.
5. Independently verify 939 unique IDs, split membership 756/94/89, shape
   `(939,1024,28)`, finite values, source hashes, event counts, and train-only
   normalization metadata.
6. Document the command, channel schema, and generated files in `README.md`.

## Final verification

- `.venv/bin/python -m unittest discover -s tests -v`
- `.venv/bin/python -m compileall -q src scripts tests`
- `.venv/bin/python -m pip check`
- Validate the real cache against the checklist in Task 4.

## Workspace constraint

This workspace is not a Git repository, so do not attempt commits or worktree
creation. Keep all changes local and report generated artifact paths.

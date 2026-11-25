# M2 Audio-First Siamese Prototype Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train and evaluate a small audio-first Siamese retrieval model using the completed M1 cache and train-fitted BPM, lyric, and genre modalities.

**Architecture:** Preserve the existing N0 model and runner. Add a separate `AudioFirstSiameseEncoder`, a CPU training function that reuses the established pair sampler and contrastive loss, and an M2 runner that aligns every modality by track ID. Keep the fixed N0 protocol and evaluate the 89-track test gallery once after fixed-epoch training.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, h5py, existing M0 parsers/pair sampler/retrieval metrics, stdlib JSON/CSV/hashlib, unittest.

---

## Fixed inputs and acceptance contract

- Audio cache: `data/processed/audio_sequences.h5`; metadata:
  `data/processed/audio_sequences_meta.json`.
- Audio HDF5 datasets: sorted UTF-8 `track_ids`, float32 `sequences` shaped
  `(939,1024,28)`, and 28 `channel_names`. The on-disk sequence shape is
  `(N,T,C)`; the encoder transposes to `(N,C,T)` internally.
- Auxiliary sources remain `msd_tracks.csv`, `mxm_counts.csv` (or the existing
  `mxm_counts/` directory), `tagtraum_genres.csv`, and `lastfm_similar/` under
  `data/processed/`.
- Read the same artist-disjoint split from
  `results/data_audit.json`: 756 train, 94 validation, 89 test. Fit BPM/IDF/
  genre transforms on train IDs only. The M1 audio cache is already normalized
  from those same train IDs.
- Keep N0 untouched. For training, provide only train+validation features and
  pair pools; do not pass test edges, relevance, or test rows into the training
  API. Use test modalities only after all seed checkpoints are fixed.
- Fixed protocol: seeds 684/685/686, 50 epochs, batch size 64, Adam `0.001`,
  contrastive margin `1.0`, one negative per positive per epoch, existing 50/50
  same-genre/random sampler, fresh train negatives per epoch, fixed validation
  negatives per seed. No validation checkpoint selection or hyperparameter
  tuning.
- Report B0/N0 comparisons descriptively only. There are 6 validation and 11
  test positive edges; unobserved pairs are not verified negatives.

## Task 1: Validated audio-cache loading and ID alignment

**Files:**
- Modify `src/data/preprocess_audio.py`
- Test `tests/test_audio_preprocessing.py`

**Steps:**
1. Add a failing test that loads a small generated HDF5 cache and metadata,
   verifies sorted IDs and row values, and rejects a requested ID set that does
   not exactly match the cache.
2. Add `load_audio_sequence_cache(hdf5_path, metadata_json_path,
   expected_ids=None)`. Return IDs, sequences, and metadata; validate the cache
   shape `(N,1024,28)`, float32 dtype, finite values, unique sorted IDs, channel
   schema, and exact ID-set agreement when `expected_ids` is supplied.
3. Add a test where an auxiliary/source ID order differs from cache order and
   prove rows can be aligned with the returned IDs, not positional assumptions.
4. Run `.venv/bin/python -m unittest tests.test_audio_preprocessing -v`.

**Expected:** Tests pass; loader does not rewrite or mutate cache files.

## Task 2: Audio-first shared encoder

**Files:**
- Modify `src/model.py` by adding `AudioFirstSiameseEncoder`; do not change
  `SiameseEncoder` or `contrastive_loss` behavior.
- Test `tests/test_model.py`

**Steps:**
1. Add failing tests for audio input `(B,1024,28)`, output `(B,64)`, unit L2
   norms, finite CPU forward/backward, and exact output determinism in eval mode.
2. Implement the audio branch: transpose to channel-first; Conv1d blocks
   `28→32 (k=7,s=2,p=3)`, `32→64 (k=5,s=2,p=2)`, `64→64 (k=3,s=2,p=1)`, each
   with ReLU; adaptive average pool to 16 bins; flatten `64*16`; Linear to 32
   with ReLU.
3. Add auxiliary branches matching N0: BPM `1→8`, lyrics `5000→32`, genre
   `G→16`, each with ReLU. Concatenate to 88 values, project `88→64`, and
   L2-normalize. The same model instance is used for both pair sides.
4. Run `.venv/bin/python -m unittest tests.test_model -v`.

**Expected:** Existing N0 model tests remain unchanged and pass; the new
encoder has finite unit-norm embeddings and gradients.

## Task 3: Fixed-protocol audio training core

**Files:**
- Create `src/train_audio.py`
- Create `tests/test_train_audio.py`

**Steps:**
1. Add a failing CPU test using synthetic audio sequences, BPM/lyrics/genre
   tensors, and train/validation pair pools. Verify a parameter changes after
   one epoch and losses are finite.
2. Add `train_audio_seed(audio_sequences, modalities, song_ids, train_pools,
   validation_pools, seed, epochs=50, batch_size=64, learning_rate=0.001,
   margin=1.0)`. Validate row/ID alignment, `(N,1024,28)` audio shape, all
   modality widths, finite real inputs, unique IDs, and disjoint train/validation
   IDs. The signature must not accept test IDs or test pairs.
3. Reuse `sample_pairs` and `contrastive_loss`; use `AudioFirstSiameseEncoder`.
   Resample train rows each epoch, keep validation rows fixed, and compute
   validation loss under `torch.no_grad()`.
4. Add a deterministic regression: run twice with the same seed and compare
   curves and sampled pair traces; confirm every sampled pair remains in its
   designated partition.
5. Run `.venv/bin/python -m unittest tests.test_train_audio -v`.

**Expected:** Same seed and same inputs produce identical traces/curves on CPU;
test data is absent from this training function.

## Task 4: M2 runner, report, and synthetic CLI integration

**Files:**
- Create `scripts/run_m2_audio.py`
- Create `tests/test_m2_audio_cli.py`

**Steps:**
1. Add a failing one-epoch CLI integration test following the fixture pattern
   in `tests/test_n0_cli.py`. Create temporary auxiliary CSV/Last.fm inputs,
   audit partitions, and a small valid audio cache; run `main(argv)` with
   temporary paths, one seed, and one epoch.
2. Implement the runner path overrides: `--data-dir`, `--audit`, `--audio-cache`,
   `--audio-metadata`, `--output`, `--checkpoint-dir`, `--epochs`, and
   `--seeds`. Defaults must target the current processed files and the fixed
   50-epoch, 684/685/686 protocol.
3. Validate partitions and cache IDs/split metadata against the audit. Build auxiliary
   modalities using `build_train_fitted_modalities` with train IDs only; align
   them to audio rows by track ID. Build train and validation pair pools with
   `build_pair_pools` and call `train_audio_seed` with only train+validation
   rows.
4. After all checkpoints are fixed, transform test modalities from train-fitted
   metadata, select test audio rows by ID, build test relevance, and evaluate
   the full test gallery with `rank_top_k` and `evaluate_rankings`.
5. Write `results/m2_audio_experiment.json`,
   `results/m2_audio_training_curves.csv`,
   `results/m2_audio_sampled_pairs.csv`, and
   `results/checkpoints/m2_audio_seed_{seed}.pt`. Include config, architecture,
   split/edge counts, fit-train IDs, per-seed history/metrics, negative policy,
   hashes for cache/metadata/audit/auxiliary data/relevant code, and Python/
   NumPy/PyTorch versions. Make the low-edge warning derive from measured test
   relevance. Checkpoint the fixed 50th-epoch state and its provenance.
6. Extend the synthetic CLI test to check report fields, finite metrics/losses,
   checkpoint and CSV contents, pair partition isolation, train-only fit IDs,
   and absence of test IDs from `training_input_ids`.
7. Run `.venv/bin/python -m unittest tests.test_m2_audio_cli -v`.

**Expected:** One-epoch synthetic run produces deterministic artifacts without
using test labels or test rows during training.

## Task 5: Fixed three-seed run and project documentation

**Files:**
- Modify `README.md`
- Generate the four M2 artifacts listed in Task 4.

**Steps:**
1. Document the new runner, fixed defaults, architecture, artifacts, and
   exploratory limitations. Keep the M1 preprocessing section, B0, and N0
   descriptions accurate; label B0/N0 as metadata-only comparisons.
2. Run `.venv/bin/python -m scripts.run_m2_audio` on the current 939-track
   cache. Do not tune based on validation/test metrics.
3. Independently check three seeds, 50 epochs per seed, finite curves/metrics,
   checkpoint/config consistency, pair partition isolation, test gallery count
   89, and measured test positive-edge count 11.
4. Confirm that report/source/code hashes match current files and that the
   artifacts remain explicitly exploratory.

## Final verification

- `.venv/bin/python -m unittest discover -s tests -v`
- `.venv/bin/python -m compileall -q src scripts tests`
- `.venv/bin/python -m pip check`
- Recheck generated reports, checkpoints, sampled-pair CSV, and the 939-track M1
  cache after all code changes.

## Workspace constraint

This workspace is not a Git repository, so do not create worktrees or commits.
Keep changes local and report artifact paths. Implementation should proceed in
small test-first tasks with review gates between model, trainer, and runner.

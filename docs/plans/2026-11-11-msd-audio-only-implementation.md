# MSD Audio-Only Supervised Retrieval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Train and evaluate a separate audio-only Siamese retriever on the expanded MSD 10K temporal-feature catalog, using Last.fm links only as labels.

**Architecture:** Build an audio-only audit and artist-disjoint split without requiring BPM, lyrics, or genres; use the existing M1 preprocessing CLI to create a distinct train-normalized cache; add a 32-D audio-only encoder, a partition-local trainer with random negatives only, and a separate runner. Preserve all M1/M2/N0 outputs and code paths.

**Tech Stack:** Python, NumPy, h5py, PyTorch CPU, existing MSD/Last.fm parsers, existing pair sampler and retrieval metrics, `unittest`.

---

## Accepted constraints

- Model input: only the 1,024-step, 28-channel temporal features read from MSD HDF5. The 939-track M1 cache remains untouched; create a separate cache for the expanded catalog.
- Labels: existing Last.fm subset similar-track edges. No musiXmatch or tagtraum data is read by the audio-only experiment.
- Negatives: one uniformly sampled unobserved pair per positive, restricted to the current partition. Do not use genre-conditioned sampling.
- Duplicate filtering: keep the lexicographically smallest track for repeated `(artist_id, song_id)` keys and official MSD duplicate groups, matching M0.
- Split: seed 684, artist-disjoint 80/10/10, using the existing fallback artist key. Preliminary expected counts: 9,939 tracks split 7,935/971/1,033; within-partition edges 5,385/94/121; 2,609 cross-partition edges excluded.
- Training: seeds 684/685/686, 50 epochs, batch size 64, Adam 0.001, margin 1.0. Record validation loss, do not select checkpoints, and do not construct test representations/relevance until all checkpoints are fixed.
- Treat all results as exploratory. Unobserved Last.fm links are not verified negatives.
- Workspace has no Git repository: do not attempt worktrees or commits.

## Task 1: Build the audio-only audit function

**Files:** Create `src/data/audio_only_audit.py`; test `tests/test_audio_only_audit.py`.

1. Write failing tests for duplicate filtering, deterministic artist-disjoint splitting, unique undirected Last.fm edge counts, within-partition counts, cross-partition exclusion, and invalid/empty inputs.
2. Run `.venv/bin/python -m unittest tests.test_audio_only_audit -v`; confirm the new import/function is missing.
3. Implement `build_audio_only_audit(tracks, edges, duplicate_groups=None, seed=684)`. Do not require a valid tempo or any lyric/genre rows. Reuse the established M0 artist split policy and duplicate policy.
4. Return an audit structure containing `splits.artist_disjoint.track_ids`, source/duplicate counts, eligible track counts, unique link counts, per-split link counts, and cross-partition link count.
5. Run `.venv/bin/python -m unittest tests.test_audio_only_audit -v`; confirm all tests pass.

## Task 2: Add and run the audit CLI

**Files:** Create `scripts/audit_msd_audio.py`; test `tests/test_audio_only_audit_cli.py`.

1. Write a synthetic CLI test with MSD rows, Last.fm edges, and duplicate groups; assert output path, split IDs, counts, and source hashes.
2. Run `.venv/bin/python -m unittest tests.test_audio_only_audit_cli -v`; confirm the CLI is missing.
3. Implement defaults for `data/processed/msd_tracks.csv`, `data/processed/lastfm_similar`, `downloads/msd_duplicates.txt`, seed 684, and `results/msd_audio_only_audit.json`, with path/seed overrides.
4. Run `.venv/bin/python -m unittest tests.test_audio_only_audit_cli -v`.
5. Generate the real audit with `.venv/bin/python -m scripts.audit_msd_audio` and verify the expected 9,939 tracks, 7,935/971/1,033 split, 5,385/94/121 within-split edges, and 2,609 excluded cross-partition edges. Stop if the real values differ; investigate before cache generation.

## Task 3: Generate a separate expanded audio cache

**Files:** No code changes expected; generate `results/msd_audio_only_audit.json`, `data/processed/msd_audio_only_sequences.h5`, and `data/processed/msd_audio_only_sequences_meta.json`.

1. Run the existing M1 preprocessor against the new audit and distinct outputs:

   ```bash
   .venv/bin/python -m scripts.preprocess_audio \
     --audit results/msd_audio_only_audit.json \
     --output data/processed/msd_audio_only_sequences.h5 \
     --metadata-output data/processed/msd_audio_only_sequences_meta.json
   ```

2. Verify the HDF5 cache has the sorted 9,939 IDs, shape `(9939, 1024, 28)`, float32 finite sequences, and the expected channel schema.
3. Verify metadata split IDs match the audio-only audit and `normalization.train_ids` match exactly the 7,935 training IDs. Confirm the existing `audio_sequences.h5` and metadata are unchanged.

## Task 4: Add the 32-D audio-only encoder

**Files:** Modify `src/model.py` by adding `AudioOnlySiameseEncoder`; modify `tests/test_model.py`.

1. Write tests for finite `(B,32)` unit-normal embeddings, deterministic evaluation, same-instance pair reuse, and finite nonzero backward gradients from audio inputs.
2. Run `.venv/bin/python -m unittest tests.test_model -v`; confirm the new class is absent.
3. Implement the separate CNN branch from the approved design: Conv1d `28→32→64→64`, strides `2`, adaptive pooling to 16 ordered bins, `Linear(1024,32)`, ReLU, L2 normalization. Keep `SiameseEncoder` and `AudioFirstSiameseEncoder` unchanged.
4. Run `.venv/bin/python -m unittest tests.test_model -v`.

## Task 5: Add the random-negative audio-only trainer

**Files:** Create `src/train_audio_only.py`; test `tests/test_train_audio_only.py`.

1. Write a synthetic test for finite losses, parameter updates, reproducibility, ID alignment, fixed validation pairs, and train/validation partition isolation. Assert no genre or auxiliary modality argument is accepted.
2. Run `.venv/bin/python -m unittest tests.test_train_audio_only -v`; confirm the module is missing.
3. Implement `train_audio_only_seed(audio_sequences, song_ids, train_pools, validation_pools, seed, epochs=50, batch_size=64, learning_rate=0.001, margin=1.0)` using `AudioOnlySiameseEncoder`, `contrastive_loss`, and `sample_pairs(..., same_genre_fraction=0.0)`. Reject malformed pool membership, partition overlap, invalid shapes, non-finite inputs, and float32 conversion overflow.
4. Run `.venv/bin/python -m unittest tests.test_train_audio_only -v`.

## Task 6: Add the separate experiment runner

**Files:** Create `scripts/run_msd_audio_only.py`; test `tests/test_msd_audio_only_cli.py`.

1. Write an end-to-end synthetic test with distinct per-ID audio sequences and non-sorted partition IDs. Assert row alignment for training and test inputs, random-only negatives, no cross-partition pairs, all checkpoints saved before test input/relevance creation, finite metrics, source/code hashes, and report/CSV/checkpoint artifacts.
2. Run `.venv/bin/python -m unittest tests.test_msd_audio_only_cli -v`; confirm the runner is missing.
3. Implement path defaults/overrides for the audio-only audit, cache, metadata, Last.fm subset, output report, and checkpoint directory. Parse only Last.fm edges and the audio-only audit/cache; do not read MXM or tagtraum inputs.
4. Build train/validation pair pools with empty genre groups and random-only negative sampling. Validate cache split and train-only normalization provenance against the audit. Train and save every seed before creating test modalities/representations or relevance.
5. Save `results/msd_audio_only_experiment.json`, `results/msd_audio_only_training_curves.csv`, `results/msd_audio_only_sampled_pairs.csv`, and `results/checkpoints/msd_audio_only_seed_{684,685,686}.pt`. Include split, fit provenance, pair policy, metrics, dynamic sparse-edge warning, source/code hashes, runtime versions, and exact output paths.
6. Run `.venv/bin/python -m unittest tests.test_msd_audio_only_cli -v`.

## Task 7: Document, run, and verify the experiment

**Files:** Modify `README.md` with a distinct MSD audio-only section; preserve the M2/N0 descriptions.

1. Document that MSD supplies input features while Last.fm supplies supervision; list the audit/cache/train commands and outputs; state that this expanded split is not directly comparable with M2's 939-track split and all results are exploratory.
2. Run `.venv/bin/python -m scripts.run_msd_audio_only` with the fixed three-seed, 50-epoch defaults after confirming the audit and cache.
3. Verify all 3 checkpoints, 150 curve rows, pair-trace partition isolation, finite losses/metrics, test gallery/edge counts, train-only normalization IDs, and report hashes against current source files.
4. Run final checks:

   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   .venv/bin/python -m compileall -q src scripts tests
   .venv/bin/python -m pip check
   ```

5. Report exact counts and metrics with the sparse-label/exploratory caveat. Do not make significance or generalization claims.

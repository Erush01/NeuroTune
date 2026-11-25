# NeuroTune A–D Audio Cache and Fresh Artist-CV Design

**Date:** 2026-11-15  
**Status:** Completed for cache and split audit; model training remains deferred  
**Scope:** Build a separate raw audio cache and three-fold artist-grouped split from the completed A–D shard audit. Do not train a model in this work unit.

## Context

The A–D audit combines the downloaded MSD-format A/B/C/D archives without modifying the existing A/B experiment. It found 13,822 lyrics+genre+positive-tempo candidate tracks, including 6,855 IDs beyond the 6,967-track A/B catalog. The audit output is under `data/msd_hf_abcd_pilot/`. All 13,822 candidates passed first-30-second temporal preprocessing, producing 1,430,546 finite float32 rows across 26 channels; no tracks were rejected.

The old A/B artist-disjoint experiment used 452 test artists, 725 test tracks, and 215 positive links. The approved policy excludes all artist groups present in the A/B catalog from the new test folds, while allowing them in new training or validation partitions. The full old A/B Last.fm test-source edge file (1,940 edges) is excluded from all new label targets, not only the 215 edges that were internal to the prior model's held-out test tracks.

## Goals and non-goals

### Goals

- Validate the A–D candidates against the established first-30-seconds, ragged `(T, 26)` float32 audio contract.
- Produce a separate, raw/unnormalized cache and a machine-readable report of tracks rejected for track-local data problems.
- Build three deterministic artist-held-out folds from artist groups absent from the old A/B catalog.
- Keep train-source and official-test-source edges separate, prevent cross-partition edge leakage, and record provenance.
- Preserve all existing A/B audit, cache, split, model, and result artifacts.

### Non-goals

- No normalization before the fold-specific train IDs are known.
- No model training, checkpoint selection, or performance claim.
- No reuse of old A/B test-source links in train, validation, or the new test cohort.

## Data and cache design

The cache builder consumes the A–D audit and its sorted candidate catalog, then streams the A, B, C, and D archives. It reuses `preprocess_audio_window` and its established HDF5 safety limits. Each accepted track stores its first 30 seconds as a finite, non-empty, ragged `(T, 26)` float32 sequence. The cache remains raw and unnormalized; training-only normalization is deferred to a later model task.

Track-local data-quality failures, such as an empty or invalid temporal window, are skipped and recorded with the track ID and a reason code. Archive/hash mismatches, unsafe tar members, HDF5 link/VLEN/resource-limit violations, metadata identity mismatches, and publication failures abort the whole build. No partial cache, metadata, or rejection report is left behind. The CV split is built only after the cache is complete and uses exactly the cache-valid track IDs; edge counts are recomputed after invalid tracks are removed.

The cache and metadata use new targets under `data/msd_hf_abcd_pilot/`, for example:

- `abcd_audio_30s_26ch.h5`
- `abcd_audio_30s_26ch_meta.json`
- `abcd_audio_30s_26ch_rejections.json`

All generated targets refuse overwrite and are published with staging and rollback cleanup. The metadata records source, audit, and code hashes, cache-valid track IDs/counts, rejected IDs/reasons, channel/window contract, and the explicit raw/no-normalization status.

## Artist-grouped CV design

Artist identity follows the existing fallback contract: `artist_id`, then `artist_name`, then `track_id`. A test-eligible artist key is one that does not occur in the old A/B catalog. After cache validation, 1,764 eligible groups containing 2,371 tracks remained; all passed preprocessing.

Sort eligible artist keys, shuffle with seed 684, and assign round-robin to three test folds. The resulting galleries contain 805, 794, and 772 tracks (588 test artists per fold). In each fold, all other cache-valid artists—including the old A/B test artists—form the non-test pool. Shuffle these artists with seed 684, place the first floor-10% in validation, and use the remainder for training. Train, validation, and test artist keys are pairwise disjoint.

Edge assignment is source-aware:

- Last.fm train-archive edges internal to train artists become training labels.
- Last.fm train-archive edges internal to validation artists become validation labels.
- Last.fm train-archive edges internal to the held-out test fold are preserved separately as held-out train-source test labels.
- Official Last.fm test-archive edges internal to the held-out test fold are preserved in a separate official-test-source file.
- Cross-partition edges are excluded. Official test-source edges are never relabeled as training or validation data. Old A/B test links cannot enter any new partition.

The completed split keeps the two sources separate. Held-out train-source edge counts are 183/221/235 and held-out official-test-source counts are 50/30/45. The new-only subset also has 1,141 train-source and 180 official-test-source cross-fold edges excluded. The source counts were recomputed from the cache-valid IDs and are recorded in the fold manifests.

Each fold manifest records artist and track IDs by partition, edge counts by source, excluded cross-partition and invalid-track edge counts, deterministic seeds, input hashes, and code hashes. A complete known-positive superset is retained for future negative-sampling exclusion only; it is not a training target or model input. This follows the existing project rule that known positives withheld from a label partition must not be sampled as negatives.

## Failure handling and publication

- Reject pre-existing targets before work begins; never overwrite A/B artifacts or prior A–D audit outputs.
- Verify archive and audit hashes before and after scanning.
- Stream archive members, enforce member-size limits, and reuse the existing HDF5 dimension, dtype, link, and VLEN checks before reading arrays.
- Skip only explicitly classified track-local data-quality failures. Treat identity, archive-integrity, security, and publication failures as fatal.
- Stage all cache outputs together and each split output set together; publish without clobbering and remove newly published files if any step fails.
- Build the CV split from cache-valid IDs so every split ID has a corresponding cached sequence.

## Verification

Unit tests cover cache extraction from all four named shards, ID/metadata agreement, ragged offsets and sequence boundaries, safe rejection of malformed HDF5 structures, track-local skip reporting, fatal archive/security errors, deterministic folds, absence of old A/B artist keys from every test fold, pairwise artist disjointness, source-specific edge routing, cross-partition exclusions, source/hash provenance, output collisions, and rollback.

The real run verified sorted cache IDs, offsets, finite `(T, 26)` values, cache/audit/catalog/source/code hashes, rejection summaries, exact cache-to-split ID agreement, edge-source routing, fold counts, and unchanged A/B hashes. The split is under `data/msd_hf_abcd_pilot/cv/` and contains the manifest, a separate known-positive exclusion set, and four source-specific edge files per fold. No model was trained or evaluated in this work unit; training and any performance analysis require a separate approved protocol.

### Completed artifact counts (2026-11-15)

The raw cache contains 13,822 tracks, 1,430,546 segments, and zero rejected tracks. The fold manifest covers 75,690 cache-valid known-positive edges (67,657 train-source and 8,033 official-test-source); invalid-endpoint exclusions were zero. Each fold has 4,622 training artists, 513 validation artists, and 588 test artists.

| Fold | Train/validation/test tracks | Train-source train/validation/test edges | Official test-source test edges |
| --- | ---: | ---: | ---: |
| 0 | 11,640 / 1,377 / 805 | 49,266 / 811 / 183 | 50 |
| 1 | 11,693 / 1,335 / 794 | 48,523 / 835 / 221 | 30 |
| 2 | 11,669 / 1,381 / 772 | 47,992 / 831 / 235 | 45 |

| Fold | Cross-partition train-source edges | Cross-partition official-test-source edges | Official-test-source edges kept out of train/validation |
| --- | ---: | ---: | ---: |
| 0 | 17,397 | 1,899 | 4,144 |
| 1 | 18,078 | 1,559 | 4,504 |
| 2 | 18,599 | 1,893 | 4,155 |

The complete 1,940-edge old A/B test-source set is excluded from all label targets and recorded separately for known-positive exclusion only. All six protected A/B audit/cache artifact hashes remain unchanged.

## Limitations

The three-fold cohort is fresh relative to the old A/B test artists and track IDs, but it still uses Last.fm labels. The held-out train-source edges are a within-source artist-grouped evaluation, not an independent external test. Missing Last.fm links remain unverified negatives. The third-party A–D archives are authorized for project use by the user; public source provenance and checksum metadata remain unverified.

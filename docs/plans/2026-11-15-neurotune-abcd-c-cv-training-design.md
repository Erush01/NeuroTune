# NeuroTune A–D C-Model Training Design

**Date:** 2026-11-15  
**Status:** Implemented; fixed CV run completed (2026-11-15)  
**Scope:** Train the existing NeuroTune v1 C model on the cache-valid A–D catalog with the approved artist-grouped three-fold split, and compare it with a matched cosine reference. Do not change prior experiments or tune on test results.

## Context

The A–D data preparation is complete. The raw, unnormalized cache at `data/msd_hf_abcd_pilot/abcd_audio_30s_26ch.h5` contains 13,822 tracks and 1,430,546 finite `(T, 26)` segment rows, with no rejected tracks. The verified artist-grouped manifests and source-separated edges are under `data/msd_hf_abcd_pilot/cv/`.

Each fold partitions all 13,822 tracks into training, validation, and held-out artist groups. The test groups contain only artists absent from the full A/B catalog. The full 1,940-edge A/B test-source set is absent from all train, validation, and test target files, but remains in the known-positive exclusion set. No model has been trained on this A–D cohort.

The existing `NeuroTuneV1Encoder` and `train_neurotune_v1` support the previously approved A–D architecture and training protocol. Variant C uses audio, BPM, and genre; it does not use lyrics. This is the selected development candidate, while the other variants remain historical ablations.

## Goals and non-goals

### Goals

- Train C independently on each of the three folds with seeds 684, 685, and 686 (nine runs total).
- Fit audio normalization, BPM statistics, and genre vocabulary on each fold's training IDs only.
- Train only from that fold's Last.fm train-source training edges; use its train-source validation edges only for checkpoint selection.
- Exclude known positive links from negative candidates, including positives withheld from the current train/validation target set.
- Compare C with the equal-modality cosine baseline using the same audio+BPM+genre inputs and the same fold-specific train-fitted transforms.
- Report held-out retrieval metrics separately for train-source and official-test-source edges.
- Save provenance, training histories, schedules, and selected checkpoints in a new output namespace with no-overwrite publication.

### Non-goals

- No A/B/D training or new modality ablations.
- No combining the two test-edge sources into a single score.
- No hyperparameter tuning based on test metrics or claims of independent generalization.
- No changes to the existing A/B pilot, official 10K CV, A–D cache, or A–D split artifacts.

## Training protocol

For each fold, use the manifest's train, validation, and test IDs. Fit transformations only on train IDs, then build the C model inputs for that fold. The model receives the 26-channel ragged audio sequence, standardized BPM, and train-vocabulary genre vector; no lyric values are loaded or supplied to the model. Use the existing shared encoder, 64-dimensional normalized output, cosine triplet-margin loss with margin 0.2, Adam at learning rate 0.001, batch size 64, 50 epochs, and CPU execution.

For every fold and seed, generate a deterministic training schedule from the fold's train-source training edges, with the existing bidirectional positive ordering and uniform negative sampling. Generate one fixed validation schedule from train-source validation edges. Restrict the full known-positive edge set to edges whose two endpoints are both in the candidate pool before passing it to the partition-local sampler; this respects the sampler's ID contract while preventing any known positive inside that pool from becoming a negative. Known-positive edges are exclusions only, never additional supervised targets.

Select each run's checkpoint by validation NDCG@10, then lower validation triplet loss, then the earlier epoch. Do not use either test-edge source for training, validation, or checkpoint selection.

## Baseline and held-out evaluation

Build one C-matched cosine baseline per fold using the fold's train-fitted audio, BPM, and genre representations with equal modality weighting. Evaluate the baseline against the same held-out gallery and the same two source-separated relevance sets used for C.

The evaluation barrier requires all nine C checkpoints, all nine compressed train/validation schedule artifacts, and the complete training-curves file to be saved before constructing any test embeddings or relevance maps. After this barrier, evaluate C and the baseline on:

1. `test_train_source_edges` — held-out positives from the Last.fm train archive.
2. `test_official_source_edges` — held-out positives from the official Last.fm test archive.

Report Recall@10, HitRate@10, MRR@10, NDCG@10, eligible query counts, and positive-edge counts separately for each source, fold, and model seed where applicable. Report the three-fold cosine observations separately from the nine fold-seed model observations; include per-fold C seed means for descriptive paired context. Do not produce a combined-source metric or significance claim.

## Artifacts and provenance

Write only under `results/neurotune_v1_abcd_cv/`, refusing any existing output namespace. The expected artifacts are:

- `experiment.json` — protocol, fold/source counts, separate model and baseline metrics, selected epochs, caveats, runtime versions, and input/code/transform/schedule/checkpoint hashes.
- `training_curves.csv` — 450 rows (three folds × three seeds × 50 epochs).
- `triplet_schedules/fold_{fold}_seed_{seed}.npz` — compressed deterministic training and validation schedules.
- `checkpoints/fold_{fold}_C_seed_{seed}.pt` — nine validation-selected checkpoints.

Load and process one fold at a time to bound memory; release fold features and schedules after saving their artifacts. Rebuild or reload fold-specific test representations only after the all-checkpoint barrier. Stage the complete result directory and publish it without clobbering; remove newly created outputs if the run fails, while preserving pre-existing artifacts.

Record hashes for the A–D audit, catalog, cache and metadata, CV manifest, known-positive set, all fold edge files, auxiliary source used for the C inputs, relevant implementation code, transforms, schedules, and checkpoints. Verify the HDF5 cache provenance attributes against its sidecar metadata and current audit/catalog before training.

## Verification

Synthetic tests cover fold isolation, train-only transform fitting, C-only active modalities, correct source routing, known-positive-safe negatives, deterministic schedules, validation-only checkpoint selection, schedule/checkpoint provenance, separate source-specific metrics, matched baseline behavior, checkpoint-before-test ordering, output collision refusal, and rollback.

Before any real training, run the focused tests and inspect the runner contract. For the real run, verify all nine checkpoints and schedules, 450 curve rows, exact fold and edge counts, source-separated metrics, current input/code hashes, and the test-evaluation barrier. Confirm prior A/B, official 10K CV, A–D cache, and split artifacts remain unchanged. Finish with the full unit suite, `compileall`, and `pip check`.

## Limitations

This is artist-grouped evaluation over the current A–D shard cohort, not an independent external test. The folds reuse Last.fm source pools. Missing links are not verified negatives, and the known-positive exclusion set includes withheld test-source edges for negative filtering only. Results are exploratory and should be reported with fold/seed variation and source-specific label counts; they do not establish broad recommendation quality or generalization.

## Completed run (2026-11-15)

The fixed run completed in about 5 hours 3 minutes on CPU. It published nine validation-selected C checkpoints, nine compressed `triplet_schedules/` artifacts, 450 training-curve rows, and `results/neurotune_v1_abcd_cv/experiment.json`. Independent verification checked every output hash, all 43,749,162 scheduled triplet rows for target membership and known-positive-safe negatives, and 9,465 pre-existing result/cache files for byte-for-byte preservation. The full project suite passed 396 tests before the run; `compileall` and `pip check` also passed.

The aggregate mean ± population standard deviation metrics were:

| Held-out label source | Method | Observations | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Last.fm train archive | C | 9 fold-seed runs | 0.1681 ± 0.0274 | 0.2232 ± 0.0347 | 0.0769 ± 0.0186 | 0.0895 ± 0.0178 |
| Last.fm train archive | C-matched cosine | 3 folds | 0.1510 ± 0.0248 | 0.2022 ± 0.0309 | 0.0682 ± 0.0089 | 0.0787 ± 0.0126 |
| Official Last.fm test archive | C | 9 fold-seed runs | 0.1521 ± 0.0556 | 0.1802 ± 0.0699 | 0.0560 ± 0.0252 | 0.0760 ± 0.0292 |
| Official Last.fm test archive | C-matched cosine | 3 folds | 0.1649 ± 0.0716 | 0.1970 ± 0.0854 | 0.0801 ± 0.0449 | 0.0920 ± 0.0446 |

C is descriptively higher on all four aggregate metrics for the held-out train-source links, while cosine is higher on all four for the official-test-source links. The different observation counts, shared source pools, small official-test-source edge counts (30–50 per fold), and unverified negatives prevent significance or generalization claims. The third-party A–D archive is authorized for project use by the user, but its public provenance and license remain unverified. Detailed fold counts and selected epochs are in the README and experiment report.

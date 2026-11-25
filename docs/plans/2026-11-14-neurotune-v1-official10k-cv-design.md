# NeuroTune v1 Official-10K Artist-Grouped CV Design

**Status:** Approved and implemented on 2026-11-14; completed as an exploratory robustness check.  
**Scope:** Evaluate the validation-selected NeuroTune v1 C variant on the official 10K MSD subset's complete-modality catalog. This is a robustness check using official Last.fm train-edge labels, not a new external test set.

## Goal and interpretation

The A/B pilot validation analysis selected C (audio+BPM+genre) as the next development candidate, while retaining cosine retrieval as a reference. Test metrics from the 6,967-track A/B pilot are frozen and must not guide this CV. The official 10K route gives a source-verified MSD audio catalog without downloading the approximately 280 GB full archive, but only 939 tracks have all required modalities.

The 939-track graph has 600 unique within-catalog Last.fm train edges. The official Last.fm test archive has no usable edges in this common catalog. Therefore the CV will use official train edges in artist-held-out folds. Of the 600 edges, the 220 whose endpoints fall in the same outer artist fold are each held out for testing in exactly one fold; the 380 cross-fold edges are excluded from every outer test set. An edge may be a training/validation edge in another fold, as in standard k-fold CV; do not describe the result as an independent external test or as an untouched project-wide label set.

## Data contract

- **Tracks:** the 939 IDs in `results/data_audit.json`'s duplicate-filtered, artist-disjoint train/validation/test union. Every track has positive BPM, musiXmatch lyrics, and tagtraum CD2C genre.
- **Audio:** read official 10K subset HDF5 records, extract raw first-30-second `[0,30)` segment sequences with `src/data/preprocess_audio_window.py`, and store them in a separate ragged 26-channel cache. Channels are 12 pitch, 12 timbre, loudness, and clipped segment duration. Do not reuse the 28-channel fixed-grid cache or any third-party A/B audio.
- **Labels:** parse official Last.fm train edges and retain unique undirected links whose endpoints both occur in the 939-track catalog. The existing subset join has 600 such links. Do not use Last.fm test labels; they have zero overlap here.
- **Auxiliary features:** use the existing train-fitted NeuroTune v1 BPM, 5,000-word TF-IDF, and genre transforms. Do not modify old artifacts or feature code without a specific need.

## Outer artist folds and inner edge validation

Sort artist keys with the existing `artist_id → artist_name → track_id` fallback, shuffle using seed 684, and assign artists round-robin to three folds. The feasibility audit measured:

| Fold | Artists | Tracks | Within-fold test edges |
| --- | ---: | ---: | ---: |
| 0 | 201 | 323 | 63 |
| 1 | 201 | 310 | 78 |
| 2 | 200 | 306 | 79 |

The folds exclude 380 cross-fold edges from their own test comparisons. For each outer fold, keep all its artists/tracks entirely out of that fold's train and validation inputs. From the 289/262/269 edges with both endpoints in the other two folds, shuffle with `random.Random(684 + fold_index)`, reserve 20% (floor) for validation, and use the rest for training:

| Held-out fold | Train edges | Validation edges | Test edges |
| --- | ---: | ---: | ---: |
| 0 | 232 | 57 | 63 |
| 1 | 210 | 52 | 78 |
| 2 | 216 | 53 | 79 |

Validation is an **edge holdout**, not an artist/track holdout: its endpoints can also occur in training edges, but the exact positive labels must be disjoint. This is intentional because strict nested artist validation produced only 5/1/6 validation edges. Fit audio/BPM/TF-IDF/genre transforms on all non-test tracks in the outer fold; no held-out test track may contribute to fitted statistics or vocabularies.

For both training and validation schedules, sample negatives only from non-test track IDs. Exclude the anchor, the current positive, and every known positive neighbor among all non-test edges (training plus validation). This prevents withheld validation positives from being treated as negatives. Training schedules contain training positives only; validation schedules contain validation positives only and are fixed per outer fold.

## Candidate model, baseline, and training

- Train **variant C only**: the existing shared 64-D masked CNN with BPM and genre branches, without lyrics. Keep the completed A/B run unchanged.
- Use Adam at 0.001, batch size 64, cosine triplet margin 0.2, 50 epochs, and seeds 684/685/686. Resample training negatives deterministically per seed/epoch and reuse the fold/seed schedule for that run.
- Select checkpoints using validation NDCG@10, then lower fixed validation triplet loss, then earlier epoch. No test data is accepted by the trainer.
- Compare against a deterministic equal-modality cosine baseline using the **same C modalities** (train-normalized mean audio, BPM, and genre), so the comparison isolates learned metric/encoder value rather than a different modality set.
- Report fold/seed results descriptively. The fixed held-out galleries contain 323/310/306 tracks and 63/78/79 positive edges. Cross-fold measures are related, and model seeds do not create independent test samples; do not make significance or generalization claims.

All nine fold/seed checkpoints must be selected and saved before any held-out test relevance or test embeddings are built. Each model's own held-out fold edges are excluded from that model's train and validation data. The fact that other folds use those edges for their training is standard cross-validation, not a single global holdout.

## Artifacts and safety

Create new targets only, refusing collisions:

- Cache: `data/processed/neurotune_v1_official10k_audio_30s_26ch.h5` and `data/processed/neurotune_v1_official10k_audio_30s_26ch_meta.json`.
- Split audit: `results/neurotune_v1_official10k_cv_audit.json`, containing the fold track IDs, edge-level train/validation/test assignments, counts, policies, and source hashes.
- CV outputs: `results/neurotune_v1_official10k_cv/experiment.json`, `training_curves.csv`, compact fold/seed sampled-triplet schedules, and nine selected checkpoints.

Record catalog/split/edge counts, fold assignments, validation edge IDs or their complete audit/hash, train-only fit-ID hashes, source/cache/code/schedule/checkpoint hashes, exact settings, all fold/seed metrics, and caveats. Stage output publication and roll back newly created files on failure or interruption. Leave the A/B pilot and all earlier reports/caches/checkpoints unchanged.

## Completed run and interpretation

The fixed C-only run completed in about 214 seconds and produced `results/neurotune_v1_official10k_cv/experiment.json`, 450 curve rows, nine fold/seed schedules, and nine selected checkpoints. The report's source, code, cache, split, schedule, checkpoint, and curve hashes were verified. Selected epochs ranged from 1 to 49.

| Method | Observation unit | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Variant C | 9 fold-seed runs | 0.2618 | 0.3270 | 0.1202 | 0.1440 |
| C-matched cosine | 3 folds | 0.2322 | 0.2910 | 0.1106 | 0.1307 |

Per-fold gallery/test-edge/query counts are 323/63/84, 310/78/97, and 306/79/107. C's fold-mean NDCG@10 was 0.1629, 0.1178, and 0.1513, compared with cosine's 0.1585, 0.1147, and 0.1189. The aggregate means favor C on all four metrics in this run, but this descriptive comparison is not inferential: it averages nine model observations against three baseline observations, and the same official train-label source is reused across folds. A fold's test links can be used as train/validation links in other folds. The run therefore remains a robustness check, not a new independent test or evidence of generalization. Missing links remain unverified negatives.

## Verification criteria

- Read-only preflight confirms 939 catalog IDs, all required modality coverage, the 600-edge source intersection, and the reported fold/edge counts before cache creation.
- Cache output is ragged, raw, finite float32 `(T,26)`, aligned to sorted catalog IDs, and byte-reproducible for sampled source tracks.
- Tests prove deterministic artist folds and edge validation splits, no train/validation positive overlap, no test IDs/edges in the corresponding fold's schedules or fit-ID hash, and exclusion of every known non-test positive from negatives.
- The CV trainer uses only non-test features and train/validation schedules, performs validation without gradients, and selects checkpoints by the fixed tie-break; no test arguments are present.
- The runner proves all nine checkpoints exist before any test evaluation; verifies hashes, collision refusal, rollback, and exact baseline/model galleries.
- Final verification includes focused tests, the full test suite, compilation, and dependency checks. The final report is explicitly a small-catalog CV robustness check, not a new independent external evaluation.

## Limitations

The official 10K subset is source-verified, but its 939-track complete-modality catalog is small. Artist grouping excludes 380 cross-fold links; each fold has only 52–57 validation and 63–79 test edges. The official Last.fm train-edge labels have been used in earlier project experiments, and missing edges remain unverified negatives. Companion-data access terms still apply. This CV cannot validate performance on the unverified A/B source or full MSD-scale catalog.

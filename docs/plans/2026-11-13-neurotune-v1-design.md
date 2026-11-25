# NeuroTune v1 Multimodal Model Design

**Status:** Approved for implementation on 2026-11-13.  
**Scope:** Implement the supplied multimodal architecture on the A/B pilot catalog only. Do not download C–Z or alter prior experiments.

## Goal and constraints

Implement the architecture in `NeuroTune_v1_Model_Mimarisi_ve_Literatur.md` as a shared single-track embedding encoder trained with triplet metric learning. Inputs are the A/B catalog's MSD temporal audio, BPM, musiXmatch word counts, and tagtraum genre. The A/B HF upload remains a local research pilot: its provenance/license/checksums are not verified, and results must not be generalized to the full MSD.

The existing M1/M2, N0, and MSD-audio-only implementations and their outputs remain unchanged. The A/B cache is raw and variable length. No train-fitted normalization is stored in the cache.

## Dataset and split contract

The A/B catalog has 6,967 tracks (6,028 beyond the original 10K feature intersection), all with audio, positive BPM, lyrics, and genre. The ragged cache stores the first `[0,30)` seconds as 26 channels in this order: 12 pitch, 12 timbre, loudness, and clipped segment duration. The 26-channel sequence is not padded on disk.

Last.fm train and test archives remain distinct. The audit found 288 undirected edges in both archives; those copies are removed from training, leaving 17,536 train edges and 1,940 official test edges. Test-source artists define the final held-out artist group. With split seed 684 and a 10% validation assignment over the remaining artists, the approved split is:

| Partition | Artists | Tracks | Positive edges used | Query tracks |
|---|---:|---:|---:|---:|
| Train | 3,157 | 5,632 | 12,916 Last.fm train edges | — |
| Validation | 350 | 610 | 185 Last.fm train edges | 238 |
| Test | 452 | 725 | 215 official Last.fm test edges | 265 |

Cross-partition edges are excluded: 4,435 train-source edges and 1,725 official-test edges. Validation and test retrieval galleries contain all tracks in their respective partitions (610 and 725). Test edges are used only for the final evaluation; they are not training or negative-sampling inputs. The 1,940-edge source count is not the number of test positives after enforcing artist disjointness.

## Train-fitted inputs

- **Audio:** Fit per-channel mean and population standard deviation over raw segment rows belonging only to training tracks. Replace zero standard deviations with one. Normalize the 26 real timesteps; padded cells are zero and masked.
- **BPM:** Use the positive catalog BPM values with train-only mean/std normalization; zero train std maps to one.
- **Lyrics:** Keep the fixed 5,000-word musiXmatch vocabulary. Fit smoothed IDF using training tracks only; transform counts to per-track L2-normalized TF-IDF. Build minibatch dense lyric rows from the sparse/count representation rather than storing a padded time dimension.
- **Genre:** Use one-hot encoding over sorted genre labels observed in training. An unseen validation/test genre maps to an all-zero vector.

No transform may fit on validation/test rows. Record fitted statistics and vocabularies in the experiment manifest.

## Encoder and objective

One encoder with shared weights maps each track to one 64-D L2-normalized embedding.

- **Audio branch:** Input `(B,26,T)` after batch padding and masking. Apply `Conv1D(26,32,k=5)`, `Conv1D(32,64,k=5)`, and `Conv1D(64,64,k=3)`, each stride 1, same-length padding, and ReLU. After every convolution, zero invalid timesteps with the mask before the next layer so convolution biases at padded cells cannot leak back into valid outputs. Use the mask for a mean over valid post-CNN timesteps, then project 64→64. The duration channel is retained; pooling is a masked mean, not duration weighted.
- **BPM branch:** `1→8` linear + ReLU.
- **Lyrics branch:** `5000→32` linear + ReLU.
- **Genre branch:** `G→16` linear + ReLU, where `G` is the training genre vocabulary size.
- **Fusion:** Concatenate dimensions `64+8+32+16=120`; apply `120→128` linear + ReLU, then `128→64` linear and L2 normalization.
- **Ablation variants:** A: audio only; B: audio+BPM; C: audio+BPM+genre; D: full audio+BPM+genre+lyrics. All use the same split, encoder settings, triplet stream, seeds, and training budget. The optional CNN+GRU variant is deferred.
- **Loss:** Cosine-distance triplet margin loss `mean(relu(d(a,p) - d(a,n) + 0.2))`, with `d(u,v)=1-dot(u,v)` for normalized embeddings. Margin 0.2 is the initial fixed value for this experiment.

## Triplet construction and training

Each undirected train positive contributes both directed `(anchor, positive)` orientations. Per epoch, sample one uniformly random negative for each directed positive from training tracks only; exclude the anchor, positive, and that anchor's known train-positive neighbors. Do not use genre to select negatives and do not mine semi-hard negatives in v1. Unobserved pairs are not verified negatives; some latent false negatives are unavoidable and must be documented. Resample negatives each epoch deterministically from the run seed, without materializing all candidate pairs. Validation uses a fixed deterministic random negative per directed validation positive drawn only from validation tracks.

Use Adam (`lr=0.001`), batch size 64 triplets, 50 epochs, and seeds 684, 685, and 686 for each A–D variant. Select each checkpoint by maximum validation NDCG@10; break ties by lower validation triplet loss, then earlier epoch. Save all selected checkpoints before reading test relationships for evaluation or producing test embeddings.

## Baseline, evaluation, and reporting

Add a new same-split equal-modality cosine baseline; do not reuse the old 939-track B0 result as a direct comparison. The baseline uses train-normalized mean audio summaries plus train-fitted BPM, TF-IDF, and genre features, with equal modality weighting.

Report Recall@10, NDCG@10, and MRR on the fixed validation and test galleries. Validation is for checkpoint selection. In the successful published run, evaluate the 215 held-out test edges and 265 query tracks after all 12 checkpoints are fixed. Report each seed and descriptive aggregates; do not make significance or generalization claims from this single pilot test cohort. Record split IDs, edge-source provenance, transform statistics, model/training settings, runtime, source/code/cache hashes, training curves, a compact sampled-triplet schedule generated once per seed and reused across variants, checkpoints, and metrics. **Operational deviation:** an earlier unchanged fixed-protocol attempt completed training and computed test metrics, then failed while writing the curves CSV due to a redundant `MRR` history key. Staged files and those metrics were rolled back/not retained and were not used for tuning. The metric alias was removed, curve serialization was moved before test evaluation, and the exact protocol was rerun successfully. Thus the held-out evaluator executed in two attempts, though no first-attempt metrics informed any model or protocol decision.

## Verification criteria

- Ragged loader/collation preserves each track's true length; padded values do not affect masked pooling.
- Encoder tests cover input/output shapes, finite values, unit norms, shared-weight triplet flow, and gradient updates for variants A–D.
- Sampler tests prove deterministic train-only membership, both positive directions, exclusions, and no test-edge use.
- Transform tests prove train-only fitting and zero handling for constant channels/features.
- Synthetic runner tests prove validation-only selection and that test relevance/embeddings are not built before checkpoints are fixed.
- The real pilot run records all artifacts and hashes under a new `results/neurotune_v1_ab/` namespace. Prior result files/caches are not overwritten.

## Known limitations

This is a research pilot from a third-party HF upload with unverified license/provenance. Last.fm links are sparse behavioral positives, not exhaustive similarity labels; random missing-link negatives may be false negatives. Only 215 official test edges remain within the held-out artist gallery. Do not redistribute extracted data or claim that A/B estimates full-MSD performance.

## Implementation results (2026-11-14)

Implementation and the fixed production run are complete. The report and all artifacts are under `results/neurotune_v1_ab/`; the production preflight confirmed 5,632/610/725 train/validation/test tracks, 12,916/185/215 within-partition edges, 265 test queries, 288 upstream train/test overlap removals, and zero overlap in the published raw edge CSVs. All 12 selected checkpoints, 600 curve rows, three schedules, source/code/cache hashes, and validation-selected epochs were verified.

The same-split equal-modality cosine baseline scored Recall@10 0.1803, HitRate@10 0.2528, MRR@10 0.1105, and NDCG@10 0.1120. Across the three seeds, A scored 0.0520/0.0780/0.0256/0.0271, B 0.0413/0.0667/0.0206/0.0220, C 0.1887/0.2629/0.0912/0.1031, and D 0.1874/0.2553/0.0887/0.1013 for those metrics; the README records sample standard deviations. C and D exceeded cosine on mean Recall@10 and HitRate@10, while cosine was higher on MRR@10 and NDCG@10. This suggests the models retrieve slightly more relevant items within the top ten, but the baseline tends to rank relevant items higher. These are descriptive results from one test cohort, not statistical evidence. The successful run took 6:03:03 wall time and peaked at 2,401,640 KiB resident memory on this environment.

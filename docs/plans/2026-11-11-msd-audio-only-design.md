# MSD Audio-Only Supervised Retrieval Design

**Status:** Approved on 2026-11-11

## Goal

Train a Siamese retrieval model using temporal audio features from the MSD 10K subset as its only model inputs. Keep Last.fm similar-track links as the supervised positive labels and held-out retrieval relevance. This is an audio-only feature experiment, not a claim that MSD itself contains similarity labels.

The existing M1/M2/N0 caches, models, reports, and checkpoints remain unchanged. The expanded catalog and new artist-disjoint split mean this experiment is not a directly controlled comparison with the 939-track M2 run.

## Dataset and split

The M1 inspection found valid aligned temporal arrays in all 10,000 subset HDF5 files. Applying the existing artist/song duplicate and official MSD duplicate-list policy leaves 9,939 audio tracks; unlike the multimodal catalog, this selection does not require lyrics, genre labels, or valid BPM.

Using seed 684 and the existing 80/10/10 artist-disjoint split policy produces 7,935 train, 971 validation, and 1,033 test tracks. The current Last.fm subset has 8,209 unique undirected links within this duplicate-filtered catalog. The split retains 5,385 train, 94 validation, and 121 test links; 2,609 cross-partition links are excluded. 4,517 catalog tracks have at least one observed link. These are audit counts, not a guarantee that unobserved pairs are true negatives.

Build a separate audit and temporal cache for this catalog. Keep the existing 939-track M1 cache and `results/data_audit.json` untouched. Fit audio normalization on train IDs only and record that provenance in the new cache metadata.

## Model

Add a separate `AudioOnlySiameseEncoder` using only the 1,024-step, 28-channel M1 temporal sequence:

1. `Conv1d(28,32,kernel=7,stride=2,padding=3)` + ReLU
2. `Conv1d(32,64,kernel=5,stride=2,padding=2)` + ReLU
3. `Conv1d(64,64,kernel=3,stride=2,padding=1)` + ReLU
4. Adaptive average pooling to 16 ordered time bins, flatten, `Linear(1024,32)` + ReLU
5. L2-normalize the 32-dimensional output.

This reuses the M2 audio branch design without BPM, lyrics, genre, or a fusion layer. A 32-D embedding is sufficient for cosine retrieval and avoids adding a projection solely to match M2's 64-D size.

## Training and evaluation

Use a separate trainer and runner; do not modify M2 or N0. Reuse the contrastive margin loss and partition-local pair infrastructure. Use one uniformly sampled unobserved negative per positive within each partition. Do not use tagtraum genres for same-genre negative selection.

Keep the fixed exploratory settings: seeds 684/685/686, 50 epochs, batch size 64, Adam learning rate 0.001, margin 1.0. Sample train negatives per epoch; fix validation pairs per seed and record validation loss without checkpoint selection. Save all fixed-epoch checkpoints before creating test representations or relevance, then evaluate each checkpoint against the complete 1,033-track test gallery using the existing Recall@10, HitRate@10, MRR@10, and NDCG@10 metrics.

The observed-link dataset is sparse and missing links are not verified negatives. The 94 validation and 121 test edges support exploratory reporting only; do not make strong performance, significance, or generalization claims.

## Outputs

- `results/msd_audio_only_audit.json`
- `data/processed/msd_audio_only_sequences.h5`
- `data/processed/msd_audio_only_sequences_meta.json`
- `results/msd_audio_only_experiment.json`
- `results/msd_audio_only_training_curves.csv`
- `results/msd_audio_only_sampled_pairs.csv`
- `results/checkpoints/msd_audio_only_seed_{684,685,686}.pt`

Reports and checkpoints record the split, fit provenance, sampled pairs, source/code hashes, and runtime versions. The experiment report and README must identify Last.fm as the label source and explicitly mark the results exploratory.

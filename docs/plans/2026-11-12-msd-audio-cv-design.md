# MSD Audio-Only Last.fm Cross-Validation Design

**Status:** Approved on 2026-11-12

## Goal

Estimate how the existing MSD-temporal-audio-only Siamese retrieval protocol performs across multiple artist-disjoint test galleries, using the current Last.fm similar-track subset as labels. This is an evaluation extension of the existing experiment, not a new model-selection or Taste Profile experiment.

## Data and artist-grouped folds

Use the 9,939-track duplicate-filtered MSD 10K temporal catalog and the current Last.fm subset. The full Last.fm train archive was scanned in place and contained no additional unique links inside this catalog; the Last.fm test archive had no matching catalog source tracks. The subset has 8,209 unique undirected catalog links.

Create three deterministic outer folds by grouping eligible tracks on the existing artist-key fallback (`artist_id`, then `artist_name`, then `track_id`), sorting artist keys, shuffling with seed 684, and distributing artist groups round-robin. A preliminary assignment produced 3,249/3,321/3,369 tracks and 1,065/965/981 within-fold positive links. The implementation must recompute and verify the actual assignment counts. Each fold is used once as the complete test gallery; the other two folds supply training IDs and within-training-fold links. Links crossing fold boundaries are excluded from both training and evaluation.

## Fold-specific audio normalization

Do not reuse the current 80/10/10 normalized audio cache for CV. Its normalization statistics were fit on that split's training IDs and would leak into test tracks in other folds. Build one separate cache per fold from raw MSD HDF5 sequences, fitting continuous-channel normalization only on that fold's training IDs. Keep event-count channels unchanged. Record the fold assignment, training IDs, normalization statistics/provenance, source hashes, and cache hashes. Leave all existing M1/M2/audio-only caches and reports untouched.

## Training and evaluation protocol

Keep the approved audio-only `AudioOnlySiameseEncoder` and fixed protocol unchanged: 32-D normalized embeddings, Last.fm positives, one uniformly sampled unobserved negative per positive within the training fold, seeds 684/685/686, 50 epochs, batch size 64, Adam 0.001, margin 1.0, and K=10. Train on all non-test folds; do not reserve an inner validation partition or select checkpoints. The CV trainer records training loss only. This evaluates the current fixed-epoch protocol and does not address its observed overfitting.

Train three seeds for each of the three folds (nine checkpoints total). Save all nine checkpoints before creating any test embeddings or relevance labels. For each fold, rank every test-fold track against the full test-fold gallery, excluding the query itself, and evaluate only observed Last.fm links internal to that fold. Report Recall@10, HitRate@10, MRR@10, and NDCG@10 per fold and seed, plus descriptive across-fold summaries. Do not make significance or generalization claims; the fold results are exploratory and not directly comparable to the prior 1,033-track gallery.

## Scope and labels

The Taste Profile triplets remain analysis-only. The expanded catalog yielded 199,488 co-listening pairs with at least one shared listener, but only 545 overlap current Last.fm links; those are a distinct weak-label target. No Taste Profile pairs enter this CV. The MSD song/track mismatch blacklist is not present locally, so previously computed Taste Profile counts remain preliminary and are irrelevant to the Last.fm-only CV labels.

## Outputs

- `results/msd_audio_only_cv_audit.json`
- `data/processed/msd_audio_only_cv_fold_{0,1,2}.h5`
- `data/processed/msd_audio_only_cv_fold_{0,1,2}_meta.json`
- `results/msd_audio_only_cv_experiment.json`
- `results/msd_audio_only_cv_training_curves.csv`
- `results/msd_audio_only_cv_sampled_pairs.csv`
- `results/checkpoints/msd_audio_only_cv_fold_{0,1,2}_seed_{684,685,686}.pt`

All CV artifacts use distinct names. Reports/checkpoints record fold IDs, source/code/cache hashes, runtime versions, sampled-pair trace hash, fixed configuration, metrics, and the exploratory caveat. The existing single-split audio-only run remains unchanged.

The runner must reject existing final artifact targets before writing and remove only artifacts created by the current invocation if an exception interrupts it. It must load and release one full fold cache at a time to keep peak memory bounded. Before training, it verifies that each fold cache references the current CV audit and that the current Last.fm directory hash matches the source hash recorded by that audit.

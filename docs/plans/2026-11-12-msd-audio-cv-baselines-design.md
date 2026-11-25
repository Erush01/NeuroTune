# MSD Audio CV Baselines Design

**Status:** Approved for implementation

## Context

The completed three-fold artist-grouped MSD audio-only experiment reports mean Recall@10 of 0.0133 across nine fold/seed runs. This is low in absolute terms, and no directly measured baseline currently shows how much the learned encoder adds over chance or simple audio similarity. The existing CV test folds and labels have already been evaluated, so this baseline comparison is post hoc and must remain descriptive; it must not be used to tune the model or make significance claims.

## Goals

- Measure a deterministic uniform-random ranking reference on the exact same three fold galleries, queries, and held-out Last.fm edges.
- Measure a simple no-training audio similarity reference using the existing fold-specific, train-normalized temporal caches.
- Keep the completed model report, checkpoints, and traces unchanged.
- Save reproducible baseline metrics and provenance in a separate artifact.

## Baselines

### Uniform random ranking

For each fold and each existing seed (684, 685, 686), independently generate a deterministic uniform ordered sample of 10 distinct gallery IDs for every query, excluding the query itself. Sampling 10 without replacement is distribution-equivalent to taking the first 10 entries of a full random permutation, while avoiding storage and shuffling of millions of unused candidates. Evaluate with the existing Recall@10, HitRate@10, MRR@10, and NDCG@10 implementation. This yields nine fold/seed measurements and a measured chance reference, rather than relying only on a theoretical estimate.

### Mean-pooled audio cosine

For each fold, use that fold's held-out sequences from its own normalization-specific cache. Average the continuous channels 0–24 over the 1,024 temporal positions, L2-normalize each resulting vector, and rank the remaining gallery tracks by cosine similarity. Exclude event-count channels 25–27 because they are intentionally left unnormalized and their raw count scale would distort this simple vector baseline. This baseline has no learned parameters and yields one measurement per fold.

## Evaluation and data flow

Use only the existing outer-fold test IDs and within-fold positive edges from the approved CV audit. Construct relevance with the same helper used by the learned-model runner and evaluate each query against its complete held-out fold gallery. Read the existing CV report as a reference and copy its fold/seed metrics into the baseline report for side-by-side descriptive comparison; validate its audit/source/cache hashes and never modify it. Do not change the split, derive negatives from cross-fold edges, or use the baseline results for model selection.

Add a standalone evaluator, `scripts/evaluate_msd_audio_cv_baselines.py`, that validates the ready audit, Last.fm source hash, model-report provenance, and fold-cache audit/split/normalization provenance before evaluating. It writes `results/msd_audio_only_cv_baselines.json` and does not alter any existing CV artifacts. The report records per-fold metrics, random-seed metrics, fold-level descriptive means, the model reference metrics/hash, overall summaries, source/code/cache hashes, runtime versions, and the post-hoc exploratory caveat.

## Validation

- Unit-test random-ranking reproducibility, query self-exclusion, and unique in-gallery top-10 samples.
- Test mean-pooling dimensions, normalization, cosine ranking, and deterministic tie handling.
- Test exact fold/cache ID alignment and that relevance uses only each held-out fold's internal edges.
- Reject stale audit/cache/Last.fm provenance and output collisions before writing.
- Use synthetic end-to-end tests to verify report metrics and hashes without touching the completed model artifacts.
- Run the real evaluator once, confirm the existing model report/checkpoints/caches remain unchanged, then run the full test suite, compilation, and dependency checks.

## Interpretation limits

The model's nine fold/seed observations, random baseline's nine observations, and cosine baseline's three fold observations have different repetition structure; report fold-level comparisons explicitly and do not treat runs as independent statistical samples. The comparison is exploratory, uses sparse Last.fm behavioral links, and treats unobserved links as non-relevant for ranking even though they are not verified dissimilarities.

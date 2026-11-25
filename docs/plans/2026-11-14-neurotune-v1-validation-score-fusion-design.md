# NeuroTune v1 Validation-Only Score-Fusion Ablation

**Status:** Approved, implemented, and executed on 2026-11-14.  
**Scope:** A small validation-only post-hoc comparison using the existing 6,967-track A/B pilot's three saved C checkpoints and its C-matched cosine baseline. It does not retrain models or use test data.

## Motivation

The A/B pilot already expanded the complete-modality catalog to 6,967 tracks and trained variants A–D. Variant C (audio+BPM+genre) had higher test Recall@10 and HitRate@10 than the equal-modality cosine baseline, but lower MRR@10 and NDCG@10. The latest official-10K CV also showed higher aggregate C means, but used reused Last.fm train labels and has different model/baseline observation counts. Another model change should first be checked without tuning on any test set.

The A/B pilot source archives were a third-party Hugging Face upload whose provenance and license remain unverified. This analysis is exploratory and does not make the A/B data suitable for redistribution or production use.

## Proposed analysis

Use the existing `C_seed_{684,685,686}.pt` selected checkpoints from `results/neurotune_v1_ab/checkpoints/` and the existing artist-disjoint A/B split. Fit audio, BPM, lyric, and genre transforms on the 5,632 training IDs only; construct feature rows only for the training and validation IDs. The validation gallery contains 610 tracks and 185 observed positive edges (238 queries). The three saved C checkpoints are not retrained.

For each seed, compute validation embeddings from its saved C checkpoint and the fixed equal-modality cosine embeddings using audio+BPM+genre. Form a symmetric blended cosine score matrix:

```text
score(λ) = λ × cosine(C embeddings) + (1 − λ) × cosine(baseline embeddings)
```

Evaluate the same full non-self validation gallery at top-10 for `λ ∈ {0, 0.25, 0.5, 0.75, 1}`. Use one shared λ across all seeds; select it by the mean validation NDCG@10 across the three seeds, breaking ties in favor of the smaller λ (the simpler, more baseline-heavy blend). Report Recall@10, HitRate@10, MRR@10, NDCG@10, query counts, and positive-edge counts for each seed/weight and their means.

## Leakage and artifact boundaries

- Do not use the test partition: no test IDs are passed to feature/model/baseline construction, no test-edge file is opened, and no test metrics are accessed or copied into the new report. The existing experiment report is read only for its C validation metrics and checkpoint hashes. Restrict feature outputs to train plus validation IDs.
- Do not change the A/B experiment report, curves, triplet schedules, checkpoints, split, or test cohort.
- Write a separate report under `results/neurotune_v1_validation_score_fusion/` with source/code/cache/split/checkpoint hashes, fitted-transform provenance, fixed λ grid, and an explicit post-hoc validation caveat.
- Verify that the `λ=0` score-matrix metrics reproduce the existing retrieval evaluator applied to the same C-matched baseline embeddings, and that `λ=1` reproduces each selected C checkpoint's recorded validation metrics. Do not compare `λ=0` with the saved A/B report's baseline: that report's baseline uses all four modalities, including lyrics, whereas C and this fusion use only audio+BPM+genre. Fail rather than silently accepting a mismatch.

## Interpretation and alternatives

This is a development experiment, not an unbiased estimate: the same validation set was already used to select each C checkpoint, and it will also select the fusion weight. A favorable weight is only a candidate for later testing on a new artist-held-out cohort. The test set remains untouched.

A retrained regularization or learning-rate sweep was deferred because repeated tuning on this same validation cohort would increase selection bias. A supervision-quality study and licensed data expansion may have greater long-term value, but they are separate workstreams.

## Completed result

The fixed run completed successfully and wrote `results/neurotune_v1_validation_score_fusion/experiment.json`. It produced the expected three-seed × five-weight validation grid for 610 gallery tracks, 185 positive links, and 238 queries. The C-matched cosine-only baseline (λ=0) scored Recall@10 0.188415, HitRate@10 0.247899, MRR@10 0.098513, and NDCG@10 0.108865. C alone (λ=1) scored 0.235683, 0.292717, 0.102130, and 0.126446, respectively; λ=1 was selected by validation NDCG. No blended weight improved NDCG over C alone in this run.

These values are not independent performance evidence: the same validation set already selected the C checkpoints and then selected λ. The A/B test set was not used by this analysis, but it has been inspected in earlier work; a new artist-held-out cohort is still required for an unbiased assessment.

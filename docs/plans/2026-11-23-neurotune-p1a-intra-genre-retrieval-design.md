# NeuroTune P1-A: Intra-genre retrieval design

**Status:** Approved for implementation on 23 November 2026.  
**Scope:** Validation-only analysis; no training and no independent-test access.

## Objective

Test whether the frozen D4 representation ranks known similar tracks within a genre differently from D0, when genre cannot itself separate the candidates. This is a changed retrieval task, so results must not be presented as a direct improvement over full-gallery retrieval.

## Data and candidate policy

- Use the existing three ABCD folds and seeds 684, 685, and 686, with the saved D0 and D4 checkpoints. No checkpoint selection or training is repeated.
- Use only each fold's validation IDs, validation features/embeddings, and validation positives sourced from the training labels. The existing validated loader may read test-source files and partition IDs solely for input/partition integrity checks; no test content may enter feature builders, query/gallery construction, relevance, metrics, or training, and none is passed beyond that validation boundary.
- The current catalog assigns one raw string genre per track. Match genres by exact, non-empty catalog label, not by the train-fitted one-hot vector: an unseen validation genre may have an all-zero one-hot vector but still has a valid catalog label.
- For query `q`, the gallery is every other validation track with exactly the same usable genre label. Remove self. Restrict relevance to known validation positive edges whose two endpoints are both in this same-genre candidate set.
- A query is eligible for ranking metrics when its same-genre gallery contains at least one known positive. Queries with missing/unusable genre, no same-genre candidates, or no in-gallery positives are counted separately and excluded from metric averages. Eligible queries with fewer than 10 candidates remain in the primary analysis and are separately reported.

## Systems and metrics

Evaluate four representations over the identical per-query candidate IDs and relevance set:

1. Frozen D0 neural embedding.
2. D0-matched cosine baseline using the single-window D0 feature representation.
3. Frozen D4 neural embedding.
4. D4-matched cosine baseline using the D4 multi-window feature representation.

Report Recall@10, MRR@10, and NDCG@10. Metrics use only in-gallery positives: ranking depth is `min(10, gallery size)`, Recall's denominator is the number of relevant candidates in that gallery, and NDCG's ideal list is limited to the same depth. Report per-query results, fold/seed summaries, per-genre results, eligible-query and eligible-edge counts, gallery-size distribution, and the `<10` gallery subgroup.

Aggregation is explicit: macro-query metrics weight eligible queries equally; genre-macro metrics weight genres equally after averaging their eligible queries. Micro Recall@10 is total retrieved relevant candidates divided by total in-gallery relevant candidates. MRR and NDCG are reported as query/genre means rather than assigning them a nonstandard edge-weighted “micro” interpretation.

## Provenance, boundaries, and interpretation

- Reuse train-fitted transforms and current selected checkpoints; build validation features only after excluding held-out test IDs from feature inputs.
- Validate checkpoint, transform, schedule, fold/seed, and source provenance; snapshot protected D0/D4/input namespaces before and after the run. Publish to a fresh no-overwrite namespace: `results/neurotune_v1_p1a_intra_genre_retrieval/`.
- Atomic no-overwrite publication currently requires Linux `renameat2` with `RENAME_NOREPLACE`; the runner must fail before loading experiment data if that facility is unavailable.
- Record the existing D0 transform-code hash allowance for `src/features_neurotune_v1.py` exactly as in P0; require all other transform fields and source hashes to match. Require exact D4 transform identity.
- The diagnostic uses known validation positives only. It does not change the inherited negative-candidate exclusion policy used during prior training; official/legacy test-source positives remain outside this evaluation's labels and query/gallery construction. Test-source files may be read only by the existing loader's integrity checks; their contents are not used as P1-A targets or candidates.
- Interpret results as validation-only development evidence. Same-genre Last.fm positives may be incomplete; missing positive edges are not evidence of dissimilarity. Do not claim causal audio contribution, superiority, significance, or generalization.

## Outputs and validation

Expected outputs are `experiment.json`, `per_query_metrics.csv`, `per_fold_seed_genre_metrics.csv`, `gallery_coverage.csv`, and `analysis.md`. Synthetic tests will cover exact-genre gallery construction, self exclusion, cross-genre positive filtering, missing labels, zero in-genre positives, small galleries, metric denominators, identical candidate sets across four systems, and strict fold/test boundaries. Before production evaluation, run focused tests, the full existing suite, static checks, and CLI help; then validate row counts, finite metrics, paired candidate/label hashes, and protected-artifact hashes after the diagnostic.

**Repository note:** NeuroTune is not a Git repository; this design is recorded in place and cannot be committed here.

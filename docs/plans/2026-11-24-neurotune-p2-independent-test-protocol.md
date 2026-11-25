# NeuroTune P2 Independent-Test Protocol

**Status:** Optional future protocol, deferred for the current course assignment. The user said further source/licensing research is unnecessary; no independent test source or labels have been opened. See `2026-11-24-neurotune-p2-test-source-research.md` for the completed initial screen.

## Objective

Determine whether the frozen D4 neural retriever generalizes to unseen artists and provides a practically meaningful ranking improvement over its matched cosine baseline. This protocol is an evaluation gate, not a new training experiment.

## Frozen comparison

- **Primary system:** the already selected, frozen D4 neural checkpoint. Do not change weights, transforms, candidate rules, or hyperparameters after viewing test labels.
- **Primary comparator:** D4-matched cosine, using the same track features and exact same query/gallery candidate IDs as D4 neural.
- **Gallery:** full-catalog retrieval within the independent test cohort: each eligible test track is a query, and all eligible test tracks except that query are candidates. Apply no genre filter. Keep this exact candidate pool fixed for both systems.
- D0 neural and D0-matched cosine may be reported as secondary context; they do not replace the primary D4-versus-D4-cosine decision.
- The independent test is run once after data provenance, cohort, labels, evaluation code, hashes, and this decision rule are frozen.

## Data and leakage boundary

- Select a licensed cohort whose artists are disjoint from all development train and validation artists; also check track-ID overlap and label-source overlap.
- Prefer similarity judgments from a source independent of the training/validation relevance links. Define what constitutes a positive/relevance label and how incomplete or unjudged query–candidate pairs are handled before accessing test labels.
- Test tracks and labels must not affect feature normalization, IDF, vocabulary construction, checkpoint selection, negative sampling/mining, model or threshold selection, or any development decision. Do not use test positives to filter negative candidates.
- Freeze the query set, full-catalog candidate gallery, relevance construction, exclusions, and label-density/coverage reporting before the test run. Exclude the query track itself, but do not filter candidates by genre. Use identical queries, galleries, and labels for D4 neural and D4-matched cosine.
- Validate dataset provenance, consent/licensing, artist and track disjointness, schema, and hashes before inference. If any gate fails, do not compute test metrics.

## Primary metric and acceptance rule

- Primary metric: macro-query **NDCG@10**.
- Primary effect: `mean(NDCG@10_D4_neural - NDCG@10_D4_matched_cosine)` over eligible test queries.
- **Pass only if both conditions hold:** the absolute mean lift is at least **+0.01 NDCG@10**, and the lower bound of a paired **95% artist-cluster bootstrap confidence interval** is greater than zero.
- The selected +0.01 is an absolute NDCG@10 difference, not a relative percentage.
- Secondary descriptive metrics: Recall@10, MRR@10, and HitRate@10, plus eligible-query count, candidate/positive coverage, and label density. Predefine any genre/popularity/cold-start slices before unblinding; do not use secondary results to replace a failed primary criterion.
- Do not treat seeds or multiple queries from the same artist as independent test units. Report the number of independent query artists and all exclusions.

## Uncertainty calculation proposal

Use paired artist-cluster resampling: each bootstrap draw samples query-artist clusters with replacement, includes all eligible query tracks for each sampled artist, and computes D4 and cosine metrics on the same draw. Keep the predeclared test gallery fixed; the resulting interval is conditional on that gallery. Use 10,000 deterministic draws and a 95% percentile interval. Document that this interval captures query-artist sampling uncertainty, not variation across independently sampled galleries. If the intended deployment claim requires generalization across catalog galleries, add multiple independent artist-disjoint test cohorts before unblinding rather than overstating this interval.

Before labels are opened, confirm that the source contains enough independent query artists and judged relevance coverage for the +0.01 criterion to be informative. If coverage is too small or uncertain, report that the test cannot decide the criterion; do not relax the threshold after observing results.

## Interpretation and follow-up

- A pass supports a bounded claim about retrieval on this independent artist-disjoint cohort; it is not by itself a production launch approval.
- A fail, invalid source, or inconclusive interval means no superiority/generalization claim. Do not tune on this test; a changed model requires a new untouched test.
- After a valid pass, evaluate serving latency/throughput, memory, feature and index refresh, missing-modality behavior, monitoring/rollback, data rights, and user-facing relevance in a separate shadow or limited pilot.
- Until P2 passes and operational checks are completed, keep NeuroTune labeled as a research/development candidate, not a production-ready recommendation system.

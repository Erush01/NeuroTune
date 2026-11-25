# M1 Fixed B0 Baseline Design

## Goal

Produce one leakage-safe, handcrafted retrieval baseline on the current official
10K MSD subset. Treat results as exploratory: the M0 audit found 939
full-feature tracks, with 416/6/11 observed positive edges in the artist-disjoint
train/validation/test splits.

## Approaches considered

1. **Fixed B0 (selected):** equal, predeclared modality weights; no parameter
   tuning. This is the simplest meaningful classical baseline and avoids
   overfitting the six validation edges.
2. **Validation-tuned B1:** tune independent BPM, lyric, and genre weights on
   validation. Rejected for now because the validation graph has only six
   positive edges.

## Approved design

- Use the split IDs already generated in `results/data_audit.json` (seed 684).
  Fit preprocessing only from the 756 training tracks. Evaluate queries and
  candidates only in the 89-track test catalog; do not rank across partitions.
- BPM: standardize raw positive BPM with training mean and standard deviation.
- Lyrics: preserve the official 5,000-word IDs. Calculate smoothed IDF from
  training tracks only (`log((1+n)/(1+df)) + 1`), multiply count TF by IDF, and
  L2-normalize each song vector. Songs with no lyrics use a zero vector.
- Genre: one-hot encode categories observed in training. An unseen test genre
  maps to all zeros.
- Multiply each modality vector by `1/sqrt(3)`, concatenate BPM, lyrics, and
  genre, then L2-normalize the combined song vector. Rank the full test gallery
  by cosine similarity; exclude the query itself.
- Relevance consists only of observed Last.fm edges whose endpoints are both in
  the test split. Report Recall@10, HitRate@10, MRR@10, and NDCG@10, alongside
  query count and eligible positive count. Do not treat absent edges as
  negatives or present the small-sample results as conclusive.

## Scope and artifacts

Implement feature preparation, fixed-vector similarity, full-gallery top-K,
and evaluation output. Save generated metrics and preprocessing metadata under
`results/`. Do not add learned weights, tune on validation/test, begin neural
training, or claim significance. Keep the official split and current M0 audit
unchanged.

## Verification

Test train-only fitting, unseen-genre handling, normalized embeddings,
symmetry, self-exclusion, top-K ordering, and no cross-partition relevance.
Run the full Python test suite and verify the generated result reports exactly
which queries and observed test edges were evaluated.

## Decision status

Approved by the user on 2026-11-07. The workspace is not a Git repository, so
this design file cannot be committed here.

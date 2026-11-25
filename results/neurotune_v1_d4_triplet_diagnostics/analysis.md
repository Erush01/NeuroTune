# D0/D4 triplet and embedding diagnostics

## Scope and sampling

This is a descriptive train/validation diagnostic using the nine existing D0 and D4 checkpoints. No model was trained, no saved schedule was regenerated, and no independent-test features, embeddings, relevance labels, or metrics were created. The D0 and D4 rows are paired by fold, seed, selected triplet indices, and pairwise-cosine sample indices.

For each fold/seed, 200 rows were sampled without replacement from each of the 50 saved training-schedule epochs (10,000 training triplets total). This represents 10,000 of 4,799,200–4,926,600 saved training rows, depending on fold. The full validation schedule was evaluated: 1,622 rows in fold 0, 1,670 in fold 1, and 1,662 in fold 2. Pairwise embedding cosine used 10,000 deterministically sampled track pairs per split and fold/seed. Sampling seeds and index hashes are recorded in `experiment.json` and the CSV files.

The distance is `1 − dot(normalized embeddings)`, with margin `m=0.2` and loss `max(0, d_ap − d_an + 0.2)`. `active_rate` is the fraction with positive loss; `easy`, `semi-hard`, and `hard` correspond to `gap >= 0.2`, `0 < gap < 0.2`, and `gap <= 0`, where `gap = d_an − d_ap`. Schedule negatives are called `sampled_negative`; unknown song pairs are not confirmed negatives. The summary values below are unweighted means across the nine fold-seed observations per model and split.

## Triplet summaries

| Split | Model | Active | Easy | Semi-hard | Hard | Mean d(a,p) | Mean d(a,n) | Mean gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train sample | D0 | 11.43% | 88.57% | 7.24% | 4.19% | 0.1853 | 0.9007 | 0.7155 |
| Train sample | D4 | 12.38% | 87.62% | 7.89% | 4.50% | 0.1768 | 0.8913 | 0.7145 |
| Validation | D0 | 31.21% | 68.79% | 11.83% | 19.39% | 0.4407 | 0.8842 | 0.4435 |
| Validation | D4 | 28.23% | 71.77% | 11.59% | 16.64% | 0.3703 | 0.8701 | 0.4998 |

Foldwise paired mean differences (D4 − D0; rates are percentage-point differences):

| Split | Active rate, folds 0 / 1 / 2 | Semi-hard rate, folds 0 / 1 / 2 | Hard rate, folds 0 / 1 / 2 | Mean gap, folds 0 / 1 / 2 |
|---|---|---|---|---|
| Train sample | −2.11 / +2.22 / +2.75 pp | −1.42 / +1.46 / +1.90 pp | −0.69 / +0.76 / +0.85 pp | +0.0504 / −0.0308 / −0.0224 |
| Validation | −2.75 / −2.83 / −3.37 pp | −0.99 / +0.12 / +0.16 pp | −1.77 / −2.95 / −3.53 pp | +0.0491 / +0.0417 / +0.0780 |

The D4 training-sample active rate is slightly higher on average, but its foldwise differences change sign. On validation, D4 has a lower active rate in each fold, driven mainly by fewer hard triplets; semi-hard rates are close. D4's mean validation gap is larger in all folds, while the training gap is similar on average and varies by fold. These comparisons describe distances to sampled negatives under the inherited schedule; they do not establish that those pairs are true negatives or that D4 is superior.

## Pairwise embedding cosine

The table gives the mean of the nine fold-seed median cosine values, with the range across those nine medians:

| Split | D0 median cosine | D4 median cosine |
|---|---:|---:|
| Train sample | 0.0666 (−0.0237 to 0.2470) | 0.0711 (−0.0211 to 0.2636) |
| Validation | 0.0782 (−0.0211 to 0.2550) | 0.0858 (−0.0202 to 0.2543) |

These sampled pairwise-cosine summaries alone do not indicate a strong global embedding collapse, but they are not a complete representation-quality assessment. Full quantiles and histograms are in `distance_distributions.csv`.

## Transform provenance and limitations

D0's frozen checkpoint transform manifest matches its recorded reference. The current D0 transform differs only in `code_sha256.src/features_neurotune_v1.py`, extended for D4; all other transform values and source hashes match. D4's current transform hash remains exact. The old/current D0 code hashes are recorded per fold/seed, and frozen/current transform hashes are recorded for both models in `experiment.json` and `triplet_summary_by_fold_seed.csv`.

Saved D0 schedules are reused as-is. Official/legacy test-source positives may remain exclusion-only in those schedules under the inherited policy; this diagnostic does not rebuild schedules, create test targets, or change that policy. All 70 protected input/reference/code hashes matched before and after the run, including the D4 physical schedule copies. The eight existing result namespaces protected by P0-A also still match P0-A's pre-run hashes after both diagnostics. No significance or superiority claim is made; the validation checkpoint-selection data are reused for diagnostics, and no independent test was run.

## Execution

Command: `.venv/bin/python scripts/run_neurotune_v1_d4_triplet_diagnostics.py`  
Exit status: 0; elapsed wall time: 2 min 15.67 s; peak resident memory: 3,296,516 KiB.  
Outputs: `triplet_summary_by_fold_seed.csv` (36 rows), `distance_distributions.csv` (4,176 rows), and `experiment.json`.

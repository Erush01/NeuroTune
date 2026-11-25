# D4 modality sensitivity

## Scope and protocol

This is a validation-only, read-only D0/D4 sensitivity analysis using the nine existing checkpoints (three folds × seeds 684, 685, and 686). No model was trained, and no test features, embeddings, relevance labels, or metrics were created. Each checkpoint was evaluated unmasked and with one input modality zeroed; for D4 audio, only normalized feature values were zeroed while window-validity and padding masks remained unchanged.

The output contains 45 conditions per model and 36 paired masked-modality comparisons. D0 and D4 use the same validation queries, positive edges, and paired fold/seed rows. The seed-level rows are in `paired_d0_d4_comparison.csv`; `condition_records.csv` contains the full condition metrics and provenance.

## Results

`delta_ndcg` is the unmasked NDCG@10 minus the masked NDCG@10. The paired difference below is `D4_delta_ndcg − D0_delta_ndcg`; a positive value means a larger NDCG drop under zero masking for D4, not better retrieval performance or causal modality importance.

Per-pair relative changes are reported separately in `paired_d0_d4_comparison.csv` as `100 × (D4_delta_ndcg − D0_delta_ndcg) / abs(D0_delta_ndcg)`. They are undefined when the D0 drop is zero and are stored as null/empty in that case.

| Zeroed input | D0 mean drop | D4 mean drop | Mean paired difference | Positive paired shifts |
|---|---:|---:|---:|---:|
| Audio | 0.007963 | 0.011883 | +0.003920 | 6/9 |
| BPM | 0.001268 | 0.001874 | +0.000606 | 6/9 |
| Genre | 0.048561 | 0.051610 | +0.003049 | 7/9 |
| Lyrics | 0.018911 | 0.011150 | −0.007761 | 1/9 |

Paired mean differences by fold (each fold averages its three seeds):

| Zeroed input | Fold 0 | Fold 1 | Fold 2 |
|---|---:|---:|---:|
| Audio | +0.006556 | −0.000726 | +0.005929 |
| BPM | +0.001442 | +0.000713 | −0.000337 |
| Genre | +0.004078 | +0.001659 | +0.003410 |
| Lyrics | −0.012256 | −0.005063 | −0.005964 |

The observed zero-mask sensitivity is higher for D4 on average for audio, BPM, and genre, and lower for lyrics. The paired changes are not uniformly positive across folds or seeds for audio and BPM. Zeroing an input is a distribution-shifted intervention, so these results describe sensitivity to this specific mask and do not establish causal modality importance, model superiority, or generalization. No significance tests were performed; the nine fold-seed observations are not treated as nine independent test sets.

## Integrity and limitations

- The 9 D4 unmasked validation results passed parity checks against the saved experiment records; all metrics are finite.
- The nine physical D4 schedule copies matched the D0 schedule bytes, array fields, and ID order. The copied schedule hashes are recorded with each condition row.
- All eight protected existing result namespaces were unchanged before and after this diagnostic.
- D0's frozen checkpoint transform manifest matches its reference. The current D0 transform differs only in `code_sha256.src/features_neurotune_v1.py`, whose source was extended for D4; all other transform fields and source hashes match. D4's current transform hash matches its frozen reference exactly. This narrow D0 provenance allowance is recorded per fold in `experiment.json`; unmasked metric parity remains strict.
- The frozen D0/D4 models retain the inherited negative-candidate exclusion policy, under which official/legacy test-source positives may be exclusion-only. This analysis created no test targets and did not change that policy.

## Execution

Command: `.venv/bin/python scripts/evaluate_neurotune_v1_d4_modality_sensitivity.py`  
Exit status: 0; elapsed wall time: 3 min 38.91 s; peak resident memory: 3,515,032 KiB.

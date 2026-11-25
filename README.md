# NeuroTune

NeuroTune is a course project exploring audio-first music similarity retrieval. The primary development candidate is v1 variant C (MSD audio features+BPM+genre); variants A, B, and D remain ablation comparisons. C is still exploratory, not an independently validated production model. Earlier M1/M2, N0, and MSD-audio-only runs remain separate exploratory experiments. Last.fm similar-track links provide supervision labels. MSD analysis features are not raw audio files. Input acquisition, formats, and access/license notes are documented in [`data/README.md`](data/README.md); source datasets are not redistributed here.

## NeuroTune v1 multimodal A/B pilot

This is the implemented multimodal architecture for the local A/B pilot. It combines MSD temporal analysis features with BPM, musiXmatch lyric counts, and tagtraum genre. The 6,967-track catalog has variable-length sequences from the first 30 seconds with 26 channels (12 pitch, 12 timbre, loudness, and clipped segment duration). The A/B MSD archives are a third-party Hugging Face upload; the user confirmed project-use permission, while public source provenance and checksum metadata have not been independently verified. These results must not be generalized to the full MSD.

The split is artist-disjoint. The final test gallery is defined by artists represented by official Last.fm test sources; remaining train-source artists are split into train and validation. Cross-partition links are excluded:

| Partition | Tracks | Positive links | Retrieval queries |
| --- | ---: | ---: | ---: |
| Train | 5,632 | 12,916 | — |
| Validation | 610 | 185 | 238 |
| Test | 725 | 215 | 265 |

All input normalization, TF-IDF IDF weights, and the genre vocabulary are fit on train tracks only. The shared encoder applies a masked 26→32→64→64 audio CNN with masked mean pooling, then combines it with BPM (1→8), lyrics (5,000→32), and genre (G→16) branches through a 120→128→64 fusion network. Its output is a normalized 64-D embedding trained with cosine triplet-margin loss (margin 0.2). The four fixed ablations are A: audio; B: audio+BPM; C: audio+BPM+genre; D: audio+BPM+genre+lyrics. Each directed positive gets one uniformly random train-only negative per epoch, excluding the anchor, positive, and anchor-known train positives. Missing Last.fm links are not verified negatives.

Run the fixed CPU experiment from the project root after the A/B audit, split, cache, and processed modalities are present:

```bash
.venv/bin/python -m scripts.run_neurotune_v1
```

The production runner is fixed to variants A–D, seeds 684/685/686, 50 epochs, batch size 64, and Adam at 0.001. It selects checkpoints by validation NDCG@10, breaking ties by lower validation triplet loss and then earlier epoch. Test labels and embeddings are constructed only after all 12 checkpoints are saved. Outputs are isolated under `results/neurotune_v1_ab/`: `experiment.json`, `training_curves.csv`, one compact sampled-triplet schedule per seed, and 12 selected checkpoints. The report records the split, source/code/cache/schedule/checkpoint hashes, runtime versions, and caveats.

The held-out test results below use the same 265 queries and 215 observed edges for every row. Model values are mean ± sample standard deviation across the three fixed seeds; the deterministic cosine baseline has one value, not a seed spread.

| Method | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Equal-modality cosine baseline | 0.1803 | 0.2528 | 0.1105 | 0.1120 |
| A: audio | 0.0520 ± 0.0292 | 0.0780 ± 0.0424 | 0.0256 ± 0.0158 | 0.0271 ± 0.0154 |
| B: audio+BPM | 0.0413 ± 0.0060 | 0.0667 ± 0.0095 | 0.0206 ± 0.0050 | 0.0220 ± 0.0028 |
| C: audio+BPM+genre | 0.1887 ± 0.0321 | 0.2629 ± 0.0293 | 0.0912 ± 0.0144 | 0.1031 ± 0.0200 |
| D: full multimodal | 0.1874 ± 0.0053 | 0.2553 ± 0.0157 | 0.0887 ± 0.0136 | 0.1013 ± 0.0084 |

On this one cohort, C has the highest mean Recall@10 and HitRate@10. The fixed cosine baseline has higher MRR@10 and NDCG@10, suggesting it tends to rank relevant items higher even though C retrieves slightly more relevant items within the top 10. Adding lyrics in D did not improve these descriptive metrics over C. This is not evidence of statistical significance or generalization: there is one artist-held-out test cohort, only 215 observed edges, and random unobserved negatives can contain latent positives. The A/B source provenance is also unverified.

**Execution note:** The first fixed-protocol attempt completed training and computed test metrics but failed while serializing training curves because the trainer emitted a redundant `MRR` alias. Staged outputs were rolled back; those unpersisted metrics were not used for tuning. After removing the alias and moving curve serialization before test evaluation, the unchanged protocol was rerun successfully. The second attempt took about 6 hours on the current CPU environment (peak RSS about 2.3 GiB).

### Validation-only C/cosine score fusion

This post-hoc development check reuses the three saved C checkpoints and the same A/B validation partition (610 tracks, 185 positive links, 238 queries); it performs no training and uses no test features or labels. It compares cosine score matrices from C embeddings and a **C-matched** equal-modality baseline over audio+BPM+genre. The saved A/B experiment's original baseline includes lyrics and is not this comparator.

The fused score is `λ × C cosine + (1 − λ) × C-matched cosine`, using one global `λ` across seeds. The selected weight maximizes mean validation NDCG@10 over seeds, with ties favoring the smaller model weight:

| Model weight λ | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 (cosine only) | 0.1884 | 0.2479 | 0.0985 | 0.1089 |
| 0.25 | 0.1824 | 0.2395 | 0.1019 | 0.1109 |
| 0.50 | 0.1935 | 0.2507 | 0.0987 | 0.1128 |
| 0.75 | 0.2052 | 0.2619 | 0.0966 | 0.1145 |
| 1.00 (C only) | 0.2357 | 0.2927 | 0.1021 | 0.1264 |

The selected value was `λ=1.0`: on this validation set, adding cosine scores did not improve NDCG@10 over C alone. This is not independent evidence because the same validation set had already selected each C checkpoint and was then used to choose λ. Do not treat it as a generalization claim or tune against the previously inspected A/B test set.

Reproduce the analysis with:

```bash
.venv/bin/python -m scripts.evaluate_neurotune_v1_validation_fusion
```

The separate report is `results/neurotune_v1_validation_score_fusion/experiment.json`; it records the fixed grid, validation metrics, source/code/checkpoint hashes, and caveats without test metrics.

## A–D shard expansion: cache and fresh artist-grouped CV

The A–D audit combines the downloaded A/B/C/D shards and applies the existing artist/song and official duplicate rules globally before joining BPM, musiXmatch lyrics, tagtraum genre, and Last.fm links. Compared with the A/B pilot, it adds **6,855** complete-modality candidate track IDs, increasing the catalog from 6,967 to **13,822** (about 1.98×); no A/B candidate IDs were removed. All 13,822 candidates passed first-30-second temporal preprocessing: the separate raw cache contains **1,430,546** finite float32 segment rows across the 26 channels, with zero rejected tracks. It is unnormalized so any later model can fit normalization only on its training partition.

| Split labels | A/B | A–D | Added | Removed from A/B |
| --- | ---: | ---: | ---: | ---: |
| Last.fm train links (after train/test overlap removal) | 17,536 | 67,657 | 50,121 | 0 |
| Last.fm test links | 1,940 | 8,033 | 6,093 | 0 |

The A–D train links comprise 16,305 new-new, 33,816 new-existing, and 17,536 existing-existing links. The test links comprise 2,060 new-new, 4,033 new-existing, and 1,940 existing-existing links. The audit removed 1,204 train/test-overlapping undirected links from the train set. The cache is `data/msd_hf_abcd_pilot/abcd_audio_30s_26ch.h5`, with metadata and track-level rejections alongside it. All existing A/B audit and cache artifacts remain unchanged.

The fresh CV split is under `data/msd_hf_abcd_pilot/cv/`: `manifest.json`, `known_positive_edges.csv`, and four source-specific edge files per fold. Test artists are drawn only from artist groups absent from the entire A/B catalog; the old A/B catalog artists, including its former test cohort, are assigned to train or validation instead. Each fold holds out 588 new-only artists, then assigns 10% of remaining artists to validation with seed 684. Train/validation targets use only Last.fm train-source edges. Test labels preserve train-source and official-test-source edges in separate files. All 1,940 old A/B test-source edges are excluded from every label target and retained only in the separate known-positive exclusion set.

| Fold | Train/val/test tracks | Train-source train/val/test edges | Official test-source test edges | Cross-partition train/test edges excluded | Official test-source train/val edges excluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 11,640 / 1,377 / 805 | 49,266 / 811 / 183 | 50 | 17,397 / 1,899 | 4,144 |
| 1 | 11,693 / 1,335 / 794 | 48,523 / 835 / 221 | 30 | 18,078 / 1,559 | 4,504 |
| 2 | 11,669 / 1,381 / 772 | 47,992 / 831 / 235 | 45 | 18,599 / 1,893 | 4,155 |

Each fold partitions all 13,822 cache-valid tracks; no invalid-endpoint edges were found. The three folds reuse the same Last.fm source pools and are a data/split robustness setup, not an independent external test. Missing Last.fm links remain unverified negatives, and the previous A/B test-source links were already inspected. The C-only cross-validation run and its limitations are summarized below. The downloaded third-party shard files were authorized for project use by the user; public license and checksum provenance remain unverified.

### C-only three-fold evaluation

The fixed experiment trains variant C (audio+BPM+genre; no lyrics) on each fold with seeds 684/685/686, 50 epochs, batch size 64, Adam at 0.001, and cosine triplet-margin loss with margin 0.2. Each fold fits transforms on its training tracks only, uses train-source edges for training and validation, and selects checkpoints by validation NDCG@10, then validation loss and earlier epoch. Uniform random negatives exclude every known positive within the candidate pool. All nine checkpoints, nine schedules, and 450 curve rows were saved before held-out test embeddings or relevance were constructed.

| Fold | Train/validation/test tracks | Train-source test edges / queries | Official-test-source edges / queries | Selected epochs (seeds 684/685/686) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 11,640 / 1,377 / 805 | 183 / 260 | 50 / 73 | 24 / 12 / 25 |
| 1 | 11,693 / 1,335 / 794 | 221 / 299 | 30 / 50 | 22 / 4 / 16 |
| 2 | 11,669 / 1,381 / 772 | 235 / 273 | 45 / 69 | 4 / 4 / 5 |

The table reports mean ± population standard deviation. C has nine fold-seed observations; the deterministic C-matched cosine baseline has three fold-level observations. The two edge sources are reported independently:

| Held-out label source | Method | Observations | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Last.fm train archive | C | 9 | 0.1681 ± 0.0274 | 0.2232 ± 0.0347 | 0.0769 ± 0.0186 | 0.0895 ± 0.0178 |
| Last.fm train archive | C-matched cosine | 3 | 0.1510 ± 0.0248 | 0.2022 ± 0.0309 | 0.0682 ± 0.0089 | 0.0787 ± 0.0126 |
| Official Last.fm test archive | C | 9 | 0.1521 ± 0.0556 | 0.1802 ± 0.0699 | 0.0560 ± 0.0252 | 0.0760 ± 0.0292 |
| Official Last.fm test archive | C-matched cosine | 3 | 0.1649 ± 0.0716 | 0.1970 ± 0.0854 | 0.0801 ± 0.0449 | 0.0920 ± 0.0446 |

For fold-level context, the C seed-mean NDCG@10 versus cosine NDCG@10 was 0.0868 vs 0.0921, 0.0753 vs 0.0618, and 0.1065 vs 0.0821 on train-archive labels for folds 0–2; on official-test-archive labels it was 0.0746 vs 0.1532, 0.1078 vs 0.0747, and 0.0458 vs 0.0481. These are descriptive comparisons only: C's aggregate is higher on all four metrics for the held-out train-source edges, while cosine is higher on all four for the official test-source edges. The nine model observations and three baseline observations are not independent or equally sized comparison units, the folds reuse the train-label pool, and the official-test-source cohort has only 30–50 links per fold. Neither result establishes generalization or model superiority; unobserved Last.fm links are not verified negatives.

The report and artifacts are isolated under `results/neurotune_v1_abcd_cv/`: `experiment.json`, `training_curves.csv`, `triplet_schedules/`, and `checkpoints/`. It records source/code/cache/manifest/transform/schedule/checkpoint hashes, per-source counts, selected epochs, and the same caveats. The fixed CPU run took about 5 hours 3 minutes.

Rebuild the isolated A–D audit, raw cache, and split in order with:

```bash
.venv/bin/python -m scripts.audit_msd_hf_abcd
.venv/bin/python -m scripts.preprocess_msd_hf_abcd
.venv/bin/python -m scripts.audit_neurotune_v1_abcd_cv
```

These commands refuse existing output targets; run them only when the relevant A–D audit, cache, and `cv/` targets are absent.

Run the separate C-only cross-validation experiment after those artifacts are present:

```bash
.venv/bin/python -m scripts.run_neurotune_v1_abcd_cv
```

The runner refuses an existing `results/neurotune_v1_abcd_cv/` namespace and preserves all previous experiment outputs.

### Full-modal P0 ablation (validation only)

The follow-up P0 run adds the musiXmatch lyric branch without changing the completed C namespace. All **13,822** A–D catalog tracks had an MXM document, with zero empty documents. A, B, and D were newly trained; the existing C checkpoints were used as read-only validation references. The variants use audio; audio+BPM; audio+BPM+genre (C); and audio+BPM+genre+lyrics (D), respectively. Training and cosine baselines use only the existing train/validation partitions. Official and legacy test-source links are retained only in the known-positive negative-exclusion provenance; no test features, embeddings, relevance, or metrics were produced.

The fixed protocol uses three artist-grouped folds, seeds 684/685/686, 50 epochs, batch size 64, Adam at 0.001, and triplet margin 0.2. Checkpoints are selected on validation NDCG@10. Each model mean covers nine fold/seed observations; each matched cosine mean covers three fold observations. Tables show means; the JSON report includes population standard deviations and fold/seed detail.

| Model | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| A — audio | 0.048795 | 0.088177 | 0.034365 | 0.030221 |
| B — audio+BPM | 0.043948 | 0.078437 | 0.028857 | 0.026365 |
| C — audio+BPM+genre (read-only) | 0.118594 | 0.193172 | 0.069910 | 0.068638 |
| D — audio+BPM+genre+lyrics | 0.119240 | 0.194019 | 0.072913 | 0.071530 |

| Matched cosine baseline | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| A — audio | 0.041842 | 0.075427 | 0.031908 | 0.027089 |
| B — audio+BPM | 0.041812 | 0.080982 | 0.035022 | 0.028011 |
| C — audio+BPM+genre | 0.109255 | 0.190992 | 0.078063 | 0.068781 |
| D — audio+BPM+genre+lyrics | 0.121607 | 0.204624 | 0.087984 | 0.077921 |

D is slightly ahead of C in the learned-model validation means, but the matched D cosine baseline is higher on all four metrics. C is above its cosine baseline on Recall@10 and HitRate@10, and below it on MRR@10 and NDCG@10. These are development results only: validation NDCG also selects checkpoints, the observation counts differ, Last.fm links are incomplete, and unobserved pairs are not verified negatives. Do not treat this ablation as evidence of generalization or model superiority. The previously inspected test cohort remains deferred until a new independent artist-held-out cohort exists.

Artifacts are isolated under `results/neurotune_v1_full_modal_ablation/` (`experiment.json`, 1,800 curve rows, nine copied schedules, and 27 new checkpoints). Reproduce from the project root with:

```bash
.venv/bin/python -m scripts.run_neurotune_v1_full_modal_ablation
```

The runner refuses an existing output namespace; run it only when that target is absent. Final verification for this work: 423 unit tests passed, `compileall` passed, and `pip check` reported no broken requirements.

### D1 — Semi-hard negative mining (validation-only development)

D1 independently trains variant D from its own initialization on the existing three folds and seeds; it does not fine-tune D0 checkpoints. The model, optimizer, margin, epoch budget, and validation-only checkpoint selection remain fixed. The first three epochs reuse the verified D0 uniform schedules. Starting before epoch 4, train-track embeddings are refreshed every five epochs. Each later epoch targets a 75% uniform / 25% semi-hard attempt mix using the strict cosine-distance band; an empty semi-hard pool falls back to uniform sampling, so the realized semi-hard share can be lower.

Mining candidates and their embeddings are restricted to the current fold's training tracks. To isolate the sampler change from D0, D1 preserves D0's known-positive exclusion policy: known-positive edges from train, validation, and official/legacy test sources may exclude candidate negatives within the train pool, but are never training targets. The inherited official/legacy test-source edges are exclusion-only; no test-track features, embeddings, relevance, or metrics are built. Labels from any future independent test cohort must not enter training or mining. This is a validation-development comparison, not an independent test or generalization result. The pilot is only a resource check and must not be used to retune the fixed protocol.

The frozen values and deterministic RNG/miner identifiers are in `config/neurotune_v1_d1_semi_hard.json`; the runner reads and hashes this file. Run from the project root after the D0 full-modal artifacts and required data inputs are present:

```bash
.venv/bin/python -m scripts.run_neurotune_v1_training_improvements --pilot
.venv/bin/python -m scripts.run_neurotune_v1_training_improvements
```

The first command runs fold 0 with seeds 684/685/686 and writes only to `results/neurotune_v1_training_improvements_pilot/`. After inspecting pilot validity and resource use, the second runs folds 0/1/2 with the same three seeds into the separate `results/neurotune_v1_training_improvements/` namespace. Existing targets are refused; pilot outputs are not copied into or merged with full-run outputs. Each namespace contains `experiment.json`, `d1_config.json`, `training_curves.csv`, `validation_metrics.csv`, `mining_diagnostics.csv`, compressed epoch triplet schedules with per-triplet sampling-type codes, and selected checkpoints. The report pairs D1 with its D0 validation record per fold/seed and the matching D cosine validation baseline; these are development observations, not independent test sets.

The fixed D1 pilot and full run are complete. Across the nine fold-seed validation pairs, the unweighted means were:

| Model | Observations | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 model | 9 | 0.1192 | 0.1940 | 0.0729 | 0.0715 |
| D1 semi-hard | 9 | 0.1230 | 0.1948 | 0.0712 | 0.0722 |
| D-matched cosine | 3 folds | 0.1216 | 0.2046 | 0.0880 | 0.0779 |

Mean paired NDCG@10 differences (three seeds per fold) were `D1 − D0` **+0.0029 / −0.0018 / +0.0009** and `D1 − D-matched cosine` **+0.0064 / −0.0108 / −0.0129** for folds 0/1/2. D1 therefore did not show a consistent NDCG gain over D0 and was below the matched cosine baseline in two folds; these results do not establish that semi-hard mining is superior. The three-fold/seed observations are development data, not independent samples, and checkpoint selection uses validation metrics.

The full run completed in 4:56:03 with a peak RSS of 2.57 GiB. All 450 epoch timings were recorded (median 39.2 s; range 35.8–42.6 s). There were 20 uniform fallbacks across the nine runs; the overall realized semi-hard fraction was about 23.5% because the first three epochs were uniform warm-up. Input, code, split, D0 artifact, pilot, checkpoint, and schedule hashes were verified unchanged or consistent after the run. Detailed fold-seed metrics and provenance are in `results/neurotune_v1_training_improvements/experiment.json` and the accompanying CSV/NPZ files. No independent test was run.

### D2 — Modality dropout (validation-only development)

D2 independently trains variant D from scratch on the same three folds and seeds, with the D0 optimizer, margin, epoch budget, uniform triplet schedules, and validation checkpoint selection unchanged. The D0 schedules are reused byte-for-byte and their hashes are checked against the D0 report. D2 does not resume D1 or D0 checkpoints. It preserves D0's known-positive exclusion policy, including official/legacy test-source positives used only to exclude candidate negatives; no test features, embeddings, relevance, or metrics are constructed.

During training only, each BPM, genre, or lyrics branch embedding is independently zeroed per track after the branch activation and immediately before fusion, with probabilities 0.10, 0.25, and 0.20, respectively. There is no inverse-probability scaling; audio and its padding mask are unchanged. Validation and inference use all branches. A dedicated CPU RNG derives each stream as `seed * 1000003 + fold * 1009 + epoch_index_zero_based`, independently of triplet-order randomness.

Run from the project root when the D0 full-modal artifacts and data inputs are present:

```bash
.venv/bin/python -m scripts.run_neurotune_v1_modality_dropout
```

The runner reads the frozen values from `config/neurotune_v1_d2_modality_dropout.json`, refuses an existing `results/neurotune_v1_d2_modality_dropout/` target, and writes the config, nine selected checkpoints, nine copied schedules, 450 training-curve rows, 450 dropout-diagnostic rows, and `experiment.json` with provenance and hashes.

The fixed 3-fold × 3-seed run completed in 4:51:55.5. Mean validation metrics were:

| Model | Observations | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 model | 9 | 0.1192 | 0.1940 | 0.0729 | 0.0715 |
| D2 modality dropout | 9 | 0.1410 | 0.2263 | 0.0820 | 0.0822 |
| D-matched cosine baseline | 3 folds | 0.1216 | 0.2046 | 0.0880 | 0.0779 |

Three-seed mean NDCG@10 by fold was:

| Fold | D2 | D0 model | D-matched cosine |
| ---: | ---: | ---: | ---: |
| 0 | 0.0969 | 0.0890 | 0.0854 |
| 1 | 0.0709 | 0.0614 | 0.0703 |
| 2 | 0.0788 | 0.0643 | 0.0780 |

The observed D2 validation means for NDCG@10 and Recall@10 were above D0 and the fold-level cosine baseline in all three folds; MRR@10 remained below cosine in folds 1 and 2. These are descriptive development results, not evidence of model superiority or generalization: validation metrics also select checkpoints, the folds are not independent samples, and no independent test was run. The realized dropout rates were 0.099991 (BPM), 0.250032 (genre), and 0.199956 (lyrics), each over 131,202,900 track-role masks. Detailed fold/seed values and provenance are in `results/neurotune_v1_d2_modality_dropout/experiment.json`, `training_curves.csv`, and `modality_dropout_diagnostics.csv`.

### D3 — Semi-hard mining + modality dropout (validation-only development)

D3 trains variant D from a fresh initialization for every fold/seed, using the same trainer seeds (684/685/686) as D0–D2; no D0, D1, or D2 weights are reused. The first three epochs use the verified D0 uniform schedules. Later epochs use the unchanged D1 semi-hard provider (25% target, five-epoch embedding refresh, uniform fallback) while the D2 branch dropout is active only during training (BPM 0.10, genre 0.25, lyrics 0.20, audio 0.00; no inverse-probability scaling). Mining embeddings are computed in eval/no-grad mode, and validation is unmasked. D0's known-positive exclusion policy is retained, including official/legacy test-source positives used only to exclude candidate negatives; no test features, embeddings, relevance, or metrics are constructed.

Run from the project root when the D0, D1, and D2 references and data inputs are present:

```bash
.venv/bin/python -m scripts.run_neurotune_v1_d3_combined
```

The runner reads the frozen protocol from `config/neurotune_v1_d3_combined.json`, validates the D0/D1/D2 reference reports and artifacts, refuses an existing `results/neurotune_v1_d3_combined/` target, and publishes the results transactionally. The output includes nine selected checkpoints and epoch schedules with sampling-type codes, 450 training-curve rows, 450 mining-diagnostic rows, 450 dropout-diagnostic rows, paired validation metrics, and provenance hashes.

The fixed 3-fold × 3-seed run completed in 8:03:57.0. Mean validation metrics over nine fold-seed runs were:

| Model | Observations | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 model | 9 | 0.1192 | 0.1940 | 0.0729 | 0.0715 |
| D1 semi-hard | 9 | 0.1230 | 0.1948 | 0.0712 | 0.0722 |
| D2 modality dropout | 9 | 0.1410 | 0.2263 | 0.0820 | 0.0822 |
| D3 combined | 9 | 0.1403 | 0.2227 | 0.0812 | 0.0822 |
| D-matched cosine baseline | 3 folds | 0.1216 | 0.2046 | 0.0880 | 0.0779 |

Three-seed mean NDCG@10 by fold was:

| Fold | D3 | D0 | D1 | D2 | D-matched cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0983 | 0.0890 | 0.0919 | 0.0969 | 0.0854 |
| 1 | 0.0709 | 0.0614 | 0.0595 | 0.0709 | 0.0703 |
| 2 | 0.0774 | 0.0643 | 0.0652 | 0.0788 | 0.0780 |

D3's mean NDCG@10 was 0.082233 versus 0.082199 for D2, a negligible difference; D3's mean Recall@10, HitRate@10, and MRR@10 were lower than D2's. By fold, D3 NDCG was higher than D2 in fold 0, equal in fold 1, and lower in fold 2. Six of the nine D3 checkpoints selected epochs 1–3, before semi-hard mining began at epoch 4. Five matched the corresponding D2 selected weights: the three fold-1 selections (epochs 2, 1, and 2), plus fold 0/seed 686 and fold 2/seed 686 (both epoch 3). For fold 2/seed 685, D3 selected epoch 1 while D2 selected epoch 29, so the selected checkpoints differ. D3 and D2 share the same seed, dropout stream, and uniform schedule during warm-up. These results do not show a consistent benefit from combining the two strategies.

The realized semi-hard share was 23.5001% of 43,734,300 triplets, as expected with three uniform warm-up epochs; there were five uniform fallbacks. Realized dropout rates were 0.099991 (BPM), 0.250032 (genre), and 0.199956 (lyrics), each over 131,202,900 track-role masks. Median epoch time was 46.35 s (range 37.46–154.25 s). Detailed metrics and provenance are in `results/neurotune_v1_d3_combined/experiment.json` and its CSV/NPZ artifacts. This remains a validation-only development comparison: validation selects checkpoints, folds are not independent samples, and no independent test was run; do not interpret it as evidence of superiority or generalization.

### D4 — Multi-window audio (validation-only development)

D4 isolates the audio-window representation against D0; it does not combine D1 semi-hard mining or D2/D3 dropout. Each fold/seed starts from a fresh model initialization, uses the original D0 seed and byte-identical uniform schedule, and keeps the existing 50-epoch Adam protocol and validation checkpoint selection. The three 30-second windows start at `0`, `max(0, (duration - 30) / 2)`, and `max(0, duration - 30)` with half-open intervals `[start, start + 30)`. Overlap is allowed. Tracks shorter than 30 seconds reuse their opening sequence in each view; empty windows are masked, and a track with no valid window is rejected. The unchanged D0 CNN/projection architecture is trained from scratch, with the same D4 weights shared across the three views; raw projected 64D vectors are averaged before the unchanged D fusion and final L2 normalization.

The additive cache is `data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch.h5` with metadata and rejection sidecars; it does not replace the D0 first-window cache or rewrite D0–D3 outputs. It contains 13,822 tracks and 4,003,002 float32 feature rows (315.18 MiB); no track was rejected. The start/middle/end windows were valid for 13,822/13,813/13,813 tracks, respectively; 13,804 tracks had all three windows and 18 had two. Fold-local audio statistics are fitted on training tracks only. Overlapping rows sharing `(track_id, source_segment_index)` count once for the statistics (using their mean), while original per-window rows remain model inputs. The three folds had 3,352,109 / 3,369,552 / 3,364,986 unique training source-segment rows.

From the project root, with the A–D archives, catalog audit, and catalog present, build the separate cache and run the fixed pilot/full protocol:

```bash
.venv/bin/python -m scripts.preprocess_neurotune_v1_d4_audio_windows
.venv/bin/python -m scripts.run_neurotune_v1_d4_multi_window_audio --pilot
.venv/bin/python -m scripts.run_neurotune_v1_d4_multi_window_audio
```

The cache command requires the A–D archives, audit, and catalog. The runner additionally requires the existing D0 report/checkpoints/schedules, original cache, folds, and processed MXM inputs. The preprocessor and runner refuse existing targets; preserve any existing cache/result namespace rather than overwriting it. The pilot is fixed to fold 0/seed 684 and is not used for tuning; the full run covers folds 0/1/2 and seeds 684/685/686. The full artifacts are under `results/neurotune_v1_d4_multi_window_audio/`: `experiment.json`, 450 training-curve rows, nine checkpoints, and nine schedules copied byte-for-byte from D0. The runner records source, code, cache, split, transform, reference, and output hashes. Cache extraction took 5:11.40 with 1,096,616 KiB peak RSS; the full run took 15:58:25 wall time with 2,959,092 KiB peak RSS.

Mean validation metrics were:

| Method | Observations | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 model | 9 fold-seed runs | 0.1192 | 0.1940 | 0.0729 | 0.0715 |
| D4 model | 9 fold-seed runs | 0.1278 | 0.2054 | 0.0769 | 0.0754 |
| D0-matched cosine | 3 folds | 0.1216 | 0.2046 | 0.0880 | 0.0779 |
| D4-matched cosine | 3 folds | 0.1290 | 0.2191 | 0.0890 | 0.0806 |

Three-seed mean NDCG@10 by fold was:

| Fold | D4 model | D0 model | D4-matched cosine | D0-matched cosine | D4 − D0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0937 | 0.0890 | 0.0892 | 0.0854 | +0.0048 |
| 1 | 0.0618 | 0.0614 | 0.0725 | 0.0703 | +0.0005 |
| 2 | 0.0707 | 0.0643 | 0.0802 | 0.0780 | +0.0064 |

The D4 model's mean validation metrics exceeded D0's in this run, and 8 of 9 paired NDCG@10 observations were higher; the remaining seed-level difference was −0.0019 in fold 1. However, the D4 model means were below the D4-matched cosine means on all four metrics, and the model's fold-mean NDCG@10 was below D4 cosine in folds 1 and 2. Model means use nine fold-seed observations while cosine means use three fold observations, so these are descriptive comparisons, not inferential evidence. Validation is also used for checkpoint selection, and folds are not independent samples. D4 therefore does not establish model superiority or generalization. The experiment is validation-only: no independent-test features, embeddings, relevance, or metrics were created. D0's inherited known-positive exclusion policy remains; official/legacy test-source positives can only exclude candidate negatives and are not targets or evaluation evidence. Full fold/seed results and provenance are in `results/neurotune_v1_d4_multi_window_audio/experiment.json`.

## Official 10K artist-grouped CV robustness check

This separate CV run uses the 939-track complete-modality intersection from the official MSD 10K subset, not the third-party A/B audio archives. Its audio input is the raw first 30 seconds in a ragged 26-channel sequence (12 pitch, 12 timbre, loudness, and clipped segment duration); BPM also comes from MSD, while lyrics and genre come from musiXmatch and tagtraum CD2C. Official Last.fm **train** links provide labels; the official Last.fm test archive contributes no links to this catalog.

Three artist-grouped outer folds hold out 323, 310, and 306 tracks. Of the 600 catalog links, 220 are within-fold test links and 380 cross-fold links are excluded from every test fold. In each fold, the other two folds form one shared non-test track pool; its positive edges are divided into disjoint train and validation labels. Validation is edge-held-out, so a track may occur in both sets, but every known non-test positive—including validation links—is excluded from negative sampling.

| Held-out fold | Gallery tracks | Train links | Validation links | Test links | Test queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 323 | 232 | 57 | 63 | 84 |
| 1 | 310 | 210 | 52 | 78 | 97 |
| 2 | 306 | 216 | 53 | 79 | 107 |

The experiment trains variant C (audio+BPM+genre) with Adam at 0.001, margin 0.2, batch size 64, 50 epochs, and seeds 684/685/686. Transforms are fitted on each fold's non-test IDs only; checkpoints are selected by validation NDCG@10, then lower validation triplet loss, then earlier epoch. All nine checkpoints, sampled schedules, and the training-curve file are saved before held-out features or relevance are constructed. The reference is the equal-modality cosine baseline using the same three C modalities and the same fold galleries.

| Method | Observations | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C model | 9 fold-seed runs | 0.2618 | 0.3270 | 0.1202 | 0.1440 |
| C-matched cosine | 3 folds | 0.2322 | 0.2910 | 0.1106 | 0.1307 |

The unweighted C means across the nine fold-seed observations exceed the cosine means across three fold observations on all four metrics in this run. Fold-mean NDCG@10 was 0.1629/0.1178/0.1513 for C versus 0.1585/0.1147/0.1189 for cosine in folds 0/1/2. These are descriptive comparisons, not inferential evidence: the observation units differ, folds reuse the same label source, and links held out in one fold can be training or validation labels in another. The selected epochs ranged from 1 to 49. The 939-track catalog remains small, and missing links are not verified negatives; this is a robustness check, not an independent external test or evidence of generalization.

After creating the separate audit and raw cache, run:

```bash
.venv/bin/python -m scripts.run_neurotune_v1_official10k_cv
```

The report and artifacts are isolated under `results/neurotune_v1_official10k_cv/`: `experiment.json`, 450 training-curve rows, nine fold/seed schedules, and nine selected checkpoints. Source, code, cache, split, schedule, and checkpoint hashes are recorded there. The completed run took about 214 seconds in the current CPU environment.

## M1 temporal audio preprocessing

M1 prepares the 939-track common catalog from the official 10K MSD subset for the temporal audio model. Each track is resampled over its duration to 1,024 time steps with 28 channels: 12 pitch, 12 timbre, one loudness, and beat/bar/section event counts. Continuous channels are standardized using statistics fit on training tracks only; event counts are left unchanged. The original scalar `msd_tracks.csv` remains unchanged.

Run from the project root after extracting the MSD 10K HDF5 subset into `data/msd_subset_raw/MillionSongSubset/`:

```bash
.venv/bin/python -m scripts.preprocess_audio
```

The command reads the artist-disjoint split in `results/data_audit.json` and writes `data/processed/audio_sequences.h5` plus `data/processed/audio_sequences_meta.json`. The cache stores sorted track IDs, float32 sequences, channel names, split provenance, train-only normalization statistics, and source hashes. `--input-root`, `--audit`, `--output`, `--metadata-output`, and `--grid-size` override the defaults. This step prepares the temporal audio input for the M2 audio-first prototype below.

The current artist-disjoint split contains 756 train, 94 validation, and 89 test tracks. Its graph has only 416 train, 6 validation, and 11 test positive Last.fm edges, so evaluation remains exploratory and does not support strong performance or significance claims. Missing Last.fm links are not verified negatives.

## M2 exploratory audio-first Siamese prototype

M2 uses the M1 temporal cache as its primary input, with BPM, musiXmatch lyric-count features, and tagtraum genre as auxiliary modalities. Its shared encoder applies a small 1D-CNN to the 1,024-step audio sequence, pools to 16 ordered temporal bins, fuses the audio representation with the three auxiliary branches, and produces a normalized 64-dimensional embedding. The audio cache and auxiliary transforms must use training IDs only; the runner checks the cache split and normalization provenance against the audit. B0 and N0 remain metadata-only comparison baselines.

Run from the project root after M1 preprocessing and the processed feature/Last.fm inputs are available:

```bash
.venv/bin/python -m scripts.run_m2_audio
```

The fixed exploratory run uses seeds 684/685/686, 50 epochs, batch size 64, Adam at 0.001, margin 1.0, and one sampled negative per positive. It records validation loss but does not use the six-edge validation graph for checkpoint selection. All seed checkpoints are fixed before test representations and relevance are created. Outputs are `results/m2_audio_experiment.json`, `results/m2_audio_training_curves.csv`, `results/m2_audio_sampled_pairs.csv`, and `results/checkpoints/m2_audio_seed_{684,685,686}.pt`. Path overrides are `--data-dir`, `--audit`, `--audio-cache`, `--audio-metadata`, `--output`, and `--checkpoint-dir`; `--epochs` and `--seeds` support smoke runs, with actual settings recorded in the report.

**Caution:** M2 is exploratory on the current 939-track 10K subset. Its 6-edge validation and 11-edge test graphs are too small for strong performance, significance, or generalization claims. Missing Last.fm links are not verified negatives. Results are compared with B0/N0 descriptively only.

## MSD-audio-only exploratory Siamese experiment

This separate experiment uses only temporal MSD features as model inputs. It retains Last.fm similar-track links as positive supervision and test relevance labels; unobserved pairs sampled as negatives are not verified dissimilarities. It does not read BPM, musiXmatch, or tagtraum features. The 32-dimensional Siamese encoder processes the same 1,024-step, 28-channel temporal sequences as M1, with an audio-only 1D CNN and train-only normalization.

Unlike M1/M2, the audio-only audit does not require complete auxiliary feature coverage. It uses all 10,000 MSD subset HDF5 tracks after the existing duplicate rules, yielding 9,939 tracks. The seed-684 artist-disjoint split contains 7,935 train, 971 validation, and 1,033 test tracks, with 5,385, 94, and 121 within-split Last.fm links. It is a new split and is not a directly controlled comparison with M2's 939-track catalog.

Run the separate audit, cache, and training steps from the project root:

```bash
.venv/bin/python -m scripts.audit_msd_audio
.venv/bin/python -m scripts.preprocess_audio \
  --input-root data/msd_subset_raw/MillionSongSubset \
  --audit results/msd_audio_only_audit.json \
  --output data/processed/msd_audio_only_sequences.h5 \
  --metadata-output data/processed/msd_audio_only_sequences_meta.json
.venv/bin/python -m scripts.run_msd_audio_only
```

The audit reads `data/processed/msd_tracks.csv`, `data/processed/lastfm_similar/`, and `downloads/msd_duplicates.txt`, and writes `results/msd_audio_only_audit.json`. The cache is separate from the existing M1 cache. The fixed training run uses seeds 684/685/686, 50 epochs, batch size 64, Adam at 0.001, margin 1.0, and one uniformly random unobserved negative per positive within each partition. Validation pairs are fixed per seed and only recorded; they do not select checkpoints. All checkpoints are fixed before test audio embeddings or relevance labels are created.

Outputs are `results/msd_audio_only_experiment.json`, `results/msd_audio_only_training_curves.csv`, `results/msd_audio_only_sampled_pairs.csv`, and `results/checkpoints/msd_audio_only_seed_{684,685,686}.pt`. The report and checkpoints record split/normalization provenance, source/code hashes, runtime versions, and the sampled-pair trace hash. The audit accepts `--tracks`, `--lastfm`, `--duplicates`, `--output`, and `--seed`; preprocessing accepts `--input-root`, `--audit`, `--output`, `--metadata-output`, and `--grid-size`; training accepts `--audit`, `--audio-cache`, `--audio-metadata`, `--lastfm`, `--output`, `--checkpoint-dir`, `--epochs`, and `--seeds`.

**Caution:** This is exploratory work on the MSD 10K subset and the available Last.fm subset. The 94 validation and 121 test links do not support strong performance, significance, or generalization claims. Missing Last.fm links are not verified negatives.

The completed three-seed run evaluated 175 queries in the 1,033-track test gallery with 121 observed test links. Mean metrics (sample standard deviation across seeds) were Recall@10 `0.0725` (`0.0183`), HitRate@10 `0.0914` (`0.0198`), MRR@10 `0.0329` (`0.0065`), and NDCG@10 `0.0387` (`0.0101`). Training loss fell from `0.202–0.214` to `0.043–0.046`, while validation loss was lowest at epochs 3–5 (`0.195–0.204`) and rose to `0.362–0.434` by epoch 50. The fixed final checkpoints were retained; no checkpoint was selected using the sparse validation graph. These figures are descriptive only and are not directly comparable with M2's smaller, different split.

## MSD-audio-only artist-grouped cross-validation

This separate experiment uses the 9,939 duplicate-filtered MSD subset tracks, grouped by artist into three deterministic folds (seed 684). Each fold is held out once as the complete retrieval gallery; the other two folds provide training tracks and within-fold Last.fm positive links. Cross-fold links are excluded. Fold sizes are 3,249, 3,321, and 3,369 tracks, with 1,065, 965, and 981 within-fold links; 5,198 cross-fold links are excluded. Each fold uses its own cache, normalized only with that fold's training tracks. The runner checks that cache metadata points to the current audit and that the Last.fm source hash matches the audit.

Build the audit and three fold-specific caches, then run the fixed experiment:

```bash
.venv/bin/python -m scripts.audit_msd_audio_cv
.venv/bin/python -m scripts.preprocess_audio_cv
.venv/bin/python -m scripts.run_msd_audio_cv
```

Training uses the same 32-dimensional audio-only Siamese encoder, 50 epochs, seeds 684/685/686, batch size 64, Adam at 0.001, margin 1.0, and one uniformly sampled unobserved negative per positive. There is no validation split or checkpoint selection. All nine checkpoints and the sampled-pair trace hash are fixed before held-out relevance or embeddings are constructed. Outputs are `results/msd_audio_only_cv_experiment.json`, `results/msd_audio_only_cv_training_curves.csv`, `results/msd_audio_only_cv_sampled_pairs.csv`, and `results/checkpoints/msd_audio_only_cv_fold_{0,1,2}_seed_{684,685,686}.pt`. The audit accepts `--tracks`, `--lastfm`, `--duplicates`, `--output`, `--seed`, and `--fold-count`; cache preprocessing accepts `--audit`, `--input-root`, `--output-dir`, and `--grid-size`; the runner accepts `--audit`, `--cache-dir`, `--metadata-dir`, `--lastfm`, `--output`, `--checkpoint-dir`, `--epochs`, and `--seeds`. Existing output targets are never overwritten; failed runs clean up outputs created by that attempt.

**Caution:** The fold galleries have different sizes and this protocol is not directly comparable to the prior 1,033-track single-split run. Last.fm links are sparse behavioral supervision, and unobserved pairs sampled as negatives are not verified dissimilarities. Treat all cross-validation metrics as exploratory, descriptive results—not as evidence of statistical significance or generalization.

The completed run evaluated all nine fixed fold/seed models. Fold means below are across the three seeds; the overall mean and sample standard deviation summarize all nine fold/seed measurements and are descriptive, not independent statistical estimates.

| Fold | Gallery | Queries | Positive links | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3,249 | 1,022 | 1,065 | 0.0100 | 0.0212 | 0.0069 | 0.0056 |
| 1 | 3,321 | 981 | 965 | 0.0125 | 0.0265 | 0.0083 | 0.0070 |
| 2 | 3,369 | 972 | 981 | 0.0173 | 0.0312 | 0.0098 | 0.0094 |
| All fold/seed runs | — | — | — | 0.0133 ± 0.0042 | 0.0263 ± 0.0058 | 0.0083 ± 0.0022 | 0.0073 ± 0.0022 |

Training loss ranged from `0.214–0.223` at epoch 1 to `0.034–0.040` at epoch 50. There is no validation split in this protocol, and no checkpoint selection is performed.

To compare this completed model against random-ranking and simple mean-pooled audio-cosine baselines without retraining or changing the CV artifacts, run:

```bash
.venv/bin/python -m scripts.evaluate_msd_audio_cv_baselines
```

The evaluator validates the current audit, Last.fm source, caches, and model report, then writes a separate `results/msd_audio_only_cv_baselines.json`. Random top-10 candidates are sampled independently per query without replacement for each fold and seed. The audio baseline averages normalized continuous channels 0–24 over time, excludes raw event channels, and ranks the whole held-out fold. This is a post-hoc exploratory comparison; it is not for tuning or significance claims.

On the existing three folds, the mean-pooled cosine baseline averaged Recall@10 `0.0394`, HitRate@10 `0.0657`, MRR@10 `0.0271`, and NDCG@10 `0.0243` across the three fold observations. The learned-model reference across nine fold/seed observations averaged `0.0133`, `0.0263`, `0.0083`, and `0.0073`, respectively; the random baseline across nine fold/seed observations averaged `0.0031`, `0.0058`, `0.0018`, and `0.0016`. Thus, for this representation and dataset, the simple audio cosine baseline outscored the learned model on all four aggregate metrics. These summaries have different observation units, are not inferential statistics, and come from a post-hoc comparison on already inspected test folds; treat them as a diagnostic signal to investigate, not proof that cosine generalizes better.

## Fixed B0 exploratory baseline

The source collection is the 10K Million Song Dataset (MSD) subset. After feature-coverage and duplicate filtering, the audit's full-feature catalog contains 939 tracks. B0 is a metadata-only comparison baseline, not the accepted audio-first model. The current baseline report ranks the 89-song held-out test gallery, excluding each query itself; it does not rank all 939 catalog tracks or the original 10K subset.

Create the project environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The B0 baseline combines standardized tempo, a train-fitted TF-IDF-style musiXmatch count vector, and a one-hot primary genre vector at fixed equal modality weights. BPM statistics, lyric inverse-document frequencies, and the genre vocabulary are fit using only the recorded training IDs. It evaluates only observed Last.fm links whose two endpoints are both in the test split.

Run from the repository root:

```bash
.venv/bin/python -m scripts.run_baseline
```

The command reads `results/data_audit.json` and processed inputs under `data/processed/`, then writes `results/baseline_b0.json`. `--data-dir`, `--audit`, `--output`, and `--k` can override those paths and the default K=10. The JSON records source SHA-256 hashes, split counts, preprocessing fit metadata, metric values, and the seed.

**Interpretation warning for the currently generated report:** its 89-song test gallery contains only 11 observed positive edges. These results are exploratory and do not support a statistical-significance claim. The CLI derives this edge count from the run's measured test relevance, so reports for other splits show their own count. Missing Last.fm links are not verified negatives. Full official Last.fm test edges are not used as relevance labels.

## Fixed N0 exploratory Siamese experiment

N0 is an exploratory metadata-only Siamese prototype; it does not implement the accepted temporal audio encoder. It trains the shared-weight 64-dimensional encoder using only training-internal observed links and sampled unobserved candidates. It fits BPM, lyrics, and genre transforms on training IDs only, records validation loss without selecting a checkpoint, then evaluates the fixed final epoch on the held-out test gallery. Test edges are used only for final evaluation. The standard run uses 50 epochs, seeds 684/685/686, batch size 64, Adam at 0.001, margin 1.0, and K=10. It samples one negative per positive: `floor(total_negatives / 2)` same-genre negatives and the remainder uniformly random, so odd totals allocate the extra negative to the random pool.

Run from the repository root:

```bash
.venv/bin/python -m scripts.run_n0
```

The run writes `results/n0_experiment.json`, `results/n0_training_curves.csv`, `results/n0_sampled_pairs.csv` (the actual per-epoch train pairs and fixed validation pairs), and one final state-dict checkpoint per seed under `results/checkpoints/`. The report and checkpoints record Python, NumPy, and PyTorch versions plus SHA-256 hashes of the runner and training/evaluation code and requirements. `--epochs`, `--seeds`, `--data-dir`, `--audit`, `--output`, and `--checkpoint-dir` support deterministic smoke runs and path overrides; actual settings are recorded in the JSON report. It reads only the processed MSD, musiXmatch, tagtraum, and current `lastfm_similar/` subset sources—not official Last.fm train/test archives. PyTorch CPU is already pinned in `requirements.txt`.

**Caution:** the current split has only 6 validation and 11 test positive edges. N0 is exploratory only; results are highly unstable, missing links are not verified negatives, and neither validation nor test results are used for model selection or hyperparameter tuning. Reported across-seed mean and sample spread are descriptive only and make no confidence or significance claims. Across the current three 50-epoch runs, training loss falls to 0.015–0.019 while validation loss rises to 0.597–0.799, consistent with overfitting; see the saved curves. N0 is not expected to improve on B0.

Run the full test suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

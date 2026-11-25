# NeuroTune Full-Modal Validation Ablation Design

**Date:** 2026-11-17  
**Status:** Implemented and run; validation-only P0 complete; final verification passed  
**Scope:** Extend the A–D artist-grouped CV path with musiXmatch lyrics and run a validation-only A/B/C/D ablation. The existing A–D test cohort is not evaluated in this work unit.

## Decision and context

The approved next-steps plan calls for a full audio+BPM+genre+lyrics model and a fair A/B/C/D ablation. The user chose to **defer all evaluation of the previously inspected A–D test cohort until a new independent test cohort exists**. Therefore this phase may train models and compare them on train/validation data, but it must not construct test embeddings/relevance or report any test metrics.

Code inspection establishes that `NeuroTuneV1Encoder` already implements all four variants and that the D lyrics branch already exists. The fusion widths are variant-specific:

| Variant | Active inputs | Fusion input |
| --- | --- | ---: |
| A | audio | no fusion layer; 64-D audio embedding |
| B | audio + BPM | 72-D (`64 + 8`) |
| C | audio + BPM + genre | 88-D (`64 + 8 + 16`) |
| D | audio + BPM + genre + lyrics | 120-D (`64 + 8 + 16 + 32`) |

Thus `120→128→64` describes D, not C. The current C checkpoint/report namespace must remain unchanged. The implementation work is primarily a lyrics-enabled A–D fold feature path and an isolated validation-only runner, not a new encoder architecture.

## Data and lyrics contract

The A–D catalog builder defines eligible IDs as the intersection of positive-tempo, duplicate-resolved audio IDs, MXM IDs, and tagtraum genre IDs. The P0 preflight reparsed the official MXM files, joined by normalized MSD `track_id`, and confirmed all 13,822 candidate records are present with zero empty documents.

- Do not use title/artist fuzzy matching.
- Keep raw word counts sparse and limit parsing to catalog IDs where practical; the train-fitted 5,000-term float32 matrix for 13,017 development tracks is 260,340,000 bytes (about 248 MiB), so process one fold at a time. A read-only fold-0 preflight measured 881.29 MiB peak RSS across loader, lyric parsing, and feature construction, with 10,213 MiB available.
- Fit document frequency/IDF on each fold's train IDs only. Transform validation documents with those fitted values.
- A catalog ID with no MXM record is a preflight error; do not silently create a zero vector or random value. A present record with no nonzero counts is an explicit empty document, represented by an all-zero vector and counted in the report.
- Preserve raw MXM file hashes, parsed catalog-lyrics provenance, and parser/transform code hashes.

## Ablation protocol

Use the existing three artist-grouped folds, seeds `684/685/686`, 50 epochs, batch size 64, Adam at `0.001`, cosine triplet-margin loss with margin `0.2`, and validation-NDCG checkpoint selection with the existing loss/earlier-epoch tie breaks. Fit audio, BPM, genre, and lyrics transformations on train IDs only. Train/validation targets come only from their existing Last.fm train-source edge sets. The full known-positive set may be used only to exclude known links from negative sampling; official test-source links are never training or validation positives.

The existing C model is frozen and used as a read-only reference only if preflight confirms identical fold IDs, seeds/settings, schedule hashes, and active audio/BPM/genre transforms. Its checkpoint metadata/validation selection metrics may be read; its test metrics must not be imported into the new report. Train A, B, and D in the new namespace on the same train/validation schedules. If the C parity check fails, stop before training and ask rather than silently substituting a different C model or split.

Compute a train-fitted, equal-modality cosine validation reference for each variant using only that variant's active modalities. These baseline and model metrics are development evidence on the validation data, not unbiased performance estimates: validation NDCG is also used for checkpoint selection.

## Test-set boundary

The runner must not construct test feature views, test embeddings, test relevance maps, or test metrics. The existing A–D held-out cohort has already been inspected and remains reserved from this ablation. A later P1 must create and precommit a new independent artist-held-out test cohort before final evaluation or performance claims. The old C results and all old experiment outputs remain read-only.

## Outputs and provenance

Write to the new namespace `results/neurotune_v1_full_modal_ablation/`, refusing any pre-existing target. Stage outputs and publish with no-overwrite protection and rollback. Expected outputs:

- `experiment.json`: protocol, validation-only per-variant/fold/seed metrics, per-variant matched-cosine metrics, counts, selected epochs, source/code/split/transform/schedule/checkpoint hashes, runtime, and explicit `test_evaluation_deferred` status.
- `training_curves.csv`: A/B/D curves plus a clearly identified read-only C reference where available.
- `triplet_schedules/`: nine fold/seed schedules, copied or reproduced deterministically and hashed.
- `checkpoints/`: 27 newly trained A/B/D checkpoints. Existing C checkpoints are referenced by path/hash and are not copied or modified.

## Verification

Synthetic tests must cover the exact A/B/C/D fusion input widths, D finite normalized 64-D output and lyrics gradients, exact MXM track-ID coverage and empty-document policy, train-only IDF fitting, known-positive-safe schedules, train/validation isolation, C checkpoint/schedule/transform parity, matched-modality cosine baselines, absence of test features/labels/metrics in the runner, provenance, and collision/rollback safety.

Before the real validation run, verify the expected input/output paths and snapshot the frozen C outputs plus all prior result/cache artifacts. After the run, independently check all report hashes, schedule contents, model variants, selected epochs, validation-only result schema, and byte-for-byte preservation of the original C namespace. Run the full unit suite, `compileall`, and `pip check`.

## Completed P0 validation result

The fixed run wrote `results/neurotune_v1_full_modal_ablation/experiment.json`, `training_curves.csv` (1,800 rows), nine copied schedules, and 27 new A/B/D checkpoints. The report contains 36 model validation observations (nine per variant) and 12 matched-cosine observations (three per variant). C checkpoints, schedules, and curves were read-only references. Independent artifact verification matched 56 checkpoint/schedule/curve hashes and found the C reference files unchanged. The completion report says `status: complete`; the host did not provide a recoverable OS exit code.

| Variant | Model Recall@10 / HitRate@10 / MRR@10 / NDCG@10 (9 fold-seed observations) | Matched cosine (3 folds) |
| --- | --- | --- |
| A — audio | 0.048795 / 0.088177 / 0.034365 / 0.030221 | 0.041842 / 0.075427 / 0.031908 / 0.027089 |
| B — audio+BPM | 0.043948 / 0.078437 / 0.028857 / 0.026365 | 0.041812 / 0.080982 / 0.035022 / 0.028011 |
| C — audio+BPM+genre, read-only | 0.118594 / 0.193172 / 0.069910 / 0.068638 | 0.109255 / 0.190992 / 0.078063 / 0.068781 |
| D — audio+BPM+genre+lyrics | 0.119240 / 0.194019 / 0.072913 / 0.071530 | 0.121607 / 0.204624 / 0.087984 / 0.077921 |

D is slightly ahead of C on the learned-model validation means, while the matched D cosine baseline is higher than D on all four metrics. Checkpoint selection uses the same validation split, so these values are development evidence only. The current test cohort was not evaluated.

## Limitations

Validation results will be used for development and checkpoint selection and must not be presented as final performance estimates. Last.fm unobserved pairs remain unverified negatives. The A–D source archive's project use is user-authorized, but public provenance and licensing remain unverified. No current test cohort will be re-evaluated in this phase.

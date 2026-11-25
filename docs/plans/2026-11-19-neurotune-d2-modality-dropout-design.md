# NeuroTune D2: Modality-Dropout Design

**Date:** 19 November 2026  
**Status:** Approved design; implementation pending.  
**Scope:** A validation-only D2 development comparison for NeuroTune variant D.

## Context and goal

D1 is complete, but its NDCG@10 change versus D0 was small and inconsistent across folds. D2 tests a separate, pre-specified training change: drop auxiliary modality contributions during training so the full-modal D model does not depend excessively on one input. D2 is not a follow-up tuning pass on D1 and must not be described as independent-test or generalization evidence.

## Fixed experiment protocol

- Train variant D independently from initialization; do not fine-tune D0 or D1 checkpoints.
- Preserve the D0 folds, seeds (684, 685, 686), 50 epochs, Adam at 0.001, batch size 64, margin 0.2, uniform triplet schedules, and validation checkpoint-selection rule.
- Reuse the D0 fold/seed schedules byte-for-byte, verify their hashes, and write D2 outputs to a new fixed namespace: `results/neurotune_v1_d2_modality_dropout/`.
- Keep the comparison validation-only. Do not construct test features, embeddings, relevance, or metrics. Preserve the existing D0 known-positive exclusion policy; official/legacy test-source edges may only exclude train-pool negative candidates.

## Approved dropout behavior

The user selected whole-branch embedding dropout immediately before fusion. Compute each branch normally, including its projection and activation, then apply a per-example keep/drop mask to the resulting auxiliary embedding before concatenation:

| Branch | Drop probability |
| --- | ---: |
| Audio | 0.00 |
| BPM | 0.10 |
| Genre | 0.25 |
| Lyrics | 0.20 |

Dropped branch embeddings are set to zero with no inverse-probability scaling. The mask is sampled independently per track instance, including anchor, positive, and negative triplet inputs. Audio embeddings and the audio padding mask remain untouched. Applying the mask after each branch's activation ensures branch biases cannot leak a nonzero contribution when that modality is dropped.

Dropout masks are applied only during the training forward pass. Validation and inference receive no masks and use all modalities. Generate masks with a dedicated deterministic Torch RNG stream derived from fold, seed, and epoch and consumed in deterministic batch order; do not perturb triplet ordering or rely on process-global random state. Record the RNG derivation and realized per-branch drop counts/rates.

## Implementation boundary

- Add an optional, default-disabled dropout-mask path through `NeuroTuneV1Encoder.forward`, `forward_triplets`, and `train_neurotune_v1`. Existing D0/D1 calls without a mask must retain the same behavior and history schema.
- Create a separate D2 config and runner using the shared fold-local feature construction and the D0 uniform schedules. Do not route D2 through D1's semi-hard schedule provider or change D0/D1 frozen configs, results, or defaults.
- Store the fixed probabilities, placement, scaling behavior, train-only scope, RNG scheme, sources/splits/code/output hashes, validation metrics, selected epochs, and checkpoints in D2 provenance. Keep outputs isolated and refuse an existing namespace.

## Verification and interpretation

Unit tests must prove that default calls are unchanged; a dropped branch is exactly zero after its biased branch layers; audio remains untouched; masks are deterministic for a fixed fold/seed/epoch/batch sequence; train uses masks while validation/inference do not; and the configured drop rates are respected. A synthetic short-run smoke check precedes the fixed nine-condition training run. Compare D2 with its matched D0 validation records and D-matched cosine baseline across folds; report all fold/seed observations and do not claim improvement or generalization from these validation data.

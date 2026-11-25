# NeuroTune D4 — Multi-window Audio Design

**Date:** 22 November 2026  
**Status:** Design approved; implementation in progress (Tasks 1–7 complete; Task 8 preflight/pilot next).

## Goal and scope

D4 tests whether representing a song with start, middle, and end audio windows improves the existing D0 variant D. It is an isolated audio-input change: do not combine D4 with D1 semi-hard mining or D2 modality dropout. Reuse the existing 13,822-track artist-grouped cohort, fold manifests, labels, galleries, and validation protocol. Do not change the cohort to accommodate missing D4 audio; fail preflight and report the coverage problem instead.

The experiment remains validation-only. Preserve the inherited D0 known-positive exclusion policy, including official/legacy test-source links used only to exclude train-local negative candidates. Do not construct independent-test features, embeddings, relevance, targets, or metrics.

## Audio-window extraction

The current cache contains the first 30 seconds only, is ragged and unnormalized, and does not retain segment timestamps. D4 therefore requires a separate cache extracted from the original HDF5 sources; the existing cache and all D0–D3 artifacts remain unchanged.

For song duration `D` seconds, use three 30-second windows with starts:

```text
start   = 0
middle  = max(0, (D - 30) / 2)
end     = max(0, D - 30)
```

Select segments by `segments_start` in the half-open interval `[window_start, window_start + 30)`. Clip each segment's duration at the next segment start, the song duration, and the window end, following the existing feature contract. Preserve the original 26-channel representation and store nonnegative source segment indices and finite timestamps within each song's `[0,duration]` range so overlap and provenance are auditable. Windows may overlap naturally for songs shorter than 90 seconds. For songs shorter than 30 seconds, reuse the available start-window sequence for all three views.

If an individual window has no segments, mark it invalid and exclude its embedding from the window average. If a track has no valid window, fail preflight rather than dropping the track or changing a split. Preflight must establish that the exact 13,822-track ID set is available from the source archives and has at least one valid window per track.

## Model and feature preprocessing

Use the existing D0 temporal audio CNN and projection, with shared weights across all three windows. Encode each valid window as a 64D vector and average the valid, unnormalized projected vectors. Pass that mean directly into the unchanged BPM/genre/lyrics fusion; keep D0's existing L2 normalization on the final fused vector, with no new normalization before fusion. This keeps the audio branch at 64 dimensions and adds no learnable parameters. Mean pooling is order-invariant; concatenation plus projection would preserve window position but add parameters, while learned attention would add more complexity and capacity. Those alternatives are out of scope for this isolated comparison.

Fit audio normalization separately within each existing fold using training tracks only. Because windows can overlap and the clipped-duration channel may differ between window copies, group rows by `(track_id, source_segment_index)`, take their elementwise mean as one statistics-only representative, and count that segment once. Keep the actual per-window rows unchanged for model input, and keep all non-audio feature construction and transforms unchanged.

## Training and comparisons

Train a fresh D4 model for each existing fold/seed with the D0 protocol: folds 0/1/2, seeds 684/685/686 passed unchanged to the trainer, 50 epochs, batch size 64, Adam at 0.001, triplet margin 0.2, byte-identical D0 uniform schedules, and the existing validation NDCG@10 checkpoint selection. Do not resume D0, D1, D2, or D3 weights.

Compare D4 against D0 on exact fold/seed pairs. Also recompute a D4-matched cosine baseline using the same three-window audio representation and unchanged auxiliary modalities; do not reuse the first-window D0 cosine as though it were matched to D4. Report per-fold/per-seed metrics, selected epoch, window coverage, runtime, extraction cost, and provenance. Do not claim superiority or generalization from validation results.

## Data integrity and execution gates

- Keep the new multi-window cache, configuration, runner, and result namespace separate from existing artifacts; refuse existing targets.
- Record source archive, audit, extraction-code, cache, fold/split, train-only transform, schedule, checkpoint, and output hashes.
- Before training, verify source archive hashes, exact track-ID coverage, window counts, timestamps, segment clipping, valid-window masks, and the training-only normalization boundary.
- Test duration/window boundaries, overlap, short songs, empty windows, segment clipping, duplicate-segment normalization, masked aggregation, and fixed 64D output with synthetic fixtures.
- Run a real model/trainer smoke on a synthetic fixture, then a fixed one-fold/one-seed resource pilot without hyperparameter tuning. If the pilot and all provenance checks pass, run the full fixed 3-fold × 3-seed experiment in a new D4 namespace.
- Preserve all existing D0–D3 results and their validation-only/test-boundary limitations.

## Evidence inspected

- `src/data/preprocess_audio_window.py` — current `[0,30)` selection and segment-duration clipping.
- `scripts/preprocess_msd_hf_ab.py` and `scripts/audit_msd_hf_ab.py` — archive, catalog, and cache provenance.
- `src/features_neurotune_v1.py` — train-only audio normalization and feature provenance.
- `src/data/neurotune_v1_dataset.py` — ragged sequence and mask contract.
- `src/model_v1.py` — shared temporal audio encoder, masked pooling, 64D projection, and fusion.
- `README.md` — 13,822-track cohort, artist-grouped folds, and D0 protocol/results.

The NeuroTune workspace is not a Git repository, so this design document cannot be committed here.

# NeuroTune v1 Multimodal Model Implementation Plan

> **For implementer:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and evaluate the approved shared 64-D multimodal triplet encoder and its required same-split baseline/ablations on the A/B pilot catalog.

**Architecture:** Create a separate v1 path so M1/M2/N0 and audio-only experiments remain intact. Build a source-aware artist-disjoint manifest, fit every transform on training IDs only, consume the raw 26-channel ragged cache with masked batch padding, train four ablation variants from shared deterministic triplet schedules, and evaluate the official Last.fm test edges only after all checkpoints are selected.

**Tech Stack:** Python, PyTorch CPU, NumPy, h5py, existing MSD/MXM parsers and retrieval utilities, `unittest`.

---

## Fixed decisions

- Use only `data/msd_hf_pilot/` A/B outputs; no C–Z download and no edits to old M1/M2/N0/CV artifacts.
- Split seed 684. The artists represented by official Last.fm test source tracks form test; shuffle the other artist keys with seed 684 and assign 10% to validation. Current expected counts: train 5,632 tracks / 12,916 train edges; validation 610 / 185 train edges; test 725 / 215 official-test edges and 265 query tracks. Drop cross-partition links (4,435 train-source and 1,725 test-source edges).
- Each undirected positive contributes both anchor directions. For each directed train positive and epoch, sample one uniform negative from train IDs, excluding the anchor, positive, and that anchor's observed train positives. Do not use genre for sampling or mine semi-hard negatives. Missing links remain unverified negatives.
- Validation negatives are fixed/deterministic and drawn only within validation IDs. Use validation NDCG@10 for checkpoint selection; tie-break by lower validation triplet loss, then earliest epoch.
- Model: audio `26→32→64→64` Conv1D with `k=5/5/3`, stride 1, same-length padding, ReLU, mask-aware temporal mean, projection to 64; BPM `1→8`; lyrics `5000→32`; genre `G→16`; full fusion `120→128→64`; final L2 normalization.
- Audio z-score (all 26 channels), BPM z-score, TF-IDF IDF, and genre vocabulary are fit on training IDs only. Lyrics use the fixed 5,000-word MXM vocabulary; genre is one-hot with train-observed sorted categories.
- Triplet loss is `mean(relu(d(a,p)-d(a,n)+0.2))`, where `d=1-dot` on normalized embeddings. Training: Adam `0.001`, batch 64 triplets, 50 epochs, seeds 684/685/686.
- Variants A–D: audio; audio+BPM; audio+BPM+genre; full audio+BPM+genre+lyrics. Add one fixed equal-modality cosine baseline on the same split. CNN+GRU is out of scope.
- Evaluate Recall@10, NDCG@10, and MRR. Build test relevance/embeddings only after all 12 variant/seed checkpoints are fixed. Report test results descriptively; no significance/generalization claims.
- The workspace has no Git repository, so work in the current directory, do not create a worktree, and do not create commits.

## Task 1: Artist-disjoint split and edge manifest

**Files:** Create `src/data/neurotune_v1_split.py`, `scripts/build_neurotune_v1_split.py`, `tests/test_neurotune_v1_split.py`, and `tests/test_build_neurotune_v1_split.py`.

1. Write synthetic tests for test-source artist assignment, deterministic 10% validation artist selection, artist disjointness, endpoint-only query construction, and train/test edge provenance.
2. Run `.venv/bin/python -m unittest tests.test_neurotune_v1_split -v`; verify the new tests fail for the missing split builder.
3. Implement a pure builder that reads catalog track/artist metadata, obtains eligible test-source IDs from `lastfm_test.zip` archive paths, and uses `artist_id→artist_name→track_id` as the grouping key.
4. Convert the other sorted artist keys to a list, shuffle with `random.Random(684)`, select `int(0.1 * artist_count)` artists for validation, and assign the rest to train.
5. Place all tracks by test-source artists in test. Retain train CSV edges only when both endpoints are in train or both in validation; retain official test CSV edges only when both endpoints are in test. Record every cross-partition exclusion and assert no edge appears in more than one partition.
6. Add CLI tests for missing sources, source-ID parsing, deterministic output, and refusal to overwrite. Write the manifest and three partition edge CSVs under `data/msd_hf_pilot/`.
7. Run `.venv/bin/python -m unittest tests.test_neurotune_v1_split tests.test_build_neurotune_v1_split -v` and run the real split CLI. Confirm the expected 5,632/610/725 tracks and 12,916/185/215 edges before training.

## Task 2: Train-only multimodal transforms

**Files:** Create `src/features_neurotune_v1.py` and `tests/test_features_neurotune_v1.py`.

1. Add failing tests for audio channel statistics over train IDs only, zero-variance channels, BPM standardization, fixed 5,000-term smoothed IDF, per-track lyric L2 normalization, sorted train-only genre vocabulary, and unseen validation/test genres mapping to zero.
2. Run `.venv/bin/python -m unittest tests.test_features_neurotune_v1 -v` and verify expected failures.
3. Implement audio-stat fitting by streaming each train sequence from `ab_audio_30s_26ch.h5`; calculate per-channel population mean/std over train segment rows and substitute 1 for zero std. Apply only to actual rows; leave padding for the collator.
4. Reuse `src/data/parse_mxm.py:parse_mxm` for MXM parsing. Implement the same B0 transform contract in the isolated v1 module instead of calling `build_train_fitted_modalities`, because that helper creates an unused full dense B0 matrix at this catalog size. Verify parity against the old helper on synthetic fixtures; do not modify existing B0 code.
5. Return per-track modalities and a serializable transform manifest containing audio/BPM statistics, IDF, genre vocabulary, fit IDs hash, split hash, and source/code hashes.
6. Run `.venv/bin/python -m unittest tests.test_features_neurotune_v1 -v`; verify that refitting with altered validation/test values does not change any training transform.

## Task 3: Ragged sequence dataset and masked collator

**Files:** Create `src/data/neurotune_v1_dataset.py` and `tests/test_neurotune_v1_dataset.py`.

1. Add synthetic transformed-feature tests for ID lookup, per-track sequence extraction, modality alignment, and right-padding/mask construction for mixed sequence lengths.
2. Run `.venv/bin/python -m unittest tests.test_neurotune_v1_dataset -v` and verify expected failures.
3. Build the dataset from `build_neurotune_v1_features` outputs, which provide requested-order per-track `(T,26)` arrays and already validate raw-cache IDs, offsets, and coverage. Keep audit/cache-metadata hash reconciliation in the experiment runner; do not duplicate cache parsing in the dataset class.
4. Implement a batch collator that pads only within each minibatch and returns `(audio, mask)` plus aligned BPM/lyrics/genre tensors. Set padded cells to zero and never mark them valid.
5. Test a sequence alone and in a batch with longer sequences; verify the valid rows and masks remain unchanged. Run the focused suite green.

## Task 4: Shared v1 encoder and ablation variants

**Files:** Create `src/model_v1.py` and `tests/test_model_v1.py`.

1. Add failing tests for variants A–D, variable lengths including length one, mask-aware pooling, expected dimensions, finite output, L2 unit norms, and missing/unexpected modality inputs.
2. Run `.venv/bin/python -m unittest tests.test_model_v1 -v` and confirm the intended failures.
3. Implement the `26→32→64→64` stride-1 same-padded CNN; after each convolution+ReLU, zero invalid positions with the unchanged mask so padded bias activations cannot leak into later layers. Use a mask-aware mean and project to a 64-D audio representation.
4. Implement BPM/lyrics/genre branches and only instantiate branches used by each variant. Fuse selected branches through `Linear(concat_dim,128)+ReLU→Linear(128,64)`; normalize the final output.
5. Add shared-weight triplet forwarding that accepts full collator dictionaries, selects only the active variant modalities, and embeds all three tracks through the same module instance; test gradients reach every active branch and padded timesteps do not enter the pooling denominator.
6. Run `.venv/bin/python -m unittest tests.test_model_v1 -v` green; do not modify `src/model.py` or prior encoder classes.

## Task 5: Triplet loss and deterministic train-only schedule

**Files:** Create `src/triplets.py` and `tests/test_triplets.py`.

1. Add tests for cosine-distance margin loss, zero loss on satisfied triplets, positive loss on violations, gradient flow, and invalid shapes/non-finite inputs.
2. Add sampler tests for both directions per undirected positive, one negative per directed positive/epoch, deterministic seeds, and exclusions of self/positive/known train neighbors; assert validation/test IDs and edges are never inputs to the train sampler.
3. Run `.venv/bin/python -m unittest tests.test_triplets -v` red.
4. Implement an index-based sampler that uses only partition-local adjacency and rejection sampling; do not materialize all song pairs. Build one compact `int32` triplet schedule per seed and reuse it across A–D variants.
5. Implement `triplet_margin_cosine_loss` using `d=1-dot` and margin 0.2. Run the focused suite green.

## Task 6: Training loop and validation checkpoint selection

**Files:** Create `src/train_neurotune_v1.py` and `tests/test_train_neurotune_v1.py`.

1. Add synthetic tests for repeatable updates, train-only schedule membership, validation without gradient updates, per-epoch curve fields, and checkpoint selection by best NDCG@10 with the documented tie-break.
2. Run `.venv/bin/python -m unittest tests.test_train_neurotune_v1 -v` red.
3. Implement one seed/variant trainer using Adam (`0.001`), batch 64, 50 epochs, the shared schedule, and the same train/validation IDs for all variants. Shuffle each epoch's schedule with a CPU `torch.Generator` seeded by `seed + epoch`, so all variants use the same deterministic batch order for a given seed.
4. At each epoch, record train triplet loss, fixed validation triplet loss, and validation Recall@10/NDCG@10/MRR over the 610-track validation gallery.
5. Save the best-validation checkpoint with model state, variant, seed, epoch, transform manifest, split hash, source/code/cache hashes, and selection metrics. Keep test input/relevance arguments out of this trainer API.
6. Run the focused suite green and test all four variants on synthetic sequences.

## Task 7: Same-split cosine baseline and retrieval evaluator

**Files:** Create `src/evaluate_neurotune_v1.py` and `tests/test_evaluate_neurotune_v1.py`.

1. Add tests for equal weighting across audio-summary/BPM/lyrics/genre modalities, normalized cosine ranking, query exclusion, deterministic tie-breaking, Recall@10, NDCG@10, and MRR.
2. Run `.venv/bin/python -m unittest tests.test_evaluate_neurotune_v1 -v` red.
3. Build the fixed baseline from train-normalized per-track mean audio summaries and the same train-fitted BPM/TF-IDF/genre inputs. L2-normalize each modality, weight each by `1/sqrt(4)`, concatenate, and normalize the final vector.
4. Add/use a batched retrieval helper in `src/retrieve.py` that normalizes the gallery once and ranks the query rows together with the existing deterministic self-exclusion/tie-break contract; reuse `src/evaluate.py` metrics and do not tune weights on validation or test.
5. Run the focused suite green and verify the baseline is evaluated on exactly the same validation/test galleries and relevance edges as the learned variants.

## Task 8: Experiment runner, provenance, and leakage-order tests

**Files:** Create `scripts/run_neurotune_v1.py` and `tests/test_run_neurotune_v1.py`.

1. Add a synthetic integration test that spies on test relevance/embedding construction and proves it occurs only after all 12 variant/seed checkpoints have been selected and saved.
2. Add tests for existing output refusal, audit/cache/split hash mismatches, source boundary, artist disjointness, and rollback of newly created outputs on failure/interruption.
3. Run `.venv/bin/python -m unittest tests.test_run_neurotune_v1 -v` red.
4. Implement a runner that executes the fixed cosine baseline and A–D × seeds 684/685/686. Generate/store each seed's triplet schedule once and reuse it across all variants. Do not retain run-wide traces in memory.
5. Write only under a new `results/neurotune_v1_ab/` namespace: `experiment.json`, `training_curves.csv`, compact `sampled_triplets_seed_*.npz`, and `checkpoints/{variant}_seed_*.pt`. Refuse any pre-existing targets.
6. Record exact split counts, 288 overlap removals, per-partition excluded edges, test gallery/query counts, transform/code/source/cache hashes, runtime versions, model settings, all seed metrics, caveats, and selected epochs.
7. Run the focused suite green and verify that all older reports, checkpoints, and caches remain outside the output path.

## Task 9: Real v1 experiment and final verification

**Files:** Update `docs/plans/2026-11-13-neurotune-v1-design.md`, `docs/plans/2026-11-13-neurotune-v1-implementation.md`, and the project README only if needed to describe the new experiment. Do not rewrite the original proposal.

1. Confirm all required v1 output targets do not exist before starting the real run.
2. Run `.venv/bin/python -m scripts.run_neurotune_v1`; retain exact stdout, exit status, report, checkpoints, curves, and hashes.
3. Recompute the report's source/code/cache/split hashes; check all metrics are finite and all selected epochs correspond to validation-only checkpoint selection.
4. Verify no test edges appear in the train/validation sampled triplets or fit-ID lists, and all test queries/gallery IDs belong to the held-out test artists.
5. Run `.venv/bin/python -m unittest discover -s tests -q`, `.venv/bin/python -m compileall -q src scripts tests`, and `.venv/bin/python -m pip check`.
6. Review the results as exploratory only; report the 215-edge test limitation, false-negative risk from absent links, and unverified third-party dataset provenance. No statistical/generalization claim.

## Execution constraints

This workspace is not a Git repository. Do not attempt worktree creation, commits, resets, or cleanup of unrelated outputs. Work sequentially where tasks depend on the new split/cache manifest; after independent model modules are stable, use subagents only for non-overlapping file scopes. Re-review each implementation lane before proceeding.

## Execution log

### Task 1 — complete (2026-11-13)

Created `src/data/neurotune_v1_split.py`, `scripts/build_neurotune_v1_split.py`, and their tests. The real manifest and partition edge CSVs are under `data/msd_hf_pilot/`. Counts: train/validation/test tracks 5,632/610/725; within-partition edges 12,916/185/215; test queries 265; cross-partition exclusions 4,435 train-source and 1,725 test-source edges. The test ZIP hash matches the catalog audit, artist groups are disjoint, and train/test edges do not overlap. Focused suite: 9 tests passed after exact seeded-selection and interruption/staging rollback coverage. Spec and quality re-reviews found no open issues.

### Task 2 — complete (2026-11-13)

Created `src/features_neurotune_v1.py` and its focused tests. Audio stats use float64 merged-Welford population variance over training-track rows, with zero-variance fallback; audio apply rejects non-finite float32 results. BPM, smoothed 5,000-word IDF/TF-IDF, and sorted genre vocabulary are fit from train IDs only. The v1 transform avoids the unused dense B0 intermediate and parity-tests against the existing helper. The manifest records fit/split, semantic source, cache, and code hashes. Focused suite: 13 tests passed; spec and quality re-reviews have no open findings. Full MXM parsing and model training are deferred to the integrated runner.

### Task 3 — complete (2026-11-13)

Created `src/data/neurotune_v1_dataset.py` and tests. The dataset accepts the feature builder's per-track ragged arrays in requested ID order; the collator pads only to each batch maximum, returns a boolean mask and lengths, and aligns auxiliary rows. Tests cover all input immutability, zero padding/masked cells, and sequence-value invariance alone vs. with longer tracks. The dataset rejects zero-width genre matrices and documents shared read-only NumPy inputs. Focused suite: 6 tests passed; spec and quality reviews found no open actionable findings.

### Task 4 — complete (2026-11-13)

Created `src/model_v1.py` and tests without changing existing model classes. Variants A–D implement the approved audio-only, audio+BPM, audio+BPM+genre, and full branch sets; invalid positions are zeroed after every convolution before masked mean pooling. The encoder validates modality shape/dtype/device and returns 64-D L2-normalized embeddings; triplet forwarding selects active modalities from full collator dictionaries and uses one shared module. Focused suite: 7 tests passed, including padding invariance at different padded lengths and active-branch gradient coverage. Spec and quality re-reviews found no open findings.

### Task 5 — complete (2026-11-13)

Created `src/triplets.py` and tests. The sampler canonicalizes undirected positives, emits both orientations, draws one partition-local uniform negative per directed positive/epoch by rejection sampling, excludes every observed anchor neighbor, and returns compact `int32` index schedules. The loss implements the specified cosine-distance margin on L2-normalized embeddings and validates its inputs. Focused suite: 9 tests passed, including deterministic approximate-uniformity, malformed endpoints, gradient, and boundary cases. Spec and quality re-reviews found no open findings.

### Task 6 — complete (2026-11-13; curve-schema correction 2026-11-14)

Created `src/train_neurotune_v1.py` and tests. Training consumes only partition-local datasets/schedules, shuffles each epoch deterministically, records fixed-schedule validation loss and full validation-gallery retrieval metrics, and selects checkpoints by NDCG@10 then validation loss with earliest-epoch exact ties. Checkpoints include selected state, config, transforms, and provenance; the API has no test inputs. Validation edges are materialized once, and the trainer uses the model's public active-modality property. During final integration, removed a redundant `MRR` history alias so curve rows match the runner's `MRR@10` schema; training and selection semantics are unchanged. Focused trainer/model suite: 16 tests initially, then all 8 trainer tests passed after the schema correction; spec and quality re-reviews found no open issues.

### Task 7 — complete (2026-11-13)

Created `src/evaluate_neurotune_v1.py` and tests for the fixed cosine baseline and matching retrieval metrics. The baseline uses mean summaries of train-normalized audio rows, train-fitted BPM/lyrics/genre features, per-modality L2 normalization, equal weights, and final normalization. Added `rank_top_k_many` to normalize the gallery once and compute batched query-gallery scores while preserving single-query ranking semantics. Focused retrieval/evaluator suite: 15 tests passed. Spec and quality re-reviews found no open issues after metric coverage and performance fixes.

### Task 8 — complete (2026-11-13; production hardening 2026-11-14)

Created `scripts/run_neurotune_v1.py` and integration tests. The runner validates audit/cache/split contracts and hashes, exact A/B source boundary, artist-disjoint membership, raw-to-partition edge derivation and exclusions; builds train-fitted transforms and contiguous partition feature views; generates one compact schedule per seed and reuses it for A–D; and stages a complete report/curves/checkpoints bundle with rollback on failure. Production settings are locked to all four variants, seeds 684/685/686, 50 epochs, batch 64, and the leakage barrier checks all 12 checkpoints before any test embedding or evaluation. Synthetic overrides are isolated to a separate test-only namespace. The real-artifact preflight passed with 5,632/610/725 tracks, 12,916/185/215 edges, 265 test queries, 288 audit-recorded upstream overlap removals, and zero remaining raw train/test edge overlap. Curves are now serialized after checkpoint completion but before test evaluation. Focused runner suite: 13 tests passed; final spec and quality re-reviews found no open issues.

### Task 9 — complete (2026-11-14)

Ran `.venv/bin/python -m scripts.run_neurotune_v1` successfully on the fixed A/B protocol; `/usr/bin/time -v` recorded exit status 0, 6:03:03 wall time, and 2,401,640 KiB maximum RSS. An initial completed attempt failed at curve serialization because trainer history included a redundant `MRR` field. Its staged artifacts were rolled back. After removing the alias and moving curve serialization before test evaluation, the unchanged protocol was rerun. The first attempt's computed metrics were not persisted or used for tuning; the design and README document that the test evaluator executed in two attempts.

The published `results/neurotune_v1_ab/experiment.json` contains 12 runs, 600 curve rows, three schedule files, and 12 selected checkpoints. The same-split cosine baseline achieved Recall@10/HitRate@10/MRR@10/NDCG@10 = 0.1803/0.2528/0.1105/0.1120. Across seeds, A averaged 0.0520/0.0780/0.0256/0.0271, B 0.0413/0.0667/0.0206/0.0220, C 0.1887/0.2629/0.0912/0.1031, and D 0.1874/0.2553/0.0887/0.1013; README records sample standard deviations. C and D exceeded cosine on mean Recall@10 and HitRate@10, while cosine was higher on MRR@10 and NDCG@10. Recomputed source/code/cache/split hashes and all schedule/checkpoint hashes match. All 12 selected epochs were checked against validation NDCG/loss curves; sampled triplets are train/validation partition-local, and checkpoint fit-ID hashes match train IDs only. Final verification: `.venv/bin/python -m unittest discover -s tests -q` passed 265 tests; compileall and pip check passed. Results remain exploratory: 215 official test edges, one cohort, incomplete Last.fm labels/possible false negatives, and unverified third-party A/B provenance.

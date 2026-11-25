# NeuroTune D3 Combined Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run an isolated, validation-only D3 experiment combining the already-fixed D1 semi-hard sampler and D2 modality dropout.

**Architecture:** Add one frozen D3 config and a separate runner at `scripts/run_neurotune_v1_d3_combined.py`. Reuse D1's `_make_schedule_provider` unchanged: D0 uniform warm-up for epochs 1–3, then the existing 75% uniform / 25% semi-hard target with the five-epoch embedding refresh and uniform fallback. Call the existing trainer with D2's post-branch, pre-fusion dropout probabilities during training only. Mining-refresh and validation/inference forwards remain in `eval()` with dropout disabled. D3 starts from a fresh initialization; D0, D1, and D2 are immutable reference artifacts.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, unittest; existing NeuroTune data, sampling, trainer, and evaluation code.

---

## Frozen D3 protocol

- Variant D; folds `[0, 1, 2]`; seeds `[684, 685, 686]`; 50 epochs; batch size 64; Adam at 0.001; triplet margin 0.2; CPU.
- Independent model initialization for every fold/seed. Do not load D0, D1, or D2 model weights.
- Pass the original configured experiment seed (684/685/686) unchanged to the trainer for initialization and shuffle behavior, matching D0/D1/D2; record `trainer_seed_derivation: "seed"`. D1 mining and D2 dropout retain their separate fold-derived RNG streams.
- D1 schedule policy: verified D0 uniform schedules for the first 3 epochs; then the existing 25% semi-hard target, strict cosine-band miner, five-epoch refresh, and uniform fallback. Keep the D1 provider implementation unchanged.
- D2 training-only branch dropout: BPM 0.10, genre 0.25, lyrics 0.20, audio 0.00; zero whole auxiliary branch embeddings after activation and before fusion; no inverse-probability scaling. The trainer's dedicated Torch RNG uses base `seed * 1000003 + fold * 1009`, then adds the zero-based epoch index. The D1 miner retains its separate Python RNG (`seed*1000003 + fold*1009 + one-based epoch`).
- Mining embeddings are computed by the existing provider while the model is in `eval()`/`no_grad()`; no D2 masks are passed for mining. Training triplet forwards use D2 masks. Validation and inference are unmasked.
- Preserve D0's known-positive exclusion policy, including official/legacy test-source positives as exclusion-only. No test features, embeddings, relevance, targets, or metrics may be built.
- Compare every D3 fold/seed with D0, D1, and D2 by exact fold/seed, and with the single deterministic D-cosine baseline for that fold. Require equal validation query and eligible-edge counts.
- Output namespace: `results/neurotune_v1_d3_combined/`; refuse an existing target and publish transactionally.

## Public test seams and evidence path

- Test the existing public `train_neurotune_v1` interface with both `train_schedule_provider` and the three D2 dropout probabilities/seed supplied together. The provider must see `model.training == False`; optimizer-step triplet forwards must see masks; validation must receive none.
- Test the D3 runner through its `_run(..., _test_only=True)` injection seam with synthetic folds/features/schedules and actual model/trainer for a four-epoch smoke (three uniform warm-up epochs plus the first semi-hard-provider epoch). Keep synthetic fixtures out of production result directories.
- Before the real 9-condition run, validate the frozen config, D0/D1/D2 reports and referenced artifact hashes, inputs, code, and absent D3 namespace. After the run, verify all 9 records, 450 epochs, schedules, checkpoints, report/output hashes, sampling/dropout diagnostics, finite validation metrics, unchanged protected hashes, and explicit absence of test-derived artifacts.

## Task 1: Add the frozen D3 configuration

**Files:**
- Create `config/neurotune_v1_d3_combined.json`
- Create or extend `tests/test_neurotune_v1_d3_combined_cli.py`
- Create `scripts/run_neurotune_v1_d3_combined.py` with config loading/validation only

**Steps:**
1. Write tests for exact D1 and D2 protocol values, both separate RNG derivations, validation-only scope, and the fixed D3 namespace.
2. Add invalid-type/value cases (including booleans in numeric fields), altered sampling/dropout values, and an unsafe output namespace.
3. Run `.venv/bin/python -m unittest discover -s tests -p 'test_neurotune_v1_d3_combined_cli.py' -q`; verify the new config/validator tests fail before implementation.
4. Add the minimal frozen JSON config and canonical type-sensitive validation in the new runner.
5. Re-run the focused command; expect all config tests to pass.

## Task 2: Validate immutable D0/D1/D2 references

**Files:**
- Modify `scripts/run_neurotune_v1_d3_combined.py`
- Extend `tests/test_neurotune_v1_d3_combined_cli.py`

**Steps:**
1. Add failing tests for incomplete/duplicate D0, D1, and D2 fold/seed records; duplicate/missing fold-level cosine rows; invalid selected epochs/metrics; missing or mismatched referenced checkpoints/schedules/config hashes; and incompatible query/eligible-edge counts.
2. Run the focused D3 tests and verify these cases fail before training is invoked.
3. Validate the complete fixed D0/D1/D2 protocols and report statuses. Identify D1 training records by their D1 schedule-file metadata, not by variant alone, because D1 `model_validation` also contains D0 reference rows.
4. Verify each D0/D1/D2 checkpoint and schedule path under its exact result namespace against the corresponding report SHA-256; verify the D1 schedule archive hash and its inner schedule digest. Verify D0/D1/D2 config files and report hashes; reject path traversal and missing artifacts.
5. Preserve D0's three fold-level cosine rows as one deterministic baseline per fold; reject duplicates rather than silently selecting the last row.
6. Re-run the focused tests; expect all reference-validation cases to pass without reading test labels as targets or constructing test features.

## Task 3: Implement the combined runner and synthetic integration tests

**Files:**
- Modify `scripts/run_neurotune_v1_d3_combined.py`
- Extend `tests/test_neurotune_v1_d3_combined_cli.py`

**Steps:**
1. Add a failing trainer-spy test that checks D3 uses variant D, `train_schedule_provider`, D2 probabilities, deterministic dropout seed, independent initialization, and fixed epochs/batch size.
2. Add failing output tests for the exact namespace, existing-target refusal, D1/D2/D0 validation comparisons, and no test-derived artifacts.
3. Run the focused test command and verify the new runner behavior fails before implementation.
4. Load fold-scoped features and D0 schedules through the existing shared data helpers. Build the D1 provider with current train IDs/edges, D0 warm-up schedule, known-positive exclusions, fold/seed, and the existing sampler factory; do not reimplement or alter D1 mining.
5. Pass the provider plus D2 BPM/genre/lyrics probabilities and the collision-resistant Torch dropout seed to `train_neurotune_v1`; keep `record_epoch_timing=True`.
6. Persist D3's generated per-epoch triplets and sampling-kind codes, validation schedules, checkpoints, curves, mining diagnostics, dropout diagnostics, and fold/seed comparison rows under the new namespace. Record D0/D1/D2 reference hashes, both RNG schemes, exclusion provenance, code/input/split hashes, and validation-only flags.
7. Use staged, no-overwrite publication; remove only the D3 staging tree on failure. Do not mutate D0/D1/D2 artifacts.
8. Re-run the focused D3 test file; expect all runner tests to pass.

## Task 4: Verify integration, run D3, and update documentation

**Files:**
- Update `README.md`
- Update `NeuroTune_v1_Performans_Iyilestirme_Deney_Plani.md`
- D3 artifacts under `results/neurotune_v1_d3_combined/`

**Steps:**
1. Run focused D1 sampler, trainer, D2 model/trainer, and D3 runner tests.
2. Run a four-epoch synthetic end-to-end D3 smoke using the real trainer/model and D1 provider; verify D0 uniform warm-up, one semi-hard-provider epoch, training-only branch masks, unmasked refresh/validation, diagnostic outputs, and cleanup/publishing in a temporary results root.
3. Run `.venv/bin/python -m unittest discover -s tests -q`, `.venv/bin/python -m compileall -q src scripts tests`, `.venv/bin/python -m pip check`, and `.venv/bin/python -m scripts.run_neurotune_v1_d3_combined --help`; do not launch real training if any fails.
4. Confirm all required D0/D1/D2 artifacts and data exist, validate their cross-references without training, and confirm the D3 output namespace is absent.
5. Run `.venv/bin/python -m scripts.run_neurotune_v1_d3_combined` once; expected output is 9 independent D3 records with 50 epochs each and no overwritten artifacts.
6. Independently audit runner exit status, 9 checkpoints/schedules, 450 curve/mining/dropout rows, reference and output hashes, configured versus realized sampling/dropout rates, per-fold/per-seed comparisons against D0/D1/D2/cosine, finite runtime/metrics, and test-data boundary flags.
7. Record D3's positive, negative, or mixed validation findings in README and the experiment plan. Do not claim model superiority or generalization without an independent test.

**Workspace constraint:** The NeuroTune directory is not a Git repository, so worktree isolation, commits, and Git diff review are unavailable. Preserve the D0/D1/D2 namespaces and make only the listed additive D3 files plus documentation updates.

## Completion record — 2026-11-21

- Tasks 1–4 completed. D3 uses the frozen trainer seed `seed` to match D0/D1/D2, while miner and dropout RNG streams remain separately fold-derived.
- The four-epoch synthetic smoke used the actual trainer/model and D1 schedule provider; it verified D0 uniform warm-up, the first miner epoch, training-only dropout, eval/no-grad mining and validation, and staged publication. It first reproduced the trainer contract error when a provider was supplied with a static schedule; the runner was corrected to pass `train_schedule=None`.
- Pre-run software verification: 501 tests passed; `compileall`, `pip check`, and D3 CLI help passed. D0/D1/D2 references validated as 9/9/9 model rows plus three cosine rows.
- The fixed runner completed with exit status 0 in 29,037.0 seconds (8:03:57). It produced 9 checkpoint/schedule pairs, 450 training-curve rows, 450 mining-diagnostic rows, 450 dropout-diagnostic rows, and 9 paired comparison rows. All 23 manifested output files, per-epoch schedule digests, warm-up schedules, input/code/reference hashes, and the unchanged D0/D1/D2 namespaces were verified. No independent test data or metrics were produced.
- Mean D3 NDCG@10 was 0.082233 versus D2 0.082199; D3's mean Recall@10, HitRate@10, and MRR@10 were lower than D2's. Six selected checkpoints came from the uniform warm-up; five had the same selected weights as D2, while fold 2/seed 685 selected D3 epoch 1 versus D2 epoch 29. The validation-only interpretation and mixed findings are recorded in `README.md` and the main experiment plan; D4 remains unstarted.

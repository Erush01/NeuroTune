# NeuroTune D2 Modality Dropout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and run a fixed validation-only D2 experiment that trains NeuroTune variant D with per-example auxiliary-branch embedding dropout while preserving D0/D1 defaults and artifacts.

**Architecture:** Add an explicit optional keep-mask path to the model and trainer. Only a new D2 runner passes masks: after each auxiliary branch projection/activation, it zeros whole BPM/genre/lyrics embeddings before fusion, without inverse-probability scaling. The runner reuses D0's fold-local features and byte-identical uniform schedules, writes to a fixed isolated namespace, and does not construct independent-test features or metrics.

**Tech Stack:** Python 3.12, PyTorch, NumPy, unittest, existing NeuroTune fold/data/schedule helpers.

---

## Fixed D2 protocol

- Variant D, independent initialization, folds `[0, 1, 2]`, seeds `[684, 685, 686]`, 50 epochs, batch size 64, Adam at 0.001, margin 0.2.
- Preserve D0's uniform triplet schedules byte-for-byte and verify schedule hashes.
- Drop probabilities: audio `0.00`, BPM `0.10`, genre `0.25`, lyrics `0.20`.
- Apply one Bernoulli keep/drop decision per auxiliary modality and track instance, including anchor, positive, and negative inputs. Zero embeddings after branch activation and before fusion; do not rescale kept embeddings.
- Apply masks only during training. Validation and inference run with all modalities enabled and no masks.
- Use a dedicated Torch RNG stream derived from fold, seed, and epoch and consumed in deterministic batch order. Keep triplet-order RNG and D0/D1 callers unchanged.
- Output namespace: `results/neurotune_v1_d2_modality_dropout/`; refuse an existing target. No D1 checkpoint continuation, hyperparameter retuning, independent-test features, independent-test labels, or independent-test metrics.

## Task 1: Add explicit post-branch model masks

**Files:**
- Modify `src/model_v1.py`
- Test `tests/test_model_v1.py`

**Step 1: Write failing model tests**

Test that a false keep-mask zeros the selected BPM, genre, or lyrics branch embedding after its biased projection/activation; a true mask preserves it; audio output and audio padding behavior are unchanged; and an omitted mask preserves current outputs and API behavior. Test invalid mask keys, shape, or dtype as rejected.

**Step 2: Run focused tests and confirm the expected failure**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_model_v1.py' -q`  
Expected: the new mask tests fail because the model has no explicit mask argument.

**Step 3: Implement the optional mask path**

Add an optional per-modality keep-mask argument to `NeuroTuneV1Encoder.forward` and route it through `forward_triplets`. Apply boolean masks to each auxiliary branch embedding after that branch's existing layers and immediately before concatenation. With no mask argument, do not add operations that alter the existing D0/D1 path.

**Step 4: Re-run focused tests**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_model_v1.py' -q`  
Expected: all model tests pass; default forward and D0/D1 behavior remain unchanged.

## Task 2: Generate deterministic train-only modality masks

**Files:**
- Modify `src/train_neurotune_v1.py`
- Test `tests/test_train_neurotune_v1.py`

**Step 1: Write failing trainer tests**

Test that configured dropout is applied to training triplet inputs, masks are reproducible for identical fold/seed/epoch/batch order, different keys produce different masks, observed rates match the Bernoulli configuration within a statistically reasonable tolerance on a sufficiently large synthetic batch, and validation/inference never receives masks. Test that the no-dropout default retains the existing history schema and deterministic checkpoint behavior.

**Step 2: Run focused tests and confirm the expected failure**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_train_neurotune_v1.py' -q`  
Expected: new tests fail because the trainer does not accept D2 dropout settings or generate masks.

**Step 3: Implement opt-in trainer support**

Add default-disabled dropout probabilities and a dedicated dropout seed input. Generate keep masks with a private CPU `torch.Generator`; seed it from the D2 fold/seed/epoch derivation and consume it in deterministic batch order. Keep this RNG separate from triplet shuffling and model initialization. Pass masks only for training calls; validation loss and ranking calls must omit masks. Record per-epoch attempted/dropped counts and realized rates for each auxiliary branch.

**Step 4: Re-run focused tests**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_train_neurotune_v1.py' -q`  
Expected: all trainer tests pass, including the legacy no-dropout behavior checks.

## Task 3: Add the frozen D2 config and isolated runner

**Files:**
- Create `config/neurotune_v1_d2_modality_dropout.json`
- Create `scripts/run_neurotune_v1_modality_dropout.py`
- Test `tests/test_neurotune_v1_d2_modality_dropout_cli.py`
- Reuse shared helpers in `src/data/neurotune_v1_abcd_cv_training.py` and the established D0/D1 artifact conventions; do not refactor D0 or alter D1 defaults.

**Step 1: Write failing runner/config tests**

Test the frozen protocol values, exact safe output namespace, config hash revalidation, refusal to overwrite an existing output target, byte-identical D0 schedule reuse, selected-checkpoint provenance, and validation-only behavior. Inject test dependencies to assert that no test features, test embeddings, test relevance, or test metrics are constructed.

**Step 2: Run focused runner tests and confirm the expected failure**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_neurotune_v1_d2_modality_dropout_cli.py' -q`  
Expected: the tests fail because the D2 config and runner do not exist.

**Step 3: Implement the fixed config and runner**

Load and validate the JSON config against exact fixed values, including the namespace. Load the same folds and train-fitted features as D0, reuse and hash the D0 uniform schedules, train D from independent initialization with dropout enabled, and select checkpoints using the existing validation NDCG@10/loss rule. Preserve D0's known-positive exclusion policy; official/legacy test-source edges remain exclusion-only. Hash inputs, code, config, splits, D0 references, schedules, checkpoints, and output files. Persist `experiment.json`, `d2_config.json`, training curves, modality-dropout diagnostics, per-fold/seed validation comparisons, schedules, and checkpoints. Include the dropout placement, no-rescaling rule, probabilities, and RNG derivation in provenance.

**Step 4: Run focused runner tests**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_neurotune_v1_d2_modality_dropout_cli.py' -q`  
Expected: all runner and configuration tests pass; D0 and D1 namespaces are untouched.

## Task 4: Validate, smoke-test, and run the fixed D2 study

**Files:**
- Update `README.md`
- Update `NeuroTune_v1_Performans_Iyilestirme_Deney_Plani.md`
- D2 artifacts under `results/neurotune_v1_d2_modality_dropout/`

**Step 1: Run the complete software preflight**

Run from the NeuroTune project root:

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
.venv/bin/python -m scripts.run_neurotune_v1_modality_dropout --help
```

Expected: all tests pass, compilation succeeds, dependencies are consistent, and CLI help exits 0. Do not train if any check fails.

**Step 2: Run a synthetic short-run smoke check**

Use synthetic features and schedules with the real model/trainer through the runner's test-only seam for a few epochs. Verify deterministic masks, training-only application, dropout diagnostics, safe output publication, and artifact provenance. Do not use test labels or production experiment outputs for the smoke test.

**Step 3: Snapshot protected hashes and run the fixed 9-condition study**

Before training, verify and protect the D0 report, referenced checkpoints and uniform schedules, data inputs, splits, config, and relevant code; confirm the D2 namespace is absent. D1 artifacts are neither inputs nor tuning targets. Run:

```bash
.venv/bin/python -m scripts.run_neurotune_v1_modality_dropout
```

Expected: nine D2 training records, each with 50 epochs, in the isolated D2 namespace; no existing output is overwritten.

**Step 4: Audit the completed artifacts**

Verify runner exit status, all nine checkpoint/schedule/output hashes, exact D0 schedule identity, fold/seed/epoch counts, finite validation metrics and total runtime, configured versus realized per-modality dropout rates, unchanged protected hashes, and absence of test metrics. Compare D2 with D0 by fold/seed and with the single fold-level D-cosine baseline (shared across that fold's three seeds). Do not infer generalization or claim D2 superiority from validation results.

**Step 5: Update the experiment documentation**

Record the fixed config, runtime, provenance, validation-only metrics, fold-wise differences, and any negative or mixed findings in `README.md` and the main experiment plan. Do not modify the D1 results or treat D1 validation values as a tuning target.

## Completion record — 2026-11-20

- The D2 implementation, frozen config, runner, and tests are complete. The full suite passed (487 tests); `compileall`, `pip check`, CLI help, and a two-epoch synthetic end-to-end smoke using the actual trainer/model also passed.
- The fixed run completed with exit 0 in 17,515.5 seconds: 9 training records, 450 curve rows, 450 dropout-diagnostic rows, 9 checkpoints, and 9 byte-identical D0 schedule copies. D0's 9 checkpoint references, 9 schedule hashes, and 3 fold-level cosine references were independently checked. Output, code, config, checkpoint, and schedule hashes verified.
- Mean validation NDCG@10 was 0.082199 for D2, 0.071530 for D0, and 0.077921 for the fold-level D-cosine baseline. Realized BPM/genre/lyrics mask rates were 0.099991/0.250032/0.199956. See the README and `results/neurotune_v1_d2_modality_dropout/experiment.json` for the full fold/seed comparisons and caveats.
- This is validation-only development evidence; no independent test features, labels, relevance, or metrics were produced. D3 remains unstarted pending a separate decision.

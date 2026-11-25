# NeuroTune D1 Semi-Hard Negative Mining Implementation Plan

> **For future agents:** Execute task-by-task. Preserve the existing D0 behavior and all prior result namespaces.

**Goal:** Add an isolated, deterministic D1 training variant for the existing full-modal D model, using a 75% uniform / 25% semi-hard negative mixture and reporting validation-only development results.

**Architecture:** Keep `sample_triplet_schedule` and the current static-schedule trainer path unchanged for D0. Add a D1 mixed sampler and an optional per-epoch schedule provider that refreshes train embeddings every five epochs after three uniform warm-up epochs. A new runner trains only variant D in a no-overwrite namespace, uses the existing fold/seed/split/transform/optimizer protocol, and records schedule, mining, checkpoint, input, and code provenance.

**Tech Stack:** Python, `unittest`, NumPy, PyTorch, existing NeuroTune v1 data, feature, model, and retrieval APIs.

---

**Approved design:** `NeuroTune_v1_Performans_Iyilestirme_Deney_Plani.md`, Sections 3 and 7–9, plus the user's choice on 2026-11-18 to preserve the existing D0 known-positive exclusion policy for this validation-development experiment. The existing policy's use of official/legacy test-source positives for negative exclusion is inherited and must be disclosed; no test features, embeddings, relevance, or metrics may be produced. This workspace is not a Git repository; do not initialize Git or claim a commit.

**Scope:** Implement D1 only. Do not implement D2, D3, or D4. Train D1 independently from initialization, not by fine-tuning D0 checkpoints. Preserve all D0 checkpoints, schedules, reports, and result namespaces.

### Task 1 — Deterministic mixed semi-hard sampler

**Files:** Modify `src/triplets.py`; modify `tests/test_triplets.py`. Read-only references: `src/triplets.py:12-90`, `tests/test_triplets.py:47-145`.

1. Add failing tests for a one-epoch mixed schedule over synthetic IDs and normalized embeddings. Cover the strict cosine-distance condition `d(a,p) < d(a,n) < d(a,p)+margin`, exactly the configured attempted semi-hard share (75% random / 25% semi-hard, subject to fallback), directed positives, and stable output for identical inputs and seed.
2. Run `.venv/bin/python -m unittest discover -s tests -p 'test_triplets.py' -q`; confirm the new tests fail before implementation.
3. Add a separate sampler helper; do not change the existing `sample_triplet_schedule` behavior or its deterministic legacy test. Inputs must bind candidate IDs to the provided train embeddings and known-positive exclusions. Never include an ID outside the supplied train partition.
4. Exclude the anchor, scheduled positive, and the same partition-local known-positive neighbors used by D0. Pick uniformly among eligible semi-hard candidates. If no semi-hard candidate exists, use the existing uniform eligibility rule and record a fallback rather than failing training.
5. Add tests for exclusion of unscheduled known positives, empty semi-hard pools, no eligible uniform negatives, invalid/nonfinite embeddings, invalid margin/fraction, and unchanged legacy schedule bytes.
6. Rerun the focused test command; expected: all sampler and loss tests pass.

### Task 2 — D1 per-epoch schedule provider and training telemetry

**Files:** Modify `src/train_neurotune_v1.py`; modify `tests/test_train_neurotune_v1.py`. Do not modify `src/model_v1.py` or dataset/feature behavior.

1. Add failing tests for an optional train-schedule provider called once per epoch with the current model and epoch index; verify its schedule is validated against the train dataset and only the train dataset is supplied for mining.
2. Test the refresh cadence: epochs 1–3 use uniform schedules; the first mining embedding refresh occurs before epoch 4, then refreshes every 5 epochs (epochs 4, 9, 14, …). Reuse each refreshed embedding snapshot only until the next refresh, while drawing a deterministic new schedule each epoch.
3. Run `.venv/bin/python -m unittest discover -s tests -p 'test_train_neurotune_v1.py' -q`; confirm the new provider tests fail before implementation.
4. Add the optional provider path while leaving the existing static-array path and D0 call sites unchanged. Provider embedding inference must use `model.eval()` and `torch.no_grad()` and restore the model's training state before optimizer updates.
5. Record per-epoch train loss, active-triplet fraction, positive/negative cosine-distance summaries, margin-violation fraction, requested and realized random/semi-hard fractions, fallback count, and mining-pool diagnostics. Keep validation evaluation, NDCG@10 checkpoint selection, and tie-breaking unchanged.
6. Test D0 static-schedule parity: with the provider omitted, the same seed, datasets, and schedules must retain the existing deterministic history/checkpoint selection. Test that validation and test IDs/embeddings are never passed to the provider.
7. Rerun the focused trainer tests; expected: existing and new tests pass.

### Task 3 — Isolated D1 runner and artifact provenance

**Files:** Add `scripts/run_neurotune_v1_training_improvements.py`; add `tests/test_neurotune_v1_training_improvements_cli.py`. Read-only references: `scripts/run_neurotune_v1_full_modal_ablation.py:44-85,148-189,191-330,377-415`; `src/data/neurotune_v1_abcd_cv_training.py:105-122,249-342`; `src/train_neurotune_v1.py:60-197`.

1. Add a synthetic runner test with tiny train/validation partitions, deterministic features, fake checkpoints, and an injected trainer. Assert that only D is trained; fold/seed, 50-epoch, batch-64, Adam `0.001`, margin `0.2`, transform and validation inputs match the approved protocol; and output collisions fail before expensive I/O.
2. Test D0 parity inputs: consume the same train/validation partition IDs, train/validation target edges, validation schedule, and partition-local known-positive exclusion set as the existing D0 schedule. The D1 train sampler may use only fold-train candidate embeddings. Preserve the inherited exclusion-only official/legacy test-source boundary and record it explicitly in the report.
3. Add runner tests proving no test-track features, embeddings, relevance, or metrics are built. Validation metrics must use the same validation query/gallery/edge set as D0; mark `independent_test_completed=false` and `evaluation_scope=validation_only_development`.
4. Implement atomic staged publication to `results/neurotune_v1_training_improvements/`; refuse if the target exists or is a symlink, and clean up only files created by a failed attempt. Snapshot and recheck D0, split, source, feature, and code hashes; do not modify existing artifacts.
5. Persist `experiment.json`, frozen D1 configuration, D0/D1 validation records, `training_curves.csv`, `validation_metrics.csv`, `mining_diagnostics.csv`, compressed epoch triplet schedules with sampling-type labels, and selected checkpoints. Include fold, seed, epoch, RNG/mining version, refresh, schedule hash, fallback and realized sampling counts. Preserve one row per fold/seed and do not treat nine runs as nine independent test sets.
6. Run `.venv/bin/python -m unittest discover -s tests -p 'test_neurotune_v1_training_improvements_cli.py' -q`; expected: synthetic runner tests pass without starting model training.

### Task 4 — Documentation and reproducible configuration

**Files:** Add a D1 config under `config/`; update `README.md`; update `NeuroTune_v1_Performans_Iyilestirme_Deney_Plani.md` with the approved exclusion decision and implementation status. Do not edit prior result reports or their numerical values.

1. Record the fixed D1 values: variant D; folds `0/1/2`; seeds `684/685/686`; 50 epochs; batch size 64; learning rate `0.001`; margin `0.2`; semi-hard fraction `0.25`; 3 warm-up epochs; refresh interval 5; CPU; deterministic RNG derivation.
2. Document that this run is validation-only and preserves D0's inherited official/legacy test-source use for negative exclusion only. State that no independent-test or generalization claim is made and that future independent-test labels must not enter training or mining.
3. Document the new isolated runner command and outputs; do not change D0 defaults or add D2/D3/D4 commands.
4. Verify every new README path and command against the filesystem and CLI; run `bash -n` on all documented shell blocks.

### Task 5 — Focused pilot, full D1 development run, and final checks

**Files:** New artifacts only under `results/neurotune_v1_training_improvements_pilot/` and `results/neurotune_v1_training_improvements/`; all historical D0 namespaces remain read-only.

1. Run the synthetic/unit suite, `compileall`, and `pip check` before any real training. Snapshot hashes for D0 checkpoints/schedules/reports, data inputs, splits, and generated figures.
2. Run the fixed fold-0 / seeds-684, 685, 686 pilot into its own no-overwrite pilot namespace. Do not alter the predeclared hyperparameters based on validation results. Inspect peak RSS, per-epoch runtime, mining fallback rates, realized sampler ratios, and output integrity.
3. If the pilot is numerically valid and resource use is manageable, run the full fixed 3-fold × 3-seed D1 evaluation once into `results/neurotune_v1_training_improvements/`. The pilot is a separate resource check and is not copied into or merged with the full-run namespace.
4. Independently verify all nine D1 checkpoint/validation/schedule records, fold-paired D0 comparisons, matched-cosine comparisons, configuration/provenance hashes, `independent_test_completed=false`, and absence of test metrics. Report means and fold/seed variation descriptively; do not claim significance or generalization.
5. Run exact final checks from the project root:

   ```bash
   .venv/bin/python -m unittest discover -s tests -q
   .venv/bin/python -m compileall -q src scripts tests
   .venv/bin/python -m pip check
   ```

   Expected: all tests pass, compileall exits 0, and pip check reports `No broken requirements found.` Also verify old D0 and source/input hashes match the pre-run snapshot. This workspace has no Git metadata, so leave changes uncommitted and do not initialize a repository.

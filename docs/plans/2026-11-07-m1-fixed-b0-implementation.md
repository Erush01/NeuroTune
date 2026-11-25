# M1 Fixed B0 Baseline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement one leakage-safe, fixed-weight handcrafted retrieval baseline and exploratory top-K metrics for the current 10K MSD subset.

**Architecture:** Fit BPM scaling, lyric IDF, and the genre vocabulary from the training split only. Build one normalized concatenated vector per eligible song, retrieve from the full held-out test gallery, and evaluate only observed Last.fm edges internal to the test partition. Save metrics and the exact split/fit metadata to JSON.

**Tech Stack:** Python 3, NumPy, existing CSV/JSON parsers, `unittest`.

---

### Task 1: Implement train-only B0 feature fitting

**Files:**
- Create: `src/features.py`
- Create: `tests/test_features.py`
- Modify: `pyproject.toml` (declare NumPy runtime dependency if it is not already declared)

**Step 1: Write failing tests**

Test raw-BPM standardization from train IDs only, smoothed IDF from train documents only, L2-normalized lyric vectors, train-observed genre vocabulary, and unseen test genre mapping to zeros.

**Step 2: Verify the tests fail**

Run: `.venv/bin/python -m unittest tests.test_features -v`
Expected: import failure for the unimplemented feature fitter.

**Step 3: Implement minimal fitting and transform functions**

Use the official fixed MXM word indices. Fit BPM mean/std, document frequencies, and sorted genre categories using only `train_ids`. Transform all requested IDs using those fitted values. Give the three modality vectors equal predeclared scalar weights of `1/sqrt(3)`, concatenate, and L2-normalize each resulting row. Do not inspect validation/test values while fitting.

**Step 4: Verify the tests pass**

Run: `.venv/bin/python -m unittest tests.test_features -v`
Expected: all feature-fitting tests pass.

### Task 2: Implement full-gallery ranking and metrics

**Files:**
- Create or modify: `src/retrieve.py`
- Modify: `src/evaluate.py`
- Create: `tests/test_retrieve.py`
- Create: `tests/test_evaluate.py`

**Step 1: Write failing tests**

Cover cosine ordering, self-exclusion, sorted top-K output, no cross-partition positives, and hand-computed Recall@10, HitRate@10, MRR@10, and NDCG@10.

**Step 2: Verify the tests fail**

Run: `.venv/bin/python -m unittest tests.test_retrieve tests.test_evaluate -v`
Expected: import failures for missing retrieval/evaluation functions.

**Step 3: Implement ranking and binary relevance metrics**

Rank all held-out test candidates; exclude only the query itself. Define eligible queries as test songs with at least one observed positive in the test gallery. Use only edges whose endpoints are both test IDs. Keep query count and total eligible positive-edge count in the metric result.

**Step 4: Verify the tests pass**

Run: `.venv/bin/python -m unittest tests.test_retrieve tests.test_evaluate -v`
Expected: all ranking and metric tests pass.

### Task 3: Wire the B0 experiment CLI and report

**Files:**
- Create: `scripts/run_baseline.py`
- Modify: `README.md`
- Create: `tests/test_baseline_cli.py`
- Output: `results/baseline_b0.json`

**Step 1: Write a failing CLI/report test**

Use a temporary data directory with a deterministic synthetic split and assert that the command writes metrics, split counts, train-only preprocessing metadata, and an exploratory warning.

**Step 2: Verify the test fails**

Run: `.venv/bin/python -m unittest tests.test_baseline_cli -v`
Expected: command/report entry point is not implemented.

**Step 3: Implement the CLI**

Read `results/data_audit.json`, the documented processed feature inputs, and the current Last.fm subset JSON. Fit on the recorded training IDs; evaluate only the recorded test IDs. Include source hashes, seed, split counts, K, metric values, query count, eligible positive count, and a warning that the measured test graph contains only 11 positive edges. Do not modify `results/data_audit.json` or use full official Last.fm test edges as test relevance for this 10K catalog.

**Step 4: Verify the test passes**

Run: `.venv/bin/python -m unittest tests.test_baseline_cli -v`
Expected: JSON output is written and parsed by the test.

### Task 4: End-to-end verification

**Files:**
- Test: all existing and newly added tests
- Output: `results/baseline_b0.json`

**Step 1: Run the full test suite**

Run: `.venv/bin/python -m unittest discover -s tests -v`
Expected: all tests pass.

**Step 2: Run compilation and the actual baseline**

Run: `.venv/bin/python -m compileall -q src scripts tests`
Then run: `.venv/bin/python -m scripts.run_baseline`
Expected: the metrics file records the full test-gallery size and a small query/positive count without claims of statistical significance.

**Step 3: Inspect the generated JSON**

Verify split isolation, training-only fitting metadata, source hashes, query count, eligible positives, and the exploratory caveat before reporting results.

---

### Constraints

- No neural model or learned weights in this milestone.
- No weight tuning on the six validation positive edges.
- No test-driven preprocessing: every fitted statistic uses training IDs only.
- The current test set has only 11 observed positive edges; metrics are exploratory.
- This workspace has no Git repository, so the plan cannot include commit steps or be committed.

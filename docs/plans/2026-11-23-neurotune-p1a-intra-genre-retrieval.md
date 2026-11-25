# NeuroTune P1-A Intra-Genre Retrieval Implementation Plan

> **For Codex:** Use `executing-plans` to execute this plan task-by-task with the validation checkpoints below.

**Goal:** Compare frozen D0 and D4 neural retrieval with each model's matched cosine baseline on identical, query-specific same-genre validation galleries.

**Architecture:** Add a P1-A evaluator module for raw-genre gallery/eligibility construction, grouped ranking, metrics, and summaries; add a fixed-protocol CLI runner that reuses the validated ABCD loader, train-fitted D0/D4 feature builders, frozen checkpoints, and existing cosine/ranking helpers. Keep query/gallery and positive labels validation-only; allow the existing loader to read test-source files only for integrity verification, while never passing test content to features, candidates, relevance, metrics, or training.

**Tech Stack:** Python, NumPy, PyTorch, existing NeuroTune retrieval/feature modules, `unittest`.

**Status:** Completed 23 November 2026. The final preflight passed all 578 tests, `compileall`, `pip check`, and CLI help. The validation-only production run completed in 135.107 seconds and published `results/neurotune_v1_p1a_intra_genre_retrieval/`; saved metrics and protected input/result hashes were independently checked. No training or independent-test evaluation was performed. The first run stopped at protected-directory hash validation before feature processing; P1-A was aligned to the canonical D0/D4 path-hash contract before the successful retry.

---

## Task 1: Define and test same-genre candidate eligibility

**Files:**
- Create: `tests/test_neurotune_v1_p1a_intra_genre_retrieval.py`
- Create: `src/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py`

**Step 1: Write failing tests for candidate and edge policy**

Test a small fixture containing two genres, one missing label, a cross-genre positive, a same-genre positive, a singleton genre, and an otherwise valid query with no same-genre positives. Assert exact-genre candidates, self-exclusion, in-gallery relevance only, and separate statuses for missing genre, empty gallery, and no in-genre positive.

**Step 2: Run the focused tests and confirm the new helper is missing**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_p1a_intra_genre_retrieval -v`  
Expected: FAIL because the candidate builder does not exist yet.

**Step 3: Implement the minimum candidate builder**

Add a helper that takes validation IDs, their raw catalog genre strings, and unique undirected validation edges. Return per-query exact-genre candidate IDs, filtered relevance, gallery size, eligible status/reason, and per-fold coverage counts. Do not use the train-fitted one-hot genre matrix to determine matches.

**Step 4: Re-run the focused tests**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_p1a_intra_genre_retrieval -v`  
Expected: PASS, including missing-label and cross-genre-edge cases.

## Task 2: Implement filtered rankings and metric aggregation

**Files:**
- Modify: `src/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py`
- Modify: `tests/test_neurotune_v1_p1a_intra_genre_retrieval.py`

**Step 1: Write failing ranking/metric tests**

Use synthetic embeddings where the full-gallery top-10 differs from the same-genre top-10. Verify that each query is ranked only against its exact-genre candidates using the existing `rank_top_k_many` on genre groups, then evaluate with `evaluate_rankings`. Test that off-gallery positives do not enter Recall@10's denominator, queries with no in-gallery positive do not enter macro metrics, and an eligible gallery smaller than ten is included but flagged.

**Step 2: Verify the focused tests fail for the absent evaluator**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_p1a_intra_genre_retrieval -v`  
Expected: FAIL on missing filtered evaluator/aggregation helpers.

**Step 3: Implement per-query, genre, and fold/seed metrics**

Return Recall@10, MRR@10, and NDCG@10 per eligible query. Add macro-query and equal-weight genre-macro summaries. Define micro Recall@10 as total top-10 relevant candidates divided by total in-gallery relevant candidates; do not invent edge-weighted micro MRR/NDCG. Produce `<10` gallery subgroup metrics and counts separately. Preserve deterministic tie-breaking from `rank_top_k_many`.

**Step 4: Verify focused tests pass**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_p1a_intra_genre_retrieval -v`  
Expected: PASS for ranking restrictions, denominators, small galleries, and macro/micro summaries.

## Task 3: Build the four-system frozen-checkpoint runner

**Files:**
- Modify: `src/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py`
- Modify: `tests/test_neurotune_v1_p1a_intra_genre_retrieval.py`

**Step 1: Add a synthetic integration fixture with four representations**

Inject loader, feature builders, checkpoint loader/embedder, and ranking seams following `tests/test_neurotune_v1_d4_modality_sensitivity_cli.py`. Include a test-ID canary and assert that only `fold`, `train`, and `validation` are passed to feature builders; only `validation_edges` are used as relevance. Include frozen D0/D4 manifest examples matching the real checkpoint schemas: D0 has `provenance.transform_sha256`; D4 may omit it while its checkpoint manifest and current transform remain exactly bound to the reference.

**Step 2: Run integration tests red**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_p1a_intra_genre_retrieval -v`  
Expected: FAIL until the runner evaluates all four systems and writes the staged artifacts.

**Step 3: Implement frozen validation-only evaluation**

Use `load_abcd_cv_training`, `build_abcd_cv_fold_mxm`, `build_abcd_cv_development_features`, and `build_abcd_cv_d4_multi_window_features` from `src/data/neurotune_v1_abcd_cv_training.py`. Build train-fitted features from train+validation IDs only; derive genre membership from `loaded["catalog"][track_id]["genre"]`. For each fold, build D0-matched and D4-matched cosine embeddings with `build_neurotune_v1_cosine_baseline` and `build_d4_neurotune_v1_cosine_baseline`. For each fold/seed, load the existing D0/D4 D checkpoints and create validation neural embeddings with `NeuroTuneV1Encoder`, `NeuroTuneV1Dataset`, and `neurotune_v1_collate`.

Reuse the strict checkpoint, manifest, D0 transform-drift, and D4 schedule-copy validation helpers from `src/evaluate_d4_modality_sensitivity.py`; do not rewrite their schema rules. Verify D0/D4 full-gallery neural parity against the saved validation records before accepting the same-genre results. Rank all four systems over the same per-query candidate IDs and filtered validation positives. Refuse to overwrite `results/neurotune_v1_p1a_intra_genre_retrieval/`; stage outputs before publication and hash protected existing inputs/references before and after.

**Step 4: Verify integration tests pass**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_p1a_intra_genre_retrieval -v`  
Expected: PASS with no test ID in feature-builder inputs, no test-source edge in relevance, identical candidate/label hashes across all four systems, and overwrite refusal.

## Task 4: Add the fixed-protocol CLI and output report

**Files:**
- Create: `scripts/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py`
- Modify: `src/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py`
- Modify: `tests/test_neurotune_v1_p1a_intra_genre_retrieval.py`
- Create at runtime: `results/neurotune_v1_p1a_intra_genre_retrieval/experiment.json`
- Create at runtime: `results/neurotune_v1_p1a_intra_genre_retrieval/per_query_metrics.csv`
- Create at runtime: `results/neurotune_v1_p1a_intra_genre_retrieval/per_fold_seed_genre_metrics.csv`
- Create at runtime: `results/neurotune_v1_p1a_intra_genre_retrieval/gallery_coverage.csv`
- Create at runtime: `results/neurotune_v1_p1a_intra_genre_retrieval/analysis.md`

**Step 1: Test CLI help and output-schema invariants**

Test `--help`, fixed folds/seeds/system names, finite metrics, missing/unusable genre counts, zero-positive query counts, small-gallery counts, edge/query coverage, and explicit validation-only/test-integrity-only scope in the report.

**Step 2: Implement CLI and staged outputs**

Use the established `--data-root` and `--results-root` conventions from `scripts/evaluate_neurotune_v1_d4_modality_sensitivity.py`; expose no protocol overrides. Record source/checkpoint/transform/schedule and P1-A module/CLI code hashes, runtime, system-to-cosine matching, all gallery/eligibility counts, macro and micro definitions, and limitations. Keep prior P0 and D0–D4 namespaces read-only. The runner requires Linux `renameat2` for atomic no-overwrite publication and fails fast elsewhere.

**Step 3: Verify CLI and focused tests**

Run: `.venv/bin/python -m unittest tests.test_neurotune_v1_p1a_intra_genre_retrieval -v`  
Run: `.venv/bin/python scripts/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py --help`  
Expected: focused tests PASS and help exits 0; help exposes only the two root options and performs no data evaluation.

## Task 5: Final preflight and production validation analysis

**Files:**
- Verify: `src/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py`
- Verify: `scripts/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py`
- Verify: all artifacts under `results/neurotune_v1_p1a_intra_genre_retrieval/`
- Modify after successful run: `NeuroTune_D4_Sonraki_Adimlar_ve_Teshis_Plani_2026-11-23.md`

**Step 1: Run the full suite and static checks**

Run: `.venv/bin/python -m unittest discover -s tests -q`  
Run: `.venv/bin/python -m compileall -q src scripts tests`  
Run: `.venv/bin/python -m pip check`  
Expected: all tests pass, compilation succeeds, and pip reports no broken requirements.

**Step 2: Run one validation-only P1-A analysis**

Run: `.venv/bin/python scripts/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py`  
Expected: Linux `renameat2` available; exit 0; no training or test metrics; output namespace created once without overwriting any prior results.

**Step 3: Validate and summarize the saved artifacts**

Check every fold/seed/system has the expected eligible-query rows; all scores are finite; four systems share each query's candidate and positive-edge hashes; <10-gallery and ineligible-query counts reconcile; micro Recall and macro summaries recompute from per-query CSV; provenance hashes and protected D0/D4 namespaces remain unchanged. Review `analysis.md` for coverage, per-genre results, macro/micro definitions, the validation-only limitation, incomplete same-genre positive labels, and no superiority/generalization claim. Mark P1-A complete in the existing next-steps plan only after these checks pass.

## Repository constraints

`NeuroTune` is not a Git repository. Work in place, do not create worktrees or attempt commits. Preserve all prior D0–D4, P0, pilot, and test artifacts.

# MSD A–D Multimodal Expansion Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Audit the combined A/B/C/D MSD shards and report how many usable multimodal candidate tracks and Last.fm train/test links were added relative to the existing A/B pilot.

**Architecture:** Preserve the existing A/B command and artifact contract while generalizing its streaming audit core to accept named shard paths and an output prefix. Add a dedicated A–D CLI that scans all four archives together for global duplicate resolution, compares its candidate catalog and edge sets with the existing A/B artifacts, and publishes only under `data/msd_hf_abcd_pilot/`. Do not build an audio cache or train a model in this phase.

**Tech Stack:** Python 3, `tarfile`, `zipfile`, `h5py`, NumPy-compatible audit helpers, `unittest`.

---

## Task 1: Generalize the audit core without changing A/B behavior

**Files:**
- Modify: `scripts/audit_msd_hf_ab.py`
- Test: `tests/test_msd_hf_abcd_audit.py`

1. Add failing synthetic tests for a named-shard audit accepting A/B/C/D archives, preserving each source hash under the right label, and resolving an artist/song or official duplicate group that spans two different shards.
2. Test the existing two-shard `run()` wrapper still writes the same `ab_*` filenames and `A`/`B` source-hash keys.
3. Add a generic internal audit entry point that receives a unique ordered mapping of shard labels to paths and an output prefix; retain `run(a_tar, b_tar, ...)` as the backward-compatible A/B wrapper.
4. Keep the current safety properties: stream tar members, reject unsafe/special members, enforce the 64 MiB member bound, reject untrusted HDF5 VLEN/VDS/external links and invalid shapes, hash in chunks, refuse output collisions, and rollback partial publication.
5. Run `.venv/bin/python -m unittest tests.test_msd_hf_ab_audit tests.test_msd_hf_abcd_audit -v`; confirm red, implement, and rerun green.

### Task 1 — complete (2026-11-15)

Added `_run_named_shards()` with ordered per-shard provenance and output-prefix validation, while retaining the original A/B `run()` wrapper and `ab_*` outputs. Synthetic tests verify A/B/C/D source hashes, global artist/song duplicate resolution across shards, unsafe-prefix rejection, and A/B compatibility. The test module imports A/B fixtures through the module so unittest discovery does not duplicate the legacy class. Focused command passed 16 tests (13 A/B and 3 named-shard). Spec and quality re-reviews found no remaining findings.

## Task 2: Add the isolated A–D CLI and A/B comparison

**Files:**
- Create: `scripts/audit_msd_hf_abcd.py`
- Modify: `scripts/audit_msd_hf_ab.py`
- Test: `tests/test_msd_hf_abcd_audit.py`

1. Add repository-rooted defaults for `data/msd_hf_pilot/{A,B,C,D}.tar.gz`, existing processed musiXmatch/Tagtraum/duplicate inputs, root Last.fm train/test archives, A/B reference catalog/edge CSVs, and `data/msd_hf_abcd_pilot/` output.
2. Call the named-shard core with labels A/B/C/D and prefix `abcd`; output `abcd_catalog_audit.json`, `abcd_catalog_tracks.csv`, `abcd_lastfm_train_edges.csv`, and `abcd_lastfm_test_edges.csv`. Never write into `data/msd_hf_pilot/`.
3. Apply duplicate resolution across all four shard record sets before modality/tempo filtering. Keep Last.fm train/test edges separate and remove undirected overlap from train.
4. Compare A–D outputs with the existing A/B catalog and edge files. Report added, removed, and net candidate tracks; added/removed/net train and test edges; and new-new/new-existing/existing-existing link counts. Hash every reference artifact used in the comparison.
5. Add tests for cross-shard duplicate resolution, edge overlap and partitioning, comparison counts, source labels/output prefix, missing sources, collision refusal, rollback, and A/B output compatibility. Run the focused audit suites.
6. Record the user's project-use authorization in the A–D report without presenting it as independent verification of upstream provenance or redistribution terms.

### Task 2 — complete (2026-11-15)

Added the isolated A–D CLI, global catalog/edge comparison with A/B, reference and source hashes, and project-use authorization metadata. The A–D wrapper hash is included while the A/B hash contract stays unchanged. The CLI parser initially routed `--genres` to `genres` instead of `genres_csv`; the first real command failed before scanning or writing outputs. Added strict-signature argument-routing coverage and fixed all parser destinations. Final focused audit suite passed 22 tests; spec and quality reviews found no remaining issues. The real A–D scan was launched only after the parser fix and A/B artifact hashes were rechecked.

## Task 3: Run and verify the combined audit

**Files:**
- New outputs only: `data/msd_hf_abcd_pilot/abcd_catalog_audit.json`, `abcd_catalog_tracks.csv`, `abcd_lastfm_train_edges.csv`, `abcd_lastfm_test_edges.csv`

1. Confirm the A–D output directory/targets are absent; record SHA-256 snapshots of the four existing A/B audit outputs.
2. Run `.venv/bin/python -m scripts.audit_msd_hf_abcd`.
3. Verify A/B/C/D archive hashes and byte sizes, combined eligible-track counts, duplicate policy, A/B comparison deltas, train/test edge counts, and zero train/test edge overlap. Verify each recorded reference hash matches the unchanged A/B artifact.
4. Confirm no cache or model-training output was created and that the four prior A/B outputs are byte-for-byte unchanged.

### Task 3 — complete (2026-11-15)

The first CLI invocation failed before scanning because the parser passed `genres` instead of `genres_csv`; no output was created. After fixing argument destinations and adding strict routing coverage, `.venv/bin/python -m scripts.audit_msd_hf_abcd` exited 0. The A–D scan found 154,801 audio records, removed 2,954 duplicates, retained 151,331 positive-tempo survivors, and identified 13,822 lyrics+genre+tempo candidate tracks. Relative to A/B, 6,855 candidate IDs were added and none removed. Train links increased from 17,536 to 67,657 (+50,121); test links increased from 1,940 to 8,033 (+6,093). A–D train edges break down as 16,305 new-new, 33,816 new-existing, and 17,536 existing-existing; test edges as 2,060, 4,033, and 1,940. The audit removed 1,204 train/test overlap edges. Independent checks verified row counts, zero output train/test overlap, source/reference/code hashes, and all six saved A/B audit/cache artifact hashes. The new namespace contains only the four `abcd_*` audit outputs; no temporal cache or training was produced. These are coverage candidates; segment arrays remain to be validated by a later cache build.

## Task 4: Document results and final verification

**Files:**
- Modify: `README.md`
- Modify: this plan with audit counts and verification results

1. Document the measured A–D catalog and edge expansion, distinguish candidate coverage from temporal-cache validation, and state that no training was run in this audit phase.
2. Run `.venv/bin/python -m unittest tests.test_msd_hf_ab_audit tests.test_msd_hf_abcd_audit -v`.
3. Run `.venv/bin/python -m unittest discover -s tests -q`.
4. Run `.venv/bin/python -m compileall -q src scripts tests` and `.venv/bin/python -m pip check`.
5. Verify A/B artifact hashes remain unchanged after documentation and tests.

### Task 4 — complete (2026-11-15)

Updated README and this plan with candidate-count, train/test-edge, edge-class, and temporal-cache caveats. Final verification: `.venv/bin/python -m unittest discover -s tests -q` passed 331 tests; `.venv/bin/python -m compileall -q src scripts tests` passed; `.venv/bin/python -m pip check` reported no broken requirements. All six snapshotted A/B audit/cache artifacts remain byte-for-byte unchanged; the A–D report/counts and four-file output namespace were rechecked.

## Execution constraints

The C/D archives were downloaded to `data/msd_hf_pilot/C.tar.gz` and `D.tar.gz` (8,186,746,659 and 8,223,584,977 bytes). The user confirmed project-use permission. Treat archive contents as untrusted input and preserve all existing A/B artifacts. The workspace has no Git repository; work in place without commits. Do not preprocess the A–D audio cache or train C until the audit counts are reviewed.

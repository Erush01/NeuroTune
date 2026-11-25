# MSD A–D Multimodal Expansion Audit

**Status:** Approved on 2026-11-15 for implementation.  
**Scope:** Safely audit all four downloaded HF shards together to measure the A/B-to-A–D change in complete-modality candidate tracks and separate Last.fm train/test links. Do not build a feature cache or train a model in this phase.

## Motivation

The existing A/B pilot contains 6,967 complete-modality candidate tracks and has already been used to train the primary C model. The user approved downloading C and D to explore whether a larger catalog and more Last.fm supervision are available. A C/D-only scan would miss artist/song and official duplicate relationships across A/B and C/D, so the exact incremental catalog count requires a joint A–D scan.

The user has confirmed permission to use the downloaded archive source for this project. That project-specific authorization is distinct from independently verifying all upstream provenance or redistribution terms.

## Proposed audit

Generalize the existing A/B audit core to accept a named set of archive paths while preserving the existing two-archive `run()` behavior, defaults, output names, and safety tests. Add a dedicated A–D CLI that scans A, B, C, and D together, using the existing streaming tar reader and HDF5 bounds/link/dtype validation. Keep A/B output artifacts untouched and publish to a new namespace, `data/msd_hf_abcd_pilot/`, with `abcd_*` names and source-hash keys `A`, `B`, `C`, and `D`.

After joining local musiXmatch IDs, tagtraum genres, and official duplicate groups, resolve duplicates globally across all four shards using the established `(artist_id, song_id)` and official duplicate-list rules. Filter to positive BPM and complete lyrics/genre coverage. Scan the existing official Last.fm train and test archives against that combined candidate catalog, retain train/test labels separately, and remove their undirected overlap from train labels.

Compare the resulting A–D catalog and edge sets with the existing A/B catalog and train/test edge files. Record added, removed, and net eligible-track counts; added/removed/net train and test edge counts; and new-new, new-existing, and existing-existing link breakdowns. Include hashes for all four archives, feature/duplicate/Last.fm sources, A/B reference artifacts, and code.

## Boundaries and interpretation

- Do not extract tar members to the filesystem; continue the bounded streaming scan and untrusted-HDF5 validation.
- Do not overwrite or rename any existing A/B cache, audit, split, training result, checkpoint, or test artifact.
- Do not create the A–D audio cache or run training yet. This phase measures the metadata/auxiliary/label intersection. Temporal array validation remains a later cache-preprocessing gate.
- Keep Last.fm train and test edges separate; report the overlap removed from train. The existing A/B test cohort was previously inspected and must not be reused for a new performance claim.
- Treat all resulting counts as coverage/supervision feasibility, not evidence that a larger dataset will improve model quality.

## Success criteria

The A–D audit produces a separate report/catalog/train-edge/test-edge set, global duplicate resolution is deterministic, the report quantifies the change relative to A/B, and all existing A/B files remain untouched. Synthetic tests cover named archive provenance, cross-shard duplicates, edge partitioning/overlap, unsafe archive inputs, rollback, and output collision refusal.

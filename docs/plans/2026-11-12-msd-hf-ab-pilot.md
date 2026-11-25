# MSD Hugging Face A/B Archive Pilot

**Status:** A/B-only feasibility audit and cache generation complete; no model training performed.  
**Date:** 12 November 2026  
**Source:** Third-party Hugging Face dataset `RichardErkhov/The_Million_Song_Dataset`, files `A.tar.gz` and `B.tar.gz` (not represented as an official MSD-maintainer release).

## Downloaded files

| File | Bytes | Human-readable size | HDF5 files |
|---|---:|---:|---:|
| `data/msd_hf_pilot/A.tar.gz` | 8,251,685,808 | 7.69 GiB | 39,100 |
| `data/msd_hf_pilot/B.tar.gz` | 8,090,956,299 | 7.53 GiB | 38,265 |
| **Total** | **16,342,642,107** | **15.22 GiB** | **77,365** |

Both archives contain only `.h5` regular files. A full member-list scan found no absolute or parent-traversal paths and no non-regular members. A and B together contain all 10,000 track IDs in the current MSD 10K CSV (7,620 in A and 2,380 in B), plus additional tracks. All 939 IDs in the existing four-feature catalog occur in these two archives (709 in A, 230 in B).

Two representative HDF5 files, one from each archive and both matching the current four-feature catalog, were extracted into `data/msd_hf_pilot/samples/` and validated. Their HDF5 analysis track IDs match their member filenames; they contain positive tempo, pitch/timbre/loudness segment arrays and beat/bar/section arrays; the existing `preprocess_audio` implementation accepted both and produced finite `float32` arrays of shape `(1024, 28)`.

## Coverage and duplicate audit

Using the already-local full musiXmatch and tagtraum CD2C source files, the A/B archive scan found:

- 77,365 unique MSD HDF5 IDs in the A/B cohort.
- 6,967 tracks with a musiXmatch ID, CD2C genre ID, and finite positive HDF5 tempo before duplicate reconciliation.
- 6,028 of those are outside the existing 10K MSD subset; all 6,028 passed the existing temporal preprocessing validation in the pilot scan.
- Existing 939 four-feature IDs are retained within this A/B cohort.
- Duplicate reconciliation identified 5 repeated `(artist_id, song_id)` exclusions and 730 official-duplicate exclusions in the full A/B audio cohort; none of the 6,967 four-feature IDs were removed by those rules.

Thus the selected A/B cohort contains **6,967 all-input candidates**, including **6,028 new tracks** beyond the current 939-track multimodal catalog. This is exact for the downloaded A/B cohort and the current local annotation snapshots. It is not the final full-MSD count: downloading C–Z could reveal cross-archive duplicates, and the HF re-pack is not yet independently authenticated against an official archive checksum.

## Last.fm supervision overlap

The existing official Last.fm train/test per-track archives were queried for the 6,967 A/B all-input candidates. Similar-track links were normalized to unique undirected pairs, self-links were dropped, and both endpoints were required to be in the A/B feature catalog.

| Source archive | Within-catalog edges before cross-archive overlap removal | New–new edges | New–existing edges | Existing–existing edges |
|---|---:|---:|---:|---:|
| Last.fm train | 17,824 | 12,914 | 4,310 | 600 |
| Last.fm test | 1,940 | 1,740 | 200 | 0 |

The audit found 288 undirected edges present in both source archives. It removed those from train and retained the test copy, so the actual train edge file has **17,536** rows and the test edge file has **1,940** rows, with zero shared pairs. The table's train breakdown is the pre-overlap count; the exact retained training count is after removing the overlap.

The Last.fm test archive remains held out and must not be used for training or model selection. A future split/audit must preserve that source boundary, or explicitly define a new evaluation protocol before using these labels. Edges are observed behavioral links, not complete similarity judgments; absent links are not verified negatives.

## Interpretation and next gate

The A/B pilot demonstrates that this third-party archive can provide valid MSD-compatible HDF5 records and appears capable of expanding the four-input feature catalog by 6,028 tracks using local lyrics/genre tables. It also provides substantial Last.fm train/test edge overlap. This makes A/B a useful **pilot cohort**, not an unbiased sample of the complete million-song corpus; do not extrapolate its feature or label density to C–Z.

Before training the newly proposed architecture:

1. The user approved A/B as the pilot cohort; the separate duplicate-aware catalog audit is complete at `data/msd_hf_pilot/ab_catalog_audit.json`, with tracks and train/test edges in sibling CSVs. It reports 6,967 eligible tracks, 730 duplicate exclusions overall (none among eligible feature tracks), 17,536 retained train edges, 1,940 test edges, and 288 train/test edge overlaps removed from train.
2. The unnormalized first-30-second, 26-channel ragged cache has been generated at `data/msd_hf_pilot/ab_audio_30s_26ch.h5` (49,420,488 bytes) with metadata at `ab_audio_30s_26ch_meta.json`; it contains 6,967 tracks and 719,866 segment rows. It stores sorted IDs, flat features, and offsets, with no padding or train-fitted normalization. Validation confirmed all values finite, durations in `(0,30]`, sequence lengths 1–228 (median 103), matching audit/source hashes, and byte-exact preprocessing round-trips for one existing 10K track and one newly added track.
3. Keep Last.fm train links for training and reserve Last.fm test links for held-out evaluation; create an artist-disjoint split within permissible source partitions.
4. The A/B pilot uses the agreed `T x 26` contract (first 30 seconds; pitch 12, timbre 12, loudness, segment duration). This is intentionally separate from the legacy M1/M2 `(1024, 28)` full-track representation (25 continuous and 3 event-count channels); no prior cache was changed.

## Provenance limitations

The HF repository is under a personal account, has no README/dataset card or license statement, and was uploaded in 2024. Sample HDF5 compatibility and ID matches are verified, but the complete archive's provenance, legal permissions, equivalence to an official MSD bundle, and any official checksum match have not been verified. Keep the download as a research pilot; do not redistribute it. Verify licensing and official correspondence before relying on it for a final submission or sharing extracted data.

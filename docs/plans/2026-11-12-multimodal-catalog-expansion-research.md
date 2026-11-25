# Multimodal Catalog Expansion: Research Notes

**Accessed:** 2026-11-12  
**Scope:** Increase the number of tracks with all four inputs: MSD temporal HDF5 features, positive BPM, musiXmatch lyrics, and tagtraum CD2C genre; estimate available overlapping Last.fm positive edges.

## Current baseline

The project reports 10,000 MSD 10K-subset HDF5 tracks, 9,939 after its duplicate policy, and 939 tracks in the current all-input catalog. The M0 audit reports 2,350 MSD∩musiXmatch tracks, 943 after intersecting CD2C, and 939 after final validity/duplicate filtering. Its feature catalog contains 600 unique undirected Last.fm edges, with 416/6/11 within train/validation/test and 167 cross-partition edges excluded. Separately, the audio-only CV audit reports 8,209 unique undirected Last.fm edges within the duplicate-filtered 9,939-track catalog; these are not additional edges among the 939 multimodal tracks. The four-feature set is MSD IDs ∩ lyric IDs ∩ CD2C IDs ∩ valid-positive-tempo IDs, followed by duplicate policy, then distinct undirected Last.fm edges whose endpoints both lie in the set (`src/data/join_audit.py`, `results/data_audit.json`).

## Practical expansion routes

### 1. Obtain more MSD temporal HDF5 coverage (largest possible gain, highest cost)

The Million Song Dataset (MSD) project describes the full collection as one million tracks and makes available a 10,000-song subset; it distributes analysis features, not music audio. Official entry point: [MSD](https://millionsongdataset.com/). The project’s previous investigation estimated the full HDF5 collection at roughly 280 GB, found no official intermediate-size HDF5 package, and declined that acquisition. Treat full-catalog acquisition as conditional on confirming current links, storage, transfer and licensing with the official page; do not assume a mirror or a third-party repack is official.

If full MSD HDF5 becomes feasible, it is the only identified route that directly raises the hard ceiling imposed by available temporal audio. Before a bulk download, run a small sample and a source-ID audit against lyrics, CD2C, BPM, and Last.fm; source catalog size is not the same as usable four-way intersection. Requesting access from the MSD maintainers is reasonable if the official page’s current download path is inaccessible, but no public official intermediate package or guaranteed access grant was verified in this research.

### 2. Add external MSD-ID annotations, not merely larger unrelated catalogs (moderate effort)

The official MSD site lists companion datasets, including musiXmatch lyrics and Last.fm-derived resources. The relevant useful property is whether records carry MSD `track_id` identifiers, which permit deterministic joins. Larger annotation files only help if their track IDs extend beyond the copies already in use. Re-check and record official download/version provenance, ID coverage, and redistribution/use terms before replacing current snapshots.

- [MSD: musiXmatch lyrics](https://millionsongdataset.com/musixmatch/): use track-ID keyed lyric bags where available. More lyrics alone cannot add tracks absent from local temporal HDF5.
- [MSD: Last.fm dataset](https://millionsongdataset.com/lastfm/): determine whether a larger official similar-track graph is available and its exact scope/format. This can increase edge overlap/label density, but cannot increase the four-input track count by itself.
- [MSD: tagtraum genre annotations](https://millionsongdataset.com/genre/): compare the current CD2C release and its identifiers against any official updated release. A larger genre vocabulary does not help if it annotates no new MSD HDF5 IDs.

### 3. Improve coverage through matching/audit policy (low effort, bounded and must be conservative)

Check local source versions, track-ID normalization and duplicate handling. MSD IDs are the primary join key; avoid fuzzy title/artist joins unless manually adjudicated, since remasters, versions, and homonyms can create false matches. The current audit keeps one lexicographically smallest track ID for duplicate artist/song and official duplicate groups. Recompute intersections with each candidate source, and report incremental IDs before adopting it. Also inspect why each MSD track misses each feature (missing lyrics, CD2C, or invalid/missing tempo), rather than reporting only the final intersection.

CD2C is a derived genre classification and has its own coverage and use conditions. Consult the primary [tagtraum CD2C dataset page](https://www.tagtraum.com/genres/) and its included documentation for version, matching identifiers, and license before adding an alternate release. It is not safe to infer that the general tagtraum catalog has more matching MSD IDs than the current file without a join audit.

## Positive-edge overlap

The current project has 8,209 unique undirected observed links among 9,939 duplicate-filtered temporal tracks, while `results/data_audit.json` reports 600 retained unique undirected Last.fm links among the 939 four-feature tracks (416/6/11 within split, 167 cross-split excluded). The count of links added by any larger Last.fm graph depends on endpoint overlap after applying the same four-way track filter and duplicate policy. Preserve source edge direction/duplicates for provenance, then deduplicate to unordered pairs and exclude self-links as the audit does.

The official Last.fm/MSD resource’s precise availability and current distribution should be verified at the linked primary source before treating it as a larger edge source. If expanding edges, compare incremental endpoint IDs and edge pairs against the current `lastfm_similar` input, and verify whether the official resource is train/test split data rather than a single freely combinable graph. Do not merge held-out test links into training labels; keep split provenance and evaluation boundaries intact.

## Recommended order

1. **Run a coverage diagnostic now** on current local files: per-source unique ID counts; pairwise and cumulative intersections; BPM-valid intersection; four-way count before/after duplicate rules; and unique Last.fm edge counts within each set. This is the fastest way to identify the actual bottleneck and the current multimodal edge overlap.
2. **Audit official companion releases** (musiXmatch, CD2C, Last.fm) for newer or broader MSD-ID coverage. Only ingest if a measured join adds usable IDs/edges; document version and terms.
3. **Revisit full MSD HDF5 acquisition only if resources permit.** It is potentially the largest catalog increase but entails ~280 GB according to prior project research, storage/transfer planning, and an explicit access/license check. No intermediate official HDF5 tier has been established.
4. **Keep an audio-only experiment as a separate cohort** if full multimodal coverage remains small. It increases the modeling sample but does not answer the four-modality experiment’s catalog question.

## What cannot be concluded without local audit

- Exact current number of tracks with all four modalities after the actual current files and duplicate policy (the project’s documented value is 939, but rerun the audit when source snapshots change).
- Exact number of Last.fm edges with both endpoints in that four-way catalog, or incremental tracks/edges from candidate companion releases.
- Whether official companion archives currently contain records beyond local snapshots; webpage availability and nominal source size do not establish ID-level overlap.
- Whether additional matched tracks have valid, usable BPM and temporal arrays, rather than merely IDs in annotation tables.
- Whether current public links, access conditions, and licenses allow the intended download/use/redistribution. Confirm on the source pages and archive documentation at acquisition time.

## Primary-source links

- MSD official portal and downloads: https://millionsongdataset.com/
- MSD musiXmatch page: https://millionsongdataset.com/musixmatch/
- MSD Last.fm page: https://millionsongdataset.com/lastfm/
- MSD genre/tagtraum page: https://millionsongdataset.com/genre/
- tagtraum CD2C primary page: https://www.tagtraum.com/genres/

The source links above are starting points for official acquisition and documentation. This report does not assert a live 2026 download was tested; re-verify page availability, archive version, terms, and exact ID coverage before acquisition.

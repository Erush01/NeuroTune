# M0 data inputs

NeuroTune does not redistribute source datasets, lyrics, or derived lyric counts. Obtain each dataset from its official source and follow the source's access terms and license. Cite the source and record the exact release/file hashes for any report. In particular, the musiXmatch supplement contains count-based bag-of-words data, not ordered/full lyrics.

The M0 script consumes local extracts in `data/processed/`:

| File | Required columns | Meaning |
|---|---|---|
| `msd_tracks.csv` | `track_id,tempo,artist_id,song_id,artist_name,title` | One MSD track per row, extracted from metadata plus tempo. `title` may instead be `song_name`; the ID must be the canonical `TR...` key. |
| `mxm_counts.csv` or `mxm_counts/` | CSV columns `track_id,word_id,count`; or extracted official `.txt` train/test files | Sparse musiXmatch counts. Long-form rows are summed by `(track_id,word_id)`; official text files are parsed from `#` comments, `%` vocabulary, and `track_id,mxm_id,index:count,...` rows. Word IDs are 1-based indices into the fixed vocabulary. |
| `tagtraum_genres.csv` | `track_id,genre` | One primary CD2C genre per track. Conflicting labels for the same ID stop the audit. |
| `lastfm_similar/` or `lastfm_similar.tsv` | Official JSON files have `track_id` and `similars: [[target_track_id, score], ...]`; normalized TSV has `track_id,similar_track_id,score` | The parser accepts the official Last.fm JSON-per-track release directory or a normalized TSV. `source_track_id,target_track_id` are accepted TSV aliases. Missing scores remain unknown; absence of an edge is not a negative label. |
| `msd_duplicates.txt` (optional) | Official MSD duplicate-group format | Duplicate groups are considered when deciding which releases to exclude. If omitted, the audit still checks repeated `(artist_id,song_id)` pairs. |

Inputs may be comma-, tab-, or semicolon-delimited. Do not join Spotify-ID variants. Normalize the canonical MSD track identifier before export. For repeated non-empty `(artist_id,song_id)` keys or official duplicate groups, the audit deterministically retains the lexicographically smallest track ID and excludes other releases from the full-feature catalog. It reports source-specific duplicate counts and their union.

Run `python -m scripts.check_data`. Missing sources produce an explicit `blocked_missing_sources` JSON report with unknown counts (`null`), not invented zeros. With all four inputs present, the audit writes sequential overlap counts, feature coverage/quality, retained unique undirected observed edges, degree/artist/genre/duplicate summaries, SHA-256 hashes, and seeded artist-disjoint 80/10/10 split edge counts to `results/data_audit.json`.

## Source and license references

- Million Song Dataset: <https://millionsongdataset.com/> (metadata/derived features; not audio recordings)
- musiXmatch MSD supplement: <https://datadryad.org/dataset/doi:10.15146/R3467H> (count-based lyrics representation; consult the dataset's terms)
- tagtraum CD2C: <https://www.tagtraum.com/msd_genre_datasets.html> (genre annotation; upstream annotations partly derive from Last.fm)
- Last.fm MSD supplement: <https://millionsongdataset.com/lastfm/> (research only, strictly non-commercial per its official page)

Do not treat “freely available” as a blanket commercial license: check component-specific terms for each dataset. The MSD metadata/tempo sources, Last.fm and musiXmatch have distinct terms; tagtraum also labels the genre set research-only/non-commercial. Obtain instructor approval for course use and retain dataset citations.

## Export caveats for official files

- **MSD tempo is not in the metadata-only `track_metadata.db` or `msd_summary_file.h5`.** The official FAQ puts tempo in the HDF5 `analysis/songs` table. `src/data/extract_msd.py` extracts tempo and metadata from an unpacked subset/full MSD directory into `msd_tracks.csv`, then joins by canonical MSD `track_id`. The metadata SQLite database alone cannot supply this audit's BPM field.
- Official musiXmatch text files have `#` comments, a `%` vocabulary line, then rows `MSD_track_id,MXM_track_id,index:count,...` with 1-based vocabulary indices. Convert these to `mxm_counts.csv` rows using `track_id,word_id,count`; preserve the vocabulary mapping for later features. Do not export full lyrics.
- The official Last.fm release contains per-track JSON in train/test zip folders. Each JSON record can contain `track_id` and `similars` as `[target_track_id, score]` pairs. The parser reads an extracted directory of these files directly; normalized TSV is also supported. Its page warns of known song/track matching issues and states research-only, strictly non-commercial use.
- tagtraum's release is distributed as a `.cls.zip`. The CD2C file has comment lines beginning with `#`, followed by tab-separated `track_id`, primary genre, and optional minority genre. `src/data/extract_tagtraum.py` exports only the primary label to `tagtraum_genres.csv`.
- The official MSD duplicates list uses `%` group-heading lines and one `TR...` track ID per line. The audit can read it as `msd_duplicates.txt`; for the 10k subset used here, none of its listed tracks overlapped the subset.

Official format references: [MSD Getting the Dataset](https://millionsongdataset.com/pages/getting-dataset/), [MSD FAQ](https://millionsongdataset.com/faq/), [Last.fm Dataset](https://millionsongdataset.com/lastfm/), [musiXmatch Dataset](https://millionsongdataset.com/musixmatch/), [tagtraum CD2C](https://www.tagtraum.com/msd_genre_datasets.html).

Confirm with the CMP684 instructor whether Python is acceptable for submitted source code; the public assignment instructions mention MATLAB.

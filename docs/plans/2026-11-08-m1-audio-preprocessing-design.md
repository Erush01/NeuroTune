# M1 MSD Temporal Audio Preprocessing

## Status

Approved design for the existing 939-track common multimodal catalog.

## Goal

Prepare the MSD 10K subset's temporal analysis features for the audio-first
Siamese model without changing the existing scalar `msd_tracks.csv` contract
or fitting normalization on validation/test data.

## Observed MSD HDF5 structure

Temporal arrays are group-level datasets under `analysis/`, not fields in the
compound `analysis/songs` record:

- `segments_pitches`: `(T, 12)`
- `segments_timbre`: `(T, 12)`
- `segments_loudness_max`: `(T,)`
- `segments_start`: `(T,)`
- `beats_start`, `bars_start`, `sections_start`: separate variable-length
  event-time arrays

All 10,000 locally available HDF5 files contain these datasets, and the four
segment arrays have matching lengths. Segment count percentiles are 4 / 563 /
766.5 / 1,029 / 1,375 / 1,726 / 2,705 / 8,419 at p0 / p25 / p50 / p75 / p90 /
p95 / p99 / max. A 100-file value sample had no non-finite values; the
preprocessor will validate values and timing for every selected track.
`segments_start` must be strictly increasing because NumPy's interpolation
contract requires increasing sample coordinates.

## Chosen representation

Process the 939 track IDs in the existing artist-disjoint audit so audio-only
and multimodal experiments use the same catalog and splits. For each track,
produce a deterministic `(1024, 28)` sequence on a uniform grid spanning its
duration. Interpolate the continuous segment features at grid times derived
from `segments_start`:

| Channels | Source | Treatment |
|---|---|---|
| 0–11 | `segments_pitches` | Linear interpolation per pitch channel |
| 12–23 | `segments_timbre` | Linear interpolation per timbre channel |
| 24 | `segments_loudness_max` | Linear interpolation |
| 25 | `beats_start` | Count events per grid bin |
| 26 | `bars_start` | Count events per grid bin |
| 27 | `sections_start` | Count events per grid bin |

Event counts, rather than binary flags, preserve multiple events mapped to one
bin. In the current 939-track catalog, multiple beat events share a bin for 37
tracks (maximum two per bin); bar collisions occur for two tracks (also maximum
two); no section collisions were observed.

Fit channel-wise mean and standard deviation for the 25 continuous channels
using only the 756 training tracks and their 1,024 time steps. Apply those
statistics unchanged to all 939 tracks. Leave event-count channels unscaled.
If a training-channel standard deviation is zero, use scale 1 so the channel
remains finite. Record the fit statistics and training-ID provenance.

## Cache contract

- Write one consolidated HDF5 cache at `data/processed/audio_sequences.h5`.
- Store sorted `track_ids`, a float32 `sequences` array of shape `(939, 1024,
  28)`, and the channel-name list.
- Store preprocessing configuration, source HDF5 hashes, split provenance,
  and train-only normalization statistics in
  `data/processed/audio_sequences_meta.json`.
- Keep the current scalar `msd_tracks.csv` unchanged. The cache can be rebuilt
  if a later data audit expands the common catalog.

The HDF5 and JSON files are each staged before replacement, but publishing two
separate files cannot be atomic as a pair. A filesystem failure between the
two final replacements can leave mismatched artifacts; after a reported
publication failure, rebuild and verify the cache/metadata pair before use.

## Validation

- Unit-test synthetic HDF5 records for channel order, output shape, irregular
  segment-time interpolation, event-count binning, and repeatability.
- Reject missing required arrays, non-finite data, non-positive duration,
  non-increasing/out-of-range times, mismatched segment lengths, or an HDF5 track ID
  that does not match the requested catalog ID.
- Verify train-only normalization by changing validation/test arrays while
  confirming fitted statistics remain unchanged.
- Run a real-cache check: 939 unique IDs, `(939, 1024, 28)` shape, finite values,
  existing split membership, and consistent metadata/statistics.

## Scope limits

This phase prepares temporal inputs only. It does not implement the 1D-CNN,
Siamese training, ablations, or a new split. It does not add Taste Profile
co-listening edges to the Last.fm relevance labels. The existing 11-edge test
graph remains a major evaluation limitation.

## Sources

- Million Song Dataset: https://millionsongdataset.com/
- MSD code and HDF5 access: https://millionsongdataset.com/pages/code/
- NumPy interpolation contract: https://numpy.org/doc/stable/reference/generated/numpy.interp.html
- Local sample inspected at `data/msd_subset_raw/MillionSongSubset/`; array
  shapes and coverage were also checked across the 10,000 local HDF5 files.

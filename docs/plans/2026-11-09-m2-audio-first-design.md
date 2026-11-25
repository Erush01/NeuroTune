# M2 Audio-First Siamese Model Design

## Status

Accepted by the user on 2026-11-09. This design continues from the completed M1
temporal preprocessing milestone.

## Context and goal

The M1 cache contains 939 tracks from the official 10K MSD subset, with an
artist-disjoint 756/94/89 train/validation/test split. Each cached track is a
float32 sequence shaped `(1024, 28)`: 12 pitch, 12 timbre, one loudness, and
beat/bar/section event-count channels. Continuous audio channels use
train-fitted normalization; event counts remain unchanged.

The final course model is audio-first. Temporal MSD features are the main input,
with BPM, musiXmatch lyric-count features, and tagtraum genre as auxiliary
modalities. Existing B0 and N0 models remain metadata-only exploratory baselines
and must not be presented as the final architecture.

The current observed Last.fm graph has 416 train, 6 validation, and 11 test
positive edges. The prototype can test the end-to-end training pipeline on this
scope, but cannot support a strong comparative, significance, or generalization
claim. Unobserved pairs are sampled candidates, not verified negatives.

## Approved model

### Inputs and encoders

- Align audio cache rows and auxiliary feature rows by track ID, never by
  assuming that different sources share row order.
- Use the already normalized M1 audio sequence as `(N, 1024, 28)` and transpose
  it to `(N, 28, 1024)` for `Conv1d`.
- Audio branch: three lightweight convolution blocks—`Conv1d(28,32,kernel=7,
  stride=2,padding=3)`, `Conv1d(32,64,kernel=5,stride=2,padding=2)`, and
  `Conv1d(64,64,kernel=3,stride=2,padding=1)`, each followed by ReLU. Adaptive
  average pooling to 16 ordered temporal bins, flattening, and
  `Linear(64*16,32) + ReLU` produce the 32-D audio representation. Coarse
  ordered bins retain more temporal structure than global averaging without
  flattening all 1,024 steps into a large dense layer.
- Reuse the existing train-fitted auxiliary transforms: BPM `1→8`, lyrics
  `5000→32`, genre `G→16`, each with ReLU. Fit BPM statistics, lyric IDF, and
  genre vocabulary from training IDs only.
- Concatenate the 32-D audio representation with the 8-D BPM, 32-D lyrics, and
  16-D genre representations. Project the resulting 88-D vector to 64-D and
  L2-normalize it.
- Use the same encoder weights on both sides of every pair. Add a separate
  audio-first model class; do not change the existing N0 `SiameseEncoder` or
  its saved results.

### Training and evaluation

- Reuse the existing partition-local pair pools, rejection-based negative
  sampler, and contrastive margin loss. Observed within-partition Last.fm edges
  are positive pairs; sampled unobserved candidates receive label zero without
  being described as verified negatives.
- Keep the N0 protocol for controlled descriptive comparison: one negative per
  positive per epoch, the existing 50/50 same-genre/random policy, seeds
  684/685/686, 50 epochs, batch size 64, Adam at 0.001, and margin 1.0.
- Resample training negatives each epoch and keep validation negatives fixed
  per seed. Record validation loss but do not use the six-edge validation graph
  for checkpoint selection or tuning. Use the fixed 50th epoch.
- Do not pass test pairs or relevance labels into training. Use the full 89-song
  test gallery for one final Top-10 evaluation per seed with the existing
  retrieval metrics. Compare B0/N0 descriptively only.

## Outputs and provenance

Add a separate audio-first runner and save:

- `results/m2_audio_experiment.json` — settings, per-seed curves and metrics,
  split counts, low-edge warning, fit provenance, input/code hashes, and runtime
  versions.
- `results/m2_audio_training_curves.csv` — per-seed train and validation loss.
- `results/m2_audio_sampled_pairs.csv` — actual sampled train pairs per epoch
  and fixed validation pairs.
- `results/checkpoints/m2_audio_seed_{seed}.pt` — fixed-epoch model state and
  its seed/configuration/provenance.

Hash the M1 cache and metadata, audit and auxiliary source data, relevant code,
and runtime versions. Preserve the existing B0/N0 artifacts and data tables.

## Alternatives considered

- **Audio-only first:** rejected for this milestone because the accepted final
  architecture includes BPM, lyrics, and genre. It can be a later ablation if
  the project scope allows it.
- **Flatten the complete sequence into a dense encoder:** rejected because it
  would create a larger parameter-heavy input layer and discard the intended
  local temporal inductive bias.
- **Recurrent or attention-heavy temporal encoder:** deferred; a small 1D-CNN
  is easier to validate and more proportionate to the 939-track catalog and
  sparse pair graph.

## Risks and limits

- Only 416 training, 6 validation, and 11 test positive edges are available;
  training may overfit and metrics are highly unstable.
- Missing Last.fm links are not confirmed negatives.
- Results apply only to the current 10K subset catalog. Do not claim
  significance or generalization beyond it.
- The M1 cache is 89 MB on disk and roughly 108 MB as an uncompressed float32
  tensor, which is appropriate for the planned CPU prototype.
- This workspace is not a Git repository; design and implementation artifacts
  remain local and are not committed.

# M2 N0 Implementation Plan

## Task 1: Train-fitted modality inputs

**Description:** Expose separate BPM, lyric TF-IDF, and genre tensors while
reusing the same train-only fitting logic as B0. Public helper contract:
`build_train_fitted_modalities(tracks, lyrics, genres, train_ids, song_ids)`
returns `(modalities, fit_metadata)`, where `modalities` has `bpm` shaped
`(N,1)`, `lyrics` shaped `(N,5000)`, and `genre` shaped `(N,G)` in `song_ids`
order. BPM is train-z-scored; lyric counts are train-IDF weighted and
per-song-L2-normalized; genres use the sorted train vocabulary.

**Acceptance criteria:**
- [x] BPM mean/std, lyric IDF, and genre vocabulary use only training IDs.
- [x] Output tensors preserve modality dimensions and unseen genre policy.
- [x] Existing B0 vectors and saved metrics remain unchanged.

**Verification:** Unit tests for exact train-only transformations and full suite.
**Dependencies:** None.

## Task 2: Pair sampler and Siamese encoder

**Description:** Build partition-contained positive/negative pairs, the shared
encoder, and the contrastive loss. Pair utility contract:
`build_pair_pools(edges, partition_ids, genres)` returns deduplicated positive
pairs, partition IDs, genre groups, and combinatorial eligible-negative counts;
it must not materialize all O(N²) candidate pairs. `sample_pairs(pools, seed,
negatives_per_positive=1, same_genre_fraction=0.5)` rejection-samples unique
unordered pairs and returns `(left_id, right_id, label)` rows.

**Acceptance criteria:**
- [x] Only train-internal positives enter training; no pair crosses partitions.
- [x] Negatives follow the fixed 50/50 policy, are deterministic by seed, and
  filter known positives, duplicates, and self-pairs.
- [x] For odd negative counts, same-genre receives floor(n/2) and random gets
  the remainder; the report records this policy.
- [x] Candidate negatives are sampled without building the full quadratic pair
  universe.
- [x] Encoder outputs unit 64-D embeddings and uses shared weights for both
  sides; contrastive loss matches the specified formula.

**Verification:** Pair isolation/sampling tests, output shape/normalization,
shared-weight symmetry, loss-value tests, CPU backward-pass smoke test.
**Dependencies:** Task 1, PyTorch dependency approval.

## Task 3: Fixed exploratory training core

**Description:** Add `src/train.py::train_seed` to train one shared encoder from
partition-specific pair pools, with fresh training negatives per epoch and one
fixed validation-negative sample per seed. Record train/validation loss per
epoch; do not select epochs from validation.

**Acceptance criteria:**
- [x] Training updates only from train rows; validation loss is computed under
  no-grad and test data is absent from the training API.
- [x] Negatives are resampled deterministically each epoch; validation negatives
  remain fixed for that seed.
- [x] Fixed seed, epoch count, margin, batch size, and optimizer are recorded.
- [x] A synthetic CPU test verifies deterministic curves, finite losses, and a
  parameter update.
- [x] `train_seed` returns the actual sampled train pair rows for each epoch
  and the fixed validation pair rows for the seed.

**Verification:** Focused unit test, including a backward/update smoke test.
**Dependencies:** Tasks 1–2.

## Task 4: N0 experiment runner and report

**Description:** Integrate input parsing, train-only transforms, pair pools,
three-seed fixed training, final test-gallery evaluation, checkpoints, curves,
and a report under `results/`.

**Acceptance criteria:**
- [x] Validation/test pairs are never used for gradient updates.
- [x] Report includes per-seed metrics, mean/spread, source hashes, split sizes,
  negative policy, hyperparameters, and low-edge warning.
- [x] The CLI defaults to the fixed 50-epoch, seeds 684/685/686 protocol; any
  smoke-test override is written into its output report.
- [x] Each final checkpoint stores model weights and its seed/configuration.
- [x] Report and checkpoint include actual sampled pair IDs, code hashes, and
  runtime dependency versions.

**Verification:** Synthetic CLI test, real 3-seed CPU training, full tests,
compileall, generated-report consistency checks.
**Dependencies:** Tasks 1–3.

## Checkpoint

- [x] All unit tests pass.
- [x] Real training completes on the current 939-track catalog.
- [x] Three seeded checkpoints and per-epoch train/validation curves are saved.
- [x] A pair-log CSV records each sampled train pair per epoch and fixed
  validation pair IDs per seed.
- [x] Test evaluation uses only the 89-song gallery and 11 observed internal
  edges, without test-tuned choices.
- [x] Results are explicitly marked exploratory.

# NeuroTune: Shallow Multimodal Song Embeddings for Similar-Music Retrieval

**Course:** CMP684 Neural Networks, final project (2026)
**Status:** Proposed architecture and experiment protocol; datasets and feasibility gates NOT yet executed.
**Research question:** Can a shallow shared-weight neural encoder trained on observed Last.fm similar-track relationships improve top-K music retrieval over handcrafted distances when using BPM, lyric term counts, and genre?

## 1. Project guardrails

- One learned similarity model: a shared-weight shallow (Siamese) feed-forward encoder. No CNNs, Transformers, pretrained text encoders, large language models, or secondary predictive recommenders.
- Offline metadata retrieval, not raw waveform analysis. The original Million Song Dataset (MSD) contains derived audio/tempo features rather than audio recordings.
- Do not promise semantic understanding of lyrics: the MSD/musiXmatch supplement distributes count-based bag-of-words with a fixed 5,000-word vocabulary, not full ordered lyrics.
- Do not claim the complete four-way overlap of datasets without measuring it. Prior published work reported a substantial MSD + musiXmatch + Last.fm intersection; tagtraum CD2C is a further filter.
- Final project's language and code requirements: the CMP684 public instructions specify a 6-page IEEE two-column English report, both PDF and DOCX, and refer to MATLAB codes. Confirm with instructor before using Python as the primary submitted source. Final submission deadline stated as Dec 31, 2026.

## 2. Primary data sources and join

| Source | Required fields | Purpose | Notes |
|---|---|---|---|
| MSD | `track_id`, `tempo`, artist and song IDs/names | BPM and catalog | Source may require an extracted metadata table from official MSD HDF5/summary. Do not accidentally join incompatible Spotify-ID variants. |
| musiXmatch MSD supplement | MSD `track_id`, 5,000-word sparse counts | Lyrics | Lyrics are bags of words, not sentences; perform TF-IDF fitting on TRAIN songs only. |
| tagtraum CD2C | MSD track ID, one genre annotation | Genre | Unambiguous primary genre mapping, not automatically multi-label. Its upstream annotations partly derive from Last.fm; disclose dependence. |
| Last.fm MSD supplement | MSD track ID and scored similar-track relationships | Similarity supervision and test relevance | Not all pairs are labeled. Absence of an edge is not proof of dissimilarity. |

Canonical join key: MSD track identifier (`TR...`), after explicit format checks. Deduplicate tracks/alternate releases as far as possible using the official MSD duplicates list and `(artist_id, song_id)` if available.

**Gate 1: data audit, before any model training.** Produce `results/data_audit.json` reporting:
- Track counts for each source, `MSD ∩ musiXmatch`, then `∩ CD2C`, then tracks covered by Last.fm.
- Number of retained *positive edges* (both endpoints have full features), distribution of positive degree, artist counts, duplicate counts, genre counts, lyrics coverage and BPM quality.
- Coverage by proposed train/validation/test partitions, specifically positive edges usable in each partition.
- A deterministic predeclared subset policy; never claim a target 10,000-20,000 catalog if its overlap/edge density is insufficient.

Start with the official 10k MSD subset to smoke-test parsers only, then extend to the verified full-feature intersection. Sampling 10k tracks at random can destroy most similarity links; build subsets by seeded expansion from tracks with observed similarity edges, with sampling rules frozen BEFORE test evaluation.

## 3. Input representations

- BPM: scalar from MSD; use a train-fitted standardization, e.g. transformed log-tempo (with an explicit policy for invalid/missing/zero BPM). A 70 vs 140 BPM half/double tempo ambiguity is a known issue; compare raw standardized BPM versus a tempo-equivalence feature as an ablation, not silent preprocessing.
- Lyrics: sparse 5,000-dim bag-of-words counts, transformed to TF-IDF using IDF fit on train songs only and optionally normalized. A 5,000 -> 32 linear/ReLU encoder is lightweight and trainable; alternatively reduce vocabulary only from the training data.
- Genre: CD2C primary genre one-hot, approximately 15 categories; check actual labels. Use a learned 15 -> 16 projection. Unseen/ambiguous/missing genres must have a documented policy.
- BPM encoder: 1 -> 8 features.

Suggested single shared encoder (starting values, to be tuned exclusively on validation):

```
BPM (1)                 -> Linear(1,8)     -> activation -> 8
Lyrics TF-IDF (5000)    -> Linear(5000,32) -> activation -> 32
Genre one-hot (~15)     -> Linear(15,16)   -> activation -> 16
                                          concatenate -> 56
                                    Linear(56,64) -> 64
                                    L2 normalization -> song embedding
```

The BPM/lyrics/genre branches and fusion layer all belong to ONE encoder, with identical weights reused for both songs in a training pair. The combined depth is small, but this is still a learned multilayer architecture: seek course instructor approval for its use in a non-deep-learning course. A stricter one-hidden-layer fallback is concatenation of fixed modality vectors -> Linear(input,64) -> nonlinearity -> Linear(64,32), trained with the same pair loss.

## 4. Training objective and pair generation

For a song `i`, `z_i = normalize(encoder(x_i))`. Similarity is cosine dot product `s(i,j) = dot(z_i,z_j)`.

Training batches contain observed Last.fm similar-track edges as positive pairs and sampled candidate negatives. Recommended starter loss is a contrastive margin objective:

```
D = ||z_i - z_j||_2
L = y * D^2 + (1-y) * max(0, margin-D)^2
```

where `y=1` is an observed similar pair; `y=0` represents a *sampled/unobserved candidate*, not verified dissimilarity. Use a fixed negative sampling policy, e.g. 50% random and 50% same-genre hard negatives, filtering known positive edges and duplicates. Compare at least one alternative negative strategy in an ablation. Track potential false-negative bias.

Training need not enumerate N^2 pairs. Resample negatives each epoch; train on O(number of observed edges + sampled negatives). Save seeds, source hashes, sampled-pair IDs, hyperparameters and model states.

## 5. No leakage: splits and evaluation

- Partition track catalog and prevent a song from appearing on both sides of train/test. Prefer artist-disjoint splits IF enough observed test-positive edges survive; report counts and fall back to a documented song-disjoint protocol if needed.
- Remove training edges that touch held-out validation/test track IDs. Do not generate negative or positive pairs crossing the partitions.
- Hold out the test query catalog; evaluate retrieval within the held-out set (or a clearly fixed held-out gallery) so the test sim graph is never used for training.
- Fit TF-IDF, feature scaling, and category encodings using training examples only. Validation may select thresholds and model hyperparameters, test may not.
- Retrieval ranking must consider the full eligible candidate gallery, not only pre-sampled negatives. Exclude self, exact duplicates and optionally the same recording.
- Last.fm edges are incomplete proxy relevance, so present retrieval metrics as *agreement with observed Last.fm similar-track links*, not human-validated absolute musical quality.
- As an optional qualitative validity check, inspect held-out human-rated Dim-Sim triplets where IDs overlap, without optimizing on them.

Metrics: Recall@10, HitRate@10 (or Precision@10 with caveat), MRR@10 and NDCG@10. Report query count, number of eligible test positives, standard deviations or bootstrap confidence intervals over queries. Include a per-artist analysis if labels support it.

## 6. Baselines and ablations

**Baseline B0:** standardized BPM, lyrics TF-IDF and genre one-hot concatenated with predeclared weights; normalized weighted cosine (no trainable parameters).
**Baseline B1:** handcrafted independent weighted similarities for BPM, genre overlap, lyric TF-IDF cosine. Tune weights on validation only.
**Neural N0:** trainable shared-weight encoder with all 3 modalities, contrastive supervision.
**Ablation N1:** BPM + Genre.
**Ablation N2:** BPM + Lyrics.
**Ablation N3:** Lyrics + Genre, optionally (if time permits).
**Ablation N4:** change negative sampling or learned embedding dimensions (optional).

Train at least three random seeds for final neural variants where feasible, and report mean/spread. Do not claim meaningful gain without matched candidate pools and the same relevance labels across baselines.

## 7. Retrieval and scale

At inference, encode each song ONCE and cache a dense row-normalized matrix `E` with shape `(N,64)` in `float32`; `E @ E[query]` returns cosine scores for the entire catalog. Find top K using `argpartition`, then sort K. This is O(N*64) per query but extremely efficient at tens of thousands of songs. ANN (FAISS/HNSW) is optional only if catalog or latency needs justify it; indexing is a search accelerator, not a second learned model.

No per-candidate neural network call is necessary. For a new query song, compute its embedding once, then reuse the same vector search.

## 8. Repository layout

```
neurotune/
  README.md
  pyproject.toml
  configs/default.yaml
  data/README.md               # data access/licenses; no raw copyrighted lyrics
  src/data/download.py
  src/data/parse_msd.py
  src/data/parse_mxm.py
  src/data/parse_tagtraum.py
  src/data/parse_lastfm.py
  src/data/join_audit.py
  src/data/splits.py
  src/features.py
  src/model.py
  src/pairs.py
  src/train.py
  src/evaluate.py
  src/retrieve.py
  scripts/check_data.py
  scripts/run_experiments.py
  scripts/build_report_tables.py
  tests/test_joins.py
  tests/test_leakage.py
  tests/test_encoder_shapes.py
  tests/test_symmetry.py
  tests/test_topk.py
  results/data_audit.json
  results/experiments.csv
  report/                         # IEEE 2-column, 6-page report
```

## 9. Milestones and acceptance gates

**M0 Data feasibility**: deterministic song-ID joins, overlap/edge report, dataset license/access notes, duplicates audit. STOP if there are insufficient usable positive edges in an artist/song-disjoint test split; revise dataset or task before building a neural net.

**M1 Baselines**: feature construction, consistent train/val/test graph split, full-gallery top-K retrieval and metrics for at least one predeclared classical baseline.

**M2 Neural encoder**: Siamese embedding encoder, contrastive learning, reproducible checkpoint, passing shape/symmetry/no-leakage tests; show learning curves with train/val separation.

**M3 Controlled experiments**: all-feature vs no-lyrics vs no-genre ablation, same test gallery for every run, >=3 random seeds if feasible, bootstrap CI/variance and informative failure cases.

**M4 Deployment demonstration**: typed track lookup -> top-10 with artist/title, score, BPM and genre. A web interface is optional; CLI and recorded examples suffice for the course.

**M5 Academic report**: no unsupported claim of novelty; explain source/license limitations, incomplete similarity graph, half/double BPM, TF-IDF non-semantic lyric representation, negative sampling bias, and train/test distribution. Generate all tables from saved experiment logs, not manual transcriptions.

## 10. High-priority risks

1. Four-way dataset intersection and density of similarity edges may be much smaller than expected; quantify FIRST.
2. Last.fm similar-track labels may reflect listener behavior, popularity and tagging rather than only audio/lyrical similarity. This is a proxy objective.
3. tagtraum annotations are partly derived from Last.fm metadata. Using a separate file does not guarantee independent underlying annotation provenance. Measure genre-only retrieval as a baseline.
4. Train/test leakage through duplicate tracks, related releases, test-tuned TF-IDF and graph edges is a major risk.
5. The assignment page mentions MATLAB for final; do not silently assume Python submission is accepted.

## 11. Primary references

- CMP684 course requirements: https://web.cs.hacettepe.edu.tr/~onderefe/cmp684/
- Million Song Dataset: https://millionsongdataset.com/
- tagtraum genre annotations: https://www.tagtraum.com/msd_genre_datasets.html
- musiXmatch MSD lyric word counts (Dryad hosting): https://datadryad.org/dataset/doi:10.15146/R3467H
- Human-annotated similarity triplets (optional evaluation): https://jongpillee.github.io/multi-dim-music-sim/

**Next action for implementation agent:** Implement M0 FIRST: all four metadata parsers (or consume clearly documented official extracted tables), song-ID normalization, deduplication, join counts, and retained positive-edge graph statistics. Report the measured numbers; do not begin neural training until the overlap is credible.

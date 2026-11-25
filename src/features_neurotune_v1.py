"""Train-only feature transforms for the isolated NeuroTune v1 experiment."""

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import h5py
import numpy as np

from .data.parse_mxm import parse_mxm
from .data.table import read_rows


CHANNEL_COUNT = 26
VOCABULARY_SIZE = 5000
CODE_PATHS = (
    "src/features_neurotune_v1.py",
    "src/features.py",
    "src/data/parse_mxm.py",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value):
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                            allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_ids(train_ids, song_ids=None):
    train_ids = list(train_ids)
    if not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("train_ids must be non-empty and unique")
    if song_ids is not None:
        song_ids = list(song_ids)
        if len(set(song_ids)) != len(song_ids):
            raise ValueError("song_ids must be unique")
        if set(train_ids) - set(song_ids):
            raise ValueError("all train_ids must be present in song_ids")
    return train_ids, song_ids


def _cache_index(h5):
    required = ("features", "offsets", "track_ids")
    if any(key not in h5 for key in required):
        raise ValueError("audio cache requires features, offsets, and track_ids")
    features = h5["features"]
    offsets = np.asarray(h5["offsets"])
    ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value)
           for value in h5["track_ids"][:]]
    if features.ndim != 2 or features.shape[1] != CHANNEL_COUNT or features.dtype != np.float32:
        raise ValueError("audio cache features must be float32 with shape (rows, 26)")
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("audio cache track_ids must be unique and sorted")
    if (offsets.dtype != np.dtype(np.int64) or offsets.ndim != 1
            or len(offsets) != len(ids) + 1 or offsets[0] != 0):
        raise ValueError("audio cache offsets must be int64, length N+1, and start at zero")
    if np.any(np.diff(offsets) <= 0) or offsets[-1] != features.shape[0]:
        raise ValueError("audio cache offsets must be strictly increasing and cover all features")
    return features, offsets, {track_id: i for i, track_id in enumerate(ids)}


def fit_audio_stats(cache_path, train_ids):
    """Fit population channel mean/std over real rows of training tracks only."""
    train_ids, _ = _validate_ids(train_ids)
    count = 0
    mean = np.zeros(CHANNEL_COUNT, dtype=np.float64)
    m2 = np.zeros(CHANNEL_COUNT, dtype=np.float64)
    with h5py.File(cache_path, "r") as h5:
        features, offsets, index = _cache_index(h5)
        missing = sorted(set(train_ids) - index.keys())
        if missing:
            raise ValueError(f"train IDs missing from audio cache: {missing[:3]}")
        for track_id in train_ids:
            row = index[track_id]
            block = np.asarray(features[int(offsets[row]):int(offsets[row + 1])], dtype=np.float64)
            if block.size == 0 or not np.isfinite(block).all():
                raise ValueError(f"invalid audio rows for track {track_id!r}")
            block_count = len(block)
            block_mean = block.mean(axis=0, dtype=np.float64)
            centered = block - block_mean
            block_m2 = np.square(centered).sum(axis=0, dtype=np.float64)
            combined_count = count + block_count
            delta = block_mean - mean
            m2 += block_m2 + np.square(delta) * (count * block_count / combined_count)
            mean += delta * (block_count / combined_count)
            count = combined_count
    std = np.sqrt(m2 / count)
    std[std == 0] = 1.0
    return {"mean": mean, "std": std, "row_count": count}


def _d4_cache_metadata(h5):
    """Validate and return only the small per-track D4 cache index."""
    required = ("features", "track_ids", "window_offsets", "segment_indices",
                "segment_starts", "window_starts", "window_valid", "song_durations")
    if any(key not in h5 for key in required):
        raise ValueError("D4 audio cache is missing required datasets")
    features = h5["features"]
    ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value)
           for value in h5["track_ids"][:]]
    offsets = np.asarray(h5["window_offsets"])
    window_starts = np.asarray(h5["window_starts"])
    valid = np.asarray(h5["window_valid"])
    durations = np.asarray(h5["song_durations"])
    n_tracks = len(ids)
    if (features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != CHANNEL_COUNT):
        raise ValueError("D4 features must be float32 with shape (rows, 26)")
    if ids != sorted(ids) or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("D4 track_ids must be unique, nonempty, and sorted")
    if (offsets.dtype != np.int64 or offsets.shape != (n_tracks, 4)
            or (offsets.size and offsets[0, 0] != 0)
            or np.any(np.diff(offsets, axis=1) < 0)
            or (n_tracks > 1 and np.any(offsets[1:, 0] != offsets[:-1, 3]))
            or (n_tracks and offsets[-1, 3] != len(features))):
        raise ValueError("D4 window_offsets must be int64 cumulative four-boundary rows covering features")
    if (h5["segment_indices"].dtype != np.int64 or h5["segment_indices"].shape != (features.shape[0],)
            or h5["segment_starts"].dtype != np.float64 or h5["segment_starts"].shape != (features.shape[0],)):
        raise ValueError("D4 segment metadata must be row-aligned int64/float64 arrays")
    if (window_starts.dtype != np.float64 or window_starts.shape != (n_tracks, 3)
            or not np.isfinite(window_starts).all()
            or valid.dtype != np.bool_ or valid.shape != (n_tracks, 3)
            or (n_tracks and np.any(~np.any(valid, axis=1)))
            or durations.dtype != np.float64 or durations.shape != (n_tracks,)
            or not np.isfinite(durations).all() or np.any(durations <= 0)):
        raise ValueError("D4 per-track window metadata has invalid dtype, shape, or values")
    for row, track_id in enumerate(ids):
        for window in range(3):
            start, end = map(int, offsets[row, window:window + 2])
            if bool(valid[row, window]) != (end > start):
                raise ValueError("D4 empty-window offsets and validity flags disagree")
    return {"track_ids": ids, "index": {track_id: row for row, track_id in enumerate(ids)},
            "offsets": offsets, "window_starts": window_starts, "window_valid": valid,
            "song_durations": durations}


def _validate_d4_window_metadata(track_id, duration, indices, starts):
    if (indices.dtype != np.int64 or starts.dtype != np.float64
            or np.any(indices < 0) or not np.isfinite(starts).all() or np.any(starts < 0)
            or np.any(starts > duration) or np.any(np.diff(indices) <= 0)
            or np.any(np.diff(starts) <= 0)):
        raise ValueError(f"invalid D4 segment indices/timestamps for track {track_id!r}")


def read_d4_audio_cache(cache_path, track_ids):
    """Read selected D4 windows only; all-track reads require explicit IDs."""
    track_ids, _ = _validate_ids(track_ids)
    windows, indices_by_track, starts_by_track = {}, {}, {}
    with h5py.File(cache_path, "r") as h5:
        metadata = _d4_cache_metadata(h5)
        missing = sorted(set(track_ids) - metadata["index"].keys())
        if missing:
            raise ValueError(f"track IDs missing from D4 audio cache: {missing[:3]}")
        features_ds = h5["features"]
        indices_ds = h5["segment_indices"]
        starts_ds = h5["segment_starts"]
        for track_id in track_ids:
            row = metadata["index"][track_id]
            duration = metadata["song_durations"][row]
            track_windows, track_indices, track_starts = [], [], []
            for window in range(3):
                start, end = map(int, metadata["offsets"][row, window:window + 2])
                block = np.asarray(features_ds[start:end])
                segment_indices = np.asarray(indices_ds[start:end])
                segment_starts = np.asarray(starts_ds[start:end])
                if not np.isfinite(block).all():
                    raise ValueError(f"nonfinite D4 feature rows for track {track_id!r}")
                _validate_d4_window_metadata(track_id, duration, segment_indices, segment_starts)
                track_windows.append(block)
                track_indices.append(segment_indices)
                track_starts.append(segment_starts)
            windows[track_id] = track_windows
            indices_by_track[track_id] = track_indices
            starts_by_track[track_id] = track_starts
    selected_rows = [metadata["index"][track_id] for track_id in track_ids]
    return {"track_ids": track_ids, "windows": windows,
            "window_valid": metadata["window_valid"][selected_rows],
            "segment_indices": indices_by_track, "segment_starts": starts_by_track,
            "window_starts": metadata["window_starts"][selected_rows],
            "song_durations": metadata["song_durations"][selected_rows]}


def fit_d4_audio_stats(cache_path, train_ids):
    """Fit channel statistics on one averaged representative per source segment."""
    train_ids, _ = _validate_ids(train_ids)
    cache = read_d4_audio_cache(cache_path, train_ids)
    return _fit_d4_audio_stats_from_data(cache, train_ids)


def _fit_d4_audio_stats_from_data(cache, train_ids):
    count = 0
    mean = np.zeros(CHANNEL_COUNT, dtype=np.float64)
    m2 = np.zeros(CHANNEL_COUNT, dtype=np.float64)
    for track_id in train_ids:
        by_segment = {}
        for window, segment_indices in zip(cache["windows"][track_id], cache["segment_indices"][track_id]):
            for feature, segment_index in zip(window, segment_indices):
                segment_index = int(segment_index)
                if segment_index not in by_segment:
                    by_segment[segment_index] = [np.asarray(feature, dtype=np.float64), 1]
                else:
                    by_segment[segment_index][0] += feature
                    by_segment[segment_index][1] += 1
        if not by_segment:
            raise ValueError(f"invalid audio rows for track {track_id!r}")
        for total, occurrences in by_segment.values():
            representative = total / occurrences
            count += 1
            delta = representative - mean
            mean += delta / count
            m2 += delta * (representative - mean)
    std = np.sqrt(m2 / count)
    std[std == 0] = 1.0
    return {"mean": mean, "std": std, "row_count": count}


def build_d4_neurotune_v1_features(cache_path, tracks, lyrics, genres, train_ids, song_ids,
                                   split_hash=None):
    """Build train-fitted modalities and aligned D4 window sequences for selected tracks."""
    if (not isinstance(split_hash, str) or len(split_hash) != 64
            or any(char not in "0123456789abcdef" for char in split_hash)):
        raise ValueError("split_hash must be a 64-character lowercase hexadecimal digest")
    tracks = list(tracks)
    train_ids, song_ids = _validate_ids(train_ids, song_ids)
    aux, fitted = build_train_fitted_modalities(tracks, lyrics, genres, train_ids, song_ids)
    cache = read_d4_audio_cache(cache_path, song_ids)
    missing = sorted(set(song_ids) - set(cache["track_ids"]))
    if missing:
        raise ValueError(f"song IDs missing from D4 audio cache: {missing[:3]}")
    stats = _fit_d4_audio_stats_from_data(cache, train_ids)
    windows = {track_id: [apply_audio_stats(sequence, stats) for sequence in cache["windows"][track_id]]
               for track_id in song_ids}
    validity = cache["window_valid"]
    fit_hash = _canonical_hash(sorted(train_ids))
    track_by_id = {track["track_id"]: track for track in tracks}
    selected_ids = sorted(song_ids)
    catalog_values = {track_id: {"bpm": float(track_by_id[track_id]["tempo"]),
                                 "genre": genres.get(track_id)} for track_id in selected_ids}
    lyric_values = {track_id: lyrics.get(track_id, {}) for track_id in selected_ids}
    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        "audio_mean": stats["mean"].tolist(), "audio_std": stats["std"].tolist(),
        "audio_fit_rows": stats["row_count"], "bpm_mean": float(fitted["bpm_mean"]),
        "bpm_std": float(fitted["bpm_std"]), "idf": fitted["idf"],
        "genre_vocabulary": fitted["genre_vocabulary"], "fit_ids_hash": fit_hash,
        "split_hash": split_hash,
        "source_sha256": {
            "d4_audio_cache": _sha256_file(cache_path),
            "catalog_bpm_genre": _canonical_hash(catalog_values),
            "lyrics": _canonical_hash(lyric_values), "split_manifest": split_hash,
        },
        "code_sha256": {relative_path: _sha256_file(repo_root / relative_path)
                         for relative_path in CODE_PATHS},
    }
    json.dumps(manifest, allow_nan=False)
    return {"audio_windows": windows, "window_valid": validity, **aux}, manifest


def apply_audio_stats(audio, stats):
    """Normalize valid audio rows to float32; caller-owned arrays are untouched."""
    values = np.asarray(audio)
    if values.ndim != 2 or values.shape[1] != CHANNEL_COUNT:
        raise ValueError("audio must have shape (T, 26)")
    if not np.isfinite(values).all():
        raise ValueError("audio must contain only finite values")
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if (mean.shape != (CHANNEL_COUNT,) or std.shape != (CHANNEL_COUNT,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("invalid audio statistics")
    normalized = (values.astype(np.float64, copy=True) - mean) / std
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = normalized.astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized audio exceeds the finite float32 range")
    return normalized


def build_train_fitted_modalities(tracks, lyrics, genres, train_ids, song_ids):
    """Build BPM, 5K TF-IDF, and genre arrays; all fitted parameters use train IDs."""
    tracks = list(tracks)
    train_ids, song_ids = _validate_ids(train_ids, song_ids)
    if not isinstance(lyrics, dict) or not isinstance(genres, dict):
        raise ValueError("lyrics and genres must be mappings")
    track_by_id = {}
    for track in tracks:
        if not isinstance(track, dict) or not track.get("track_id"):
            raise ValueError("each track must contain a track_id")
        track_id = track["track_id"]
        if track_id in track_by_id:
            raise ValueError(f"duplicate catalog track record for {track_id!r}")
        track_by_id[track_id] = track
    absent = set(song_ids) - track_by_id.keys()
    if absent:
        raise ValueError(f"missing catalog BPM data for {sorted(absent)[:3]}")

    def bpm_for(track_id):
        try:
            bpm = float(track_by_id[track_id]["tempo"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing or invalid tempo for track {track_id!r}") from exc
        if not np.isfinite(bpm):
            raise ValueError(f"missing or invalid tempo for track {track_id!r}")
        return bpm

    def lyric_count(count, track_id):
        try:
            value = Decimal(str(count))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"invalid lyric entry for track {track_id!r}") from exc
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            raise ValueError(f"invalid lyric entry for track {track_id!r}")
        return int(value)

    train_bpm = np.asarray([bpm_for(track_id) for track_id in train_ids], dtype=np.float64)
    bpm_mean = float(train_bpm.mean())
    bpm_std = float(train_bpm.std())
    bpm_divisor = bpm_std if bpm_std > 0 else 1.0

    document_frequency = np.zeros(VOCABULARY_SIZE, dtype=np.int64)
    for track_id in train_ids:
        document = lyrics.get(track_id, {})
        if not isinstance(document, dict):
            raise ValueError(f"invalid lyric document for track {track_id!r}")
        seen = set()
        for word_id, count in document.items():
            try:
                index = int(word_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid lyric entry for track {track_id!r}") from exc
            value = lyric_count(count, track_id)
            if str(index) != str(word_id) or not 1 <= index <= VOCABULARY_SIZE:
                raise ValueError(f"invalid lyric entry for track {track_id!r}")
            if value > 0:
                seen.add(index - 1)
        if seen:
            document_frequency[list(seen)] += 1
    idf = np.log((1 + len(train_ids)) / (1 + document_frequency)) + 1

    train_genres = []
    for track_id in train_ids:
        genre = genres.get(track_id)
        if not isinstance(genre, str) or not genre:
            raise ValueError(f"missing or invalid genre for track {track_id!r}")
        train_genres.append(genre)
    genre_vocabulary = sorted(set(train_genres))
    genre_index = {genre: index for index, genre in enumerate(genre_vocabulary)}

    bpm = np.empty((len(song_ids), 1), dtype=np.float32)
    lyric_vectors = np.zeros((len(song_ids), VOCABULARY_SIZE), dtype=np.float32)
    genre_vectors = np.zeros((len(song_ids), len(genre_vocabulary)), dtype=np.float32)
    for row, track_id in enumerate(song_ids):
        bpm[row, 0] = (bpm_for(track_id) - bpm_mean) / bpm_divisor
        document = lyrics.get(track_id, {})
        if not isinstance(document, dict):
            raise ValueError(f"invalid lyric document for track {track_id!r}")
        lyric_row = lyric_vectors[row]
        for word_id, count in document.items():
            try:
                index = int(word_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid lyric entry for track {track_id!r}") from exc
            value = lyric_count(count, track_id)
            if str(index) != str(word_id) or not 1 <= index <= VOCABULARY_SIZE:
                raise ValueError(f"invalid lyric entry for track {track_id!r}")
            lyric_row[index - 1] = value * idf[index - 1]
        norm = np.linalg.norm(lyric_row)
        if norm:
            lyric_row /= norm
        genre = genres.get(track_id)
        if isinstance(genre, str) and genre in genre_index:
            genre_vectors[row, genre_index[genre]] = 1.0

    return {"bpm": bpm, "lyrics": lyric_vectors, "genre": genre_vectors}, {
        "bpm_mean": bpm_mean, "bpm_std": bpm_std, "idf": idf.tolist(),
        "genre_vocabulary": genre_vocabulary,
    }


def build_neurotune_v1_features(cache_path, tracks, lyrics, genres, train_ids, song_ids,
                                split_hash=None):
    """Return aligned normalized audio/auxiliary arrays and JSON-safe fit manifest."""
    if (not isinstance(split_hash, str) or len(split_hash) != 64
            or any(char not in "0123456789abcdef" for char in split_hash)):
        raise ValueError("split_hash must be a 64-character lowercase hexadecimal digest")
    tracks = list(tracks)
    train_ids, song_ids = _validate_ids(train_ids, song_ids)
    aux, fitted = build_train_fitted_modalities(tracks, lyrics, genres, train_ids, song_ids)
    audio_stats = fit_audio_stats(cache_path, train_ids)
    audio = []
    with h5py.File(cache_path, "r") as h5:
        features, offsets, index = _cache_index(h5)
        missing = sorted(set(song_ids) - index.keys())
        if missing:
            raise ValueError(f"song IDs missing from audio cache: {missing[:3]}")
        for row, track_id in enumerate(song_ids):
            cache_row = index[track_id]
            seq = features[int(offsets[cache_row]):int(offsets[cache_row + 1])]
            if len(seq) == 0:
                raise ValueError(f"empty audio sequence for track {track_id!r}")
            audio.append(apply_audio_stats(seq, audio_stats))
    fit_hash = _canonical_hash(sorted(train_ids))
    track_by_id = {track["track_id"]: track for track in tracks}
    selected_ids = sorted(song_ids)
    catalog_values = {
        track_id: {
            "bpm": float(track_by_id[track_id]["tempo"]),
            "genre": genres.get(track_id),
        }
        for track_id in selected_ids
    }
    lyric_values = {track_id: lyrics.get(track_id, {}) for track_id in selected_ids}
    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        "audio_mean": audio_stats["mean"].tolist(), "audio_std": audio_stats["std"].tolist(),
        "audio_fit_rows": audio_stats["row_count"], "bpm_mean": float(fitted["bpm_mean"]),
        "bpm_std": float(fitted["bpm_std"]), "idf": fitted["idf"],
        "genre_vocabulary": fitted["genre_vocabulary"], "fit_ids_hash": fit_hash,
        "split_hash": split_hash,
        "source_sha256": {
            "audio_cache": _sha256_file(cache_path),
            "catalog_bpm_genre": _canonical_hash(catalog_values),
            "lyrics": _canonical_hash(lyric_values),
            "split_manifest": split_hash,
        },
        "code_sha256": {
            relative_path: _sha256_file(repo_root / relative_path) for relative_path in CODE_PATHS
        },
    }
    json.dumps(manifest, allow_nan=False)
    return {"audio": audio, **aux}, manifest


def load_neurotune_v1_inputs(data_dir):
    """Read the approved split/catalog/cache auxiliary sources; parse official MXM files."""
    data_dir = Path(data_dir)
    split_path = data_dir / "neurotune_v1_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = split.get("track_ids", split.get("ids", split))
    train_ids = ids.get("train")
    if not isinstance(train_ids, list):
        raise ValueError("split manifest must provide train track IDs")
    partitions = [ids.get(name, []) for name in ("train", "validation", "test")]
    flattened = [track_id for part in partitions for track_id in part]
    if any(len(part) != len(set(part)) for part in partitions) or len(flattened) != len(set(flattened)):
        raise ValueError("split track IDs must be unique and non-overlapping")
    rows = read_rows(data_dir / "ab_catalog_tracks.csv")
    tracks = [{"track_id": row.get("track_id"), "tempo": row.get("tempo", row.get("bpm"))}
              for row in rows]
    genres = {row["track_id"]: row.get("genre", "") for row in rows if row.get("track_id")}
    lyrics = parse_mxm(data_dir.parent / "processed" / "mxm_counts")
    return split, tracks, lyrics, genres, train_ids

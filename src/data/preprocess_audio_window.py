"""Extract raw segment features from the opening window of one MSD HDF5 track."""

import math

import numpy as np

MAX_SEGMENTS_PER_TRACK = 100_000
MAX_SEGMENT_ARRAY_BYTES = 16 * 1024 * 1024
MAX_SONG_DTYPE_BYTES = 8 * 1024


class AudioWindowTrackError(ValueError):
    """A track's audio data cannot produce a valid window."""


class AudioWindowUnsafeError(ValueError):
    """The HDF5 structure is unsafe or violates a resource bound."""


def _hard_link(group, name, h5py):
    if name not in group:
        error = AudioWindowTrackError if name.startswith("segments_") else AudioWindowUnsafeError
        raise error(f"missing required HDF5 field {group.name}/{name}")
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise AudioWindowUnsafeError(f"{group.name}/{name} must be an internal hard link")
    return group[name]


def _song_dataset(group, expected_fields, h5py):
    dataset = _hard_link(group, "songs", h5py)
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != (1,):
        raise AudioWindowUnsafeError(f"{group.name}/songs must be a single-row dataset")
    if dataset.is_virtual:
        raise AudioWindowUnsafeError(f"{group.name}/songs must not be a virtual dataset")
    if dataset.dtype.names is None or not set(expected_fields).issubset(dataset.dtype.names):
        raise AudioWindowUnsafeError(f"{group.name}/songs must contain fields {expected_fields}")
    if dataset.dtype.hasobject:
        raise AudioWindowUnsafeError(f"{group.name}/songs must not contain variable-length fields")
    if dataset.dtype.itemsize > MAX_SONG_DTYPE_BYTES:
        raise AudioWindowUnsafeError(f"{group.name}/songs compound dtype exceeds {MAX_SONG_DTYPE_BYTES} bytes")
    return dataset


def _text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value.strip() if isinstance(value, str) else value


def _track_id(value):
    return str(_text(value)).upper()


def _preprocess_audio(h5_path, expected_track_id, window_seconds, window_starts=None):
    """Load and validate the track once, then extract requested windows."""
    if isinstance(window_seconds, (bool, np.bool_)) or not isinstance(
        window_seconds, (int, float, np.integer, np.floating)
    ):
        raise ValueError("window_seconds must be finite and greater than zero")
    window_seconds = float(window_seconds)
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and greater than zero")

    import h5py

    try:
        with h5py.File(h5_path, "r") as handle:
            analysis = _hard_link(handle, "analysis", h5py)
            metadata_group = _hard_link(handle, "metadata", h5py)
            if not isinstance(analysis, h5py.Group) or not isinstance(metadata_group, h5py.Group):
                raise AudioWindowUnsafeError("analysis and metadata must be groups")
            song_dataset = _song_dataset(analysis, ("track_id", "duration", "tempo"), h5py)
            meta_dataset = _song_dataset(metadata_group, ("artist_id", "song_id"), h5py)
            segment_datasets = {}
            for name, dimensions in (("segments_start", 1), ("segments_pitches", 2),
                                     ("segments_timbre", 2), ("segments_loudness_max", 1)):
                dataset = _hard_link(analysis, name, h5py)
                if not isinstance(dataset, h5py.Dataset) or dataset.is_virtual:
                    raise AudioWindowUnsafeError(f"{name} must be a non-virtual dataset")
                if dataset.dtype.kind not in "iuf" or dataset.dtype.itemsize > 8:
                    raise AudioWindowUnsafeError(f"{name} must have a real numeric dtype with itemsize at most 8")
                if dataset.ndim != dimensions or (dimensions == 2 and dataset.shape[1:] != (12,)):
                    raise AudioWindowTrackError(f"{name} has an invalid segment shape")
                count = dataset.shape[0]
                if count < 1:
                    raise AudioWindowTrackError(f"{name} must contain at least one segment")
                if count > MAX_SEGMENTS_PER_TRACK:
                    raise AudioWindowUnsafeError(f"{name} segment count must be between 1 and {MAX_SEGMENTS_PER_TRACK}")
                if count * dataset.dtype.itemsize * (12 if dimensions == 2 else 1) > MAX_SEGMENT_ARRAY_BYTES:
                    raise AudioWindowUnsafeError(f"{name} logical bytes exceed {MAX_SEGMENT_ARRAY_BYTES}")
                segment_datasets[name] = dataset
            if len({dataset.shape[0] for dataset in segment_datasets.values()}) != 1:
                raise AudioWindowTrackError("segment arrays must be aligned")
            song = song_dataset[0]
            actual_track_id = _track_id(song["track_id"])
            normalized_expected_id = _track_id(expected_track_id)
            if actual_track_id != normalized_expected_id:
                raise ValueError(
                    f"track ID mismatch: expected {normalized_expected_id!r}, "
                    f"found {actual_track_id!r}"
                )

            duration = float(song["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise AudioWindowTrackError("duration must be finite and greater than zero")
            tempo = float(song["tempo"])
            if not math.isfinite(tempo) or tempo <= 0:
                raise AudioWindowTrackError("tempo must be finite and greater than zero")

            arrays = {}
            for name, dataset in segment_datasets.items():
                arrays[name] = np.asarray(dataset[...])
            meta_song = meta_dataset[0]
            metadata = {
                "track_id": actual_track_id,
                "tempo": tempo,
                "duration": duration,
                "artist_id": _text(meta_song["artist_id"]),
                "song_id": _text(meta_song["song_id"]),
            }
    except KeyError as error:
        raise AudioWindowUnsafeError(f"missing required HDF5 field {error}") from error

    starts = arrays["segments_start"]
    pitches = arrays["segments_pitches"]
    timbre = arrays["segments_timbre"]
    loudness = arrays["segments_loudness_max"]
    if starts.ndim != 1 or starts.size == 0:
        raise AudioWindowTrackError("segments_start must be a non-empty one-dimensional array")
    if pitches.ndim != 2 or pitches.shape[1] != 12:
        raise AudioWindowTrackError("segments_pitches must have shape (T, 12)")
    if timbre.ndim != 2 or timbre.shape[1] != 12:
        raise AudioWindowTrackError("segments_timbre must have shape (T, 12)")
    if loudness.ndim != 1:
        raise AudioWindowTrackError("segments_loudness_max must be one-dimensional")
    if not (len(starts) == len(pitches) == len(timbre) == len(loudness)):
        raise AudioWindowTrackError("segment arrays must be aligned")

    for name, values in (("segments_start", starts), ("segments_pitches", pitches),
                         ("segments_timbre", timbre), ("segments_loudness_max", loudness)):
        if values.dtype.kind not in "iuf":
            raise AudioWindowTrackError(f"{name} must contain real numeric values")
        if not np.all(np.isfinite(values)):
            raise AudioWindowTrackError(f"{name} must contain only finite values")
    if np.any(starts[1:] <= starts[:-1]):
        raise AudioWindowTrackError("segments_start times must be ordered and strictly increasing")
    if np.any(starts < 0) or np.any(starts > duration):
        raise AudioWindowTrackError("segments_start times must be within duration")

    if window_starts == "three":
        last_start = max(0.0, duration - window_seconds)
        window_starts = [0.0, max(0.0, (duration - window_seconds) / 2.0), last_start]
        require_nonempty = False
    elif window_starts is None:
        window_starts = [0.0]
        require_nonempty = True
    else:
        require_nonempty = False

    windows = []
    for window_start in window_starts:
        window_end = window_start + window_seconds
        included = np.flatnonzero((starts >= window_start) & (starts < window_end))
        if require_nonempty and included.size == 0:
            raise AudioWindowTrackError("audio window must contain at least one segment")
        rows = []
        for index in included:
            next_start = starts[index + 1] if index + 1 < len(starts) else duration
            segment_duration = min(float(next_start), duration, window_end) - float(starts[index])
            if segment_duration <= 0:
                raise AudioWindowTrackError("clipped segment durations must be positive")
            rows.append(np.concatenate((
                pitches[index], timbre[index], np.asarray([loudness[index], segment_duration])
            )))

        features = np.asarray(rows, dtype=np.float32).reshape((-1, 26))
        if not np.all(np.isfinite(features)):
            raise AudioWindowTrackError("output features must contain only finite float32 values")
        windows.append({
            "features": features,
            "valid": bool(included.size),
            "source_indices": included.astype(np.int64, copy=False),
            "timestamps": starts[included].astype(np.float64, copy=False),
            "window_start": float(window_start),
        })
    return windows, metadata


def preprocess_audio_window(h5_path, expected_track_id, window_seconds=30.0):
    """Return ``(features, metadata)`` for the opening ``[0, window_seconds)``.

    Features are an unnormalized float32 matrix containing pitch (12), timbre
    (12), loudness (1), and clipped segment duration (1) columns.
    """
    windows, metadata = _preprocess_audio(h5_path, expected_track_id, window_seconds)
    return windows[0]["features"], metadata


def preprocess_audio_three_windows(h5_path, expected_track_id, window_seconds=30.0):
    """Return start, middle, and end windows with row provenance and validity.

    Each window is a dictionary containing ``features``, ``valid``, source
    segment ``source_indices``, their ``timestamps``, and ``window_start``.
    The track is opened and validated once, and overlapping windows are allowed.
    """
    windows, metadata = _preprocess_audio(
        h5_path, expected_track_id, window_seconds, window_starts="three"
    )
    return windows, metadata

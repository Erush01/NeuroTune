"""Extract a deterministic temporal feature matrix from one MSD HDF5 file."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


def _validate_normalization_sequences(sequences):
    values = np.asarray(sequences)
    if values.ndim != 3 or values.shape[1] == 0 or values.shape[2] != 28:
        raise ValueError("sequences must have shape (N, T, 28) with non-empty T")
    if values.dtype.kind not in "iuf":
        raise ValueError("sequences must contain real numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError("sequences must contain only finite values")
    return values


def fit_audio_normalization(sequences, track_ids, train_ids):
    """Fit continuous-channel statistics using only the specified training IDs."""
    values = _validate_normalization_sequences(sequences)
    track_ids = list(track_ids)
    train_ids = list(train_ids)
    if len(track_ids) != values.shape[0]:
        raise ValueError("track_ids must be matching the number of sequences")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("track_ids must be unique")
    if not train_ids:
        raise ValueError("train_ids must be non-empty")

    positions = {track_id: index for index, track_id in enumerate(track_ids)}
    missing = [track_id for track_id in train_ids if track_id not in positions]
    if missing:
        raise ValueError(f"every train ID must be present in track_ids: {missing!r}")

    training = values[[positions[track_id] for track_id in train_ids], :, :25].astype(
        np.float64, copy=False
    )
    return {
        "mean": training.mean(axis=(0, 1)).tolist(),
        "std": training.std(axis=(0, 1)).tolist(),
        "continuous_channel_count": 25,
        "train_ids": [str(track_id) for track_id in train_ids],
    }


def apply_audio_normalization(sequences, metadata):
    """Apply fitted continuous-channel statistics without changing event channels."""
    values = _validate_normalization_sequences(sequences)
    event_values = values[:, :, 25:]
    event_float32 = event_values.astype(np.float32)
    if not np.all(np.isfinite(event_float32)):
        raise ValueError("event channels must be exactly representable as float32")
    if values.dtype.kind in "iu":
        limits = np.iinfo(values.dtype)
        event_as_float64 = event_float32.astype(np.float64)
        if np.any(event_as_float64 < limits.min) or np.any(
            event_as_float64 >= float(limits.max + 1)
        ):
            raise ValueError("event channels must be exactly representable as float32")
    if not np.array_equal(event_float32.astype(values.dtype), event_values):
        raise ValueError("event channels must be exactly representable as float32")
    if metadata.get("continuous_channel_count") != 25:
        raise ValueError("normalization metadata must describe 25 continuous channels")
    mean = np.asarray(metadata.get("mean"), dtype=np.float64)
    std = np.asarray(metadata.get("std"), dtype=np.float64)
    if mean.shape != (25,) or std.shape != (25,):
        raise ValueError("normalization mean and std must each contain 25 values")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0):
        raise ValueError("normalization statistics must be finite with non-negative std")

    result = values.astype(np.float32, copy=True)
    scale = np.where(std == 0, 1.0, std)
    result[:, :, :25] = (values[:, :, :25] - mean) / scale
    if not np.all(np.isfinite(result)):
        raise ValueError("normalized sequences must remain finite")
    return result


def _track_id(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip().upper()


def load_audio_sequence_cache(hdf5_path, metadata_json_path, expected_ids=None):
    """Load and validate an M1 audio sequence cache and its metadata."""
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        track_ids = [_track_id(value) for value in handle["track_ids"][...]]
        sequences = np.asarray(handle["sequences"][...])
        channel_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["channel_names"][...]
        ]
    with open(metadata_json_path, encoding="utf-8") as source:
        metadata = json.load(source)
    if not isinstance(metadata, dict):
        raise ValueError("metadata JSON must be an object")

    if track_ids != sorted(track_ids):
        raise ValueError("track IDs must be sorted")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("track IDs must be unique")
    if sequences.ndim != 3 or sequences.shape != (len(track_ids), 1024, 28):
        raise ValueError("sequences must have shape (N, 1024, 28)")
    if sequences.dtype != np.dtype(np.float32):
        raise ValueError("sequences must have float32 dtype")
    if not np.all(np.isfinite(sequences)):
        raise ValueError("sequences must contain only finite values")

    expected_channels = (
        [f"pitch_{index + 1}" for index in range(12)]
        + [f"timbre_{index + 1}" for index in range(12)]
        + ["loudness_max", "beat_count", "bar_count", "section_count"]
    )
    if channel_names != expected_channels or metadata.get("channel_names") != expected_channels:
        raise ValueError("channel names do not match the 28-channel schema")
    if metadata.get("grid_size") != 1024:
        raise ValueError("metadata grid_size must be 1024")

    if expected_ids is not None:
        normalized_expected_ids = [_track_id(value) for value in expected_ids]
        if len(set(normalized_expected_ids)) != len(normalized_expected_ids):
            raise ValueError("expected IDs must be unique")
        if set(normalized_expected_ids) != set(track_ids):
            raise ValueError("expected IDs must match cache track IDs")

    return track_ids, sequences, metadata


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_audio(h5_path, expected_track_id, grid_size=1024):
    """Return interpolated segment features and binned event counts.

    Channels are pitches (12), timbre (12), loudness (1), then beat, bar,
    and section event counts. Continuous values are returned unnormalized.
    """
    if not isinstance(grid_size, (int, np.integer)) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")

    import h5py

    try:
        with h5py.File(h5_path, "r") as handle:
            analysis = handle["analysis"]
            song = analysis["songs"][0]
            actual_track_id = _track_id(song["track_id"])
            expected_track_id = _track_id(expected_track_id)
            if actual_track_id != expected_track_id:
                raise ValueError(
                    f"track ID mismatch: expected {expected_track_id!r}, "
                    f"found {actual_track_id!r}"
                )

            duration = float(song["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration must be finite and greater than zero")

            required = (
                "segments_pitches",
                "segments_timbre",
                "segments_loudness_max",
                "segments_start",
                "beats_start",
                "bars_start",
                "sections_start",
            )
            arrays = {}
            for name in required:
                if name not in analysis:
                    raise ValueError(f"missing required array {name}")
                arrays[name] = np.asarray(analysis[name][...])
    except KeyError as error:
        raise ValueError(f"missing required HDF5 field {error}") from error

    starts = arrays["segments_start"]
    pitches = arrays["segments_pitches"]
    timbre = arrays["segments_timbre"]
    loudness = arrays["segments_loudness_max"]

    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("segments_start must be a non-empty one-dimensional array")
    if pitches.ndim != 2 or pitches.shape[1] != 12:
        raise ValueError("segments_pitches must have shape (T, 12)")
    if timbre.ndim != 2 or timbre.shape[1] != 12:
        raise ValueError("segments_timbre must have shape (T, 12)")
    if loudness.ndim != 1:
        raise ValueError("segments_loudness_max must be one-dimensional")
    if not (len(starts) == len(pitches) == len(timbre) == len(loudness)):
        raise ValueError("segment arrays must be aligned")

    for name, values in (("segments_start", starts), ("segments_pitches", pitches),
                         ("segments_timbre", timbre), ("segments_loudness_max", loudness)):
        if values.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(starts) <= 0):
        raise ValueError("segments_start times must be ordered and strictly increasing")
    if np.any(starts < 0) or np.any(starts > duration):
        raise ValueError("segments_start times must be within duration")

    grid_times = np.linspace(0.0, duration, num=grid_size)
    continuous = np.column_stack(
        [np.interp(grid_times, starts, pitches[:, channel]) for channel in range(12)]
        + [np.interp(grid_times, starts, timbre[:, channel]) for channel in range(12)]
        + [np.interp(grid_times, starts, loudness)]
    )

    event_counts = []
    for name in ("beats_start", "bars_start", "sections_start"):
        events = arrays[name]
        if events.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if events.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
        if not np.all(np.isfinite(events)):
            raise ValueError(f"{name} must contain only finite values")
        if np.any(events < 0) or np.any(events > duration):
            raise ValueError(f"{name} times must be within duration")
        event_fractions = events.astype(np.float64, copy=False) / duration
        bins = np.floor(event_fractions * (grid_size - 1) + 0.5).astype(np.intp)
        event_counts.append(np.bincount(bins, minlength=grid_size))

    return np.column_stack([continuous, *event_counts]).astype(np.float32, copy=False)


def build_audio_sequence_cache(
    input_root,
    split_ids,
    audit_json_path,
    hdf5_output_path,
    metadata_json_path,
    grid_size=1024,
):
    """Build a sorted, normalized HDF5 cache and JSON provenance record."""
    import h5py

    required_splits = ("train", "validation", "test")
    if set(split_ids) != set(required_splits):
        raise ValueError("split_ids must contain train, validation, and test")
    normalized_splits = {name: [_track_id(value) for value in split_ids[name]]
                         for name in required_splits}
    all_split_ids = [track_id for name in required_splits for track_id in normalized_splits[name]]
    if len(set(all_split_ids)) != len(all_split_ids):
        raise ValueError("split IDs must be unique and non-overlapping")
    if not normalized_splits["train"]:
        raise ValueError("train split must be non-empty")
    selected_ids = sorted(all_split_ids)

    hdf5_target = Path(hdf5_output_path).resolve()
    metadata_target = Path(metadata_json_path).resolve()
    if hdf5_target == metadata_target:
        raise ValueError("HDF5 and metadata output paths must be different")

    source_index = {track_id: [] for track_id in selected_ids}
    for path in sorted(Path(input_root).rglob("*.h5")):
        try:
            with h5py.File(path, "r") as handle:
                track_id = _track_id(handle["analysis"]["songs"][0]["track_id"])
        except (OSError, KeyError, IndexError, ValueError):
            continue
        if track_id in source_index:
            source_index[track_id].append(path)
    missing = [track_id for track_id in selected_ids if not source_index[track_id]]
    duplicates = [track_id for track_id in selected_ids if len(source_index[track_id]) > 1]
    if missing:
        raise ValueError(f"missing source HDF5 files for selected IDs: {missing!r}")
    if duplicates:
        raise ValueError(f"duplicate source HDF5 files for selected IDs: {duplicates!r}")

    sequences = np.stack([
        preprocess_audio(source_index[track_id][0], track_id, grid_size=grid_size)
        for track_id in selected_ids
    ]).astype(np.float32, copy=False)
    normalization = fit_audio_normalization(
        sequences, selected_ids, normalized_splits["train"]
    )
    sequences = apply_audio_normalization(sequences, normalization)
    channel_names = (
        [f"pitch_{index + 1}" for index in range(12)]
        + [f"timbre_{index + 1}" for index in range(12)]
        + ["loudness_max", "beat_count", "bar_count", "section_count"]
    )

    metadata = {
        "split_counts": {name: len(normalized_splits[name]) for name in required_splits},
        "split_ids": normalized_splits,
        "grid_size": int(grid_size),
        "grid_definition": {
            "type": "uniform",
            "start_seconds": 0.0,
            "end_seconds": "track_duration",
            "points": int(grid_size),
            "event_bin_rounding": "floor(fraction * (grid_size - 1) + 0.5)",
        },
        "channel_names": channel_names,
        "continuous_channel_count": 25,
        "event_channel_count": 3,
        "normalization": normalization,
        "audit_sha256": _sha256_file(audit_json_path),
        "source_sha256": {
            track_id: _sha256_file(source_index[track_id][0])
            for track_id in selected_ids
        },
    }
    serialized_metadata = json.dumps(metadata, indent=2, sort_keys=True) + "\n"

    hdf5_output_path = Path(hdf5_output_path)
    metadata_json_path = Path(metadata_json_path)
    hdf5_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json_path.parent.mkdir(parents=True, exist_ok=True)
    staged_paths = []
    try:
        for target in (hdf5_output_path, metadata_json_path):
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            os.close(descriptor)
            staged_paths.append(Path(temporary_path))

        staged_hdf5, staged_metadata = staged_paths
        track_id_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(staged_hdf5, "w") as handle:
            handle.create_dataset("track_ids", data=np.asarray(selected_ids, dtype=track_id_dtype))
            handle.create_dataset(
                "sequences",
                data=sequences,
                dtype=np.float32,
                chunks=(1, grid_size, 28),
                compression="gzip",
            )
            handle.create_dataset("channel_names", data=np.asarray(channel_names, dtype=track_id_dtype))
        staged_metadata.write_text(serialized_metadata, encoding="utf-8")
        os.replace(staged_hdf5, hdf5_output_path)
        os.replace(staged_metadata, metadata_json_path)
    finally:
        for staged_path in staged_paths:
            staged_path.unlink(missing_ok=True)

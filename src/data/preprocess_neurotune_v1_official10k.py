"""Build a raw ragged audio cache from the official MSD 10K subset."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np

from .preprocess_audio_window import preprocess_audio_window


CHANNEL_SCHEMA = [
    "pitch_0", "pitch_1", "pitch_2", "pitch_3", "pitch_4", "pitch_5",
    "pitch_6", "pitch_7", "pitch_8", "pitch_9", "pitch_10", "pitch_11",
    "timbre_0", "timbre_1", "timbre_2", "timbre_3", "timbre_4", "timbre_5",
    "timbre_6", "timbre_7", "timbre_8", "timbre_9", "timbre_10", "timbre_11",
    "loudness_max", "clipped_segment_duration",
]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_h5_path(subset_root, track_id):
    """Resolve one MSD track ID using the official MillionSongSubset layout."""
    if not isinstance(track_id, str) or re.fullmatch(r"TR[A-Z0-9]{16}", track_id) is None:
        raise ValueError(f"track ID must be a canonical 18-character MSD ID: {track_id!r}")
    root = Path(subset_root).resolve()
    source = (root / track_id[2] / track_id[3] / track_id[4] / f"{track_id}.h5").resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"official HDF5 source escapes the official subset root: {source}")
    return source


def _write_metadata(path, metadata):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _audit_track_ids(audit_path):
    with Path(audit_path).open(encoding="utf-8") as stream:
        audit = json.load(stream)
    ids = audit.get("track_ids") if isinstance(audit, dict) else None
    if not isinstance(ids, list) or not ids or any(not isinstance(track_id, str) for track_id in ids):
        raise ValueError("audit must contain a non-empty track_ids list")
    if len(ids) != len(set(ids)):
        raise ValueError("audit track_ids must be unique")
    return sorted(ids)


def build_official10k_audio_cache(audit_path, subset_root, output_path, metadata_path):
    """Build and publish the sorted raw float32 audio cache and its metadata."""
    audit_path, subset_root = Path(audit_path), Path(subset_root)
    output_path, metadata_path = Path(output_path), Path(metadata_path)
    if output_path.resolve() == metadata_path.resolve():
        raise ValueError("output paths must be different")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite an existing cache or metadata target")
    ids = _audit_track_ids(audit_path)
    records = []
    source_hashes = {}
    for track_id in ids:
        source_path = official_h5_path(subset_root, track_id)
        if not source_path.is_file():
            raise FileNotFoundError(f"official HDF5 source missing for {track_id}: {source_path}")
        features, extracted = preprocess_audio_window(source_path, track_id, window_seconds=30)
        features = np.asarray(features)
        if (features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 26
                or features.shape[0] == 0 or not np.isfinite(features).all()):
            raise ValueError(f"invalid raw audio features for {track_id}")
        if not isinstance(extracted, dict) or extracted.get("track_id") != track_id:
            raise ValueError(f"preprocessing returned an unexpected track ID for {track_id}")
        records.append(features)
        source_hashes[track_id] = _sha256_file(source_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    staged_h5 = staged_meta = None
    published = []
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output_path.name}.", suffix=".stage", dir=output_path.parent, delete=False) as stream:
            staged_h5 = Path(stream.name)
        with tempfile.NamedTemporaryFile(prefix=f".{metadata_path.name}.", suffix=".stage", dir=metadata_path.parent, delete=False) as stream:
            staged_meta = Path(stream.name)
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        for index, values in enumerate(records):
            offsets[index + 1] = offsets[index] + len(values)
        all_features = np.concatenate(records, axis=0).astype(np.float32, copy=False)
        with h5py.File(staged_h5, "w") as h5:
            h5.create_dataset("features", data=all_features, dtype=np.float32)
            h5.create_dataset("offsets", data=offsets, dtype=np.int64)
            h5.create_dataset("track_ids", data=np.asarray(ids, dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
        metadata = {
            "status": "complete",
            "window_seconds": 30.0,
            "interval": "[0,30)",
            "channel_count": 26,
            "channel_schema": CHANNEL_SCHEMA,
            "track_count": len(ids),
            "segment_count": int(offsets[-1]),
            "audit_sha256": _sha256_file(audit_path),
            "source_sha256": source_hashes,
            "code_sha256": {
                "preprocess_neurotune_v1_official10k": _sha256_file(Path(__file__)),
                "preprocess_audio_window": _sha256_file(Path(__file__).with_name("preprocess_audio_window.py")),
            },
            "storage": {"features": "raw unnormalized float32 (T, 26)",
                        "offsets": "int64, sorted track_ids index boundaries",
                        "track_ids": "sorted UTF-8"},
        }
        _write_metadata(staged_meta, metadata)
        # Refuse races as well as pre-existing targets; rollback a partial publish.
        if output_path.exists() or metadata_path.exists():
            raise FileExistsError("refusing to overwrite an existing cache or metadata target")
        os.link(staged_h5, output_path)
        published.append(output_path)
        os.link(staged_meta, metadata_path)
        published.append(metadata_path)
        return {"track_count": len(ids), "segment_count": int(offsets[-1]),
                "output_path": str(output_path), "metadata_path": str(metadata_path)}
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        if staged_h5 is not None:
            staged_h5.unlink(missing_ok=True)
        if staged_meta is not None:
            staged_meta.unlink(missing_ok=True)

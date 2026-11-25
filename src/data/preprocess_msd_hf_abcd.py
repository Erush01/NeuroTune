"""Build a raw ragged 30-second audio cache for the audited MSD A-D catalog."""

import csv
import hashlib
import io
import json
import math
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

from .preprocess_audio_window import (AudioWindowTrackError, AudioWindowUnsafeError,
                                      preprocess_audio_three_windows, preprocess_audio_window)

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = REPO_ROOT / "data/msd_hf_abcd_pilot"
ARCHIVE_DIR = REPO_ROOT / "data/msd_hf_pilot"
TRACK_RE = re.compile(r"^TR[A-Z0-9]{16}$")
CHANNELS = [*(f"pitch_{i}" for i in range(12)), *(f"timbre_{i}" for i in range(12)), "loudness", "duration"]
WINDOW_SECONDS = 30
EXPECTED_TRACK_COUNT = 13_822
MAX_HDF5_MEMBER_BYTES = 64 * 1024 * 1024
TEMP_DIR = Path("/tmp/opencode")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip().upper()
    if not TRACK_RE.fullmatch(value):
        raise ValueError(f"invalid MSD track ID {value!r}")
    return value


def _text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def _load_inputs(audit_path, catalog_path):
    audit_bytes = Path(audit_path).read_bytes()
    catalog_bytes = Path(catalog_path).read_bytes()
    audit = json.loads(audit_bytes.decode("utf-8"))
    if audit.get("status") != "complete":
        raise ValueError("audit status must be complete")
    if audit.get("window") != {"seconds": 30, "interval": "[0,30)"}:
        raise ValueError("audit window contract must be 30 seconds [0,30)")
    if audit.get("channels") != CHANNELS:
        raise ValueError("audit channel names do not match the 26-channel contract")
    hashes = audit.get("source_sha256")
    shards = ("A", "B", "C", "D")
    if not isinstance(hashes, dict) or any(
        not isinstance(hashes.get(shard), str) or re.fullmatch(r"[0-9a-f]{64}", hashes[shard]) is None
        for shard in shards
    ):
        raise ValueError("audit must contain valid A, B, C, and D archive SHA-256 hashes")
    records, seen = {}, []
    with io.StringIO(catalog_bytes.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"track_id", "tempo", "artist_id", "song_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("catalog is missing required track metadata columns")
        for row in reader:
            track_id = _canonical(row["track_id"])
            if track_id in records:
                raise ValueError(f"duplicate catalog track ID: {track_id}")
            try:
                tempo = float(row["tempo"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid catalog tempo for {track_id}") from error
            if not math.isfinite(tempo) or tempo <= 0:
                raise ValueError(f"catalog tempo must be positive for {track_id}")
            records[track_id] = {"tempo": tempo, "artist_id": _text(row["artist_id"]), "song_id": _text(row["song_id"])}
            seen.append(track_id)
    if not seen:
        raise ValueError("catalog contains no tracks")
    if seen != sorted(seen):
        raise ValueError("catalog track IDs must be sorted")
    expected_count = audit.get("counts", {}).get("eligible_tracks")
    if expected_count is not None and expected_count != len(seen):
        raise ValueError("catalog track count does not match audit eligible_tracks count")
    return audit, audit_bytes, records, hashlib.sha256(catalog_bytes).hexdigest()


def _validate_hashes(paths, expected):
    for shard, path in paths.items():
        if _sha256(path) != expected[shard]:
            raise ValueError(f"source archive {shard} SHA-256 does not match audit")


def _member_track_id(name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name:
        raise ValueError(f"unsafe tar member path: {name}")
    if not name.lower().endswith(".h5"):
        return None
    track_id = _canonical(pure.parts[-1][:-3])
    if len(pure.parts) < 4 or pure.parts[-4:-1] != (track_id[2], track_id[3], track_id[4]):
        raise ValueError(f"tar member path does not match track ID: {name}")
    return track_id


def _read_member(archive, member, track_id, row):
    if member.size > MAX_HDF5_MEMBER_BYTES:
        raise ValueError(f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {member.name}")
    source = archive.extractfile(member)
    if source is None:
        raise OSError(f"cannot read tar member: {member.name}")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix="msd-hf-abcd-", suffix=".h5", dir=TEMP_DIR)
    path = Path(filename)
    try:
        total = 0
        with os.fdopen(fd, "wb") as output, source:
            while True:
                chunk = source.read(min(1024 * 1024, MAX_HDF5_MEMBER_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_HDF5_MEMBER_BYTES:
                    raise ValueError(f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {member.name}")
                output.write(chunk)
        if total != member.size:
            raise ValueError(f"HDF5 member size mismatch for {member.name}: declared {member.size}, read {total}")
        _validate_catalog_metadata(path, track_id, row)
        return preprocess_audio_window(path, track_id, window_seconds=WINDOW_SECONDS)
    finally:
        source.close()
        path.unlink(missing_ok=True)


def _read_three_window_member(archive, member, track_id, row):
    if member.size > MAX_HDF5_MEMBER_BYTES:
        raise ValueError(f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {member.name}")
    source = archive.extractfile(member)
    if source is None:
        raise OSError(f"cannot read tar member: {member.name}")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix="msd-hf-abcd-d4-", suffix=".h5", dir=TEMP_DIR)
    path = Path(filename)
    try:
        total = 0
        with os.fdopen(fd, "wb") as output, source:
            while True:
                chunk = source.read(min(1024 * 1024, MAX_HDF5_MEMBER_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_HDF5_MEMBER_BYTES:
                    raise ValueError(f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {member.name}")
                output.write(chunk)
        if total != member.size:
            raise ValueError(f"HDF5 member size mismatch for {member.name}: declared {member.size}, read {total}")
        _validate_catalog_metadata(path, track_id, row)
        return preprocess_audio_three_windows(path, track_id, window_seconds=WINDOW_SECONDS)
    finally:
        source.close()
        path.unlink(missing_ok=True)


def _validate_catalog_metadata(path, track_id, row):
    with h5py.File(path, "r") as handle:
        def hard_link(group, key):
            if key not in group or not isinstance(group.get(key, getlink=True), h5py.HardLink):
                raise AudioWindowUnsafeError(f"{group.name}/{key} must be an internal hard link")
            return group[key]

        analysis = hard_link(handle, "analysis")
        metadata = hard_link(handle, "metadata")
        if not isinstance(analysis, h5py.Group) or not isinstance(metadata, h5py.Group):
            raise AudioWindowUnsafeError("analysis and metadata must be groups")
        for group, expected in ((analysis, ("track_id", "tempo")), (metadata, ("artist_id", "song_id"))):
            dataset = hard_link(group, "songs")
            if (not isinstance(dataset, h5py.Dataset) or dataset.shape != (1,) or dataset.is_virtual
                    or dataset.dtype.names is None or not set(expected).issubset(dataset.dtype.names)
                    or dataset.dtype.hasobject or dataset.dtype.itemsize > 8 * 1024):
                raise AudioWindowUnsafeError(f"{group.name}/songs is not a safe single-row metadata dataset")
        song = analysis["songs"][0]
        meta_song = metadata["songs"][0]
        actual_id = _canonical(song["track_id"])
        if actual_id != track_id:
            raise ValueError(f"track ID mismatch: expected {track_id!r}, found {actual_id!r}")
        tempo = float(song["tempo"])
        if not math.isfinite(tempo) or tempo <= 0 or not math.isclose(tempo, row["tempo"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"metadata tempo mismatch for {track_id}")
        if _text(meta_song["artist_id"]) != row["artist_id"] or _text(meta_song["song_id"]) != row["song_id"]:
            raise ValueError(f"metadata artist/song mismatch for {track_id}")


def _scan_archives(paths, records):
    found, rejected = {}, {}
    for shard in ("A", "B", "C", "D"):
        with tarfile.open(paths[shard], "r|*") as archive:
            for member in archive:
                track_id = _member_track_id(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"special tar member is not allowed: {member.name}")
                if track_id is None:
                    continue
                if member.size > MAX_HDF5_MEMBER_BYTES:
                    raise ValueError(f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {member.name}")
                if track_id not in records:
                    continue
                if track_id in found or track_id in rejected:
                    raise ValueError(f"duplicate source track ID across A-D archives: {track_id}")
                try:
                    features, metadata = _read_member(archive, member, track_id, records[track_id])
                except AudioWindowTrackError as error:
                    rejected[track_id] = str(error)
                    continue
                if _canonical(metadata.get("track_id")) != track_id:
                    raise ValueError(f"track ID mismatch for {track_id}")
                tempo = float(metadata.get("tempo", 0))
                if not math.isfinite(tempo) or tempo <= 0 or not math.isclose(tempo, records[track_id]["tempo"], rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"metadata tempo mismatch for {track_id}")
                row = records[track_id]
                if _text(metadata.get("artist_id", "")) != row["artist_id"] or _text(metadata.get("song_id", "")) != row["song_id"]:
                    raise ValueError(f"metadata artist/song mismatch for {track_id}")
                features = np.asarray(features)
                if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 26 or not len(features) or not np.isfinite(features).all():
                    raise ValueError(f"invalid feature matrix for {track_id}")
                found[track_id] = features
    missing = sorted(set(records) - found.keys() - rejected.keys())
    if missing:
        raise ValueError(f"missing catalog tracks in A-D archives: {missing[:5]}")
    return found, rejected


def _write_hdf5(path, features, provenance):
    ids = sorted(features)
    lengths = [len(features[key]) for key in ids]
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    matrix = (np.concatenate([features[key] for key in ids], axis=0).astype(np.float32, copy=False)
              if ids else np.empty((0, 26), dtype=np.float32))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=matrix, dtype=np.float32, chunks=True, compression="gzip")
        handle.create_dataset("offsets", data=offsets, dtype=np.int64)
        handle.create_dataset("track_ids", data=np.asarray(ids, dtype=object), dtype=h5py.string_dtype("utf-8"))
        for key, value in {"window_seconds": WINDOW_SECONDS, "interval": "[0,30)", "channels": json.dumps(CHANNELS),
                           "normalization": "raw/unnormalized; deferred until train split", **provenance}.items():
            handle.attrs[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def run(a_tar, b_tar, c_tar, d_tar, audit_path, catalog_path, output_h5, output_meta, output_rejections):
    paths = {key: Path(path) for key, path in zip(("A", "B", "C", "D"), (a_tar, b_tar, c_tar, d_tar))}
    audit_path, catalog_path = Path(audit_path), Path(catalog_path)
    outputs = [Path(output_h5), Path(output_meta), Path(output_rejections)]
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("cache output paths must be different")
    collision = [str(p) for p in outputs if p.exists()]
    if collision:
        raise FileExistsError("refusing to overwrite: " + ", ".join(collision))
    if len({p.parent.resolve() for p in outputs}) != 1:
        raise ValueError("staged outputs must be in the same directory")
    audit, audit_bytes, records, catalog_sha256 = _load_inputs(audit_path, catalog_path)
    _validate_hashes(paths, audit["source_sha256"])
    features, rejected = _scan_archives(paths, records)
    _validate_hashes(paths, audit["source_sha256"])
    if _sha256(catalog_path) != catalog_sha256:
        raise ValueError("catalog changed during archive scanning")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"audit_sha256": hashlib.sha256(audit_bytes).hexdigest(), "catalog_sha256": catalog_sha256,
                  "source_sha256": audit["source_sha256"],
                  "code_sha256": {"src/data/preprocess_msd_hf_abcd.py": _sha256(Path(__file__)),
                                  "scripts/preprocess_msd_hf_abcd.py": _sha256(REPO_ROOT / "scripts/preprocess_msd_hf_abcd.py"),
                                  "src/data/preprocess_audio_window.py": _sha256(Path(__file__).with_name("preprocess_audio_window.py"))}}
    ids = sorted(features)
    lengths = {key: len(features[key]) for key in ids}
    meta = {"status": "complete", "window_seconds": 30, "interval": "[0,30)", "channel_count": 26,
            "channels": CHANNELS, "dtype": "float32", "normalization": "raw/unnormalized; deferred until train split",
            "track_count": len(ids), "rejected_track_count": len(rejected), "segment_count": sum(lengths.values()),
            "valid_track_ids": ids, "sequence_lengths": lengths,
            "rejected": [{"track_id": k, "reason_code": "audio_window_track_error", "reason": rejected[k]}
                        for k in sorted(rejected)], **provenance}
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix="abcd-audio-stage-", dir=output_h5.parent) as stage:
            stage = Path(stage)
            staged = [stage / "cache.h5", stage / "metadata.json", stage / "rejections.json"]
            _write_hdf5(staged[0], features, provenance)
            staged[1].write_text(json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            staged[2].write_text(json.dumps(meta["rejected"], indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            for source, target in zip(staged, outputs):
                os.link(source, target)
                published.append(target)
    except BaseException:
        for target in published:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return {"track_count": len(ids), "rejected_track_count": len(rejected), "segment_count": meta["segment_count"],
            "output_h5": str(outputs[0]), "output_meta": str(outputs[1]), "output_rejections": str(outputs[2])}


def _scan_three_window_archives(paths, records):
    found = {}
    for shard in ("A", "B", "C", "D"):
        with tarfile.open(paths[shard], "r|*") as archive:
            for member in archive:
                track_id = _member_track_id(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"special tar member is not allowed: {member.name}")
                if member.size > MAX_HDF5_MEMBER_BYTES:
                    raise ValueError(f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {member.name}")
                if track_id is None or track_id not in records:
                    continue
                if track_id in found:
                    raise ValueError(f"duplicate source track ID across A-D archives: {track_id}")
                try:
                    windows, metadata = _read_three_window_member(archive, member, track_id, records[track_id])
                except AudioWindowTrackError as error:
                    raise ValueError(f"track {track_id} has zero valid windows: {error}") from error
                if _canonical(metadata.get("track_id")) != track_id:
                    raise ValueError(f"track ID mismatch for {track_id}")
                row = records[track_id]
                tempo = float(metadata.get("tempo", 0))
                if (not math.isfinite(tempo) or tempo <= 0
                        or not math.isclose(tempo, row["tempo"], rel_tol=1e-6, abs_tol=1e-6)):
                    raise ValueError(f"metadata tempo mismatch for {track_id}")
                if (_text(metadata.get("artist_id", "")) != row["artist_id"]
                        or _text(metadata.get("song_id", "")) != row["song_id"]):
                    raise ValueError(f"metadata artist/song mismatch for {track_id}")
                if len(windows) != 3 or not any(window["valid"] for window in windows):
                    raise ValueError(f"track {track_id} has zero valid windows")
                for window in windows:
                    features = np.asarray(window["features"])
                    if (features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 26
                            or not np.isfinite(features).all()):
                        raise ValueError(f"invalid feature matrix for {track_id}")
                found[track_id] = {"windows": windows, "duration": float(metadata["duration"])}
    missing = sorted(set(records) - found.keys())
    if missing:
        raise ValueError(f"missing catalog tracks in A-D archives: {missing[:5]}")
    return found


def _write_three_window_hdf5(path, tracks, provenance):
    ids = sorted(tracks)
    flat_features, flat_indices, flat_starts = [], [], []
    offsets = np.zeros((len(ids), 4), dtype=np.int64)
    starts = np.zeros((len(ids), 3), dtype=np.float64)
    valid = np.zeros((len(ids), 3), dtype=np.bool_)
    durations = np.zeros(len(ids), dtype=np.float64)
    cursor = 0
    for row, track_id in enumerate(ids):
        for index, window in enumerate(tracks[track_id]["windows"]):
            offsets[row, index] = cursor
            features = window["features"]
            flat_features.append(features)
            flat_indices.append(window["source_indices"])
            flat_starts.append(window["timestamps"])
            cursor += len(features)
            if index < 3:
                starts[row, index] = window["window_start"]
                valid[row, index] = window["valid"]
        offsets[row, 3] = cursor
        durations[row] = tracks[track_id]["duration"]
    matrix = np.concatenate(flat_features, axis=0).astype(np.float32, copy=False) if cursor else np.empty((0, 26), dtype=np.float32)
    indices = np.concatenate(flat_indices).astype(np.int64, copy=False) if cursor else np.empty(0, dtype=np.int64)
    timestamps = np.concatenate(flat_starts).astype(np.float64, copy=False) if cursor else np.empty(0, dtype=np.float64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=matrix, dtype=np.float32, chunks=True, compression="gzip")
        handle.create_dataset("track_ids", data=np.asarray(ids, dtype=object), dtype=h5py.string_dtype("utf-8"))
        handle.create_dataset("window_offsets", data=offsets, dtype=np.int64)
        handle.create_dataset("segment_indices", data=indices, dtype=np.int64)
        handle.create_dataset("segment_starts", data=timestamps, dtype=np.float64)
        handle.create_dataset("window_starts", data=starts, dtype=np.float64)
        handle.create_dataset("window_valid", data=valid, dtype=np.bool_)
        handle.create_dataset("song_durations", data=durations, dtype=np.float64)
        for key, value in {"window_seconds": WINDOW_SECONDS, "window_formula": "start=0; middle=max(0,(duration-30)/2); end=max(0,duration-30)",
                           "interval": "[start,start+30)", "channels": json.dumps(CHANNELS),
                           "normalization": "raw/unnormalized; deferred until train split", **provenance}.items():
            handle.attrs[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def run_three_windows(a_tar, b_tar, c_tar, d_tar, audit_path, catalog_path, output_h5, output_meta, output_rejections):
    """Build an isolated, provenance-checked start/middle/end audio cache."""
    paths = {key: Path(path) for key, path in zip(("A", "B", "C", "D"), (a_tar, b_tar, c_tar, d_tar))}
    audit_path, catalog_path = Path(audit_path), Path(catalog_path)
    outputs = [Path(output_h5), Path(output_meta), Path(output_rejections)]
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("cache output paths must be different")
    protected = {path.resolve() for path in (PILOT_DIR / "abcd_audio_30s_26ch.h5",
                                              PILOT_DIR / "abcd_audio_30s_26ch_meta.json",
                                              PILOT_DIR / "abcd_audio_30s_26ch_rejections.json")}
    if any(path.resolve() in protected for path in outputs):
        raise ValueError("D4 outputs must not target the existing opening-window cache namespace")
    collision = [str(p) for p in outputs if p.exists()]
    if collision:
        raise FileExistsError("refusing to overwrite: " + ", ".join(collision))
    if len({p.parent.resolve() for p in outputs}) != 1:
        raise ValueError("staged outputs must be in the same directory")
    audit, audit_bytes, records, catalog_sha256 = _load_inputs(audit_path, catalog_path)
    if len(records) != EXPECTED_TRACK_COUNT:
        raise ValueError(f"D4 cache requires exactly {EXPECTED_TRACK_COUNT} expected catalog IDs; found {len(records)}")
    _validate_hashes(paths, audit["source_sha256"])
    tracks = _scan_three_window_archives(paths, records)
    _validate_hashes(paths, audit["source_sha256"])
    if _sha256(catalog_path) != catalog_sha256:
        raise ValueError("catalog changed during archive scanning")
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    provenance = {"audit_sha256": hashlib.sha256(audit_bytes).hexdigest(), "catalog_sha256": catalog_sha256,
                  "source_sha256": audit["source_sha256"],
                  "code_sha256": {"src/data/preprocess_msd_hf_abcd.py": _sha256(Path(__file__)),
                                  "scripts/preprocess_msd_hf_abcd.py": _sha256(REPO_ROOT / "scripts/preprocess_msd_hf_abcd.py"),
                                  "scripts/preprocess_neurotune_v1_d4_audio_windows.py": _sha256(REPO_ROOT / "scripts/preprocess_neurotune_v1_d4_audio_windows.py"),
                                  "src/data/preprocess_audio_window.py": _sha256(Path(__file__).with_name("preprocess_audio_window.py"))}}
    ids = sorted(tracks)
    window_counts = {key: [len(window["features"]) for window in tracks[key]["windows"]] for key in ids}
    segment_count = sum(sum(counts) for counts in window_counts.values())
    meta = {"status": "complete", "window_seconds": 30, "window_formula": "start=0; middle=max(0,(duration-30)/2); end=max(0,duration-30)",
            "interval": "[start,start+30)", "channel_count": 26, "channels": CHANNELS,
            "dtype": "float32", "normalization": "raw/unnormalized; deferred until train split",
            "track_count": len(ids), "rejected_track_count": 0, "segment_count": segment_count,
            "valid_track_ids": ids, "window_segment_counts": window_counts,
            "window_valid_counts": [int(sum(tracks[key]["windows"][i]["valid"] for key in ids)) for i in range(3)],
            "rejected": [], **provenance}
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix="abcd-audio-d4-stage-", dir=outputs[0].parent) as stage:
            stage = Path(stage)
            staged = [stage / "cache.h5", stage / "metadata.json", stage / "rejections.json"]
            _write_three_window_hdf5(staged[0], tracks, provenance)
            staged[1].write_text(json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            staged[2].write_text("[]\n", encoding="utf-8")
            for source, target in zip(staged, outputs):
                os.link(source, target)
                published.append(target)
    except BaseException:
        for target in published:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return {"track_count": len(ids), "rejected_track_count": 0, "segment_count": segment_count,
            "output_h5": str(outputs[0]), "output_meta": str(outputs[1]), "output_rejections": str(outputs[2])}

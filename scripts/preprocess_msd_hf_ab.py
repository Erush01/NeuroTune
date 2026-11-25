"""Build the raw ragged 30-second audio cache for the audited MSD A/B catalog."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

from scripts.audit_msd_hf_ab import CHANNELS, MAX_HDF5_MEMBER_BYTES
from src.data.preprocess_audio_window import preprocess_audio_window

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_RE = re.compile(r"^TR[A-Z0-9]{16}$")
DEFAULT_PILOT_DIR = REPO_ROOT / "data/msd_hf_pilot"
WINDOW_SECONDS = 30
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
    audit_path, catalog_path = Path(audit_path), Path(catalog_path)
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes.decode("utf-8"))
    if audit.get("status") != "complete":
        raise ValueError("audit status must be complete")
    if audit.get("window") != {"seconds": 30, "interval": "[0,30)"}:
        raise ValueError("audit window contract must be 30 seconds [0,30)")
    if audit.get("channels") != CHANNELS:
        raise ValueError("audit channel names do not match the 26-channel contract")
    source_hashes = audit.get("source_sha256")
    if not isinstance(source_hashes, dict) or not all(
        isinstance(source_hashes.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", source_hashes[key])
        for key in ("A", "B")
    ):
        raise ValueError("audit must contain valid A and B archive SHA-256 hashes")
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"track_id", "tempo", "artist_id", "song_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("catalog is missing required track metadata columns")
        records = {}
        seen = []
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
            records[track_id] = {
                "tempo": tempo,
                "artist_id": _text(row["artist_id"]),
                "song_id": _text(row["song_id"]),
            }
            seen.append(track_id)
    if not seen:
        raise ValueError("catalog contains no tracks")
    if seen != sorted(seen):
        raise ValueError("catalog track IDs must be sorted")
    expected_count = audit.get("counts", {}).get("eligible_tracks")
    if expected_count is not None and expected_count != len(seen):
        raise ValueError("catalog track count does not match audit eligible_tracks count")
    return audit, audit_bytes, records


def _validate_archive_hashes(paths, expected):
    for key, path in paths.items():
        actual = _sha256(path)
        if actual != expected[key]:
            raise ValueError(f"source archive {key} SHA-256 does not match audit")


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


def _stream_member(archive, member, track_id):
    if member.size > MAX_HDF5_MEMBER_BYTES:
        raise ValueError(f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {member.name}")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"cannot read tar member: {member.name}")
    path = None
    fd = None
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        fd, filename = tempfile.mkstemp(prefix="msd-hf-ab-", suffix=".h5", dir=TEMP_DIR)
        path = Path(filename)
        total = 0
        output = os.fdopen(fd, "wb")
        fd = None
        with output, source:
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
        features, metadata = preprocess_audio_window(path, track_id, window_seconds=WINDOW_SECONDS)
        return features, metadata
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        source.close()
        if path is not None:
            path.unlink(missing_ok=True)


def _scan_archives(paths, expected):
    found = {}
    for key in ("A", "B"):
        with tarfile.open(paths[key], "r|*") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"special tar member is not allowed: {member.name}")
                track_id = _member_track_id(member.name)
                if track_id is None:
                    continue
                if member.size > MAX_HDF5_MEMBER_BYTES:
                    raise ValueError(
                        f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {member.name}"
                    )
                if track_id not in expected:
                    continue
                if track_id in found:
                    raise ValueError(f"duplicate source track ID across A/B archives: {track_id}")
                features, metadata = _stream_member(archive, member, track_id)
                row = expected[track_id]
                if _canonical(metadata.get("track_id")) != track_id:
                    raise ValueError(f"track ID mismatch for {track_id}")
                if not math.isfinite(float(metadata.get("tempo", 0))) or float(metadata["tempo"]) <= 0:
                    raise ValueError(f"tempo must be positive for {track_id}")
                if not math.isclose(float(metadata["tempo"]), row["tempo"], rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"metadata tempo mismatch for {track_id}")
                if _text(metadata.get("artist_id", "")) != row["artist_id"] or _text(
                    metadata.get("song_id", "")
                ) != row["song_id"]:
                    raise ValueError(f"metadata artist/song mismatch for {track_id}")
                features = np.asarray(features)
                if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 26:
                    raise ValueError(f"invalid feature matrix for {track_id}")
                if not len(features) or not np.all(np.isfinite(features)):
                    raise ValueError(f"invalid or empty feature sequence for {track_id}")
                found[track_id] = features
    missing = sorted(set(expected) - found.keys())
    if missing:
        raise ValueError(f"missing catalog tracks in A/B archives: {missing[:5]}")
    return found


def _write_hdf5(path, records, features, provenance):
    import h5py

    track_ids = sorted(records)
    lengths = [len(features[track_id]) for track_id in track_ids]
    offsets = np.empty(len(track_ids) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    matrix = np.concatenate([features[track_id] for track_id in track_ids], axis=0).astype(np.float32, copy=False)
    if np.any(np.diff(offsets) <= 0) or matrix.shape != (int(offsets[-1]), 26):
        raise ValueError("invalid ragged feature offsets or matrix shape")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=matrix, dtype=np.float32, chunks=True, compression="gzip")
        handle.create_dataset("offsets", data=offsets, dtype=np.int64, chunks=True, compression="gzip")
        handle.create_dataset("track_ids", data=np.asarray(track_ids, dtype=object), dtype=string_dtype,
                              chunks=True, compression="gzip")
        attrs = {
            "window_seconds": WINDOW_SECONDS, "interval": "[0,30)", "channels": json.dumps(CHANNELS),
            "dtype": "float32", "normalization": "raw/unnormalized; deferred until train split",
            "track_count": len(track_ids), "segment_count": int(offsets[-1]),
            "sequence_lengths": json.dumps(dict(zip(track_ids, lengths)), sort_keys=True),
            "audit_sha256": provenance["audit_sha256"],
            "source_sha256": json.dumps(provenance["source_sha256"], sort_keys=True),
            "code_sha256": json.dumps(provenance["code_sha256"], sort_keys=True),
        }
        for key, value in attrs.items():
            handle.attrs[key] = value


def run(a_tar, b_tar, audit_path, catalog_path, output_h5, output_meta):
    paths = {"A": Path(a_tar), "B": Path(b_tar)}
    audit_path, catalog_path = Path(audit_path), Path(catalog_path)
    output_h5, output_meta = Path(output_h5), Path(output_meta)
    if output_h5.resolve() == output_meta.resolve():
        raise ValueError("HDF5 and metadata outputs must be different paths")
    existing = [str(path) for path in (output_h5, output_meta) if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    audit, audit_bytes, records = _load_inputs(audit_path, catalog_path)
    _validate_archive_hashes(paths, audit["source_sha256"])
    extracted = _scan_archives(paths, records)
    _validate_archive_hashes(paths, audit["source_sha256"])
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    if output_h5.parent.resolve() != output_meta.parent.resolve():
        raise ValueError("staged outputs must be in the same directory")
    provenance = {
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "source_sha256": audit["source_sha256"],
        "code_sha256": {
            "scripts/preprocess_msd_hf_ab.py": _sha256(Path(__file__)),
            "src/data/preprocess_audio_window.py": _sha256(REPO_ROOT / "src/data/preprocess_audio_window.py"),
        },
    }
    track_ids = sorted(records)
    lengths = {track_id: len(extracted[track_id]) for track_id in track_ids}
    metadata = {
        "window_seconds": WINDOW_SECONDS, "interval": "[0,30)", "channels": CHANNELS,
        "dtype": "float32", "normalization": "raw/unnormalized; deferred until train split",
        "track_count": len(track_ids), "segment_count": sum(lengths.values()),
        "sequence_lengths": lengths, "audit_sha256": provenance["audit_sha256"],
        "source_sha256": provenance["source_sha256"], "code_sha256": provenance["code_sha256"],
    }
    staged = []
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix="msd-hf-ab-stage-", dir=output_h5.parent) as tempdir:
            tempdir = Path(tempdir)
            staged_h5, staged_json = tempdir / "cache.h5", tempdir / "metadata.json"
            _write_hdf5(staged_h5, records, extracted, provenance)
            staged_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            staged = [staged_h5, staged_json]
            for source, target in zip(staged, (output_h5, output_meta)):
                os.link(source, target)
                published.append(target)
    except BaseException:
        for target in published:
            try:
                target.unlink(missing_ok=True)
            except BaseException:
                pass
        raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-tar", type=Path, default=DEFAULT_PILOT_DIR / "A.tar.gz")
    parser.add_argument("--b-tar", type=Path, default=DEFAULT_PILOT_DIR / "B.tar.gz")
    parser.add_argument("--audit", type=Path, default=DEFAULT_PILOT_DIR / "ab_catalog_audit.json")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_PILOT_DIR / "ab_catalog_tracks.csv")
    parser.add_argument("--output-h5", type=Path, default=DEFAULT_PILOT_DIR / "ab_audio_30s_26ch.h5")
    parser.add_argument("--output-meta", type=Path, default=DEFAULT_PILOT_DIR / "ab_audio_30s_26ch_meta.json")
    args = parser.parse_args()
    result = run(args.a_tar, args.b_tar, args.audit, args.catalog, args.output_h5, args.output_meta)
    print(json.dumps({"track_count": result["track_count"], "segment_count": result["segment_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()

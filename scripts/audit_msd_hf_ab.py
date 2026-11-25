"""Build a reproducible catalog and Last.fm edge audit for the MSD A/B shards."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from src.data.parse_msd_duplicates import parse_msd_duplicates
from src.data.parse_tagtraum import parse_tagtraum

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (
    "ab_catalog_audit.json", "ab_catalog_tracks.csv",
    "ab_lastfm_train_edges.csv", "ab_lastfm_test_edges.csv",
)
CHANNELS = [*(f"pitch_{i}" for i in range(12)), *(f"timbre_{i}" for i in range(12)),
            "loudness", "duration"]
TRACK_RE = re.compile(r"^TR[A-Z0-9]{16}$")
# Bound decompression of untrusted archive members: 64 MiB per HDF5, 16 MiB per JSON.
MAX_HDF5_MEMBER_BYTES = 64 * 1024 * 1024
MAX_LASTFM_JSON_BYTES = 16 * 1024 * 1024
MAX_SONG_DTYPE_BYTES = 8 * 1024
THIRD_PARTY_CAVEAT = "Third-party HF provenance, license, and checksums are unverified; local research pilot only."


def _hard_link(group, name, h5py):
    if name not in group or not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"{group.name}/{name} must be an internal hard link")
    return group[name]


def _song_dataset(group, expected_fields, h5py):
    dataset = _hard_link(group, "songs", h5py)
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != (1,):
        raise ValueError(f"{group.name}/songs must be a single-row dataset")
    if dataset.is_virtual:
        raise ValueError(f"{group.name}/songs must not be a virtual dataset")
    if dataset.dtype.names is None or not set(expected_fields).issubset(dataset.dtype.names):
        raise ValueError(f"{group.name}/songs must contain fields {expected_fields}")
    if dataset.dtype.hasobject:
        raise ValueError(f"{group.name}/songs must not contain variable-length fields")
    if dataset.dtype.itemsize > MAX_SONG_DTYPE_BYTES:
        raise ValueError(f"{group.name}/songs compound dtype exceeds {MAX_SONG_DTYPE_BYTES} bytes")
    return dataset


def _sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for child in paths:
        if path.is_dir():
            relative = child.relative_to(path).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8").strip("\x00 ")
    return str(value).strip()


def _canonical(value):
    value = _text(value).upper()
    if not TRACK_RE.fullmatch(value):
        raise ValueError(f"invalid MSD track ID {value!r}")
    return value


def _hdf5_metadata(payload, expected_id):
    import h5py

    with h5py.File(io.BytesIO(payload), "r") as handle:
        analysis_group = _hard_link(handle, "analysis", h5py)
        metadata_group = _hard_link(handle, "metadata", h5py)
        if not isinstance(analysis_group, h5py.Group) or not isinstance(metadata_group, h5py.Group):
            raise ValueError("analysis and metadata must be groups")
        analysis_dataset = _song_dataset(analysis_group, ("track_id", "tempo"), h5py)
        metadata_dataset = _song_dataset(
            metadata_group, ("artist_id", "song_id", "artist_name", "title"), h5py
        )
        analysis = analysis_dataset[0]
        meta = metadata_dataset[0]
        actual = _canonical(analysis["track_id"])
        if actual != expected_id:
            raise ValueError(f"track ID/path mismatch: {expected_id} != {actual}")
        tempo = float(analysis["tempo"])
        return {
            "track_id": actual, "tempo": tempo,
            "artist_id": _text(meta["artist_id"]), "song_id": _text(meta["song_id"]),
            "artist_name": _text(meta["artist_name"]), "title": _text(meta["title"]),
        }


def _scan_tar(path, records):
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            name = member.name
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise ValueError(f"unsafe tar member path: {name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"special tar member is not allowed: {name}")
            if not name.lower().endswith(".h5"):
                continue
            parts = pure.parts
            filename = parts[-1]
            expected = _canonical(filename[:-3])
            if len(parts) < 4 or parts[-4:-1] != (expected[2], expected[3], expected[4]):
                raise ValueError(f"tar member path does not match track ID: {name}")
            if expected in records:
                raise ValueError(f"duplicate track ID across shard archives: {expected}")
            if member.size > MAX_HDF5_MEMBER_BYTES:
                raise ValueError(
                    f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {name}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read tar member: {name}")
            with extracted:
                # Only this one HDF5 payload is resident; the h5py handle closes before scanning onward.
                payload = extracted.read(MAX_HDF5_MEMBER_BYTES + 1)
            if len(payload) > MAX_HDF5_MEMBER_BYTES:
                raise ValueError(
                    f"HDF5 member size exceeds {MAX_HDF5_MEMBER_BYTES} bytes: {name}"
                )
            if len(payload) != member.size:
                raise ValueError(
                    f"HDF5 member size mismatch for {name}: declared {member.size}, read {len(payload)}"
                )
            records[expected] = _hdf5_metadata(payload, expected)
            del payload


def _mxm_ids(directory):
    import csv as csv_module

    ids = set()
    files = sorted(Path(directory).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no musiXmatch text files in {directory}")
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for line in stream:
                line = line.strip()
                if not line or line.startswith(("#", "%")):
                    continue
                row = next(csv_module.reader([line]))
                if row:
                    ids.add(_canonical(row[0]))
    return ids


def _representatives(records, duplicate_groups):
    parent = {track_id: track_id for track_id in records}

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(values):
        present = sorted(set(values) & parent.keys())
        for value in present[1:]:
            left, right = root(present[0]), root(value)
            if left != right:
                parent[max(left, right)] = min(left, right)

    artist_songs = defaultdict(list)
    for track_id, row in records.items():
        if row["artist_id"] and row["song_id"]:
            artist_songs[(row["artist_id"], row["song_id"])].append(track_id)
    for values in artist_songs.values():
        union(values)
    for group in duplicate_groups:
        union(group)
    groups = defaultdict(list)
    for track_id in records:
        groups[root(track_id)].append(track_id)
    return {min(group) for group in groups.values()}, groups


def _lastfm_edges(path, split, eligible):
    edges = set()
    sources_present = set()
    with zipfile.ZipFile(path) as archive:
        for track_id in sorted(eligible):
            member = f"lastfm_{split}/{track_id[2]}/{track_id[3]}/{track_id[4]}/{track_id}.json"
            try:
                info = archive.getinfo(member)
            except KeyError:
                continue
            if info.file_size > MAX_LASTFM_JSON_BYTES:
                raise ValueError(
                    f"Last.fm JSON member size exceeds {MAX_LASTFM_JSON_BYTES} bytes: {member}"
                )
            with archive.open(info, "r") as stream:
                payload = stream.read(MAX_LASTFM_JSON_BYTES + 1)
            if len(payload) > MAX_LASTFM_JSON_BYTES:
                raise ValueError(
                    f"Last.fm JSON member size exceeds {MAX_LASTFM_JSON_BYTES} bytes: {member}"
                )
            if len(payload) != info.file_size:
                raise ValueError(
                    f"Last.fm JSON member size mismatch for {member}: declared {info.file_size}, read {len(payload)}"
                )
            data = json.loads(payload.decode("utf-8"))
            if not isinstance(data, dict) or _canonical(data.get("track_id") or track_id) != track_id:
                raise ValueError(f"Last.fm source ID mismatch for {member}")
            sources_present.add(track_id)
            similars = data.get("similars", [])
            if not isinstance(similars, list):
                raise ValueError(f"invalid similars list in {member}")
            for entry in similars:
                target = entry[0] if isinstance(entry, list) and entry else entry
                target = _canonical(target)
                if target == track_id or target not in eligible:
                    continue
                edges.add(tuple(sorted((track_id, target))))
    return edges, sources_present


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_reference_csv(path, fields):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) \
                or not set(fields).issubset(reader.fieldnames):
            raise ValueError(f"invalid reference CSV header in {path}; expected fields {fields}")
        rows = []
        for row in reader:
            if None in row or any(row.get(field) is None for field in fields):
                raise ValueError(f"invalid reference CSV row in {path}")
            rows.append(row)
    return rows


def _reference_sets(comparison_paths):
    expected = {"catalog", "train_edges", "test_edges"}
    if not isinstance(comparison_paths, dict) or set(comparison_paths) != expected:
        raise ValueError(f"comparison paths must contain exactly {sorted(expected)}")
    paths = {name: Path(path) for name, path in comparison_paths.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"comparison reference {name} must be a regular file: {path}")
    catalog = {_canonical(row["track_id"]) for row in _read_reference_csv(paths["catalog"], ("track_id",))}

    def edges_for(name):
        edges = set()
        for row in _read_reference_csv(paths[name], ("track_id_a", "track_id_b")):
            left, right = _canonical(row["track_id_a"]), _canonical(row["track_id_b"])
            if left == right:
                raise ValueError(f"self-edge in comparison reference {name}: {left}")
            edges.add(tuple(sorted((left, right))))
        return edges

    edges = {"train_edges": edges_for("train_edges"), "test_edges": edges_for("test_edges")}
    hashes = {name: _sha256(path) for name, path in paths.items()}
    return catalog, edges, hashes, paths


def _set_delta(current, reference):
    added = current - reference
    removed = reference - current
    return {"added_count": len(added), "removed_count": len(removed),
            "net_count": len(added) - len(removed)}


def _comparison_report(eligible, train_edges, test_edges, reference_catalog, reference_edges,
                       reference_hashes):
    added_ids = sorted(eligible - reference_catalog)
    removed_ids = sorted(reference_catalog - eligible)
    new_ids = eligible - reference_catalog

    def classes(edges):
        counts = {"new_new": 0, "new_existing": 0, "existing_existing": 0}
        for left, right in edges:
            left_new, right_new = left in new_ids, right in new_ids
            key = "new_new" if left_new and right_new else (
                "new_existing" if left_new or right_new else "existing_existing"
            )
            counts[key] += 1
        return counts

    return {
        "eligible_tracks": {
            **_set_delta(eligible, reference_catalog), "added_ids": added_ids,
            "removed_ids": removed_ids,
        },
        "train_edges": _set_delta(train_edges, reference_edges["train_edges"]),
        "test_edges": _set_delta(test_edges, reference_edges["test_edges"]),
        "edge_classes": {"train": classes(train_edges), "test": classes(test_edges)},
        "reference_sha256": reference_hashes,
    }


def _run_named_shards(shard_paths, output_prefix, lyrics_dir, genres_csv, duplicates_path,
                      lastfm_train_zip, lastfm_test_zip, output_dir, *, comparison_paths=None,
                      third_party_caveat=THIRD_PARTY_CAVEAT, project_use_authorization=None,
                      additional_code_paths=()):
    if not isinstance(output_prefix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", output_prefix):
        raise ValueError("output prefix must be a filename-safe component")
    if not shard_paths or any(not isinstance(label, str) or not label for label in shard_paths):
        raise ValueError("shard paths must be a nonempty mapping with nonempty string labels")
    sources = {
        **{label: Path(path) for label, path in shard_paths.items()},
        "mxm": Path(lyrics_dir),
        "genres": Path(genres_csv), "duplicates": Path(duplicates_path),
        "lastfm_train": Path(lastfm_train_zip), "lastfm_test": Path(lastfm_test_zip),
    }
    comparison = None
    reference_catalog = reference_edges = reference_hashes = {}
    references = {}
    if comparison_paths is not None:
        reference_catalog, reference_edges, reference_hashes, references = _reference_sets(comparison_paths)
        sources.update({f"ab_reference_{key}": path for key, path in references.items()})
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing sources: " + ", ".join(missing))
    output_dir = Path(output_dir).resolve()
    for source in sources.values():
        resolved = source.resolve()
        if resolved == output_dir or resolved in output_dir.parents or output_dir in resolved.parents:
            raise ValueError(f"output directory cannot be inside input source {source}")
    output_names = (
        f"{output_prefix}_catalog_audit.json", f"{output_prefix}_catalog_tracks.csv",
        f"{output_prefix}_lastfm_train_edges.csv", f"{output_prefix}_lastfm_test_edges.csv",
    )
    targets = [output_dir / name for name in output_names]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))

    created_dir = False
    output_dir_existed = output_dir.exists()
    temp_paths = []
    published = []
    try:
        records = {}
        for label in shard_paths:
            _scan_tar(sources[label], records)
        lyrics = _mxm_ids(sources["mxm"])
        genres = parse_tagtraum(sources["genres"])
        duplicate_groups = parse_msd_duplicates(sources["duplicates"])
        artist_song_keys = defaultdict(list)
        for track_id, row in records.items():
            if row["artist_id"] and row["song_id"]:
                artist_song_keys[(row["artist_id"], row["song_id"])].append(track_id)
        representatives, components = _representatives(records, duplicate_groups)
        survivors = {track_id for track_id in representatives if math.isfinite(records[track_id]["tempo"])
                     and records[track_id]["tempo"] > 0}
        eligible = survivors & lyrics & genres.keys()
        tracks = [dict(records[track_id], genre=genres[track_id]) for track_id in sorted(eligible)]
        train_edges, train_sources = _lastfm_edges(sources["lastfm_train"], "train", eligible)
        test_edges, test_sources = _lastfm_edges(sources["lastfm_test"], "test", eligible)
        overlap = train_edges & test_edges
        train_edges -= overlap
        edge_rows = lambda edges: [{"track_id_a": a, "track_id_b": b} for a, b in sorted(edges)]
        if comparison_paths is not None:
            comparison = _comparison_report(
                eligible, train_edges, test_edges, reference_catalog, reference_edges,
                reference_hashes,
            )

        duplicates_removed = len(records) - len(representatives)
        duplicate_component_sizes = sorted((len(group) for group in components.values() if len(group) > 1), reverse=True)
        audit = {
            "status": "complete",
            "source_sha256": {name: _sha256(path) for name, path in sources.items()},
            "code_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in (
                Path(__file__), REPO_ROOT / "src/data/preprocess_audio_window.py",
                REPO_ROOT / "src/data/parse_mxm.py", REPO_ROOT / "src/data/parse_tagtraum.py",
                REPO_ROOT / "src/data/parse_msd_duplicates.py", REPO_ROOT / "src/data/parse_lastfm.py",
                *(Path(path) for path in additional_code_paths),
            )},
            "counts": {
                "audio_tracks": len(records), "artist_song_duplicate_groups": sum(len(v) > 1 for v in artist_song_keys.values()),
                "duplicate_tracks_removed": duplicates_removed,
                "duplicate_components": len(duplicate_component_sizes),
                "duplicate_component_sizes": duplicate_component_sizes,
                "positive_tempo_after_duplicates": len(survivors), "lyrics_ids": len(lyrics),
                "genre_ids": len(genres), "eligible_tracks": len(eligible),
                "train_sources_found": len(train_sources), "test_sources_found": len(test_sources),
                "train_edges": len(train_edges), "test_edges": len(test_edges),
                "train_test_overlap_removed": len(overlap),
            },
            "coverage": {
                "audio_with_lyrics": len(set(records) & lyrics),
                "audio_with_genre": len(set(records) & genres.keys()),
                "audio_with_lyrics_and_genre": len(set(records) & lyrics & genres.keys()),
                "deduplicated_with_lyrics": len(representatives & lyrics),
                "deduplicated_with_genre": len(representatives & genres.keys()),
                "deduplicated_with_lyrics_and_genre": len(representatives & lyrics & genres.keys()),
            },
            "window": {"seconds": 30, "interval": "[0,30)"},
            "channels": CHANNELS,
            "feature_contract": "unnormalized raw 26-channel segment features; cache is built separately",
            "duplicate_policy": "union repeated nonempty (artist_id,song_id) and official duplicate groups; retain lexicographically smallest track ID",
        }
        if third_party_caveat is not None:
            audit["third_party_caveat"] = third_party_caveat
        if project_use_authorization is not None:
            audit["project_use_authorization"] = project_use_authorization
        if comparison is not None:
            audit["ab_comparison"] = comparison
        output_dir.mkdir(parents=True, exist_ok=True)
        created_dir = True
        with tempfile.TemporaryDirectory(dir=output_dir) as tempdir:
            tempdir = Path(tempdir)
            contents = [json.dumps(audit, indent=2, sort_keys=True) + "\n"]
            specs = [
                (f"{output_prefix}_catalog_audit.json", "json"),
                (f"{output_prefix}_catalog_tracks.csv", "tracks"),
                (f"{output_prefix}_lastfm_train_edges.csv", "train"),
                (f"{output_prefix}_lastfm_test_edges.csv", "test"),
            ]
            for filename, kind in specs:
                temp = tempdir / filename
                if kind == "json":
                    temp.write_text(contents[0], encoding="utf-8")
                elif kind == "tracks":
                    _write_csv(temp, ["track_id", "tempo", "artist_id", "song_id", "artist_name", "title", "genre"], tracks)
                else:
                    _write_csv(temp, ["track_id_a", "track_id_b"], edge_rows(train_edges if kind == "train" else test_edges))
                temp_paths.append(temp)
            for source, target in zip(temp_paths, targets):
                os.link(source, target)
                published.append(target)
        return audit
    except BaseException:
        for path in published:
            try:
                path.unlink(missing_ok=True)
            except BaseException:
                # Continue rollback while retaining the exception that caused it.
                pass
        if created_dir and not output_dir_existed:
            try:
                output_dir.rmdir()
            except BaseException:
                pass
        raise


def run(a_tar, b_tar, lyrics_dir, genres_csv, duplicates_path,
        lastfm_train_zip, lastfm_test_zip, output_dir):
    return _run_named_shards(
        {"A": a_tar, "B": b_tar}, "ab", lyrics_dir, genres_csv, duplicates_path,
        lastfm_train_zip, lastfm_test_zip, output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-tar", type=Path, default=REPO_ROOT / "data/msd_hf_pilot/A.tar.gz")
    parser.add_argument("--b-tar", type=Path, default=REPO_ROOT / "data/msd_hf_pilot/B.tar.gz")
    parser.add_argument("--lyrics-dir", type=Path, default=REPO_ROOT / "data/processed/mxm_counts")
    parser.add_argument("--genres", type=Path, default=REPO_ROOT / "data/processed/tagtraum_genres.csv")
    parser.add_argument("--duplicates", type=Path, default=REPO_ROOT / "data/processed/msd_duplicates.txt")
    parser.add_argument("--lastfm-train", type=Path, default=REPO_ROOT / "lastfm_train.zip")
    parser.add_argument("--lastfm-test", type=Path, default=REPO_ROOT / "lastfm_test.zip")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/msd_hf_pilot")
    args = parser.parse_args()
    audit = run(args.a_tar, args.b_tar, args.lyrics_dir, args.genres, args.duplicates,
                args.lastfm_train, args.lastfm_test, args.output_dir)
    print(json.dumps(audit["counts"], sort_keys=True))


if __name__ == "__main__":
    main()

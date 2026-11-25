"""Build the fixed artist-disjoint NeuroTune v1 split and edge manifest."""

import csv
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path


OUTPUT_NAMES = (
    "neurotune_v1_split.json",
    "neurotune_v1_train_edges.csv",
    "neurotune_v1_validation_edges.csv",
    "neurotune_v1_test_edges.csv",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path, required_columns):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(required_columns).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {required_columns}")
        return list(reader)


def _artist_key(row):
    return row.get("artist_id", "").strip() or row.get("artist_name", "").strip() or row["track_id"]


def _test_source_ids(zip_path, track_ids):
    expected = set(track_ids)
    result = set()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.startswith("lastfm_test/") or not member.endswith(".json"):
                continue
            parts = member.split("/")
            if len(parts) != 5:
                continue
            track_id = parts[-1][:-5]
            if track_id not in expected:
                continue
            if parts[1:4] != [track_id[2], track_id[3], track_id[4]]:
                raise ValueError(f"invalid Last.fm test member path: {member}")
            result.add(track_id)
    return result


def build_split(audit_path, catalog_path, train_edges_path, test_edges_path, test_zip_path, output_dir):
    paths = {
        "audit": Path(audit_path), "catalog": Path(catalog_path),
        "train_edges": Path(train_edges_path), "test_edges": Path(test_edges_path),
        "lastfm_test": Path(test_zip_path),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = Path(output_dir)
    targets = [output_dir / name for name in OUTPUT_NAMES]
    collisions = [str(path) for path in targets if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite output(s): {', '.join(collisions)}")

    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise ValueError("catalog audit is not complete")
    if audit.get("window", {}).get("seconds") != 30 or len(audit.get("channels", [])) != 26:
        raise ValueError("catalog audit does not satisfy the 30s/26-channel contract")
    actual_test_hash = sha256_file(paths["lastfm_test"])
    if audit.get("source_sha256", {}).get("lastfm_test") != actual_test_hash:
        raise ValueError("lastfm_test.zip SHA-256 does not match catalog audit")

    catalog = _read_csv(paths["catalog"], ("track_id", "artist_id", "artist_name"))
    track_to_artist = {}
    artists = defaultdict(list)
    for row in catalog:
        track_id = row["track_id"].strip()
        if not track_id or track_id in track_to_artist:
            raise ValueError(f"empty or duplicate catalog track ID: {track_id!r}")
        key = _artist_key({**row, "track_id": track_id})
        track_to_artist[track_id] = key
        artists[key].append(track_id)
    catalog_ids = set(track_to_artist)
    test_source_ids = _test_source_ids(paths["lastfm_test"], catalog_ids)
    test_artists = {track_to_artist[track_id] for track_id in test_source_ids}
    remaining = sorted(set(artists) - test_artists)
    random.Random(684).shuffle(remaining)
    validation_artists = set(remaining[:int(0.1 * len(remaining))])
    train_artists = set(remaining) - validation_artists
    partition_by_artist = {**{key: "train" for key in train_artists},
                           **{key: "validation" for key in validation_artists},
                           **{key: "test" for key in test_artists}}
    partitions = {name: sorted(track for artist in keys for track in artists[artist])
                  for name, keys in (("train", train_artists), ("validation", validation_artists),
                                     ("test", test_artists))}
    track_partition = {track: partition_by_artist[key] for track, key in track_to_artist.items()}
    if sum(map(len, partitions.values())) != len(catalog_ids):
        raise ValueError("not every catalog track was assigned exactly once")
    if (train_artists & validation_artists or train_artists & test_artists or
            validation_artists & test_artists):
        raise ValueError("artist partitions overlap")

    edge_rows = {
        name: _read_csv(paths[name], ("track_id_a", "track_id_b"))
        for name in ("train_edges", "test_edges")
    }
    output_edges = {"train": [], "validation": [], "test": []}
    excluded = {"train_source": 0, "test_source": 0}
    seen = {"train": set(), "validation": set(), "test": set()}
    for source, rows in edge_rows.items():
        for row in rows:
            a, b = row["track_id_a"].strip(), row["track_id_b"].strip()
            if a not in catalog_ids or b not in catalog_ids:
                raise ValueError(f"edge endpoint absent from catalog: {a}, {b}")
            pa, pb = track_partition[a], track_partition[b]
            dest = pa if pa == pb and (source == "train_edges" and pa in ("train", "validation") or source == "test_edges" and pa == "test") else None
            if dest is None:
                if pa != pb:
                    excluded["train_source" if source == "train_edges" else "test_source"] += 1
                continue
            edge = tuple(sorted((a, b)))
            if edge in seen[dest]:
                raise ValueError(f"duplicate edge in {dest} partition: {edge}")
            seen[dest].add(edge)
            output_edges[dest].append(edge)
    if seen["train"] & seen["test"]:
        raise ValueError("train/test edge overlap")
    for rows in output_edges.values():
        rows.sort()
    test_queries = sorted({endpoint for edge in output_edges["test"] for endpoint in edge})
    manifest = {
        "seed": 684,
        "contract": {"window_seconds": 30, "channels": 26},
        "source_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "counts": {
            "tracks": {name: len(ids) for name, ids in partitions.items()},
            "artists": {name: len(keys) for name, keys in (("train", train_artists), ("validation", validation_artists), ("test", test_artists))},
            "edges": {name: len(rows) for name, rows in output_edges.items()},
            "excluded_cross_partition_edges": excluded,
            "test_queries": len(test_queries),
        },
        "track_ids": partitions,
        "artist_keys": {"train": sorted(train_artists), "validation": sorted(validation_artists), "test": sorted(test_artists)},
        "test_source_ids": sorted(test_source_ids),
        "test_query_ids": test_queries,
    }
    return manifest, output_edges


def publish_split(manifest, edges, output_dir):
    """Stage all outputs and publish without overwriting, rolling back partial publication."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in OUTPUT_NAMES]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite NeuroTune v1 split outputs")
    staged = []
    published = []
    try:
        payloads = [json.dumps(manifest, indent=2, sort_keys=True) + "\n"]
        for name in ("train", "validation", "test"):
            import io
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow(("track_id_a", "track_id_b"))
            writer.writerows(edges[name])
            payloads.append(buffer.getvalue())
        for target, payload in zip(targets, payloads):
            stage = output_dir / (target.name + ".stage")
            if stage.exists():
                raise FileExistsError(stage)
            staged.append(stage)
            stage.write_text(payload, encoding="utf-8")
        for stage, target in zip(staged, targets):
            target.hardlink_to(stage)
            published.append(target)
        return targets
    except BaseException:
        for target in published:
            try:
                target.unlink(missing_ok=True)
            except BaseException:
                pass
        raise
    finally:
        for stage in staged:
            try:
                stage.unlink(missing_ok=True)
            except BaseException:
                pass

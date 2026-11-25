"""Build deterministic artist-grouped CV labels for the audited MSD A-D cache."""

import csv
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import h5py
import numpy as np


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _edges(path):
    rows = _csv_rows(path)
    if not {"track_id_a", "track_id_b"}.issubset(rows[0] if rows else {}):
        # Header-only edge lists are valid when they declare the standard columns.
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            fields = next(csv.reader(stream), [])
        if not {"track_id_a", "track_id_b"}.issubset(fields):
            raise ValueError(f"edge CSV missing track_id_a/track_id_b: {path}")
    result = set()
    for row in rows:
        left, right = row["track_id_a"], row["track_id_b"]
        if not left or not right or left == right:
            raise ValueError(f"invalid edge in {path}")
        edge = tuple(sorted((left, right)))
        if edge in result:
            raise ValueError(f"duplicate edge in {path}: {edge}")
        result.add(edge)
    return result


def _artist_key(track_id, row):
    artist_id = (row.get("artist_id") or "").strip()
    artist_name = (row.get("artist_name") or "").strip()
    return "artist_id:" + artist_id if artist_id else ("artist_name:" + artist_name if artist_name else "track_id:" + track_id)


def _legacy_artist_key(track_id, row):
    return (row.get("artist_id") or "").strip() or (row.get("artist_name") or "").strip() or track_id


def _write_edges(path, edges):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("track_id_a", "track_id_b"))
        writer.writerows(sorted(edges))


def build_abcd_cv(*, audit_path, catalog_path, train_edges_path, test_edges_path,
                  cache_path, cache_metadata_path, rejection_path, ab_catalog_path,
                  ab_split_path, ab_test_edges_path, output_dir, seed=684):
    """Validate A-D provenance and publish a manifest plus known-positive and fold edge files."""
    paths = {name: Path(value) for name, value in locals().copy().items() if name.endswith("_path")}
    output_dir = Path(output_dir)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    pilot_dir = paths["ab_catalog_path"].resolve().parent
    resolved_output = output_dir.resolve()
    if resolved_output == pilot_dir or pilot_dir in resolved_output.parents:
        raise ValueError("output directory must not equal or be inside the legacy A/B pilot directory")
    audit = json.loads(paths["audit_path"].read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("window") != {"seconds": 30, "interval": "[0,30)"} or len(audit.get("channels", [])) != 26:
        raise ValueError("audit must be complete and declare the [0,30) 26-channel contract")
    catalog_rows = _csv_rows(paths["catalog_path"])
    catalog = {row["track_id"]: row for row in catalog_rows}
    if len(catalog) != len(catalog_rows) or not catalog:
        raise ValueError("catalog has duplicate or no track IDs")
    metadata = json.loads(paths["cache_metadata_path"].read_text(encoding="utf-8"))
    if (metadata.get("status") != "complete" or metadata.get("window_seconds") != 30
            or metadata.get("interval") != "[0,30)" or metadata.get("channel_count") != 26):
        raise ValueError("cache metadata must be complete and declare the [0,30) 26-channel contract")
    rejected_obj = json.loads(paths["rejection_path"].read_text(encoding="utf-8"))
    if not isinstance(rejected_obj, list) or any(not isinstance(entry, dict) or not isinstance(entry.get("track_id"), str) for entry in rejected_obj):
        raise ValueError("rejection file must be a list of track rejection records")
    rejected_ids = [entry["track_id"] for entry in rejected_obj]
    rejected = set(rejected_ids)
    if len(rejected) != len(rejected_ids):
        raise ValueError("rejection file contains duplicate track IDs")
    with h5py.File(paths["cache_path"], "r") as cache:
        if not {"track_ids", "features", "offsets"}.issubset(cache.keys()):
            raise ValueError("cache HDF5 is missing track_ids/features/offsets")
        raw_cache_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in cache["track_ids"][:]]
        cache_ids = set(raw_cache_ids)
        if raw_cache_ids != sorted(raw_cache_ids):
            raise ValueError("cache HDF5 track_ids must be sorted")
        if len(cache_ids) != len(raw_cache_ids):
            raise ValueError("cache HDF5 contains duplicate track IDs")
        feature_shape, offset_shape = cache["features"].shape, cache["offsets"].shape
        offsets = cache["offsets"][:]
        features = cache["features"]
        if len(feature_shape) != 2 or feature_shape[1] != 26 or features.dtype != np.dtype("float32"):
            raise ValueError("cache features must have float32 shape (N,26)")
        for start in range(0, feature_shape[0], 4096):
            if not np.isfinite(features[start:start + 4096]).all():
                raise ValueError("cache features contain non-finite values")
        attrs = dict(cache.attrs)
    candidate = set(catalog)
    if cache_ids & rejected or cache_ids | rejected != candidate:
        raise ValueError("cache-valid IDs must equal catalog candidates minus rejected IDs")
    if metadata.get("valid_track_ids") != sorted(cache_ids):
        raise ValueError("cache metadata valid_track_ids do not match HDF5 cache")
    if (metadata.get("track_count") != len(cache_ids) or metadata.get("rejected_track_count") != len(rejected)
            or len(feature_shape) != 2 or feature_shape[1] != 26 or offset_shape != (len(cache_ids) + 1,)
            or metadata.get("dtype") != "float32" or metadata.get("channels") != audit["channels"]
            or metadata.get("window_seconds") != 30 or metadata.get("interval") != "[0,30)"
            or metadata.get("normalization") != "raw/unnormalized; deferred until train split"
            or metadata.get("segment_count") != feature_shape[0]
            or not len(offsets) or offsets[0] != 0 or offsets[-1] != feature_shape[0]
            or offsets.dtype.kind not in "iu" or any(left >= right for left, right in zip(offsets, offsets[1:]))):
        raise ValueError("cache metadata counts or HDF5 feature/offset shapes are inconsistent")
    def attr_text(value):
        return value.decode("utf-8") if isinstance(value, bytes) else value

    try:
        hdf_channels = json.loads(attr_text(attrs["channels"]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("cache HDF5 channel attribute is invalid") from error
    if (int(attrs.get("window_seconds", -1)) != metadata["window_seconds"]
            or attr_text(attrs.get("interval")) != metadata["interval"]
            or hdf_channels != metadata["channels"]
            or attr_text(attrs.get("normalization")) != metadata["normalization"]):
        raise ValueError("cache HDF5 attributes do not match cache metadata")
    lengths = metadata.get("sequence_lengths")
    expected_offsets = [0]
    if isinstance(lengths, dict) and set(lengths) == cache_ids and all(
            isinstance(lengths[track_id], int) and lengths[track_id] >= 0 for track_id in cache_ids):
        for track_id in sorted(cache_ids):
            expected_offsets.append(expected_offsets[-1] + lengths[track_id])
    if (not isinstance(lengths, dict) or set(lengths) != cache_ids or expected_offsets != list(offsets)
            or [lengths[track_id] for track_id in sorted(cache_ids)] != list(np.diff(offsets))):
        raise ValueError("cache metadata sequence_lengths do not match HDF5 offsets")
    if metadata.get("rejected") != rejected_obj:
        raise ValueError("cache metadata rejection records do not match rejection file")
    if metadata.get("audit_sha256") != _sha(paths["audit_path"]) or metadata.get("catalog_sha256") != _sha(paths["catalog_path"]):
        raise ValueError("cache metadata audit/catalog hashes do not match input files")

    # Enforce the full cache provenance, including actual A-D archive hashes recorded by the audit.
    cache_provenance = metadata.get("source_sha256")
    if not isinstance(cache_provenance, dict) or cache_provenance != audit.get("source_sha256"):
        raise ValueError("cache metadata archive hashes do not match audit")
    try:
        hdf_source_hashes = json.loads(attr_text(attrs["source_sha256"]))
        hdf_code_hashes = json.loads(attr_text(attrs["code_sha256"]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("cache HDF5 provenance attributes are missing or invalid") from error
    if (attr_text(attrs.get("audit_sha256")) != metadata.get("audit_sha256")
            or attr_text(attrs.get("catalog_sha256")) != metadata.get("catalog_sha256")
            or hdf_source_hashes != metadata.get("source_sha256")
            or hdf_code_hashes != metadata.get("code_sha256")):
        raise ValueError("cache HDF5 provenance attributes do not match cache metadata")
    train_source, official_source = _edges(paths["train_edges_path"]), _edges(paths["test_edges_path"])
    if train_source & official_source:
        raise ValueError("train-source and official-test-source edges overlap")
    ab_test_source = _edges(paths["ab_test_edges_path"])
    ab_rows = _csv_rows(paths["ab_catalog_path"])
    ab_catalog = {row["track_id"]: row for row in ab_rows}
    if len(ab_catalog) != len(ab_rows) or not ab_catalog:
        raise ValueError("legacy A/B catalog has duplicate or no track IDs")
    if any(not set(edge) <= set(ab_catalog) for edge in ab_test_source):
        raise ValueError("legacy A/B test-source edge references a track outside the legacy A/B catalog")
    ab_split = json.loads(paths["ab_split_path"].read_text(encoding="utf-8"))
    if not isinstance(ab_split, dict) or not isinstance(ab_split.get("track_ids"), dict) or not isinstance(ab_split.get("artist_keys"), dict):
        raise ValueError("legacy A/B split is missing track_ids/artist_keys mappings")
    split_tracks = ab_split["track_ids"]
    if any(not isinstance(split_tracks.get(partition), list) for partition in ("train", "validation", "test")):
        raise ValueError("legacy A/B split must list train, validation, and test track IDs")
    for partition in ("train", "validation", "test"):
        ids = split_tracks[partition]
        if any(not isinstance(track_id, str) or not track_id for track_id in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"legacy A/B split {partition} track IDs are invalid or duplicated")
        if not set(ids) <= set(ab_catalog):
            raise ValueError(f"legacy A/B split {partition} references tracks outside its catalog")
    if any(set(split_tracks[left]) & set(split_tracks[right]) for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("legacy A/B split track partitions overlap")
    observed_legacy_keys = {}
    for partition in ("train", "validation", "test"):
        old_partition_tracks = set(split_tracks[partition])
        recorded_keys = ab_split["artist_keys"].get(partition)
        if (not isinstance(recorded_keys, list) or any(not isinstance(key, str) or not key for key in recorded_keys)
                or len(set(recorded_keys)) != len(recorded_keys)):
            raise ValueError(f"legacy A/B split {partition} artist keys are invalid or duplicated")
        observed_keys = {_legacy_artist_key(track_id, ab_catalog[track_id]) for track_id in old_partition_tracks}
        if set(recorded_keys) != observed_keys:
            raise ValueError(f"legacy A/B split {partition} artist keys do not match partition-track artists")
        observed_legacy_keys[partition] = observed_keys
    if any(observed_legacy_keys[left] & observed_legacy_keys[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("legacy A/B split artist partitions overlap")
    old_test_tracks = set(split_tracks["test"])
    old_test = {_artist_key(track_id, ab_catalog[track_id]) for track_id in old_test_tracks}
    tracks = cache_ids
    valid_train = {e for e in train_source if set(e) <= tracks}
    valid_test = {e for e in official_source if set(e) <= tracks}
    known = valid_train | valid_test
    artist_groups = {tid: _artist_key(tid, catalog[tid]) for tid in sorted(tracks)}
    old_artists = {_artist_key(tid, row) for tid, row in ab_catalog.items()}
    eligible = sorted(set(artist_groups.values()) - old_artists - old_test)
    random.Random(seed).shuffle(eligible)
    fold_artists = [set(eligible[i::3]) for i in range(3)]
    # Existing artists are placed in a non-test partition per fold; they remain eligible for train/validation.
    folds = []
    exclusions = {"invalid_endpoint": {"train_source": [], "official_test_source": []},
                  "cross_partition": {"train_source": [], "official_test_source": []},
                  "source_policy": {"train_source": [], "official_test_source": []}}
    for source_name, source in (("train_source", train_source), ("official_test_source", official_source)):
        exclusions["invalid_endpoint"][source_name] = sorted(source - (valid_train if source_name == "train_source" else valid_test))
    for fold in range(3):
        test_artists = fold_artists[fold]
        # Existing A/B artists and all other eligible artists are outside this test fold.
        test_tracks = {tid for tid in tracks if artist_groups[tid] in test_artists}
        remaining = sorted(set(artist_groups.values()) - test_artists)
        random.Random(seed).shuffle(remaining)
        validation_artists = set(remaining[:int(.10 * len(remaining))])
        val_tracks = {tid for tid in tracks if artist_groups[tid] in validation_artists}
        train_tracks = tracks - test_tracks - val_tracks
        assignment = {tid: ("test" if tid in test_tracks else "validation" if tid in val_tracks else "train") for tid in tracks}
        for src_name, src in (("train_source", valid_train), ("official_test_source", valid_test)):
            for edge in sorted(src):
                parts = {assignment[x] for x in edge}
                if len(parts) != 1:
                    exclusions["cross_partition"][src_name].append(list(edge)); continue
                part = next(iter(parts))
                if part != "test" and src_name == "official_test_source":
                    exclusions["source_policy"][src_name].append(list(edge))
        folds.append({"fold": fold, "test_artist_keys": sorted(test_artists), "validation_artist_keys": sorted(validation_artists),
                      "train_artist_keys": sorted(set(artist_groups.values()) - test_artists - validation_artists),
                      "test_track_ids": sorted(test_tracks), "validation_track_ids": sorted(val_tracks), "train_track_ids": sorted(train_tracks),
                      "counts": {}})
    # Fold-specific edge targets are materialized separately below.
    return _publish(paths, output_dir, seed, audit, metadata, catalog_rows, artist_groups, folds, known,
                    train_source, official_source, ab_test_source, exclusions)


def _publish(paths, output_dir, seed, audit, metadata, catalog_rows, artist_groups, folds, known,
             train_source, official_source, ab_test_source, exclusions):
    # Reconstruct assignments for each fold and route independently; stage all outputs before atomic no-clobber links.
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"known_positive_edges.csv": sorted(known)}
    per_fold = []
    fold_exclusions = []
    for fold in range(3):
        test_artists = set(folds[fold]["test_artist_keys"])
        val_artists = set(folds[fold]["validation_artist_keys"])
        groups = artist_groups
        assign = {tid: ("test" if groups[tid] in test_artists else "validation" if groups[tid] in val_artists else "train") for tid in groups}
        targets = {"train": [], "validation": [], "test_train_source": [], "test_official_source": []}
        excluded = {"cross_partition": {"train_source": [], "official_test_source": []},
                    "source_policy": {"train_source": [], "official_test_source": []}}
        for source, name in ((train_source - ab_test_source, "train_source"), (official_source - ab_test_source, "official_test_source")):
            for edge in sorted(source & known):
                if edge[0] not in assign or edge[1] not in assign or assign[edge[0]] != assign[edge[1]]:
                    excluded["cross_partition"][name].append(list(edge))
                    continue
                partition = assign[edge[0]]
                if partition == "test": targets["test_train_source" if name == "train_source" else "test_official_source"].append(edge)
                elif name == "train_source": targets[partition].append(edge)
                else: excluded["source_policy"][name].append(list(edge))
        row = {"fold": fold, "test_artist_keys": sorted(test_artists),
               "validation_artist_keys": sorted(val_artists),
               "train_artist_keys": sorted(set(groups.values()) - test_artists - val_artists),
               "test_track_ids": sorted(t for t in groups if assign[t] == "test"),
               "validation_track_ids": sorted(t for t in groups if assign[t] == "validation"),
               "train_track_ids": sorted(t for t in groups if assign[t] == "train"),
               "counts": {k: len(v) for k, v in targets.items()},
               **{k: [list(e) for e in v] for k, v in targets.items()}}
        per_fold.append(row)
        fold_exclusions.append(excluded)
        for name, edges in targets.items(): files[f"fold_{fold}_{name}_edges.csv"] = edges
    known_digest = hashlib.sha256(json.dumps([list(edge) for edge in sorted(known)], separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest = {"status": "complete", "seed": seed, "fold_count": 3, "validation_fraction": .10,
                 "artist_groups": artist_groups, "track_ids": sorted(artist_groups), "folds": per_fold,
                 "known_positive_edge_count": len(known), "known_positive_edges_sha256": known_digest,
                 "source_edge_counts": {"train_source": len(train_source), "official_test_source": len(official_source)},
                "exclusions": {"invalid_endpoint": exclusions["invalid_endpoint"], "by_fold": fold_exclusions},
                "ab_test_source_policy_exclusions": [list(e) for e in sorted(ab_test_source)],
        "input_sha256": {k: _sha(v) for k, v in sorted(paths.items())},
                 "audit_source_sha256": audit.get("source_sha256"), "cache_metadata_provenance": {k: metadata.get(k) for k in ("audit_sha256", "catalog_sha256", "code_sha256")},
                 "code_sha256": {"src/data/neurotune_v1_abcd_cv.py": _sha(Path(__file__)),
                                 "src/data/preprocess_msd_hf_abcd.py": _sha(Path(__file__).with_name("preprocess_msd_hf_abcd.py")),
                                 "scripts/audit_neurotune_v1_abcd_cv.py": _sha(Path(__file__).resolve().parents[2] / "scripts/audit_neurotune_v1_abcd_cv.py")},
                "policy": "Only artists absent from old A/B catalog are test-eligible; 3 shuffled artist folds; per-fold 10% shuffled validation; train-source labels only outside test; test source lists remain separate."}
    files["manifest.json"] = manifest
    output_paths = [output_dir / name for name in files]
    collision = [str(path) for path in output_paths if path.exists()]
    if collision:
        raise FileExistsError("refusing to overwrite: " + ", ".join(collision))
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix="abcd-cv-stage-", dir=output_dir) as temp:
            stage = Path(temp)
            for name, content in files.items():
                target = stage / name
                if name == "manifest.json": target.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
                else: _write_edges(target, content)
            for name in files:
                target = output_dir / name
                os.link(stage / name, target)
                published.append(target)
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        raise
    return {"manifest": str(output_dir / "manifest.json"), "files": [str(p) for p in output_paths]}

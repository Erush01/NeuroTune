"""Preflight and load the verified A-D artist-grouped CV training inputs."""

import hashlib
import json
import re
from pathlib import Path

import h5py

from src.data.neurotune_v1_abcd_cv import _artist_key, _csv_rows, _edges, _legacy_artist_key, _sha
from src.features_neurotune_v1 import build_neurotune_v1_features, build_d4_neurotune_v1_features
from src.triplets import sample_triplet_schedule
from src.data.parse_mxm import parse_mxm


def build_abcd_cv_fold_mxm(loaded, mxm_path):
    """Load only catalog lyric documents and record their source/mapping provenance."""
    path = Path(mxm_path)
    audit_sources = loaded.get("manifest", {}).get("audit_source_sha256")
    recorded_mxm_hash = audit_sources.get("mxm") if isinstance(audit_sources, dict) else None
    if not isinstance(recorded_mxm_hash, str) or re.fullmatch(r"[0-9a-f]{64}", recorded_mxm_hash) is None:
        raise ValueError("A-D audit MXM source hash is missing or invalid")
    if not path.is_dir():
        raise ValueError("MXM audit provenance requires the source directory")
    input_paths = sorted(source for source in path.rglob("*") if source.is_file())
    source_hashes = {str(source.relative_to(path)): _sha(source) for source in input_paths}
    directory_hash = _mxm_directory_sha256(path)
    if directory_hash != recorded_mxm_hash:
        raise ValueError("MXM source hash does not match A-D audit")

    catalog_ids = set(loaded["catalog"])
    lyrics = parse_mxm(path, track_ids=catalog_ids)
    current_paths = sorted(source for source in path.rglob("*") if source.is_file())
    current_source_hashes = {str(source.relative_to(path)): _sha(source) for source in current_paths}
    current_directory_hash = _mxm_directory_sha256(path)
    if current_source_hashes != source_hashes or current_directory_hash != directory_hash:
        raise ValueError("MXM source changed during parsing")
    missing = catalog_ids - set(lyrics)
    if missing:
        raise ValueError(f"MXM is missing catalog track IDs: {sorted(missing)[:5]}")

    mapping_sha = hashlib.sha256(
        json.dumps(lyrics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return lyrics, {
        "source_sha256": source_hashes,
        "directory_sha256": directory_hash,
        "mapping_sha256": mapping_sha,
        "empty_document_count": sum(not any(counts.values()) for counts in lyrics.values()),
    }


def _mxm_directory_sha256(path):
    """Match scripts/audit_msd_hf_ab.py directory hash framing exactly."""
    digest = hashlib.sha256()
    path = Path(path)
    for source in sorted(source for source in path.rglob("*") if source.is_file()):
        relative = source.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_abcd_cv_fold_features(loaded, fold):
    """Build train-fitted C features for every track in one validated fold."""
    train_ids = fold["train"]
    song_ids = sorted(fold["train"] + fold["validation"] + fold["test"])
    catalog = loaded["catalog"]
    tracks = [catalog[track_id] for track_id in song_ids]
    genres = {track_id: catalog[track_id].get("genre") for track_id in song_ids}
    split_hash = hashlib.sha256(
        json.dumps(loaded["manifest"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    features, manifest = build_neurotune_v1_features(
        loaded["cache_path"], tracks, {}, genres, train_ids, song_ids, split_hash=split_hash
    )
    features.pop("lyrics", None)
    return features, manifest


def build_abcd_cv_development_features(loaded, fold, lyrics):
    """Build train-fitted multimodal features for train/validation tracks only."""
    train_ids = fold["train"]
    song_ids = sorted(fold["train"] + fold["validation"])
    missing_lyrics = sorted(set(song_ids) - set(lyrics))
    if missing_lyrics:
        raise ValueError(f"MXM lyrics are missing train/validation track IDs: {missing_lyrics[:5]}")

    catalog = loaded["catalog"]
    tracks = [catalog[track_id] for track_id in song_ids]
    genres = {track_id: catalog[track_id].get("genre") for track_id in song_ids}
    split_hash = hashlib.sha256(
        json.dumps(loaded["manifest"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    development_lyrics = {track_id: lyrics[track_id] for track_id in song_ids}
    return build_neurotune_v1_features(
        loaded["cache_path"], tracks, development_lyrics, genres, train_ids, song_ids,
        split_hash=split_hash,
    )


def build_abcd_cv_d4_multi_window_features(loaded, fold, lyrics, d4_cache_path):
    """Build fold-local D4 features for training and validation tracks only."""
    train_ids = fold["train"]
    song_ids = sorted(fold["train"] + fold["validation"])
    if set(song_ids) & set(fold.get("test", [])):
        raise ValueError("D4 feature selection must not include test track IDs")
    missing_lyrics = sorted(set(song_ids) - set(lyrics))
    if missing_lyrics:
        raise ValueError(f"MXM lyrics are missing train/validation track IDs: {missing_lyrics[:5]}")

    catalog = loaded["catalog"]
    tracks = [catalog[track_id] for track_id in song_ids]
    genres = {track_id: catalog[track_id].get("genre") for track_id in song_ids}
    split_hash = hashlib.sha256(
        json.dumps(loaded["manifest"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    development_lyrics = {track_id: lyrics[track_id] for track_id in song_ids}
    return build_d4_neurotune_v1_features(
        d4_cache_path, tracks, development_lyrics, genres, train_ids, song_ids,
        split_hash=split_hash,
    )


def build_abcd_cv_fold_schedules(loaded, fold, seed):
    """Build independently indexed train and validation schedules for one fold."""
    schedules = []
    for partition, epochs in (("train", 50), ("validation", 1)):
        candidate_ids = fold[partition]
        candidate_set = set(candidate_ids)
        known_positive_edges = [
            edge for edge in loaded["known_positive_edges"]
            if set(edge) <= candidate_set
        ]
        schedules.append(sample_triplet_schedule(
            candidate_ids,
            fold[partition + "_edges"],
            epochs=epochs,
            seed=seed,
            known_positive_edges=known_positive_edges,
        ))
    return tuple(schedules)


def load_abcd_cv_training(*, audit_path, catalog_path, train_edges_path, test_edges_path,
                          cache_path, cache_metadata_path, rejection_path, ab_catalog_path,
                          ab_split_path, ab_test_edges_path, cv_dir):
    """Validate input provenance and return source-separated fold edge targets."""
    paths = {name: Path(value) for name, value in locals().copy().items() if name.endswith("_path")}
    cv_dir = Path(cv_dir)
    manifest_path = cv_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("fold_count") != 3:
        raise ValueError("A-D CV manifest must be complete and contain three folds")
    recorded_hashes = manifest.get("input_sha256")
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != set(paths):
        raise ValueError("manifest input hash set does not match current inputs")
    for name, path in paths.items():
        if _sha(path) != recorded_hashes[name]:
            raise ValueError(f"manifest input hash mismatch for {name}")

    audit = json.loads(paths["audit_path"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["cache_metadata_path"].read_text(encoding="utf-8"))
    if (manifest.get("audit_source_sha256") != audit.get("source_sha256")
            or manifest.get("cache_metadata_provenance") != {k: metadata.get(k) for k in ("audit_sha256", "catalog_sha256", "code_sha256")}):
        raise ValueError("manifest cache provenance does not match current audit/metadata")

    catalog_rows = _csv_rows(paths["catalog_path"])
    catalog = {row["track_id"]: row for row in catalog_rows}
    if not catalog or len(catalog) != len(catalog_rows):
        raise ValueError("catalog has duplicate or no track IDs")
    rejected_obj = json.loads(paths["rejection_path"].read_text(encoding="utf-8"))
    if not isinstance(rejected_obj, list) or any(not isinstance(row, dict) or not isinstance(row.get("track_id"), str) for row in rejected_obj):
        raise ValueError("rejection file must be a list of track rejection records")
    rejected = {row["track_id"] for row in rejected_obj}
    if len(rejected) != len(rejected_obj) or metadata.get("rejected") != rejected_obj:
        raise ValueError("rejection records are duplicated or differ from cache metadata")

    with h5py.File(paths["cache_path"], "r") as cache:
        if not {"track_ids", "features", "offsets"}.issubset(cache.keys()):
            raise ValueError("cache HDF5 is missing track_ids/features/offsets")
        cache_ids = [v.decode() if isinstance(v, bytes) else str(v) for v in cache["track_ids"][:]]
        attrs = dict(cache.attrs)
        if cache_ids != sorted(cache_ids):
            raise ValueError("cache HDF5 track_ids must be sorted")
        if len(cache_ids) != len(set(cache_ids)):
            raise ValueError("cache HDF5 contains duplicate track IDs")
        offsets = cache["offsets"][:]
        if (cache["features"].ndim != 2 or cache["features"].shape[1] != 26
                or cache["features"].dtype != "float32" or offsets.shape != (len(cache_ids) + 1,)):
            raise ValueError("cache HDF5 features/offset shapes are invalid")
    cache_set = set(cache_ids)
    if cache_set & rejected or cache_set | rejected != set(catalog):
        raise ValueError("cache-valid IDs do not exactly cover catalog candidates minus rejections")
    if manifest.get("track_ids") != cache_ids or metadata.get("valid_track_ids") != cache_ids:
        raise ValueError("manifest/cache metadata track IDs do not match sorted cache track IDs")
    if metadata.get("track_count") != len(cache_ids) or metadata.get("rejected_track_count") != len(rejected):
        raise ValueError("cache metadata counts do not match cache IDs and rejections")

    def text(value):
        return value.decode("utf-8") if isinstance(value, bytes) else value

    try:
        hdf_sources = json.loads(text(attrs["source_sha256"]))
        hdf_code = json.loads(text(attrs["code_sha256"]))
        hdf_channels = json.loads(text(attrs["channels"]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("cache HDF5 provenance attributes are missing or invalid") from error
    if (text(attrs.get("audit_sha256")) != metadata.get("audit_sha256")
            or text(attrs.get("catalog_sha256")) != metadata.get("catalog_sha256")
            or hdf_sources != metadata.get("source_sha256")
            or hdf_code != metadata.get("code_sha256")
            or hdf_channels != metadata.get("channels")
            or text(attrs.get("interval")) != metadata.get("interval")
            or int(attrs.get("window_seconds", -1)) != metadata.get("window_seconds")
            or text(attrs.get("normalization")) != metadata.get("normalization")):
        raise ValueError("cache HDF5 provenance attributes do not match cache metadata")
    if (metadata.get("audit_sha256") != _sha(paths["audit_path"])
            or metadata.get("catalog_sha256") != _sha(paths["catalog_path"])
            or metadata.get("source_sha256") != audit.get("source_sha256")):
        raise ValueError("cache metadata audit/catalog/source provenance mismatch")

    ab_rows = _csv_rows(paths["ab_catalog_path"])
    ab_catalog = {row["track_id"]: row for row in ab_rows}
    if not ab_catalog or len(ab_catalog) != len(ab_rows):
        raise ValueError("legacy A/B catalog has duplicate or no track IDs")
    old_artist_keys = {_artist_key(track_id, row) for track_id, row in ab_catalog.items()}
    legacy_split = json.loads(paths["ab_split_path"].read_text(encoding="utf-8"))
    if not isinstance(legacy_split, dict) or not isinstance(legacy_split.get("track_ids"), dict) or not isinstance(legacy_split.get("artist_keys"), dict):
        raise ValueError("legacy A/B split is missing track_ids/artist_keys mappings")
    split_tracks = legacy_split["track_ids"]
    split_artist_keys = legacy_split["artist_keys"]
    split_sets = {}
    split_artists = {}
    for partition in ("train", "validation", "test"):
        ids = split_tracks.get(partition)
        keys = split_artist_keys.get(partition)
        if not isinstance(ids, list) or any(not isinstance(track_id, str) or not track_id for track_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"legacy A/B split {partition} track IDs are invalid or duplicated")
        if not set(ids) <= set(ab_catalog):
            raise ValueError(f"legacy A/B split {partition} references tracks outside its catalog")
        if not isinstance(keys, list) or any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError(f"legacy A/B split {partition} artist keys are invalid or duplicated")
        observed_keys = {_legacy_artist_key(track_id, ab_catalog[track_id]) for track_id in ids}
        if set(keys) != observed_keys:
            raise ValueError(f"legacy A/B split {partition} artist keys do not match tracks")
        split_sets[partition] = set(ids)
        split_artists[partition] = observed_keys
    if (split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"]
            or split_sets["validation"] & split_sets["test"]
            or set().union(*split_sets.values()) != set(ab_catalog)):
        raise ValueError("legacy A/B split track partitions must be disjoint and cover the catalog")
    if (split_artists["train"] & split_artists["validation"] or split_artists["train"] & split_artists["test"]
            or split_artists["validation"] & split_artists["test"]):
        raise ValueError("legacy A/B split artist partitions overlap")
    expected_groups = {track_id: _artist_key(track_id, catalog[track_id]) for track_id in cache_ids}
    if manifest.get("artist_groups") != expected_groups:
        raise ValueError("manifest artist groups do not match catalog artist assignments")
    ab_test_source = _edges(paths["ab_test_edges_path"])
    if any(not set(edge) <= set(ab_catalog) for edge in ab_test_source):
        raise ValueError("legacy A/B test-source edge references a track outside the legacy A/B catalog")
    recorded_ab_exclusions = manifest.get("ab_test_source_policy_exclusions")
    if (not isinstance(recorded_ab_exclusions, list)
            or any(not isinstance(edge, list) or len(edge) != 2 for edge in recorded_ab_exclusions)
            or len({tuple(edge) for edge in recorded_ab_exclusions}) != len(recorded_ab_exclusions)
            or {tuple(edge) for edge in recorded_ab_exclusions} != ab_test_source):
        raise ValueError("manifest old A/B test-source exclusion set does not match full input")

    train_source = _edges(paths["train_edges_path"])
    official_source = _edges(paths["test_edges_path"])
    if train_source & official_source:
        raise ValueError("train-source and official-test-source edges overlap")
    if manifest.get("source_edge_counts") != {"train_source": len(train_source), "official_test_source": len(official_source)}:
        raise ValueError("manifest source edge counts do not match input sources")
    expected_known = {edge for edge in train_source | official_source if set(edge) <= cache_set}
    known_positive_edges = _edges(cv_dir / "known_positive_edges.csv")
    if not ab_test_source <= known_positive_edges:
        raise ValueError("legacy A/B test-source edge missing from known-positive set")
    known_digest = hashlib.sha256(json.dumps([list(edge) for edge in sorted(known_positive_edges)], separators=(",", ":")).encode("utf-8")).hexdigest()
    if (known_positive_edges != expected_known
            or manifest.get("known_positive_edge_count") != len(known_positive_edges)
            or manifest.get("known_positive_edges_sha256") != known_digest):
        raise ValueError("known-positive set does not match manifest digest/count and cache-valid source edge union")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("manifest must declare three folds")
    loaded_folds = []
    target_names = ("train", "validation", "test_train_source", "test_official_source")
    for index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError(f"fold {index} entry must be a mapping")
        if not isinstance(fold.get("counts"), dict):
            raise ValueError(f"fold {index} counts must be a mapping")
        partitions = {name: fold.get(name + "_track_ids") for name in ("train", "validation", "test")}
        if any(not isinstance(ids, list) for ids in partitions.values()):
            raise ValueError(f"fold {index} must list train/validation/test track IDs")
        for ids in partitions.values():
            if len(ids) != len(set(ids)):
                raise ValueError(f"fold {index} contains duplicate partition track IDs")
        if (set(partitions["train"]) & set(partitions["validation"])
                or set(partitions["train"]) & set(partitions["test"])
                or set(partitions["validation"]) & set(partitions["test"])
                or set().union(*(set(v) for v in partitions.values())) != cache_set):
            raise ValueError(f"fold {index} track partitions must be disjoint and cover all cache IDs")
        if any(not set(ids) <= cache_set for ids in partitions.values()):
            raise ValueError(f"fold {index} contains track IDs outside the cache")
        groups = manifest.get("artist_groups")
        if not isinstance(groups, dict) or set(groups) != cache_set:
            raise ValueError("manifest artist groups do not cover cache track IDs")
        recorded_artists = {name: fold.get(name + "_artist_keys") for name in partitions}
        if any(not isinstance(keys, list) or len(keys) != len(set(keys)) for keys in recorded_artists.values()):
            raise ValueError(f"fold {index} artist partition keys are invalid or duplicated")
        artist_partitions = {name: set(recorded_artists[name]) for name in partitions}
        observed = {name: {groups[track_id] for track_id in ids} for name, ids in partitions.items()}
        if any(artist_partitions[name] != observed[name] for name in partitions):
            raise ValueError(f"fold {index} artist keys do not match track partitions")
        if (artist_partitions["train"] & artist_partitions["validation"]
                or artist_partitions["train"] & artist_partitions["test"]
                or artist_partitions["validation"] & artist_partitions["test"]):
            raise ValueError(f"fold {index} artist partitions overlap")
        if artist_partitions["test"] & old_artist_keys:
            raise ValueError(f"fold {index} test artists overlap the legacy A/B catalog")
        targets = {name: _edges(cv_dir / f"fold_{index}_{name}_edges.csv") for name in target_names}
        if (not targets["train"] <= train_source or not targets["validation"] <= train_source
                or not targets["test_train_source"] <= train_source
                or not targets["test_official_source"] <= official_source):
            raise ValueError(f"fold {index} target violates source policy")
        all_targets = set().union(*targets.values())
        if all_targets & ab_test_source:
            raise ValueError(f"fold {index} target contains an old A/B test-source edge")
        for name in ("train", "validation"):
            if any(not set(edge) <= set(partitions[name]) for edge in targets[name]):
                raise ValueError(f"fold {index} {name} target references tracks outside its partition")
        for source_name, name in (("test_train_source", "test"), ("test_official_source", "test")):
            if any(not set(edge) <= set(partitions[name]) for edge in targets[source_name]):
                raise ValueError(f"fold {index} test target references tracks outside its partition")
        assignment = {track_id: partition for partition, ids in partitions.items() for track_id in ids}
        expected_targets = {name: set() for name in target_names}
        for source_name, source_edges in (("train_source", train_source), ("official_test_source", official_source)):
            for edge in (source_edges & expected_known) - ab_test_source:
                left_partition, right_partition = assignment[edge[0]], assignment[edge[1]]
                if left_partition != right_partition:
                    continue
                partition = left_partition
                if source_name == "train_source":
                    target_name = {"train": "train", "validation": "validation", "test": "test_train_source"}[partition]
                    expected_targets[target_name].add(edge)
                elif partition == "test":
                    expected_targets["test_official_source"].add(edge)
        for name in target_names:
            recorded_edges = fold.get(name)
            expected_list = [list(edge) for edge in sorted(expected_targets[name])]
            if (targets[name] != expected_targets[name] or recorded_edges != expected_list
                    or fold.get("counts", {}).get(name) != len(expected_targets[name])):
                raise ValueError(f"fold {index} recomputed targets do not match files and manifest")
        loaded_folds.append({"fold": index, **partitions, **{name + "_edges": sorted(edges) for name, edges in targets.items()}})

    return {"manifest": manifest, "cache_ids": cache_ids, "catalog": catalog,
            "cache_path": paths["cache_path"], "known_positive_edges": known_positive_edges,
            "ab_test_source_edges": ab_test_source,
            "train_source_edges": train_source, "official_test_source_edges": official_source,
            "folds": loaded_folds}

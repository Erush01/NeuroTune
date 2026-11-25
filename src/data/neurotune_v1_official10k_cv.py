"""Pure deterministic artist-grouped CV and edge-validation split construction."""

import random
from collections.abc import Mapping


def _artist_key(track_id, metadata):
    artist_id = (metadata.get("artist_id") or "").strip()
    artist_name = (metadata.get("artist_name") or "").strip()
    if artist_id:
        return "artist_id:" + artist_id
    if artist_name:
        return "artist_name:" + artist_name
    return "track_id:" + track_id


def _edge(edge):
    if isinstance(edge, (str, bytes)):
        raise ValueError("each positive edge must be an iterable of exactly two track IDs")
    try:
        endpoints = tuple(edge)
    except TypeError as exc:
        raise ValueError("each positive edge must be an iterable of exactly two track IDs") from exc
    if len(endpoints) != 2:
        raise ValueError("each positive edge must contain exactly two endpoints")
    if any(not isinstance(endpoint, str) or not endpoint.strip() for endpoint in endpoints):
        raise ValueError("positive edge endpoints must be nonempty strings")
    left, right = endpoints
    if left == right:
        raise ValueError("positive edges must connect distinct tracks")
    return tuple(sorted((left, right)))


def build_cv_audit(track_metadata, positive_edges, seed=684):
    """Return three artist-grouped folds, with deterministic 20% edge validation.

    ``track_metadata`` maps track IDs to ``artist_id``/``artist_name`` values;
    ``positive_edges`` contains canonicalizable undirected track pairs.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(track_metadata, Mapping):
        raise ValueError("track metadata must be a mapping of track IDs to metadata mappings")
    for track_id, metadata in track_metadata.items():
        if not isinstance(track_id, str) or not track_id.strip():
            raise ValueError("track IDs must be nonempty strings")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"metadata for track {track_id!r} must be a mapping")
        for field in ("artist_id", "artist_name"):
            value = metadata.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"metadata field {field!r} for track {track_id!r} must be a string or None")
    tracks = set(track_metadata)
    if len(tracks) != len(track_metadata) or not tracks:
        raise ValueError("track metadata must contain at least one unique track")
    try:
        edge_iterator = iter(positive_edges)
    except TypeError as exc:
        raise ValueError("positive_edges must be an iterable of two-endpoint edges") from exc
    edges = {_edge(edge) for edge in edge_iterator}
    unknown = {track for pair in edges for track in pair} - tracks
    if unknown:
        raise ValueError(f"positive edges reference tracks outside catalog: {sorted(unknown)[:5]}")

    artist_tracks = {}
    artist_groups = {}
    for track_id in sorted(tracks):
        key = _artist_key(track_id, track_metadata[track_id])
        artist_groups[track_id] = key
        artist_tracks.setdefault(key, []).append(track_id)
    artists = sorted(artist_tracks)
    random.Random(seed).shuffle(artists)
    fold_tracks = [set() for _ in range(3)]
    for index, key in enumerate(artists):
        fold_tracks[index % 3].update(artist_tracks[key])

    folds = []
    for fold_index, test_tracks in enumerate(fold_tracks):
        test_edges = sorted(edge for edge in edges if edge[0] in test_tracks and edge[1] in test_tracks)
        non_test_tracks = tracks - test_tracks
        pool = sorted(edge for edge in edges if edge[0] in non_test_tracks and edge[1] in non_test_tracks)
        shuffled_pool = list(pool)
        random.Random(seed + fold_index).shuffle(shuffled_pool)
        validation_count = int(0.20 * len(pool))
        validation_edges = sorted(shuffled_pool[:validation_count])
        validation_set = set(validation_edges)
        train_edges = sorted(edge for edge in pool if edge not in validation_set)
        folds.append({
            "fold": fold_index,
            "track_ids": sorted(test_tracks),
            "test_edges": [list(edge) for edge in test_edges],
            "non_test_pool_edges": [list(edge) for edge in pool],
            "train_edges": [list(edge) for edge in train_edges],
            "validation_edges": [list(edge) for edge in validation_edges],
            "counts": {
                "test_tracks": len(test_tracks), "test_edges": len(test_edges),
                "non_test_pool_edges": len(pool), "train_edges": len(train_edges),
                "validation_edges": len(validation_edges),
            },
        })
    fold_by_track = {track: index for index, values in enumerate(fold_tracks) for track in values}
    cross_edges = sorted(edge for edge in edges if fold_by_track[edge[0]] != fold_by_track[edge[1]])
    return {
        "status": "complete", "seed": seed, "fold_count": 3,
        "track_ids": sorted(tracks), "artist_groups": artist_groups,
        "source_edges": [list(edge) for edge in sorted(edges)],
        "cross_fold_edges": [list(edge) for edge in cross_edges],
        "folds": folds,
        "counts": {"tracks": len(tracks), "source_edges": len(edges),
                   "cross_fold_edges": len(cross_edges)},
        "split_policy": "sorted artist keys shuffled with seed; round-robin across folds; per-fold edge validation shuffled with seed+fold and floor(20%)",
    }

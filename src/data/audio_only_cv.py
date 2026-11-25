import random

from src.data.audio_only_audit import _artist_key, build_audio_only_audit


def build_audio_only_cv_audit(tracks, edges, duplicate_groups=None, fold_count=3, seed=684):
    tracks = list(tracks)
    edges = list(edges)
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError("fold_count must be an integer greater than or equal to 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    audit = build_audio_only_audit(tracks, edges, duplicate_groups=duplicate_groups, seed=seed)
    eligible_ids = {
        track_id
        for track_ids in audit["splits"]["artist_disjoint"]["track_ids"].values()
        for track_id in track_ids
    }
    tracks_by_id = {track["track_id"]: track for track in tracks}

    artists_to_tracks = {}
    for track_id in sorted(eligible_ids):
        artists_to_tracks.setdefault(_artist_key(tracks_by_id[track_id]), []).append(track_id)
    artists = sorted(artists_to_tracks)
    random.Random(seed).shuffle(artists)
    artist_folds = [artists[index::fold_count] for index in range(fold_count)]

    track_fold = {}
    folds = {}
    for fold_index, artist_keys in enumerate(artist_folds):
        track_ids = sorted(
            track_id
            for artist_key in artist_keys
            for track_id in artists_to_tracks[artist_key]
        )
        for track_id in track_ids:
            track_fold[track_id] = fold_index
        folds[str(fold_index)] = {
            "track_ids": track_ids,
            "track_count": len(track_ids),
            "artist_count": len(artist_keys),
            "positive_edges": [],
            "positive_edge_count": 0,
        }

    unique_edges = {
        tuple(sorted((edge["source"], edge["target"])))
        for edge in edges
        if edge["source"] != edge["target"]
        and edge["source"] in eligible_ids
        and edge["target"] in eligible_ids
    }
    cross_fold_edges = 0
    for source, target in sorted(unique_edges):
        source_fold, target_fold = track_fold[source], track_fold[target]
        if source_fold == target_fold:
            fold = folds[str(source_fold)]
            fold["positive_edges"].append([source, target])
            fold["positive_edge_count"] += 1
        else:
            cross_fold_edges += 1

    return {
        "status": "ready",
        "seed": seed,
        "fold_count": fold_count,
        "source_counts": audit["source_counts"],
        "duplicate_counts": audit["duplicates"],
        "catalog_counts": {
            "eligible_tracks": len(eligible_ids),
            "unique_edges_within_catalog": len(unique_edges),
        },
        "fold_assignment_policy": (
            "sort artist keys; shuffle with seed; assign artists[i::fold_count]"
        ),
        "folds": folds,
        "cross_fold_edges": cross_fold_edges,
    }

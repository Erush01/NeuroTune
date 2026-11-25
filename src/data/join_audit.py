import random
import statistics
from collections import Counter


def _artist_key(track):
    return track.get("artist_id") or track.get("artist_name") or track["track_id"]


def _split_tracks(tracks, seed):
    by_artist = {}
    for track in tracks:
        by_artist.setdefault(_artist_key(track), []).append(track["track_id"])
    artists = sorted(by_artist)
    random.Random(seed).shuffle(artists)
    n = len(artists)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    groups = {
        "train": artists[:n_train],
        "validation": artists[n_train:n_train + n_val],
        "test": artists[n_train + n_val:],
    }
    return {
        "track_ids": {
            part: sorted(track_id for artist in artist_ids for track_id in by_artist[artist])
            for part, artist_ids in groups.items()
        },
        "track_counts": {
            part: sum(len(by_artist[artist]) for artist in artist_ids)
            for part, artist_ids in groups.items()
        },
        "artist_counts": {part: len(artist_ids) for part, artist_ids in groups.items()},
    }


def build_audit(tracks, lyrics, genres, edges, seed=684, duplicate_groups=None):
    msd = {track["track_id"]: track for track in tracks}
    msd_ids = set(msd)
    lyric_ids = set(lyrics)
    genre_ids = set(genres)
    mxm_overlap = msd_ids & lyric_ids
    genre_overlap = mxm_overlap & genre_ids
    lastfm_ids = {edge[key] for edge in edges for key in ("source", "target")}
    final_overlap = genre_overlap & lastfm_ids

    bpm_valid = {
        track_id for track_id, track in msd.items()
        if isinstance(track.get("tempo"), (int, float)) and track["tempo"] > 0
    }
    song_tracks = {}
    for track in tracks:
        artist_id, song_id = track.get("artist_id"), track.get("song_id")
        if artist_id and song_id:
            song_tracks.setdefault((artist_id, song_id), []).append(track["track_id"])
    artist_song_duplicate_track_ids = {
        track_id
        for track_ids in song_tracks.values() if len(track_ids) > 1
        for track_id in sorted(track_ids)[1:]
    }
    official_duplicate_track_ids = set()
    official_groups_in_catalog = 0
    for group in duplicate_groups or ():
        matching_ids = sorted(set(group) & msd_ids)
        if matching_ids:
            official_groups_in_catalog += 1
        if len(matching_ids) > 1:
            official_duplicate_track_ids.update(matching_ids[1:])
    duplicate_track_ids = artist_song_duplicate_track_ids | official_duplicate_track_ids
    full_feature_ids = (msd_ids & lyric_ids & genre_ids & bpm_valid) - duplicate_track_ids
    unique_edges = set()
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source != target:
            unique_edges.add(tuple(sorted((source, target))))
    retained_edges = sorted(
        (source, target) for source, target in unique_edges
        if source in full_feature_ids and target in full_feature_ids
    )
    degree = Counter(track_id for pair in retained_edges for track_id in pair)
    split = _split_tracks([msd[track_id] for track_id in sorted(full_feature_ids)], seed)
    split_for = {
        track_id: part for part, ids in split["track_ids"].items() for track_id in ids
    }
    edge_counts = {part: 0 for part in ("train", "validation", "test")}
    cross_partition_edges = 0
    for source, target in retained_edges:
        if split_for[source] == split_for[target]:
            edge_counts[split_for[source]] += 1
        else:
            cross_partition_edges += 1
    tempos = [msd[track_id]["tempo"] for track_id in msd_ids if track_id in bpm_valid]
    duplicate_track_rows = sum(track.get("duplicate_rows", 0) for track in tracks)
    artist_song_keys = [
        (track.get("artist_id"), track.get("song_id")) for track in tracks
        if track.get("artist_id") and track.get("song_id")
    ]
    duplicate_artist_song_rows = len(artist_song_keys) - len(set(artist_song_keys))
    degree_values = list(degree.values())
    degree_distribution = {
        "tracks_with_degree": len(degree_values),
        "min": min(degree_values) if degree_values else 0,
        "median": statistics.median(degree_values) if degree_values else 0,
        "max": max(degree_values) if degree_values else 0,
        "mean": statistics.mean(degree_values) if degree_values else 0,
    }
    return {
        "status": (
            "blocked_no_artist_disjoint_test_edges"
            if not edge_counts["test"] else "ready"
        ),
        "seed": seed,
        "source_counts": {
            "msd": len(msd_ids),
            "mxm": len(lyric_ids),
            "tagtraum_cd2c": len(genre_ids),
            "lastfm_edges_raw": len(edges),
            "lastfm_tracks": len(lastfm_ids),
        },
        "intersections": {
            "msd_and_mxm": len(mxm_overlap),
            "msd_mxm_cd2c": len(genre_overlap),
            "covered_by_lastfm": len(final_overlap),
            "full_feature_tracks": len(full_feature_ids),
        },
        "duplicates": {
            "repeated_msd_track_id_rows": duplicate_track_rows,
            "repeated_artist_song_rows": duplicate_artist_song_rows,
            "artist_song_duplicate_tracks_excluded": len(artist_song_duplicate_track_ids),
            "official_duplicate_groups_in_catalog": official_groups_in_catalog,
            "official_duplicate_tracks_excluded": len(official_duplicate_track_ids),
            "unique_duplicate_tracks_excluded": len(duplicate_track_ids),
            "policy": "For repeated (artist_id,song_id) keys and official duplicate groups, keep the lexicographically smallest track_id.",
        },
        "artists": {
            "count": len({_artist_key(track) for track in tracks}),
            "unique_artist_ids": len({track["artist_id"] for track in tracks if track.get("artist_id")}),
        },
        "genres": {
            "source_counts": dict(sorted(Counter(genres.values()).items())),
            "msd_mxm_cd2c_overlap_counts": dict(
                sorted(Counter(genres[track_id] for track_id in genre_overlap).items())
            ),
        },
        "lyrics": {
            "coverage_tracks": len(msd_ids & lyric_ids),
            "coverage_fraction_of_msd": len(msd_ids & lyric_ids) / len(msd_ids) if msd_ids else 0,
        },
        "bpm": {
            "valid_positive_tracks": len(bpm_valid),
            "missing_or_invalid_tracks": len(msd_ids - bpm_valid),
            "min": min(tempos) if tempos else None,
            "median": statistics.median(tempos) if tempos else None,
            "max": max(tempos) if tempos else None,
        },
        "positive_edges": {
            "unique_undirected_observed_edges": len(unique_edges),
            "retained_full_feature_edges": len(retained_edges),
            "degree_distribution": degree_distribution,
        },
        "splits": {
            "policy": "artist-disjoint; artists shuffled with the recorded seed; 80/10/10 by artist count",
            "artist_disjoint": {
                **split,
                "usable_positive_edges": edge_counts,
                "cross_partition_edges_excluded": cross_partition_edges,
            },
        },
        "feasibility_review": {
            "test_positive_edges": edge_counts["test"],
            "minimum_test_edges_predeclared": None,
            "note": (
                "Assess the measured artist-disjoint test-edge count before relying "
                "on retrieval metrics; the project plan does not predeclare a minimum."
            ),
        },
        "catalog_policy": "Use all tracks with valid BPM, lyrics, and primary genre; no random catalog subsampling.",
        "limitations": [
            "Counts describe observed Last.fm links, not verified dissimilarity for absent links.",
            "Genre annotations may partly derive from Last.fm metadata.",
            "Repeated IDs and artist/song keys are counted; official MSD duplicate-list reconciliation remains a separate input audit.",
        ],
    }

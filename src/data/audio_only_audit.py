import random
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
    n_validation = int(n * 0.1)
    groups = {
        "train": artists[:n_train],
        "validation": artists[n_train:n_train + n_validation],
        "test": artists[n_train + n_validation:],
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


def build_audio_only_audit(tracks, edges, duplicate_groups=None, seed=684):
    msd = {track["track_id"]: track for track in tracks}
    msd_ids = set(msd)

    song_tracks = {}
    for track in tracks:
        artist_id, song_id = track.get("artist_id"), track.get("song_id")
        if artist_id and song_id:
            song_tracks.setdefault((artist_id, song_id), []).append(track["track_id"])
    artist_song_duplicate_ids = {
        track_id
        for track_ids in song_tracks.values() if len(track_ids) > 1
        for track_id in sorted(track_ids)[1:]
    }

    official_duplicate_ids = set()
    official_groups_in_catalog = 0
    for group in duplicate_groups or ():
        matching_ids = sorted(set(group) & msd_ids)
        if matching_ids:
            official_groups_in_catalog += 1
        if len(matching_ids) > 1:
            official_duplicate_ids.update(matching_ids[1:])
    duplicate_ids = artist_song_duplicate_ids | official_duplicate_ids
    eligible_ids = msd_ids - duplicate_ids

    unique_edges = set()
    lastfm_ids = set()
    for edge in edges:
        source, target = edge["source"], edge["target"]
        lastfm_ids.update((source, target))
        if source != target:
            unique_edges.add(tuple(sorted((source, target))))
    eligible_edges = sorted(
        (source, target)
        for source, target in unique_edges
        if source in eligible_ids and target in eligible_ids
    )
    endpoints_with_any_link = {endpoint for pair in eligible_edges for endpoint in pair}

    split = _split_tracks([msd[track_id] for track_id in sorted(eligible_ids)], seed)
    split_for = {
        track_id: part
        for part, track_ids in split["track_ids"].items()
        for track_id in track_ids
    }
    edge_counts = {part: 0 for part in ("train", "validation", "test")}
    cross_partition_edges = 0
    for source, target in eligible_edges:
        if split_for[source] == split_for[target]:
            edge_counts[split_for[source]] += 1
        else:
            cross_partition_edges += 1

    return {
        "status": "ready" if edge_counts["test"] else "blocked_no_artist_disjoint_test_edges",
        "seed": seed,
        "source_counts": {
            "msd": len(msd_ids),
            "lastfm_edges_raw": len(edges),
            "lastfm_tracks": len(lastfm_ids),
        },
        "duplicates": {
            "repeated_msd_track_id_rows": len(tracks) - len(msd_ids),
            "repeated_artist_song_rows": sum(len(ids) - 1 for ids in song_tracks.values()),
            "artist_song_duplicate_tracks_excluded": len(artist_song_duplicate_ids),
            "official_duplicate_groups_in_catalog": official_groups_in_catalog,
            "official_duplicate_tracks_excluded": len(official_duplicate_ids),
            "unique_duplicate_tracks_excluded": len(duplicate_ids),
            "policy": "For repeated (artist_id,song_id) keys and official duplicate groups, keep the lexicographically smallest track_id.",
        },
        "eligible_track_count": len(eligible_ids),
        "positive_edges": {
            "unique_undirected_observed_edges": len(unique_edges),
            "unique_edges_within_catalog": len(eligible_edges),
            "endpoints_with_any_link": len(endpoints_with_any_link),
        },
        "splits": {
            "policy": "artist-disjoint; artists shuffled with the recorded seed; 80/10/10 by artist count",
            "artist_disjoint": {
                **split,
                "usable_positive_edges": edge_counts,
                "cross_partition_edges_excluded": cross_partition_edges,
            },
        },
        "catalog_policy": "Use all MSD tracks after duplicate reconciliation; BPM, lyrics, and genre are not filters.",
    }

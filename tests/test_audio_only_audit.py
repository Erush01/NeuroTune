import json
import unittest

from src.data.audio_only_audit import build_audio_only_audit


def track(track_id, artist_id=None, song_id=None, tempo=120):
    return {
        "track_id": track_id,
        "artist_id": artist_id,
        "song_id": song_id,
        "artist_name": artist_id,
        "tempo": tempo,
    }


def edge(source, target):
    return {"source": source, "target": target}


class AudioOnlyAuditTests(unittest.TestCase):
    def test_duplicate_policies_and_invalid_bpm_retained(self):
        tracks = [
            track("B", "artist-1", "song-1", tempo=None),
            track("A", "artist-1", "song-1"),
            track("D", "artist-2", "song-2"),
            track("C", "artist-2", "song-3"),
            track("E", "artist-3", "song-4"),
        ]

        result = build_audio_only_audit(
            tracks, [], duplicate_groups=[["E", "D"]]
        )

        self.assertEqual(result["source_counts"]["msd"], 5)
        self.assertEqual(result["duplicates"]["artist_song_duplicate_tracks_excluded"], 1)
        self.assertEqual(result["duplicates"]["official_duplicate_tracks_excluded"], 1)
        self.assertEqual(result["duplicates"]["unique_duplicate_tracks_excluded"], 2)
        self.assertEqual(
            sorted(sum(result["splits"]["artist_disjoint"]["track_ids"].values(), [])),
            ["A", "C", "D"],
        )

    def test_deterministic_artist_disjoint_splits_and_counts(self):
        tracks = [track(f"T{i:02}", f"artist-{i}", f"song-{i}") for i in range(20)]
        first = build_audio_only_audit(tracks, [], seed=684)
        second = build_audio_only_audit(list(reversed(tracks)), [], seed=684)
        split = first["splits"]["artist_disjoint"]

        self.assertEqual(first, second)
        self.assertEqual(split["track_counts"], {"train": 16, "validation": 2, "test": 2})
        artist_for = {item["track_id"]: item["artist_id"] for item in tracks}
        split_artists = [
            {artist_for[track_id] for track_id in ids}
            for ids in split["track_ids"].values()
        ]
        self.assertEqual(set().union(*split_artists), {f"artist-{i}" for i in range(20)})
        self.assertTrue(all(not (left & right) for i, left in enumerate(split_artists) for right in split_artists[i + 1 :]))
        json.dumps(first)

    def test_edges_are_unique_undirected_and_cross_split_edges_excluded(self):
        tracks = [track(f"T{i:02}", f"artist-{i}", f"song-{i}") for i in range(20)]
        split = build_audio_only_audit(tracks, []) ["splits"]["artist_disjoint"]["track_ids"]
        train_a, train_b = split["train"][:2]
        validation = split["validation"][0]
        outside = "not-in-catalog"
        edges = [
            edge(train_a, train_b),
            edge(train_b, train_a),
            edge(train_a, train_a),
            edge(train_a, validation),
            edge(train_a, outside),
        ]

        result = build_audio_only_audit(tracks, edges)

        self.assertEqual(result["positive_edges"]["unique_undirected_observed_edges"], 3)
        self.assertEqual(result["positive_edges"]["unique_edges_within_catalog"], 2)
        self.assertEqual(result["positive_edges"]["endpoints_with_any_link"], 3)
        self.assertEqual(
            result["splits"]["artist_disjoint"]["usable_positive_edges"],
            {"train": 1, "validation": 0, "test": 0},
        )
        self.assertEqual(
            result["splits"]["artist_disjoint"]["cross_partition_edges_excluded"], 1
        )


if __name__ == "__main__":
    unittest.main()

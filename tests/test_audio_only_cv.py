import json
import unittest

from src.data.audio_only_cv import build_audio_only_cv_audit


def track(track_id, artist_id=None, artist_name=None, song_id=None):
    return {
        "track_id": track_id,
        "artist_id": artist_id,
        "artist_name": artist_name,
        "song_id": song_id,
    }


def edge(source, target):
    return {"source": source, "target": target}


class AudioOnlyCVTests(unittest.TestCase):
    def test_deterministic_round_robin_artist_assignment_and_fallback(self):
        tracks = [
            track("T-a1", "a", "ignored"),
            track("T-a2", "a", "ignored"),
            track("T-b", None, "b"),
            track("T-b2", None, "b"),
            track("T-c", None, None),
            track("T-d", "d"),
            track("T-e", "e"),
        ]
        first = build_audio_only_cv_audit(tracks, [])
        second = build_audio_only_cv_audit(list(reversed(tracks)), [])

        self.assertEqual(first, second)
        self.assertEqual(first["fold_count"], 3)
        self.assertEqual(first["fold_assignment_policy"],
                         "sort artist keys; shuffle with seed; assign artists[i::fold_count]")
        fold_for = {
            track_id: int(fold_index)
            for fold_index, fold in first["folds"].items()
            for track_id in fold["track_ids"]
        }
        self.assertEqual(set(fold_for), {item["track_id"] for item in tracks})
        self.assertEqual(fold_for["T-a1"], fold_for["T-a2"])
        self.assertEqual(fold_for["T-b"], fold_for["T-b2"])
        self.assertNotEqual(fold_for["T-b"], fold_for["T-c"])
        self.assertEqual(json.loads(json.dumps(first)), first)

    def test_each_artist_is_assigned_to_exactly_one_fold(self):
        tracks = [
            track(f"T{artist}-{i}", f"artist-{artist}", song_id=f"song-{artist}-{i}")
            for artist in range(8) for i in range(2)
        ]
        result = build_audio_only_cv_audit(tracks, [])
        fold_for = {
            track_id: fold_index
            for fold_index, fold in result["folds"].items()
            for track_id in fold["track_ids"]
        }
        by_artist = {}
        for item in tracks:
            by_artist.setdefault(item["artist_id"], set()).add(fold_for[item["track_id"]])
        self.assertTrue(all(len(folds) == 1 for folds in by_artist.values()))
        self.assertEqual(sorted(fold["track_count"] for fold in result["folds"].values()), [4, 6, 6])

    def test_undirected_deduplication_within_and_cross_fold_counts(self):
        tracks = [track(f"T{i}", f"artist-{i}") for i in range(12)]
        initial = build_audio_only_cv_audit(tracks, [])
        fold_ids = {key: value["track_ids"] for key, value in initial["folds"].items()}
        inside = [fold_ids["0"][0], fold_ids["0"][1]]
        across = [fold_ids["0"][0], fold_ids["1"][0]]
        edges = [edge(*inside), edge(*reversed(inside)), edge(*across), edge(*reversed(across)), edge(*across)]

        result = build_audio_only_cv_audit(tracks, edges)

        self.assertEqual(result["catalog_counts"]["unique_edges_within_catalog"], 2)
        self.assertEqual(result["cross_fold_edges"], 1)
        self.assertEqual(result["folds"]["0"]["positive_edge_count"], 1)
        self.assertEqual(result["folds"]["1"]["positive_edge_count"], 0)
        self.assertEqual(result["folds"]["2"]["positive_edge_count"], 0)
        self.assertEqual(result["folds"]["0"]["positive_edges"], [sorted(inside)])

    def test_one_shot_track_and_edge_iterators_match_lists(self):
        tracks = [track(f"T{i}", f"artist-{i}") for i in range(12)]
        baseline = build_audio_only_cv_audit(tracks, [])
        fold_ids = {key: value["track_ids"] for key, value in baseline["folds"].items()}
        edges = [
            edge(fold_ids["0"][0], fold_ids["0"][1]),
            edge(fold_ids["0"][1], fold_ids["0"][0]),
            edge(fold_ids["0"][0], fold_ids["1"][0]),
        ]

        expected = build_audio_only_cv_audit(tracks, edges)
        actual = build_audio_only_cv_audit(iter(tracks), iter(edges))

        self.assertEqual(actual, expected)

    def test_duplicate_filtering_is_inherited_from_audio_only_audit(self):
        tracks = [
            track("keep", "artist", song_id="same"),
            track("drop", "artist", song_id="same"),
            track("official-a", "a", song_id="a"),
            track("official-b", "b", song_id="b"),
            track("other", "other", song_id="other"),
        ]
        result = build_audio_only_cv_audit(tracks, [], duplicate_groups=[["official-b", "official-a"]])
        ids = [track_id for fold in result["folds"].values() for track_id in fold["track_ids"]]
        self.assertEqual(sorted(ids), ["drop", "official-a", "other"])
        self.assertEqual(result["duplicate_counts"]["unique_duplicate_tracks_excluded"], 2)

    def test_invalid_fold_count_and_seed(self):
        with self.assertRaises(ValueError):
            build_audio_only_cv_audit([], [], fold_count=1)
        with self.assertRaises(ValueError):
            build_audio_only_cv_audit([], [], seed="684")


if __name__ == "__main__":
    unittest.main()

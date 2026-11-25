import unittest
from unittest.mock import patch

import numpy as np

from src.evaluate_neurotune_v1 import (
    build_d4_neurotune_v1_cosine_baseline,
    build_neurotune_v1_cosine_baseline,
)
from src.evaluate_neurotune_v1_cv import (
    build_neurotune_v1_c_cosine_baseline,
    evaluate_neurotune_v1_source_separated_retrieval,
)


class NeuroTuneV1CrossValidationTests(unittest.TestCase):
    @staticmethod
    def _d4_features():
        return {
            "audio_windows": {
                "a": [
                    np.array([[1] * 26, [3] * 26], dtype=np.float32),
                    np.array([[5] * 26], dtype=np.float32),
                    np.empty((0, 26), dtype=np.float32),
                ],
                "b": [
                    np.array([[2] * 26], dtype=np.float32),
                    np.empty((0, 26), dtype=np.float32),
                    np.array([[6] * 26], dtype=np.float32),
                ],
            },
            "window_valid": np.array([[True, True, False], [True, False, True]], dtype=bool),
            "bpm": np.array([[1], [2]], dtype=np.float32),
            "lyrics": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "genre": np.array([[1, 0], [0, 1]], dtype=np.float32),
        }

    def test_d4_uses_per_window_means_then_mean_of_valid_windows(self):
        features = self._d4_features()
        features["audio_windows"]["a"] = [
            np.array([[2, 0] + [0] * 24, [0, 2] + [0] * 24], dtype=np.float32),
            np.array([[3, 1] + [0] * 24], dtype=np.float32),
            np.empty((0, 26), dtype=np.float32),
        ]
        features["audio_windows"]["b"] = [
            np.array([[0, 3] + [0] * 24], dtype=np.float32),
            np.empty((0, 26), dtype=np.float32),
            np.array([[0, 1] + [0] * 24], dtype=np.float32),
        ]
        result = build_d4_neurotune_v1_cosine_baseline(
            features, ["a", "b"], modalities=("audio",)
        )
        # a: window means [1, 1] and [3, 1], then [2, 1]; b pools [0, 3] and [0, 1].
        expected = np.zeros((2, 26), dtype=np.float32)
        expected[0, :2] = np.array([2, 1], dtype=np.float32) / np.sqrt(5)
        expected[1, 1] = 1.0
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_d4_invalid_windows_are_excluded_from_audio_pooling(self):
        features = self._d4_features()
        actual = build_d4_neurotune_v1_cosine_baseline(
            features, ["a", "b"], modalities=("audio",)
        )
        expected = build_neurotune_v1_cosine_baseline(
            {"audio": [np.full((1, 26), 3, dtype=np.float32),
                       np.full((1, 26), 4, dtype=np.float32)]},
            ["a", "b"], modalities=("audio",),
        )
        np.testing.assert_array_equal(actual, expected)

    def test_d4_reuses_equal_modality_weighting_and_returns_finite_normalized_rows(self):
        features = self._d4_features()
        audio_summaries = [
            np.full((1, 26), 3, dtype=np.float32),
            np.full((1, 26), 4, dtype=np.float32),
        ]
        reference_features = {
            "audio": audio_summaries,
            "bpm": features["bpm"],
            "lyrics": features["lyrics"],
            "genre": features["genre"],
        }
        modalities = ("audio", "bpm", "genre")
        expected = build_neurotune_v1_cosine_baseline(
            reference_features, ["a", "b"], modalities=modalities
        )
        actual = build_d4_neurotune_v1_cosine_baseline(
            features, ["a", "b"], modalities=modalities
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, rtol=1e-6)

    def test_d4_rejects_malformed_windows_and_validity(self):
        bad_features = self._d4_features()
        bad_features["audio_windows"]["a"] = bad_features["audio_windows"]["a"][:2]
        with self.assertRaises(ValueError):
            build_d4_neurotune_v1_cosine_baseline(bad_features, ["a", "b"])

        bad_features = self._d4_features()
        bad_features["audio_windows"]["a"][0] = np.ones((1, 26), dtype=np.float64)
        with self.assertRaises(ValueError):
            build_d4_neurotune_v1_cosine_baseline(bad_features, ["a", "b"])

        bad_features = self._d4_features()
        bad_features["window_valid"] = np.array([[True, True, True], [True, False, True]])
        with self.assertRaises(ValueError):
            build_d4_neurotune_v1_cosine_baseline(bad_features, ["a", "b"])

        bad_features = self._d4_features()
        bad_features["window_valid"][0] = False
        with self.assertRaises(ValueError):
            build_d4_neurotune_v1_cosine_baseline(bad_features, ["a", "b"])

        bad_features = self._d4_features()
        bad_features["audio_windows"]["a"] = [np.empty((0, 26), dtype=np.float32)] * 3
        bad_features["window_valid"][0] = False
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            build_d4_neurotune_v1_cosine_baseline(bad_features, ["a", "b"])

        bad_features = self._d4_features()
        bad_features["audio_windows"]["a"][0][0, 0] = np.inf
        with self.assertRaises(ValueError):
            build_d4_neurotune_v1_cosine_baseline(bad_features, ["a", "b"])

    def test_existing_d0_baseline_output_is_unchanged(self):
        features = {
            "audio": [np.ones((1, 26), dtype=np.float32)] * 2,
            "bpm": np.array([[1], [2]], dtype=np.float32),
            "genre": np.array([[1, 0], [0, 1]], dtype=np.float32),
        }
        actual = build_neurotune_v1_cosine_baseline(
            features, ["a", "b"], modalities=("audio", "bpm", "genre")
        )
        expected = np.zeros((2, 29), dtype=np.float32)
        expected[:, :26] = 1 / np.sqrt(3 * 26)
        expected[0, 26] = 1 / np.sqrt(3)
        expected[1, 26] = 1 / np.sqrt(3)
        expected[0, 27] = 1 / np.sqrt(3)
        expected[1, 28] = 1 / np.sqrt(3)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_c_baseline_matches_existing_explicit_c_modality_baseline(self):
        features = {
            "audio": [np.ones((1, 26), dtype=np.float32), np.eye(1, 26, dtype=np.float32)],
            "bpm": np.array([[2], [4]], dtype=np.float32),
            "genre": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "lyrics": np.array([[np.nan], [np.inf]]),
        }
        expected = build_neurotune_v1_cosine_baseline(
            features, ["a", "b"], modalities=("audio", "bpm", "genre")
        )
        actual = build_neurotune_v1_c_cosine_baseline(features, ["a", "b"])
        np.testing.assert_array_equal(actual, expected)

    def test_source_sets_produce_distinct_metric_records(self):
        embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
        result = evaluate_neurotune_v1_source_separated_retrieval(
            embeddings, ["a", "b", "c", "d"],
            held_out_train_source_edges=[("a", "b")],
            official_test_source_edges=[("a", "c")],
        )

        self.assertEqual(set(result), {"held_out_train_source", "official_test_source"})
        train = result["held_out_train_source"]
        official = result["official_test_source"]
        self.assertEqual(train["Recall@10"], 1.0)
        self.assertLess(official["MRR@10"], train["MRR@10"])
        self.assertEqual(train["query_count"], 2)
        self.assertEqual(official["query_count"], 2)
        self.assertEqual(train["unique_undirected_edge_count"], 1)
        self.assertEqual(official["unique_undirected_edge_count"], 1)

    def test_empty_source_has_zero_metrics_and_counts(self):
        result = evaluate_neurotune_v1_source_separated_retrieval(
            np.eye(2, dtype=np.float32), ["a", "b"], [], [("a", "b")]
        )
        empty = result["held_out_train_source"]
        self.assertEqual(
            {key: empty[key] for key in ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10")},
            {"Recall@10": 0.0, "HitRate@10": 0.0, "MRR@10": 0.0, "NDCG@10": 0.0},
        )
        self.assertEqual(empty["query_count"], 0)
        self.assertEqual(empty["unique_undirected_edge_count"], 0)

    def test_rejects_overlapping_and_invalid_edges(self):
        embeddings = np.eye(3, dtype=np.float32)
        ids = ["a", "b", "c"]
        with self.assertRaises(ValueError):
            evaluate_neurotune_v1_source_separated_retrieval(
                embeddings, ids, [("a", "b")], [("b", "a")]
            )
        for edges in ([('a', 'a')], [('a', 'missing')], [('a', 'b'), ('b', 'a')], [('a', 'b'), ('a', 'b')], ["ab"]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                evaluate_neurotune_v1_source_separated_retrieval(
                    embeddings, ids, edges, []
                )

    def test_rejects_overlapping_sources_before_ranking(self):
        with patch("src.evaluate_neurotune_v1_cv.evaluate_neurotune_v1_retrieval",
                   side_effect=AssertionError("ranking must not run")):
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate_neurotune_v1_source_separated_retrieval(
                    np.eye(2, dtype=np.float32), ["a", "b"], [("a", "b")], [("b", "a")]
                )

    def test_one_shot_song_id_iterable_is_reused_for_both_sources(self):
        result = evaluate_neurotune_v1_source_separated_retrieval(
            np.eye(2, dtype=np.float32), (song_id for song_id in ("a", "b")), [("a", "b")], []
        )
        self.assertEqual(result["held_out_train_source"]["query_count"], 2)
        self.assertEqual(result["official_test_source"]["query_count"], 0)

    def test_source_separated_evaluator_preserves_non_string_hashable_ids(self):
        result = evaluate_neurotune_v1_source_separated_retrieval(
            np.eye(3, dtype=np.float32), [1, 2, 3], [(1, 2)], [(1, 3)]
        )
        self.assertEqual(result["held_out_train_source"]["query_count"], 2)
        self.assertEqual(result["official_test_source"]["query_count"], 2)


if __name__ == "__main__":
    unittest.main()

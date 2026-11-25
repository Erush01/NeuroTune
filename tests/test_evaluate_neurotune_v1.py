import unittest
from unittest.mock import patch

import numpy as np

from src.evaluate_neurotune_v1 import (
    build_neurotune_v1_cosine_baseline,
    evaluate_neurotune_v1_retrieval,
)


class NeuroTuneV1CosineBaselineTests(unittest.TestCase):
    def test_selected_c_modalities_are_equal_weighted_and_omit_lyrics(self):
        features = {
            "audio": [np.eye(1, 26, dtype=np.float32)[0:1], np.zeros((1, 26), dtype=np.float32)],
            "bpm": np.array([[3], [0]], dtype=np.float32),
            "genre": np.array([[0, 4], [0, 0]], dtype=np.float32),
            "lyrics": np.array([[np.nan], [np.inf]]),
        }

        result = build_neurotune_v1_cosine_baseline(
            features, ["a", "zero"], modalities=("audio", "bpm", "genre")
        )

        self.assertEqual(result.shape, (2, 29))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0, atol=1e-6)
        # All three non-zero unit modality vectors contribute equally.
        np.testing.assert_allclose(result[0, :26], [1 / np.sqrt(3)] + [0.0] * 25)
        np.testing.assert_allclose(result[0, 26], 1 / np.sqrt(3))
        np.testing.assert_allclose(result[0, 27:], [0.0, 1 / np.sqrt(3)])
        np.testing.assert_array_equal(result[1], np.zeros(29, dtype=np.float32))

    def test_selected_modalities_validate_only_selected_features(self):
        features = {
            "audio": [np.ones((1, 26), dtype=np.float32)],
            "bpm": np.ones((1, 1), dtype=np.float32),
            "genre": np.ones((1, 2), dtype=np.float32),
        }
        result = build_neurotune_v1_cosine_baseline(
            features, ["a"], modalities=("audio", "bpm", "genre")
        )
        self.assertEqual(result.shape, (1, 29))

        for key, value in (
            ("bpm", np.ones((2, 1), dtype=np.float32)),
            ("genre", np.zeros((1, 0), dtype=np.float32)),
            ("audio", [np.ones((1, 25), dtype=np.float32)]),
        ):
            invalid = dict(features)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_neurotune_v1_cosine_baseline(
                    invalid, ["a"], modalities=("audio", "bpm", "genre")
                )

        del features["genre"]
        with self.assertRaises(ValueError):
            build_neurotune_v1_cosine_baseline(
                features, ["a"], modalities=("audio", "bpm", "genre")
            )

    def test_rejects_unhashable_or_non_string_modality_entries(self):
        features = {
            "audio": [np.ones((1, 26), dtype=np.float32)],
            "bpm": np.ones((1, 1), dtype=np.float32),
            "genre": np.ones((1, 1), dtype=np.float32),
        }
        for modalities in ([[]], [1], ("audio", []), ("audio", None)):
            with self.subTest(modalities=modalities), self.assertRaises(ValueError):
                build_neurotune_v1_cosine_baseline(
                    features, ["a"], modalities=modalities
                )

    def test_default_four_modality_output_is_unchanged(self):
        features = {
            "audio": [np.eye(1, 26, dtype=np.float32)[0:1]],
            "bpm": np.array([[2]], dtype=np.float32),
            "lyrics": np.eye(1, 5000, dtype=np.float32),
            "genre": np.array([[0, 1]], dtype=np.float32),
        }
        default = build_neurotune_v1_cosine_baseline(features, ["a"])
        explicit = build_neurotune_v1_cosine_baseline(
            features, ["a"], modalities=("audio", "bpm", "lyrics", "genre")
        )
        self.assertEqual(default.shape, (1, 5029))
        np.testing.assert_array_equal(default, explicit)

    def test_audio_mean_slice_order_equal_modality_weight_and_output_contract(self):
        audio = [
            np.vstack((np.eye(1, 26, dtype=np.float32)[0], np.eye(1, 26, dtype=np.float32)[0])),
            np.eye(1, 26, dtype=np.float32),
        ]
        features = {
            "audio": audio,
            "bpm": np.array([[2], [0]], dtype=np.float32),
            "lyrics": np.zeros((2, 5000), dtype=np.float32),
            "genre": np.array([[0, 1], [1, 0]], dtype=np.float32),
        }
        features["lyrics"][0, 0] = 1
        features["lyrics"][1, 1] = 1

        result = build_neurotune_v1_cosine_baseline(features, ["a", "b"])

        self.assertEqual(result.shape, (2, 5029))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)
        # Mean audio summary and modality order are audio, BPM, lyrics, genre.
        # Four unit modality vectors have norm 1/2 each after final normalization.
        np.testing.assert_allclose(result[0, :26], [0.5] + [0.0] * 25)
        np.testing.assert_allclose(result[0, 26], 0.5)
        np.testing.assert_allclose(result[0, 27:5027], [0.5] + [0.0] * 4999)
        np.testing.assert_allclose(result[0, 5027:5029], [0.0, 0.5])

    def test_zero_modalities_and_all_zero_final_rows_are_safe(self):
        features = {
            "audio": [np.zeros((2, 26), dtype=np.float32), np.ones((1, 26), dtype=np.float32)],
            "bpm": np.zeros((2, 1), dtype=np.float32),
            "lyrics": np.zeros((2, 5000), dtype=np.float32),
            "genre": np.zeros((2, 1), dtype=np.float32),
        }
        result = build_neurotune_v1_cosine_baseline(features, ["zero", "audio"])
        np.testing.assert_array_equal(result[0], np.zeros(result.shape[1], dtype=np.float32))
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(float(np.linalg.norm(result[1])), 1.0)

    def test_rejects_misaligned_duplicate_or_invalid_feature_inputs(self):
        base = {
            "audio": [np.ones((1, 26), dtype=np.float32)],
            "bpm": np.ones((1, 1), dtype=np.float32),
            "lyrics": np.ones((1, 5000), dtype=np.float32),
            "genre": np.ones((1, 1), dtype=np.float32),
        }
        for ids in ([], ["a", "a"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_neurotune_v1_cosine_baseline(base, ids)
        for key, value in (
            ("audio", [np.ones((0, 26), dtype=np.float32)]),
            ("audio", [np.ones((1, 26), dtype=np.float64)]),
            ("audio", [np.full((1, 26), np.nan, dtype=np.float32)]),
            ("bpm", np.ones((2, 1), dtype=np.float32)),
            ("lyrics", np.full((1, 5000), np.inf, dtype=np.float32)),
            ("genre", np.zeros((1, 0), dtype=np.float32)),
        ):
            invalid = dict(base)
            invalid[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build_neurotune_v1_cosine_baseline(invalid, ["a"])


class NeuroTuneV1RetrievalTests(unittest.TestCase):
    def test_retrieval_uses_batch_ranking_helper_once(self):
        embeddings = np.eye(3, dtype=np.float32)
        ids = ["a", "b", "c"]
        rankings = {
            "a": [{"song_id": "b", "score": 0.0}],
            "b": [{"song_id": "a", "score": 0.0}],
        }
        with patch("src.evaluate_neurotune_v1.rank_top_k_many", return_value=rankings) as batch_rank:
            result = evaluate_neurotune_v1_retrieval(embeddings, ids, [("a", "b")], k=1)
        batch_rank.assert_called_once_with(embeddings, ids, ["a", "b"], k=1)
        self.assertEqual(result["Recall@1"], 1.0)

    def test_mirrors_edges_excludes_query_and_returns_known_metrics(self):
        # a is closest to b (positive) at rank 1; c is closest to d (positive).
        embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
        ids = ["a", "b", "c", "d"]
        result = evaluate_neurotune_v1_retrieval(embeddings, ids, [("a", "b"), ("c", "d")], k=1)
        self.assertEqual(result["query_count"], 4)
        self.assertEqual(result["unique_undirected_edge_count"], 2)
        self.assertEqual(result["Recall@1"], 1.0)
        self.assertEqual(result["NDCG@1"], 1.0)
        self.assertEqual(result["MRR@1"], 1.0)

        result_at_10 = evaluate_neurotune_v1_retrieval(
            embeddings, ids, [("a", "b"), ("c", "d")], k=10
        )
        self.assertEqual(result_at_10["Recall@10"], 1.0)
        self.assertEqual(result_at_10["HitRate@10"], 1.0)
        self.assertEqual(result_at_10["NDCG@10"], 1.0)
        self.assertEqual(result_at_10["MRR@10"], 1.0)
        self.assertEqual(result_at_10["query_count"], 4)
        self.assertEqual(result_at_10["eligible_positive_edges"], 2)
        self.assertEqual(result_at_10["unique_undirected_edge_count"], 2)

    def test_deterministic_song_id_ties_and_k_handling(self):
        embeddings = np.array([[1, 0], [0, 1], [0, -1]], dtype=np.float32)
        ids = ["query", "z", "a"]
        edges = [("query", "z")]
        result = evaluate_neurotune_v1_retrieval(embeddings, ids, edges, k=1)
        self.assertEqual(result["Recall@1"], 0.5)
        self.assertEqual(result["query_count"], 2)
        all_result = evaluate_neurotune_v1_retrieval(embeddings, ids, edges, k=10)
        self.assertEqual(all_result["Recall@10"], 1.0)
        for invalid_k in (-1, True, 1.5):
            with self.subTest(k=invalid_k), self.assertRaises(ValueError):
                evaluate_neurotune_v1_retrieval(embeddings, ids, edges, k=invalid_k)

    def test_one_shot_edges_and_invalid_edges(self):
        embeddings = np.eye(3, dtype=np.float32)
        ids = ["a", "b", "c"]
        edges = (edge for edge in [("a", "b"), ("b", "c")])
        result = evaluate_neurotune_v1_retrieval(embeddings, ids, edges, k=2)
        self.assertEqual(result["unique_undirected_edge_count"], 2)
        for edges in ([('a', 'a')], [('a', 'missing')], [('a', 'b'), ('b', 'a')], [('a', 'b'), ('a', 'b')]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                evaluate_neurotune_v1_retrieval(embeddings, ids, edges)


if __name__ == "__main__":
    unittest.main()

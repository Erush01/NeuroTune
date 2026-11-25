import unittest
import math

import numpy as np

from src.evaluate import build_test_relevance, evaluate_rankings


class EvaluationTests(unittest.TestCase):
    def test_relevance_is_symmetric_and_excludes_cross_partition_edges(self):
        edges = [
            {"source": "a", "target": "b", "score": 1},
            {"source": "a", "target": "outside", "score": 1},
            {"source": "outside", "target": "b", "score": 1},
        ]

        relevance = build_test_relevance(edges, {"a", "b", "c"})

        self.assertEqual(relevance, {"a": {"b"}, "b": {"a"}})

    def test_eligible_positive_edges_counts_unique_unordered_pairs(self):
        relevance = build_test_relevance(
            [{"source": "a", "target": "b", "score": 1}], {"a", "b"}
        )

        metrics = evaluate_rankings({"a": [{"song_id": "b"}]}, relevance, k=10)

        self.assertEqual(metrics["eligible_positive_edges"], 1)

    def test_hand_computed_binary_ranking_metrics_at_ten(self):
        # Query a has relevant items at ranks 1 and 3; query b at rank 2.
        # Query c is not eligible and therefore does not enter the macro averages.
        rankings = {
            "a": [
                {"song_id": "a1"}, {"song_id": "a2"}, {"song_id": "a3"},
                {"song_id": "a4"}, {"song_id": "a5"}, {"song_id": "a6"},
                {"song_id": "a7"}, {"song_id": "a8"}, {"song_id": "a9"},
            ],
            "b": [
                {"song_id": "b1"}, {"song_id": "b2"}, {"song_id": "b3"},
            ],
            "c": [{"song_id": "c1"}],
        }
        relevance = {
            "a": {"a1", "a3"},
            "b": {"b2"},
            "c": set(),
        }

        metrics = evaluate_rankings(rankings, relevance, k=10)

        self.assertAlmostEqual(metrics["Recall@10"], (1.0 + 1.0) / 2)
        self.assertAlmostEqual(metrics["HitRate@10"], 1.0)
        self.assertAlmostEqual(metrics["MRR@10"], (1.0 + 0.5) / 2)
        expected_ndcg_a = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        expected_ndcg_b = 1 / math.log2(3)
        self.assertAlmostEqual(metrics["NDCG@10"], (expected_ndcg_a + expected_ndcg_b) / 2)
        self.assertEqual(metrics["query_count"], 2)
        self.assertEqual(metrics["eligible_positive_edges"], 3)

    def test_k_must_be_integer_like_and_accepts_numpy_integer(self):
        rankings = {"a": [{"song_id": "b"}]}
        relevance = {"a": {"b"}}

        with self.assertRaisesRegex(ValueError, "k must be a non-negative integer"):
            evaluate_rankings(rankings, relevance, k=1.5)

        metrics = evaluate_rankings(rankings, relevance, k=np.int64(1))
        self.assertEqual(metrics["Recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()

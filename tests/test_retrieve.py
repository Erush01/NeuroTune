import unittest

import numpy as np

from src.retrieve import rank_top_k, rank_top_k_many


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.song_ids = ["query", "same-direction", "orthogonal", "opposite"]
        self.embeddings = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )

    def test_cosine_ordering_excludes_self_and_returns_sorted_top_k(self):
        results = rank_top_k(self.embeddings, self.song_ids, "query", k=2)

        self.assertEqual([item["song_id"] for item in results], ["same-direction", "orthogonal"])
        self.assertEqual([item["score"] for item in results], [0.8, 0.0])
        self.assertNotIn("query", [item["song_id"] for item in results])
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])

    def test_k_is_clamped_to_gallery_size(self):
        results = rank_top_k(self.embeddings, self.song_ids, "query", k=100)

        self.assertEqual(len(results), len(self.song_ids) - 1)
        self.assertEqual([item["song_id"] for item in results], ["same-direction", "orthogonal", "opposite"])

    def test_equal_scores_are_ordered_by_song_id(self):
        results = rank_top_k(
            np.array([[1.0, 0.0], [0.6, 0.8], [0.6, -0.8]]),
            ["query", "z-candidate", "a-candidate"],
            "query",
            k=2,
        )

        self.assertEqual([item["song_id"] for item in results], ["a-candidate", "z-candidate"])

    def test_duplicate_song_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "song_ids must be unique"):
            rank_top_k(np.array([[1.0, 0.0], [1.0, 0.0]]), ["query", "query"], "query")

    def test_k_must_be_integer_like_and_accepts_numpy_integer(self):
        with self.assertRaisesRegex(ValueError, "k must be a non-negative integer"):
            rank_top_k(self.embeddings, self.song_ids, "query", k=1.5)

        results = rank_top_k(self.embeddings, self.song_ids, "query", k=np.int64(1))
        self.assertEqual(len(results), 1)


class BatchRetrievalTests(unittest.TestCase):
    def test_batch_matches_repeated_single_query_for_ties_and_non_ties(self):
        cases = (
            (
                np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=float),
                ["query", "same", "orthogonal", "opposite"],
            ),
            (
                np.array([[1, 0], [0.6, 0.8], [0.6, -0.8], [0, 0]], dtype=float),
                ["query", "z", "a", "zero"],
            ),
        )
        for embeddings, ids in cases:
            with self.subTest(ids=ids):
                queries = [ids[0], ids[-1]]
                rankings = rank_top_k_many(embeddings, ids, queries, k=10)
                for query_id in queries:
                    self.assertEqual(rankings[query_id], rank_top_k(embeddings, ids, query_id, k=10))
                    self.assertNotIn(query_id, [item["song_id"] for item in rankings[query_id]])

    def test_large_k_excludes_query_and_zero_vectors_tie_by_id(self):
        embeddings = np.zeros((3, 2), dtype=np.float32)
        ids = ["q", "z", "a"]
        rankings = rank_top_k_many(embeddings, ids, ["q", "a"], k=100)
        self.assertEqual([item["song_id"] for item in rankings["q"]], ["a", "z"])
        self.assertTrue(all(item["score"] == 0.0 for item in rankings["q"] + rankings["a"]))
        self.assertNotIn("a", [item["song_id"] for item in rankings["a"]])

    def test_rejects_invalid_embeddings_queries_and_k(self):
        valid = np.eye(2)
        for embeddings, ids in ((valid[:1], ["a", "b"]), (valid, ["a", "a"]),
                                (np.array([[1, np.inf], [0, 1]]), ["a", "b"])):
            with self.subTest(embeddings=embeddings, ids=ids), self.assertRaises(ValueError):
                rank_top_k_many(embeddings, ids, ["a"])
        for queries in (("a", "a"), ("missing",)):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                rank_top_k_many(valid, ["a", "b"], queries)
        for k in (-1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                rank_top_k_many(valid, ["a", "b"], ["a"], k=k)


if __name__ == "__main__":
    unittest.main()

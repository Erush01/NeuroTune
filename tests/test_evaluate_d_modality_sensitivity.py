import unittest

import numpy as np
import torch

from src.evaluate_d_modality_sensitivity import mask_one_modality, measure_behavior_change
from src.retrieve import rank_top_k_many


class MaskOneModalityTests(unittest.TestCase):
    def setUp(self):
        self.batch = {
            "audio": torch.arange(3 * 4 * 26, dtype=torch.float32).reshape(3, 4, 26) + 1,
            "mask": torch.tensor([[True, True, False, False], [True, True, True, False], [True] * 4]),
            "lengths": torch.tensor([2, 3, 4]),
            "track_ids": ["track-c", "track-a", "track-b"],
            "bpm": torch.tensor([[1.], [2.], [3.]]),
            "genre": torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1,
            "lyrics": torch.arange(18, dtype=torch.float32).reshape(3, 6) + 1,
        }

    def test_masks_only_selected_modality_for_all_rows_without_mutating_input(self):
        snapshots = {key: value.clone() if torch.is_tensor(value) else list(value)
                     for key, value in self.batch.items()}
        for modality in ("audio", "bpm", "genre", "lyrics"):
            with self.subTest(modality=modality):
                masked = mask_one_modality(self.batch, modality)
                self.assertIsNot(masked, self.batch)
                for key, value in self.batch.items():
                    if key == modality:
                        self.assertTrue(torch.equal(masked[key], torch.zeros_like(value)))
                        self.assertIsNot(masked[key], value)
                    elif torch.is_tensor(value):
                        self.assertIs(masked[key], value)
                        self.assertTrue(torch.equal(masked[key], snapshots[key]))
                    else:
                        self.assertEqual(masked[key], value)
                self.assertTrue(torch.equal(masked["mask"], snapshots["mask"]))
                self.assertTrue(torch.equal(masked["lengths"], snapshots["lengths"]))
        for key, value in self.batch.items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(value, snapshots[key]))

    def test_rejects_unknown_modality(self):
        with self.assertRaises(ValueError):
            mask_one_modality(self.batch, "mask")


class BehaviorChangeTests(unittest.TestCase):
    def test_retrieval_metrics_overlap_and_embedding_cosine_are_deterministic(self):
        ids = ["c", "a", "b", "d"]
        original = np.asarray([[1., 0.], [1., 0.], [0., 1.], [0., -1.]])
        masked = np.asarray([[1., 0.], [0., -1.], [1., 0.], [0., -1.]])
        edges = [("c", "a"), ("a", "b")]

        rankings = rank_top_k_many(original, ids, ["c", "a"], k=10)
        self.assertNotIn("c", [row["song_id"] for row in rankings["c"]])
        self.assertEqual([row["song_id"] for row in rankings["c"]], ["a", "b", "d"])

        result = measure_behavior_change(original, masked, ids, edges, k=2)
        self.assertEqual(result["eligible_query_count"], 3)
        self.assertEqual(result["unique_undirected_edge_count"], 2)
        self.assertAlmostEqual(result["mean_top_k_overlap"], 5 / 6)
        self.assertAlmostEqual(result["mean_embedding_cosine"], 0.5)
        self.assertAlmostEqual(result["mean_embedding_cosine_shift"], 0.5)
        for value in result.values():
            if isinstance(value, (int, float)):
                self.assertTrue(np.isfinite(value))

    def test_rejects_nonfinite_or_misaligned_embeddings(self):
        ids = ["a", "b"]
        edges = [("a", "b")]
        vectors = np.eye(2)
        invalid = vectors.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            measure_behavior_change(vectors, invalid, ids, edges)
        with self.assertRaises(ValueError):
            measure_behavior_change(vectors, vectors[:1], ids, edges)

    def test_rejects_complex_embeddings(self):
        ids = ["a", "b"]
        edges = [("a", "b")]
        vectors = np.eye(2, dtype=np.complex128)
        with self.assertRaises(ValueError):
            measure_behavior_change(vectors, vectors, ids, edges)

    def test_zero_vector_cosine_convention_and_short_gallery_overlap(self):
        ids = ["a", "b", "c"]
        original = np.asarray([[0., 0.], [1., 0.], [0., 0.]])
        masked = np.asarray([[0., 0.], [0., 1.], [1., 0.]])
        result = measure_behavior_change(original, masked, ids, [("a", "b")], k=10)

        # Identical zero vectors have cosine 1; one zero and one nonzero has cosine 0.
        self.assertAlmostEqual(result["mean_embedding_cosine"], 1 / 3)
        self.assertAlmostEqual(result["mean_embedding_cosine_shift"], 2 / 3)
        # k is clamped to the two non-self gallery tracks, not the requested ten.
        self.assertAlmostEqual(result["mean_top_k_overlap"], 1.0)


if __name__ == "__main__":
    unittest.main()

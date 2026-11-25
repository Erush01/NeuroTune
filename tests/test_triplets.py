import unittest
from unittest import mock

import numpy as np
import torch

from src.triplets import (
    MixedSemiHardTripletSampler,
    sample_mixed_semi_hard_triplets,
    sample_triplet_schedule,
    triplet_margin_cosine_loss,
)


class TripletLossTests(unittest.TestCase):
    def test_satisfied_triplet_has_zero_loss(self):
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])
        negative = torch.tensor([[0.0, 1.0]])
        self.assertEqual(triplet_margin_cosine_loss(anchor, positive, negative).item(), 0.0)

    def test_violating_triplet_has_expected_loss_and_gradients(self):
        anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
        positive = torch.tensor([[0.0, 1.0]], requires_grad=True)
        negative = torch.tensor([[1.0, 0.0]], requires_grad=True)
        loss = triplet_margin_cosine_loss(anchor, positive, negative, margin=0.2)
        self.assertAlmostEqual(loss.item(), 1.2)
        loss.backward()
        for tensor in (anchor, positive, negative):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_rejects_bad_shapes_nonfinite_values_and_margins(self):
        valid = torch.ones(2, 3)
        invalid = (
            (torch.ones(3), valid, valid),
            (valid, torch.ones(1, 3), valid),
            (torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)),
            (valid, valid, torch.ones(2, 2)),
            (torch.tensor([[float("nan")]]), torch.ones(1, 1), torch.ones(1, 1)),
            (torch.ones(1, 1), torch.ones(1, 1), torch.tensor([[float("inf")]])),
            (torch.ones(1, 1, dtype=torch.int64), torch.ones(1, 1), torch.ones(1, 1)),
        )
        for args in invalid:
            with self.subTest(shapes=[tuple(x.shape) for x in args]), self.assertRaises(ValueError):
                triplet_margin_cosine_loss(*args)
        for margin in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                triplet_margin_cosine_loss(torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2), margin)


class TripletScheduleTests(unittest.TestCase):
    def test_both_directions_shape_and_index_order(self):
        ids = ["a", "b", "c", "d"]
        edges = [("b", "a"), ("c", "d")]
        schedule = sample_triplet_schedule(ids, edges, epochs=2, seed=19)
        self.assertEqual(schedule.shape, (2, 4, 3))
        self.assertEqual(schedule.dtype, np.int32)
        index = {track_id: i for i, track_id in enumerate(ids)}
        for epoch in schedule:
            self.assertEqual([tuple(row[:2]) for row in epoch], [
                (index["a"], index["b"]), (index["b"], index["a"]),
                (index["c"], index["d"]), (index["d"], index["c"]),
            ])

    def test_deterministic_and_one_partition_local_negative_per_directed_edge(self):
        ids = ["a", "b", "c", "d", "e", "f"]
        edges = [("a", "b"), ("a", "c"), ("d", "e")]
        schedule = sample_triplet_schedule(ids, edges, epochs=8, seed=684)
        np.testing.assert_array_equal(schedule, sample_triplet_schedule(ids, edges, epochs=8, seed=684))
        self.assertEqual(schedule.shape, (8, 6, 3))
        id_by_index = dict(enumerate(ids))
        neighbors = {track: set() for track in ids}
        for left, right in edges:
            neighbors[left].add(right)
            neighbors[right].add(left)
        for epoch in schedule:
            for anchor_idx, positive_idx, negative_idx in epoch:
                anchor, positive, negative = map(id_by_index.__getitem__, (anchor_idx, positive_idx, negative_idx))
                self.assertNotIn(negative, {anchor, positive} | neighbors[anchor])
                self.assertIn(negative, ids)

    def test_rejects_invalid_ids_edges_and_impossible_negatives(self):
        with self.assertRaisesRegex(ValueError, "track_ids must be iterable"):
            sample_triplet_schedule(None, [], epochs=1, seed=0)
        with self.assertRaises(ValueError):
            sample_triplet_schedule([], [], epochs=1, seed=0)
        with self.assertRaises(ValueError):
            sample_triplet_schedule(["a", "a"], [], epochs=1, seed=0)
        with self.assertRaises(ValueError):
            sample_triplet_schedule(["a", "b"], [("a", "a")], epochs=1, seed=0)
        with self.assertRaises(ValueError):
            sample_triplet_schedule(["a", "b"], [("a", "b"), ("b", "a")], epochs=1, seed=0)
        with self.assertRaises(ValueError):
            sample_triplet_schedule(["a", "b"], [("a", "missing")], epochs=1, seed=0)
        with self.assertRaises(ValueError):
            sample_triplet_schedule(["a", "b"], [("a", "b")], epochs=1, seed=0)
        for epochs, seed in ((0, 1), (True, 1), (1, -1), (1, 1.5)):
            with self.subTest(epochs=epochs, seed=seed), self.assertRaises(ValueError):
                sample_triplet_schedule(["a", "b", "c"], [("a", "b")], epochs, seed)

    def test_rejects_array_endpoints_with_endpoint_validation_error(self):
        edge = (np.array([1, 2]), np.array([3, 4]))
        with self.assertRaisesRegex(ValueError, "positive edge endpoints must be strings"):
            sample_triplet_schedule(["a", "b", "c"], [edge], epochs=1, seed=0)

    def test_schedule_uses_only_supplied_partition_ids(self):
        train_ids = ["train-a", "train-b", "train-c"]
        schedule = sample_triplet_schedule(train_ids, [("train-a", "train-b")], epochs=1, seed=7)
        self.assertTrue(np.all(schedule < len(train_ids)))
        self.assertNotIn("validation", train_ids)
        self.assertNotIn("test", train_ids)

    def test_known_positive_edges_are_excluded_even_when_not_scheduled(self):
        ids = ["a", "b", "c", "d"]
        schedule = sample_triplet_schedule(
            ids,
            [("a", "b")],
            epochs=100,
            seed=7,
            known_positive_edges=[("a", "b"), ("a", "c")],
        )
        index_to_id = dict(enumerate(ids))
        for epoch in schedule:
            for anchor_idx, positive_idx, negative_idx in epoch:
                anchor, positive, negative = map(
                    index_to_id.__getitem__, (anchor_idx, positive_idx, negative_idx)
                )
                self.assertNotEqual({anchor, negative}, {"a", "c"})

    def test_rejects_scheduled_positive_absent_from_known_positive_edges(self):
        with self.assertRaisesRegex(ValueError, "positive edges must be a subset"):
            sample_triplet_schedule(
                ["a", "b", "c"],
                [("a", "b")],
                epochs=1,
                seed=0,
                known_positive_edges=[("a", "c")],
            )

    def test_omitted_known_positive_edges_preserves_legacy_schedule(self):
        ids = ["a", "b", "c", "d"]
        schedule = sample_triplet_schedule(ids, [("a", "b")], epochs=2, seed=7)
        expected = np.array(
            [
                [[0, 1, 2], [1, 0, 3]],
                [[0, 1, 2], [1, 0, 3]],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(schedule, expected)

    def test_negative_sampling_is_uniform_over_eligible_ids(self):
        ids = ["a", "b", "c", "d"]
        schedule = sample_triplet_schedule(ids, [("a", "b")], epochs=10_000, seed=684)
        first_anchor_negatives = schedule[:, 0, 2]
        counts = np.bincount(first_anchor_negatives, minlength=len(ids))
        for eligible_id in ("c", "d"):
            proportion = counts[ids.index(eligible_id)] / len(first_anchor_negatives)
            with self.subTest(negative=eligible_id):
                self.assertGreaterEqual(proportion, 0.45)
                self.assertLessEqual(proportion, 0.55)


class MixedSemiHardTripletTests(unittest.TestCase):
    def setUp(self):
        # From anchor a, b is closer (cos=.95), c is in the strict band
        # (cos=.85), and d/e are farther than the band (cos=.6/.2).
        self.ids = ["a", "b", "c", "d", "e", "f"]
        self.embeddings = np.array(
            [
                [1.0, 0.0],
                [0.95, np.sqrt(1.0 - 0.95**2)],
                [0.85, np.sqrt(1.0 - 0.85**2)],
                [0.6, np.sqrt(1.0 - 0.6**2)],
                [0.2, np.sqrt(1.0 - 0.2**2)],
                [-0.5, np.sqrt(1.0 - 0.5**2)],
            ],
            dtype=np.float64,
        )

    def sample(self, **kwargs):
        return sample_mixed_semi_hard_triplets(
            self.ids, [("a", "b"), ("d", "e")], self.embeddings,
            seed=684, **kwargs,
        )

    def test_deterministic_directed_rows_strict_band_and_requested_share(self):
        result = self.sample(margin=0.2, semi_hard_fraction=0.25)
        repeated = self.sample(margin=0.2, semi_hard_fraction=0.25)
        np.testing.assert_array_equal(result.triplets, repeated.triplets)
        self.assertEqual(result.sampling_kinds, repeated.sampling_kinds)
        self.assertEqual(result.triplets.shape, (4, 3))
        self.assertEqual(result.triplets[:, :2].tolist(), [[0, 1], [1, 0], [3, 4], [4, 3]])
        self.assertEqual(result.attempted_semi_hard_count, 1)
        self.assertEqual(result.realized_semi_hard_fraction, 0.25)
        self.assertEqual(result.realized_uniform_fraction, 0.75)
        for row, kind in zip(result.triplets, result.sampling_kinds):
            if kind == "semi_hard":
                a, p, n = self.embeddings[row]
                d_ap = 1.0 - np.dot(a, p)
                d_an = 1.0 - np.dot(a, n)
                self.assertLess(d_ap, d_an)
                self.assertLess(d_an, d_ap + 0.2)

    def test_excludes_all_known_positives_and_only_uses_train_candidates(self):
        result = sample_mixed_semi_hard_triplets(
            self.ids, [("a", "b")], self.embeddings, seed=1,
            known_positive_edges=[("a", "b"), ("a", "c"), ("a", "d")],
        )
        self.assertNotIn(self.ids[result.triplets[0, 2]], {"a", "b", "c", "d"})
        self.assertTrue(np.all(result.triplets[:, 2] < len(self.ids)))

    def test_empty_semi_hard_pool_falls_back_and_reports_pool_counts(self):
        embeddings = np.array([[1., 0.], [0.99, 0.1], [-1., 0.], [-0.5, 0.8660254]])
        result = sample_mixed_semi_hard_triplets(
            ["a", "b", "c", "d"], [("a", "b")], embeddings, seed=1,
            semi_hard_fraction=1.0,
        )
        self.assertEqual(result.sampling_kinds, ("fallback_uniform", "fallback_uniform"))
        self.assertEqual(result.fallback_count, 2)
        self.assertTrue(np.all(result.eligible_semi_hard_counts == 0))
        self.assertTrue(np.all(result.eligible_uniform_counts > 0))

    def test_zero_margin_has_empty_band_and_falls_back_with_realized_diagnostics(self):
        result = self.sample(margin=0.0, semi_hard_fraction=1.0)
        self.assertEqual(result.sampling_kinds, ("fallback_uniform",) * 4)
        self.assertEqual(result.attempted_semi_hard_count, 4)
        self.assertEqual(result.fallback_count, 4)
        self.assertEqual(result.realized_semi_hard_fraction, 0.0)
        self.assertEqual(result.realized_uniform_fraction, 1.0)
        np.testing.assert_array_equal(result.eligible_semi_hard_counts, np.zeros(4, dtype=np.int32))

    def test_rejects_empty_directed_positive_schedule(self):
        with self.assertRaisesRegex(ValueError, "at least one positive edge"):
            sample_mixed_semi_hard_triplets(
                self.ids, [], self.embeddings, seed=0,
            )

    def test_uniform_only_rows_do_not_request_semi_hard_scoring(self):
        sampler = MixedSemiHardTripletSampler(
            self.ids, [("a", "b"), ("d", "e")], self.embeddings,
            semi_hard_fraction=0.0,
        )
        with mock.patch.object(sampler, "_semi_hard_candidates", wraps=sampler._semi_hard_candidates) as score:
            result = sampler.sample(seed=3)
        score.assert_not_called()
        self.assertTrue(np.all(result.semi_hard_evaluated == 0))
        self.assertTrue(np.all(result.eligible_semi_hard_counts == -1))

    def test_exclusions_are_compactly_stored_once_per_anchor(self):
        sampler = MixedSemiHardTripletSampler(
            self.ids, [("a", "b"), ("a", "c"), ("d", "e")], self.embeddings,
        )
        self.assertFalse(hasattr(sampler, "_uniform_candidates"))
        self.assertEqual(
            set(sampler._forbidden_by_anchor),
            {sampler.id_to_index[track_id] for track_id in ("a", "b", "c", "d", "e")},
        )
        self.assertLess(
            sum(map(len, sampler._forbidden_by_anchor.values())),
            len(sampler.directed_edges) * len(self.ids),
        )
        self.assertFalse(hasattr(sampler, "_anchor_similarity_cache"))
        self.assertFalse(hasattr(sampler, "_semi_hard_pool_cache"))

    def test_similarity_matrix_is_exact_float32_and_train_partition_sized(self):
        sampler = MixedSemiHardTripletSampler(
            self.ids, [("a", "b")], self.embeddings,
        )
        normalized = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized = normalized.astype(np.float32)
        expected = normalized @ normalized.T
        self.assertEqual(sampler._similarity_matrix.shape, (len(self.ids), len(self.ids)))
        self.assertEqual(sampler._similarity_matrix.dtype, np.float32)
        np.testing.assert_array_equal(sampler._similarity_matrix, expected)

    def test_reusable_sampler_reuses_bounded_snapshot_and_is_deterministic_per_seed(self):
        sampler = MixedSemiHardTripletSampler(
            self.ids, [("a", "b"), ("a", "c"), ("d", "e"), ("d", "f")],
            self.embeddings, semi_hard_fraction=1.0,
        )
        similarity_snapshot = sampler._similarity_matrix
        first = sampler.sample(seed=1)
        repeated = sampler.sample(seed=1)
        other_epoch = sampler.sample(seed=2)
        np.testing.assert_array_equal(first.triplets, repeated.triplets)
        self.assertFalse(np.array_equal(first.triplets, other_epoch.triplets))
        self.assertIs(sampler._similarity_matrix, similarity_snapshot)
        self.assertFalse(hasattr(sampler, "_anchor_similarity_cache"))
        self.assertFalse(hasattr(sampler, "_semi_hard_pool_cache"))

    def test_mixed_sampler_does_not_call_legacy_schedule_sampler(self):
        with mock.patch("src.triplets.sample_triplet_schedule", side_effect=AssertionError("legacy called")):
            result = self.sample()
        self.assertEqual(result.triplets.shape, (4, 3))

    def test_rejects_impossible_uniform_negative_and_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "no eligible negative"):
            sample_mixed_semi_hard_triplets(
                ["a", "b"], [("a", "b")], np.eye(2), seed=0,
            )
        bad_embeddings = (
            np.ones((3, 2, 1)),
            np.array([[1., 0.], [0., 1.], [np.nan, 0.]]),
            np.array([[1., 0.], [0., 1.], [np.inf, 0.]]),
            np.ones((2, 2)),
        )
        for embeddings in bad_embeddings:
            with self.subTest(embeddings=embeddings.shape), self.assertRaises(ValueError):
                sample_mixed_semi_hard_triplets(
                    ["a", "b", "c"], [("a", "b")], embeddings, seed=0,
                )
        for kwargs in ({"margin": -0.1}, {"margin": np.nan}, {"margin": np.inf},
                       {"semi_hard_fraction": -0.1}, {"semi_hard_fraction": 1.1},
                       {"semi_hard_fraction": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sample(**kwargs)


if __name__ == "__main__":
    unittest.main()

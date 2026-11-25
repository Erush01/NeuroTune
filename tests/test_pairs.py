import unittest
from math import comb

from src.pairs import build_pair_pools, sample_pairs


class PairSamplerTests(unittest.TestCase):
    def test_builds_partition_local_deduplicated_undirected_positives(self):
        edges = [
            {"source": "b", "target": "a"},
            {"source": "a", "target": "b"},
            {"source": "a", "target": "a"},
            {"source": "a", "target": "c"},
            {"source": "a", "target": "outside"},
            {"source": "outside", "target": "a"},
        ]
        pools = build_pair_pools(edges, ["a", "b", "c"], {"a": "rock", "b": "rock", "c": "jazz"})

        self.assertEqual(pools["positives"], [("a", "b"), ("a", "c")])
        self.assertEqual(pools["partition_ids"], ["a", "b", "c"])
        self.assertEqual(pools["genre_groups"], {"rock": ["a", "b"], "jazz": ["c"]})
        self.assertEqual(pools["random_pair_count"], 1)
        self.assertEqual(pools["same_genre_pair_count"], 0)

    def test_filters_known_positives_and_self_edges_from_negative_pools(self):
        edges = [
            {"source": "a", "target": "b"},
            {"source": "a", "target": "a"},
        ]
        pools = build_pair_pools(edges, ["a", "b", "c"], {"a": "rock", "b": "rock", "c": "rock"})

        self.assertEqual(pools["positives"], [("a", "b")])
        self.assertEqual(pools["random_pair_count"], 2)
        self.assertEqual(pools["same_genre_pair_count"], 2)

    def test_fixed_half_policy_has_exact_counts_and_no_duplicate_negative_pairs(self):
        ids = [f"track-{index:03}" for index in range(30)]
        edges = [{"source": ids[index], "target": ids[index + 1]} for index in range(0, 24, 2)]
        genres = {track_id: "rock" if index % 2 == 0 else "jazz" for index, track_id in enumerate(ids)}
        pools = build_pair_pools(edges, ids, genres)

        rows = sample_pairs(pools, seed=684)
        negatives = [row[:2] for row in rows if row[2] == 0.0]
        self.assertEqual(sum(row[2] == 1.0 for row in rows), 12)
        self.assertEqual(len(negatives), 12)
        same_genre_pairs = {pair for pair in negatives[-6:] if genres[pair[0]] == genres[pair[1]]}
        self.assertEqual(len(same_genre_pairs), 6)
        self.assertEqual(len(negatives), len(set(negatives)))

    def test_odd_negative_count_assigns_remainder_to_random_deterministically(self):
        ids = list("abcdefgh")
        edges = [
            {"source": "a", "target": "e"},
            {"source": "b", "target": "f"},
            {"source": "c", "target": "g"},
        ]
        genres = {track_id: "rock" if track_id in "abcd" else "jazz" for track_id in ids}
        pools = build_pair_pools(edges, ids, genres)

        rows = sample_pairs(pools, seed=684)
        repeated_rows = sample_pairs(pools, seed=684)
        negatives = [row[:2] for row in rows if row[2] == 0.0]
        random_negatives = negatives[:2]
        same_genre_negatives = negatives[2:]

        self.assertEqual(rows, repeated_rows)
        self.assertEqual(len(random_negatives), 2)
        self.assertEqual(len(same_genre_negatives), 1)
        self.assertEqual(len(negatives), len(set(negatives)))
        self.assertEqual(sum(genres[left] == genres[right] for left, right in same_genre_negatives), 1)

    def test_sampling_is_deterministic_for_seed(self):
        ids = ["a", "b", "c", "d", "e", "f"]
        pools = build_pair_pools(
            [{"source": "a", "target": "b"}, {"source": "c", "target": "d"}],
            ids,
            {track_id: f"genre-{index % 2}" for index, track_id in enumerate(ids)},
        )

        self.assertEqual(sample_pairs(pools, seed=12), sample_pairs(pools, seed=12))
        self.assertNotEqual(sample_pairs(pools, seed=12), sample_pairs(pools, seed=13))

    def test_416_positives_yield_208_negatives_per_pool_at_half_fraction(self):
        # Use all 832 endpoints plus candidates, with a same-genre class large enough for sampling.
        ids = [f"track-{index:04}" for index in range(900)]
        edges = [{"source": ids[index * 2], "target": ids[index * 2 + 1]} for index in range(416)]
        genres = {track_id: f"genre-{index % 2}" for index, track_id in enumerate(ids)}
        pools = build_pair_pools(edges, ids, genres)

        rows = sample_pairs(pools, seed=684)

        self.assertEqual(sum(row[2] == 1.0 for row in rows), 416)
        self.assertEqual(sum(row[2] == 0.0 for row in rows), 416)
        negatives = [row[:2] for row in rows if row[2] == 0.0]
        self.assertTrue(all(genres[left] == genres[right] for left, right in negatives[-208:]))

    def test_large_partition_pool_is_compact_and_reports_combinatorial_counts(self):
        ids = [f"track-{index:04}" for index in range(500)]
        edges = [{"source": ids[index], "target": ids[index + 1]} for index in range(0, 20, 2)]
        genres = {track_id: f"genre-{index % 5}" for index, track_id in enumerate(ids)}

        pools = build_pair_pools(edges, ids, genres)

        known_same_genre = sum(genres[left] == genres[right] for left, right in pools["positives"])
        expected_same_genre = sum(
            comb(sum(genre == value for genre in genres.values()), 2)
            for value in set(genres.values())
        ) - known_same_genre
        self.assertEqual(pools["random_pair_count"], comb(len(ids), 2) - len(pools["positives"]))
        self.assertEqual(pools["same_genre_pair_count"], expected_same_genre)
        self.assertEqual(set(pools), {
            "positives", "partition_ids", "genre_groups", "random_pair_count", "same_genre_pair_count"
        })
        self.assertEqual(len(pools["partition_ids"]), len(ids))
        self.assertEqual(sum(map(len, pools["genre_groups"].values())), len(ids))

    def test_raises_when_requested_candidate_pool_is_too_small(self):
        pools = build_pair_pools(
            [{"source": "a", "target": "b"}, {"source": "c", "target": "d"}],
            ["a", "b", "c", "d"],
            {"a": "g1", "b": "g2", "c": "g3", "d": "g4"},
        )

        with self.assertRaisesRegex(ValueError, "candidate pool"):
            sample_pairs(pools, seed=1)

    def test_rejects_negative_counts_or_fractions_that_cannot_be_preserved(self):
        pools = build_pair_pools(
            [{"source": "a", "target": "b"}, {"source": "c", "target": "d"}],
            ["a", "b", "c", "d", "e"],
            {track_id: "rock" for track_id in "abcde"},
        )

        with self.assertRaisesRegex(ValueError, "fraction"):
            sample_pairs(pools, seed=1, same_genre_fraction=0.3)


if __name__ == "__main__":
    unittest.main()

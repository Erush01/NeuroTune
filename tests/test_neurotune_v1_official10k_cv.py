import unittest

from src.data.neurotune_v1_official10k_cv import build_cv_audit


def tid(number):
    return f"TR{number:016d}"


class Official10kCvAuditTests(unittest.TestCase):
    def setUp(self):
        self.tracks = {
            tid(i): {"artist_id": f"artist-{i // 2}", "artist_name": f"Artist {i // 2}"}
            for i in range(12)
        }
        self.edges = {
            tuple(sorted((tid(i), tid(j))))
            for i, j in ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (6, 7),
                         (8, 9), (10, 11), (0, 8), (2, 10), (4, 6), (5, 9))
        }

    def test_build_is_order_independent_and_deterministic(self):
        first = build_cv_audit(self.tracks, self.edges)
        second = build_cv_audit(dict(reversed(list(self.tracks.items()))), set(reversed(sorted(self.edges))))
        self.assertEqual(first, second)
        self.assertEqual([len(fold["track_ids"]) for fold in first["folds"]], [4, 4, 4])
        self.assertTrue(all(first["artist_groups"].values()))

    def test_validation_floor_and_positive_edge_partitions(self):
        audit = build_cv_audit(self.tracks, self.edges)
        for fold in audit["folds"]:
            self.assertEqual(len(fold["validation_edges"]), int(.2 * len(fold["non_test_pool_edges"])))
            self.assertFalse(set(map(tuple, fold["train_edges"])) & set(map(tuple, fold["validation_edges"])))
            self.assertFalse(set(map(tuple, fold["test_edges"])) & set(map(tuple, fold["train_edges"])))
            self.assertFalse(set(map(tuple, fold["test_edges"])) & set(map(tuple, fold["validation_edges"])))

    def test_cross_fold_edges_are_excluded_from_each_test_and_training_pool(self):
        audit = build_cv_audit(self.tracks, self.edges)
        fold_by_id = {track_id: fold["fold"] for fold in audit["folds"] for track_id in fold["track_ids"]}
        expected_cross = {edge for edge in self.edges if fold_by_id[edge[0]] != fold_by_id[edge[1]]}
        self.assertEqual(set(map(tuple, audit["cross_fold_edges"])), expected_cross)
        for fold in audit["folds"]:
            self.assertFalse(expected_cross & set(map(tuple, fold["test_edges"])))

    def test_artist_fallback_order(self):
        tracks = {tid(1): {"artist_id": "", "artist_name": "Named"},
                  tid(2): {"artist_id": "", "artist_name": "Named"},
                  tid(3): {"artist_id": "", "artist_name": ""}}
        audit = build_cv_audit(tracks, set())
        groups = audit["artist_groups"]
        self.assertEqual(groups[tid(1)], groups[tid(2)])
        self.assertNotEqual(groups[tid(1)], groups[tid(3)])

    def test_rejects_invalid_track_ids_and_metadata(self):
        for metadata in ({"": {}}, {None: {}}, {1: {}}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "track ID"):
                build_cv_audit(metadata, set())
        for value in (None, "artist", 3):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "metadata"):
                build_cv_audit({tid(1): value}, set())
        for field_value in (3, [], {}):
            with self.subTest(field_value=field_value), self.assertRaisesRegex(ValueError, "artist_id"):
                build_cv_audit({tid(1): {"artist_id": field_value}}, set())
            with self.subTest(field_value=field_value), self.assertRaisesRegex(ValueError, "artist_name"):
                build_cv_audit({tid(1): {"artist_name": field_value}}, set())

    def test_rejects_malformed_positive_edges_and_endpoints(self):
        for edges in (None, 7):
            with self.subTest(edges=edges), self.assertRaisesRegex(ValueError, "positive_edges"):
                build_cv_audit(self.tracks, edges)
        for edge in ((), (tid(1),), (tid(1), tid(2), tid(3)), "ab", None):
            with self.subTest(edge=edge), self.assertRaisesRegex(ValueError, "edge"):
                build_cv_audit(self.tracks, [edge])
        for edge in (("", tid(1)), (tid(1), None), (tid(1), 2), (tid(1), tid(1)),
                     (tid(1), "TR9999999999999999")):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                build_cv_audit(self.tracks, [edge])

    def test_rejects_invalid_seed(self):
        for seed in (-1, True, 1.5, "684", None):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                build_cv_audit(self.tracks, self.edges, seed=seed)


if __name__ == "__main__":
    unittest.main()

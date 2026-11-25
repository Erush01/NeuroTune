import unittest

import numpy as np

from src.features import build_b0_vectors


class B0FeatureTests(unittest.TestCase):
    def setUp(self):
        self.tracks = [
            {"track_id": "train-a", "tempo": 100.0},
            {"track_id": "train-b", "tempo": 140.0},
            {"track_id": "test", "tempo": 1000.0},
        ]
        self.lyrics = {
            "train-a": {"1": 2, "5000": 1},
            "train-b": {"1": 1, "2": 3},
            "test": {"2": 1, "5000": 9},
        }
        self.genres = {"train-a": "Rock", "train-b": "Jazz", "test": "Unseen"}
        self.train_ids = ["train-a", "train-b"]

    def test_bpm_uses_train_only_mean_and_std(self):
        vectors, metadata = build_b0_vectors(
            self.tracks, self.lyrics, self.genres, self.train_ids, ["train-a", "train-b", "test"]
        )
        self.assertEqual(metadata["bpm_mean"], 120.0)
        self.assertEqual(metadata["bpm_std"], 20.0)
        # The raw BPM block is weighted and then normalized with the full row.
        self.assertGreater(vectors[2, 0], vectors[1, 0])

    def test_smoothed_idf_uses_train_documents_only(self):
        _, metadata = build_b0_vectors(
            self.tracks, self.lyrics, self.genres, self.train_ids, ["train-a", "train-b", "test"]
        )
        idf = metadata["idf"]
        self.assertEqual(idf[0], np.log(3 / 3) + 1)
        self.assertEqual(idf[1], np.log(3 / 2) + 1)
        self.assertEqual(idf[4999], np.log(3 / 2) + 1)

    def test_lyric_vectors_are_l2_normalized_and_use_fixed_word_indices(self):
        vectors, metadata = build_b0_vectors(
            self.tracks, self.lyrics, self.genres, self.train_ids, ["train-a", "train-b", "test"]
        )
        # Verify train-a's normalized TF-IDF vector occupies official indices 1 and 5000.
        self.assertEqual(len(metadata["idf"]), 5000)
        self.assertEqual(vectors.shape[1], 1 + 5000 + 2)
        lyric = np.zeros(5000)
        lyric[0] = 2 * metadata["idf"][0]
        lyric[4999] = metadata["idf"][4999]
        lyric /= np.linalg.norm(lyric)
        expected = np.zeros(1 + 5000 + 2)
        expected[0] = -1 / np.sqrt(3)
        expected[1 : 1 + 5000] = lyric / np.sqrt(3)
        expected[-1] = 1 / np.sqrt(3)  # Rock is the second sorted train genre.
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vectors[0], expected)

    def test_genre_vocabulary_is_train_observed_sorted_and_unseen_maps_to_zero(self):
        vectors, metadata = build_b0_vectors(
            self.tracks, self.lyrics, self.genres, self.train_ids, ["train-a", "train-b", "test"]
        )
        self.assertEqual(metadata["genre_vocabulary"], ["Jazz", "Rock"])
        genres = vectors[:, 1 + 5000 :]
        self.assertEqual(genres.shape, (3, 2))
        self.assertNotEqual(np.linalg.norm(genres[0]), 0)
        self.assertNotEqual(np.linalg.norm(genres[1]), 0)
        np.testing.assert_array_equal(genres[2], np.zeros(2))

    def test_rejects_fractional_training_lyric_counts(self):
        for count in (1.9, -0.5):
            with self.subTest(count=count):
                lyrics = {**self.lyrics, "train-a": {"1": count}}
                with self.assertRaisesRegex(ValueError, "invalid lyric entry"):
                    build_b0_vectors(self.tracks, lyrics, self.genres, self.train_ids, ["train-a"])

    def test_rejects_fractional_transformed_lyric_counts(self):
        for count in (1.9, -0.5):
            with self.subTest(count=count):
                lyrics = {**self.lyrics, "test": {"1": count}}
                with self.assertRaisesRegex(ValueError, "invalid lyric entry"):
                    build_b0_vectors(self.tracks, lyrics, self.genres, self.train_ids, ["test"])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from src.features import build_b0_vectors, build_train_fitted_modalities


class N0FeatureTests(unittest.TestCase):
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
        self.song_ids = ["train-a", "train-b", "test"]

    def test_modality_shapes_train_only_fits_and_unseen_genre(self):
        modalities, metadata = build_train_fitted_modalities(
            self.tracks, self.lyrics, self.genres, self.train_ids, self.song_ids
        )

        self.assertEqual(set(modalities), {"bpm", "lyrics", "genre"})
        self.assertEqual(modalities["bpm"].shape, (3, 1))
        self.assertEqual(modalities["lyrics"].shape, (3, 5000))
        self.assertEqual(modalities["genre"].shape, (3, 2))

        self.assertEqual(metadata["bpm_mean"], 120.0)
        self.assertEqual(metadata["bpm_std"], 20.0)
        np.testing.assert_allclose(modalities["bpm"][:, 0], [-1.0, 1.0, 44.0])

        idf = metadata["idf"]
        self.assertEqual(idf[0], np.log(3 / 3) + 1)
        self.assertEqual(idf[1], np.log(3 / 2) + 1)
        self.assertEqual(idf[4999], np.log(3 / 2) + 1)
        expected_lyrics = np.zeros(5000)
        expected_lyrics[0] = 2 * idf[0]
        expected_lyrics[4999] = idf[4999]
        expected_lyrics /= np.linalg.norm(expected_lyrics)
        np.testing.assert_allclose(modalities["lyrics"][0], expected_lyrics)
        self.assertAlmostEqual(np.linalg.norm(modalities["lyrics"][0]), 1.0)

        self.assertEqual(metadata["genre_vocabulary"], ["Jazz", "Rock"])
        np.testing.assert_array_equal(modalities["genre"][0], [0.0, 1.0])
        np.testing.assert_array_equal(modalities["genre"][1], [1.0, 0.0])
        np.testing.assert_array_equal(modalities["genre"][2], [0.0, 0.0])

    def test_b0_output_remains_numerically_unchanged(self):
        vectors, _ = build_b0_vectors(
            self.tracks, self.lyrics, self.genres, self.train_ids, self.song_ids
        )

        expected = np.zeros(1 + 5000 + 2)
        expected[0] = -1 / np.sqrt(3)
        lyric = np.zeros(5000)
        lyric[0] = 2.0
        lyric[4999] = np.log(3 / 2) + 1
        lyric /= np.linalg.norm(lyric)
        expected[1 : 1 + 5000] = lyric / np.sqrt(3)
        expected[-1] = 1 / np.sqrt(3)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vectors[0], expected)


if __name__ == "__main__":
    unittest.main()

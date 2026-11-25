import unittest

import numpy as np
import torch

from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate


class NeuroTuneV1DatasetTests(unittest.TestCase):
    def setUp(self):
        self.track_ids = ["B", "A"]
        self.audio = [
            np.full((1, 26), 2, dtype=np.float32),
            np.full((3, 26), 7, dtype=np.float32),
        ]
        self.features = {
            "audio": self.audio,
            "bpm": np.asarray([[20], [10]], dtype=np.float32),
            "lyrics": np.stack([np.full(5000, 2), np.full(5000, 1)]).astype(np.float32),
            "genre": np.asarray([[0, 1], [1, 0]], dtype=np.float32),
        }

    def test_lookup_preserves_requested_id_order_and_aligns_modalities(self):
        dataset = NeuroTuneV1Dataset(self.track_ids, self.features)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["track_id"], "B")
        self.assertEqual(dataset[1]["track_id"], "A")
        self.assertEqual(float(dataset[0]["bpm"][0]), 20)
        self.assertEqual(float(dataset[1]["lyrics"][0]), 1)
        self.assertEqual(dataset.track_to_index, {"B": 0, "A": 1})

    def test_collator_right_pads_masks_and_preserves_aux_alignment(self):
        dataset = NeuroTuneV1Dataset(self.track_ids, self.features)
        original_audio = [array.copy() for array in self.audio]
        original_bpm = self.features["bpm"].copy()
        original_lyrics = self.features["lyrics"].copy()
        original_genre = self.features["genre"].copy()
        batch = neurotune_v1_collate([dataset[0], dataset[1]])
        alone = neurotune_v1_collate([dataset[0]])
        self.assertEqual(tuple(batch["audio"].shape), (2, 3, 26))
        self.assertEqual(tuple(batch["mask"].shape), (2, 3))
        self.assertTrue(batch["mask"].dtype == torch.bool)
        torch.testing.assert_close(batch["mask"], torch.tensor([[True, False, False], [True, True, True]]))
        torch.testing.assert_close(batch["lengths"], torch.tensor([1, 3]))
        self.assertTrue(torch.count_nonzero(batch["audio"][0, 1:]) == 0)
        torch.testing.assert_close(alone["audio"][0], batch["audio"][0, :1])
        self.assertTrue(torch.all(batch["audio"][0, 1:] == 0))
        self.assertTrue(torch.all(batch["audio"][~batch["mask"]] == 0))
        torch.testing.assert_close(batch["audio"][0, 0], torch.full((26,), 2.0))
        self.assertEqual(batch["track_ids"], ["B", "A"])
        torch.testing.assert_close(batch["bpm"], torch.tensor([[20.], [10.]]))
        torch.testing.assert_close(batch["genre"], torch.tensor([[0., 1.], [1., 0.]]))
        torch.testing.assert_close(batch["lyrics"][:, 0], torch.tensor([2., 1.]))
        for actual, expected in zip(self.audio, original_audio):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.features["bpm"], original_bpm)
        np.testing.assert_array_equal(self.features["lyrics"], original_lyrics)
        np.testing.assert_array_equal(self.features["genre"], original_genre)

    def test_explicit_modalities_allow_c_dataset_without_lyrics(self):
        features = {key: value for key, value in self.features.items() if key != "lyrics"}
        dataset = NeuroTuneV1Dataset(self.track_ids, features, modalities=("bpm", "genre"))
        self.assertEqual(set(dataset[0]), {"track_id", "audio", "bpm", "genre"})
        batch = neurotune_v1_collate([dataset[0], dataset[1]])
        self.assertEqual(set(batch), {"audio", "mask", "lengths", "bpm", "genre", "track_ids"})
        from src.model_v1 import NeuroTuneV1Encoder
        embeddings = NeuroTuneV1Encoder("C", 2).forward_triplets(batch, batch, batch)
        self.assertEqual(tuple(embeddings[0].shape), (2, 64))

    def test_collator_rejects_inconsistent_auxiliary_keys(self):
        dataset = NeuroTuneV1Dataset(self.track_ids, self.features)
        with self.assertRaisesRegex(ValueError, "modality"):
            neurotune_v1_collate([dataset[0], {key: value for key, value in dataset[1].items() if key != "lyrics"}])

    def test_rejects_unhashable_modality_names_with_value_error(self):
        with self.assertRaisesRegex(ValueError, "modalities"):
            NeuroTuneV1Dataset(self.track_ids, self.features, modalities=[[]])

    def test_single_sequence_batch_has_no_extra_padding(self):
        dataset = NeuroTuneV1Dataset(self.track_ids, self.features)
        batch = neurotune_v1_collate([dataset[0]])
        self.assertEqual(tuple(batch["audio"].shape), (1, 1, 26))
        torch.testing.assert_close(batch["mask"], torch.tensor([[True]]))

    def test_rejects_duplicate_or_empty_ids(self):
        for ids in (["A", "A"], ["", "B"], []):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                NeuroTuneV1Dataset(ids, self.features)

    def test_rejects_invalid_audio_sequences(self):
        for audio in ([np.zeros((0, 26), np.float32), self.audio[1]],
                      [np.zeros((2, 25), np.float32), self.audio[1]],
                      [np.zeros((1, 26), np.float64), self.audio[1]],
                      [np.full((1, 26), np.nan, np.float32), self.audio[1]]):
            with self.subTest(shape=audio[0].shape, dtype=audio[0].dtype), self.assertRaises(ValueError):
                NeuroTuneV1Dataset(self.track_ids, {**self.features, "audio": audio})

    def test_rejects_auxiliary_dtype_nonfinite_and_shape_or_row_mismatch(self):
        invalid = [
            {"bpm": self.features["bpm"][:-1]},
            {"bpm": np.zeros((2, 2), dtype=np.float32)},
            {"bpm": np.asarray([[1], [np.inf]], dtype=np.float32)},
            {"bpm": self.features["bpm"].astype(np.float64)},
            {"lyrics": np.zeros((2, 4999), dtype=np.float32)},
            {"genre": np.zeros((1, 2), dtype=np.float32)},
            {"genre": np.zeros((2, 0), dtype=np.float32)},
        ]
        for replacement in invalid:
            with self.subTest(key=next(iter(replacement))), self.assertRaises(ValueError):
                NeuroTuneV1Dataset(self.track_ids, {**self.features, **replacement})

    def test_multi_window_lookup_and_collation_align_validity_and_pad_ragged_windows(self):
        windows = {
            "A": [np.full((1, 26), 7, np.float32), np.full((4, 26), 8, np.float32),
                  np.zeros((0, 26), np.float32)],
            "B": [np.full((2, 26), 2, np.float32), np.zeros((0, 26), np.float32),
                  np.full((1, 26), 3, np.float32)],
        }
        valid = np.asarray([[True, False, True], [True, True, False]], dtype=np.bool_)
        features = {key: value for key, value in self.features.items() if key != "audio"}
        features.update(audio_windows=windows, window_valid=valid)
        originals = {track_id: [window.copy() for window in track_windows]
                     for track_id, track_windows in windows.items()}
        dataset = NeuroTuneV1Dataset(self.track_ids, features)

        row = dataset[0]
        self.assertEqual(set(row), {"track_id", "audio_windows", "window_valid", "bpm", "lyrics", "genre"})
        self.assertEqual(row["track_id"], "B")
        self.assertIs(row["audio_windows"], windows["B"])
        np.testing.assert_array_equal(row["window_valid"], valid[0])
        batch = neurotune_v1_collate([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["audio_windows"].shape), (2, 3, 4, 26))
        self.assertEqual(tuple(batch["mask"].shape), (2, 3, 4))
        self.assertEqual(tuple(batch["window_mask"].shape), (2, 3))
        self.assertEqual(batch["audio_windows"].dtype, torch.float32)
        self.assertEqual(batch["mask"].dtype, torch.bool)
        torch.testing.assert_close(batch["window_mask"], torch.tensor(valid))
        torch.testing.assert_close(batch["lengths"], torch.tensor([[2, 0, 1], [1, 4, 0]]))
        self.assertTrue(torch.all(batch["audio_windows"][0, 1] == 0))
        self.assertTrue(torch.all(batch["audio_windows"][1, 2] == 0))
        self.assertTrue(torch.all(batch["mask"][~batch["window_mask"]] == 0))
        torch.testing.assert_close(batch["mask"], torch.tensor([
            [[True, True, False, False], [False] * 4, [True, False, False, False]],
            [[True, False, False, False], [True] * 4, [False] * 4],
        ]))
        self.assertEqual(batch["track_ids"], self.track_ids)
        torch.testing.assert_close(batch["bpm"], torch.tensor([[20.], [10.]]))
        torch.testing.assert_close(batch["genre"], torch.tensor([[0., 1.], [1., 0.]]))
        for track_id, track_windows in windows.items():
            for actual, expected in zip(track_windows, originals[track_id]):
                np.testing.assert_array_equal(actual, expected)

    def test_multi_window_short_track_can_repeat_same_valid_view(self):
        repeated = np.full((1, 26), 5, dtype=np.float32)
        features = {key: value for key, value in self.features.items() if key != "audio"}
        features.update(audio_windows={"B": [repeated, repeated, repeated],
                                      "A": [repeated, repeated, repeated]},
                        window_valid=np.ones((2, 3), dtype=np.bool_))
        dataset = NeuroTuneV1Dataset(self.track_ids, features)
        batch = neurotune_v1_collate([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["audio_windows"].shape), (2, 3, 1, 26))
        torch.testing.assert_close(batch["lengths"], torch.ones((2, 3), dtype=torch.long))
        torch.testing.assert_close(batch["mask"], torch.ones((2, 3, 1), dtype=torch.bool))

    def test_multi_window_rejects_missing_extra_or_malformed_mappings(self):
        windows = {"B": [self.audio[0]] * 3, "A": [self.audio[1]] * 3}
        base = {key: value for key, value in self.features.items() if key != "audio"}
        valid = np.ones((2, 3), dtype=np.bool_)
        invalid_mappings = [
            {"B": windows["B"]},
            {**windows, "C": [self.audio[0]] * 3},
            {"B": [self.audio[0]] * 2, "A": windows["A"]},
            {"B": [self.audio[0]] * 3, "A": [self.audio[1], np.zeros((0, 26), np.float32), self.audio[1]]},
            {"B": [self.audio[0]] * 3, "A": [self.audio[1], np.zeros((1, 25), np.float32), self.audio[1]]},
            {"B": [self.audio[0]] * 3, "A": [self.audio[1], np.full((1, 26), np.nan, np.float32), self.audio[1]]},
        ]
        for bad_windows in invalid_mappings:
            with self.subTest(windows=bad_windows), self.assertRaises(ValueError):
                NeuroTuneV1Dataset(self.track_ids, {**base, "audio_windows": bad_windows, "window_valid": valid})
        for bad_valid in (np.ones((2, 2), dtype=np.bool_), np.ones((2, 3), dtype=np.int64),
                          np.asarray([[True, False, True], [True, True, True]], dtype=np.bool_)):
            with self.subTest(valid=bad_valid), self.assertRaises(ValueError):
                NeuroTuneV1Dataset(self.track_ids, {**base, "audio_windows": windows, "window_valid": bad_valid})

    def test_multi_window_rejects_zero_valid_tracks_and_ambiguous_audio(self):
        features = {key: value for key, value in self.features.items() if key != "audio"}
        windows = {track_id: [np.zeros((0, 26), np.float32)] * 3 for track_id in self.track_ids}
        with self.assertRaises(ValueError):
            NeuroTuneV1Dataset(self.track_ids, {**features, "audio_windows": windows,
                                                 "window_valid": np.zeros((2, 3), dtype=np.bool_)})
        with self.assertRaises(ValueError):
            NeuroTuneV1Dataset(self.track_ids, {**self.features, "audio_windows": windows,
                                                 "window_valid": np.zeros((2, 3), dtype=np.bool_)})

    def test_collator_rejects_mixed_single_and_multi_window_rows(self):
        dataset = NeuroTuneV1Dataset(self.track_ids, self.features)
        windows = {track_id: [sequence, sequence, sequence]
                   for track_id, sequence in zip(self.track_ids, self.audio)}
        multi_features = {key: value for key, value in self.features.items() if key != "audio"}
        multi_features.update(audio_windows=windows, window_valid=np.ones((2, 3), dtype=np.bool_))
        multi = NeuroTuneV1Dataset(self.track_ids, multi_features)
        with self.assertRaisesRegex(ValueError, "mix single- and multi-window"):
            neurotune_v1_collate([dataset[0], multi[1]])

    def test_multi_window_collator_rejects_malformed_rows_before_collation(self):
        windows = {track_id: [sequence, sequence, np.zeros((0, 26), np.float32)]
                   for track_id, sequence in zip(self.track_ids, self.audio)}
        features = {key: value for key, value in self.features.items() if key != "audio"}
        features.update(audio_windows=windows,
                        window_valid=np.asarray([[True, True, False], [True, True, False]], dtype=np.bool_))
        dataset = NeuroTuneV1Dataset(self.track_ids, features)
        good_rows = [dataset[0], dataset[1]]

        malformed = [
            [good_rows[0], "not a row mapping"],
            [{**good_rows[0], "audio_windows": good_rows[0]["audio_windows"][:2]}, good_rows[1]],
            [{**good_rows[0], "window_valid": np.ones(2, dtype=np.bool_)}, good_rows[1]],
            [{**good_rows[0], "audio_windows": [np.zeros((0, 26), np.float32),
                                                 *good_rows[0]["audio_windows"][1:]]}, good_rows[1]],
        ]
        for rows in malformed:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                neurotune_v1_collate(rows)


if __name__ == "__main__":
    unittest.main()

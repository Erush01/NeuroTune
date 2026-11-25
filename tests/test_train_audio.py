import unittest
from copy import deepcopy

import numpy as np
import torch

from src.model import AudioFirstSiameseEncoder
from src.pairs import build_pair_pools
from src.train_audio import train_audio_seed


class AudioTrainingTests(unittest.TestCase):
    def setUp(self):
        self.train_ids = [f"train-{index}" for index in range(6)]
        self.validation_ids = [f"validation-{index}" for index in range(6)]
        self.song_ids = self.train_ids + self.validation_ids
        rng = np.random.default_rng(10)
        self.audio = rng.normal(size=(12, 1024, 28)).astype(np.float32)
        self.modalities = {
            "bpm": np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(-1, 1),
            "lyrics": rng.normal(size=(12, 5000)).astype(np.float32),
            "genre": np.eye(2, dtype=np.float32)[np.arange(12) % 2],
        }
        genres = {song_id: "rock" for song_id in self.song_ids}
        self.train_pools = build_pair_pools(
            [
                {"source": self.train_ids[0], "target": self.train_ids[1]},
                {"source": self.train_ids[2], "target": self.train_ids[3]},
            ],
            self.train_ids,
            genres,
        )
        self.validation_pools = build_pair_pools(
            [
                {"source": self.validation_ids[0], "target": self.validation_ids[1]},
                {"source": self.validation_ids[2], "target": self.validation_ids[3]},
            ],
            self.validation_ids,
            genres,
        )

    def test_fixed_protocol_updates_model_and_repeats(self):
        seed = 684
        torch.manual_seed(seed)
        initial_model = AudioFirstSiameseEncoder(num_genres=2)
        initial_parameters = [parameter.detach().clone() for parameter in initial_model.parameters()]

        first_model, first_history, first_trace = train_audio_seed(
            self.audio,
            self.modalities,
            self.song_ids,
            self.train_pools,
            self.validation_pools,
            seed,
            epochs=2,
            batch_size=4,
        )
        second_model, second_history, second_trace = train_audio_seed(
            self.audio,
            self.modalities,
            self.song_ids,
            self.train_pools,
            self.validation_pools,
            seed,
            epochs=2,
            batch_size=4,
        )

        self.assertEqual(len(first_history), 2)
        self.assertTrue(
            all(
                np.isfinite(row["train_loss"]) and np.isfinite(row["validation_loss"])
                for row in first_history
            )
        )
        self.assertTrue(
            any(
                not torch.equal(initial, trained)
                for initial, trained in zip(initial_parameters, first_model.parameters())
            )
        )
        self.assertEqual(first_history, second_history)
        self.assertEqual(first_trace, second_trace)
        for first, second in zip(first_model.parameters(), second_model.parameters()):
            torch.testing.assert_close(first, second)

        train_partition = set(self.train_ids)
        validation_partition = set(self.validation_ids)
        for rows in first_trace["train_by_epoch"]:
            self.assertTrue(all(left in train_partition and right in train_partition for left, right, _ in rows))
        self.assertTrue(
            all(
                left in validation_partition and right in validation_partition
                for left, right, _ in first_trace["validation"]
            )
        )

    def test_rejects_pool_references_outside_own_partition(self):
        malformed_train_pools = deepcopy(self.train_pools)
        malformed_train_pools["positives"][0] = (
            self.train_ids[0], self.validation_ids[0]
        )
        with self.subTest(reference="positive"):
            with self.assertRaisesRegex(ValueError, "train.*partition"):
                train_audio_seed(
                    self.audio,
                    self.modalities,
                    self.song_ids,
                    malformed_train_pools,
                    self.validation_pools,
                    seed=684,
                    epochs=1,
                )

        malformed_train_pools = deepcopy(self.train_pools)
        malformed_train_pools["genre_groups"]["rock"].append(self.validation_ids[0])
        with self.subTest(reference="genre group"):
            with self.assertRaisesRegex(ValueError, "train.*partition"):
                train_audio_seed(
                    self.audio,
                    self.modalities,
                    self.song_ids,
                    malformed_train_pools,
                    self.validation_pools,
                    seed=684,
                    epochs=1,
                )

    def test_rejects_values_that_overflow_float32_conversion(self):
        oversized = float(np.finfo(np.float32).max) * 2.0
        invalid_audio = self.audio.astype(np.float64)
        invalid_audio[0, 0, 0] = oversized
        with self.subTest(input="audio"):
            with self.assertRaisesRegex(ValueError, "remain finite after float32 conversion"):
                train_audio_seed(
                    invalid_audio,
                    self.modalities,
                    self.song_ids,
                    self.train_pools,
                    self.validation_pools,
                    seed=684,
                    epochs=1,
                )

        for name in ("bpm", "lyrics", "genre"):
            invalid_modalities = {
                key: value.astype(np.float64) for key, value in self.modalities.items()
            }
            invalid_modalities[name][0, 0] = oversized
            with self.subTest(input=name):
                with self.assertRaisesRegex(
                    ValueError, "remain finite after float32 conversion"
                ):
                    train_audio_seed(
                        self.audio,
                        invalid_modalities,
                        self.song_ids,
                        self.train_pools,
                        self.validation_pools,
                        seed=684,
                        epochs=1,
                    )

if __name__ == "__main__":
    unittest.main()

import inspect
import unittest
from copy import deepcopy

import numpy as np
import torch

from src.pairs import build_pair_pools
from src.model import SiameseEncoder
from src.pairs import sample_pairs
from src.train import train_seed


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.train_ids = [f"train-{index}" for index in range(6)]
        self.validation_ids = [f"validation-{index}" for index in range(6)]
        self.song_ids = self.train_ids + self.validation_ids
        self.modalities = {
            "bpm": np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(-1, 1),
            "lyrics": np.random.default_rng(10).normal(size=(12, 5000)).astype(np.float32),
            "genre": np.eye(2, dtype=np.float32)[np.arange(12) % 2],
        }
        genres = {song_id: "rock" for song_id in self.song_ids}
        self.train_pools = build_pair_pools(
            [{"source": self.train_ids[0], "target": self.train_ids[1]},
             {"source": self.train_ids[2], "target": self.train_ids[3]}],
            self.train_ids,
            genres,
        )
        self.validation_pools = build_pair_pools(
            [{"source": self.validation_ids[0], "target": self.validation_ids[1]},
             {"source": self.validation_ids[2], "target": self.validation_ids[3]}],
            self.validation_ids,
            genres,
        )

    def test_fixed_epochs_finite_losses_and_parameter_update(self):
        torch.manual_seed(684)
        initial_parameters = [
            parameter.detach().clone() for parameter in SiameseEncoder(num_genres=2).parameters()
        ]
        model, history, _ = train_seed(
            self.modalities, self.song_ids, self.train_pools, self.validation_pools,
            seed=684, epochs=3,
        )

        self.assertEqual(len(history), 3)
        self.assertEqual([row["epoch"] for row in history], [1, 2, 3])
        self.assertTrue(all(set(row) == {"epoch", "train_loss", "validation_loss"} for row in history))
        self.assertTrue(all(np.isfinite(row["train_loss"]) and np.isfinite(row["validation_loss"]) for row in history))
        self.assertTrue(
            any(not torch.equal(initial, trained) for initial, trained in zip(initial_parameters, model.parameters()))
        )

    def test_same_seed_reproduces_model_and_history(self):
        first_model, first_history, first_trace = train_seed(
            self.modalities, self.song_ids, self.train_pools, self.validation_pools,
            seed=31, epochs=2,
        )
        second_model, second_history, second_trace = train_seed(
            self.modalities, self.song_ids, self.train_pools, self.validation_pools,
            seed=31, epochs=2,
        )

        self.assertEqual(first_history, second_history)
        self.assertEqual(first_trace, second_trace)
        for first, second in zip(first_model.parameters(), second_model.parameters()):
            torch.testing.assert_close(first, second)

    def test_returns_exact_sampled_training_and_fixed_validation_pair_trace(self):
        _, history, pair_trace = train_seed(
            self.modalities, self.song_ids, self.train_pools, self.validation_pools,
            seed=31, epochs=3,
        )

        expected_train = [
            sample_pairs(self.train_pools, seed=31 + epoch, negatives_per_positive=1,
                         same_genre_fraction=0.5)
            for epoch in range(3)
        ]
        expected_validation = sample_pairs(
            self.validation_pools, seed=31 + 1_000_000, negatives_per_positive=1,
            same_genre_fraction=0.5,
        )
        self.assertEqual(pair_trace, {
            "train_by_epoch": expected_train,
            "validation": expected_validation,
        })
        self.assertEqual(len(pair_trace["train_by_epoch"]), len(history))

    def test_api_has_no_test_data_argument(self):
        signature = inspect.signature(train_seed)
        self.assertEqual(
            list(signature.parameters),
            [
                "modalities", "song_ids", "train_pools", "validation_pools", "seed",
                "epochs", "batch_size", "learning_rate", "margin",
            ],
        )
        self.assertEqual(
            [signature.parameters[name].default for name in ("epochs", "batch_size", "learning_rate", "margin")],
            [50, 64, 0.001, 1.0],
        )

    def test_rejects_train_and_validation_pools_without_positives(self):
        for partition in ("train", "validation"):
            with self.subTest(partition=partition):
                train_pools = deepcopy(self.train_pools)
                validation_pools = deepcopy(self.validation_pools)
                pools = train_pools if partition == "train" else validation_pools
                pools["positives"] = []
                with self.assertRaisesRegex(ValueError, partition):
                    train_seed(
                        self.modalities, self.song_ids, train_pools, validation_pools,
                        seed=684, epochs=1,
                    )

    def test_rejects_nonpositive_or_noninteger_epoch_and_batch_values(self):
        for argument in ("epochs", "batch_size"):
            for value in (0, -1, 1.5, True):
                with self.subTest(argument=argument, value=value):
                    with self.assertRaisesRegex(ValueError, argument):
                        train_seed(
                            self.modalities, self.song_ids,
                            self.train_pools, self.validation_pools,
                            seed=684, **{argument: value},
                        )

    def test_rejects_invalid_modality_keys_shapes_and_row_counts(self):
        cases = []
        for name in ("bpm", "lyrics", "genre"):
            missing = dict(self.modalities)
            del missing[name]
            cases.append((missing, name))
        for name, values in (
            ("bpm", np.zeros(12, dtype=np.float32)),
            ("lyrics", np.zeros((12, 5000, 1), dtype=np.float32)),
            ("genre", np.zeros((12, 0), dtype=np.float32)),
            ("bpm", np.zeros((12, 2), dtype=np.float32)),
            ("lyrics", np.zeros((12, 4999), dtype=np.float32)),
            ("genre", np.zeros((11, 2), dtype=np.float32)),
        ):
            invalid = dict(self.modalities)
            invalid[name] = values
            cases.append((invalid, name))
        for modalities, expected_name in cases:
            shape = modalities.get(expected_name)
            with self.subTest(modality=expected_name, shape=getattr(shape, "shape", None)):
                with self.assertRaisesRegex(ValueError, expected_name):
                    train_seed(
                        modalities, self.song_ids, self.train_pools, self.validation_pools,
                        seed=684, epochs=1,
                    )

    def test_rejects_nonfinite_values_in_each_modality(self):
        for name, values in self.modalities.items():
            for invalid_value in (np.nan, np.inf, -np.inf):
                with self.subTest(modality=name, value=invalid_value):
                    invalid_modalities = dict(self.modalities)
                    invalid = values.copy()
                    invalid[0, 0] = invalid_value
                    invalid_modalities[name] = invalid
                    with self.assertRaisesRegex(ValueError, rf"{name}.*finite|finite.*{name}"):
                        train_seed(
                            invalid_modalities, self.song_ids,
                            self.train_pools, self.validation_pools,
                            seed=684, epochs=1,
                        )

    def test_rejects_duplicate_song_ids_and_partition_overlap(self):
        duplicate_ids = self.song_ids[:-1] + [self.song_ids[0]]
        with self.assertRaisesRegex(ValueError, "song_ids"):
            train_seed(
                self.modalities, duplicate_ids, self.train_pools, self.validation_pools,
                seed=684, epochs=1,
            )

        overlapping_validation = deepcopy(self.validation_pools)
        overlapping_validation["partition_ids"].append(self.train_ids[0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            train_seed(
                self.modalities, self.song_ids, self.train_pools, overlapping_validation,
                seed=684, epochs=1,
            )


if __name__ == "__main__":
    unittest.main()

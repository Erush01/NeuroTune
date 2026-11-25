import inspect
import unittest

import numpy as np
import torch

from src.model import AudioOnlySiameseEncoder
from src.pairs import build_pair_pools
from src.train_audio_only import train_audio_only_seed


def _fixture(audio_dtype=np.float32):
    song_ids = [f"song-{index}" for index in range(10)]
    audio = np.random.default_rng(7).normal(size=(10, 1024, 28)).astype(audio_dtype)
    train_ids = song_ids[:5]
    validation_ids = song_ids[5:]
    train_edges = [
        {"source": train_ids[0], "target": train_ids[1]},
        {"source": train_ids[2], "target": train_ids[3]},
    ]
    validation_edges = [
        {"source": validation_ids[0], "target": validation_ids[1]},
        {"source": validation_ids[2], "target": validation_ids[3]},
    ]
    return (
        audio,
        song_ids,
        build_pair_pools(train_edges, train_ids, {}),
        build_pair_pools(validation_edges, validation_ids, {}),
    )


def _run(seed=17, epochs=2):
    audio, song_ids, train_pools, validation_pools = _fixture()
    return train_audio_only_seed(
        audio, song_ids, train_pools, validation_pools, seed, epochs=epochs, batch_size=2
    )


class TrainAudioOnlyTests(unittest.TestCase):
    def test_audio_only_training_losses_updates_parameters_and_pair_contract(self):
        torch.manual_seed(17)
        initial = AudioOnlySiameseEncoder()
        initial_parameters = [parameter.detach().clone() for parameter in initial.parameters()]

        model, history, trace = _run()

        self.assertEqual(len(history), 2)
        self.assertTrue(
            all(
                np.isfinite(row["train_loss"]) and np.isfinite(row["validation_loss"])
                for row in history
            )
        )
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(initial_parameters, model.parameters()))
        )
        self.assertEqual(len(trace["train_by_epoch"]), 2)
        for rows in trace["train_by_epoch"] + [trace["validation"]]:
            self.assertEqual(sum(row[2] == 1.0 for row in rows), 2)
            self.assertEqual(sum(row[2] == 0.0 for row in rows), 2)
        known_positives = {
            tuple(pair) for pools in _fixture()[2:] for pair in pools["positives"]
        }
        for rows in trace["train_by_epoch"] + [trace["validation"]]:
            self.assertTrue(
                all(
                    tuple(sorted(row[:2])) not in known_positives
                    for row in rows
                    if row[2] == 0
                )
            )
        self.assertNotEqual(trace["train_by_epoch"][0], trace["train_by_epoch"][1])
        self.assertEqual(trace["validation"], _run()[2]["validation"])

    def test_same_seed_reproduces_model_history_and_trace_and_partitions_are_isolated(self):
        model_a, history_a, trace_a = _run(seed=29)
        model_b, history_b, trace_b = _run(seed=29)
        self.assertEqual(history_a, history_b)
        self.assertEqual(trace_a, trace_b)
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(model_a.parameters(), model_b.parameters()))
        )
        train_ids = set(_fixture()[2]["partition_ids"])
        validation_ids = set(_fixture()[3]["partition_ids"])
        for row in trace_a["train_by_epoch"]:
            self.assertTrue(all(left in train_ids and right in train_ids for left, right, _ in row))
        self.assertTrue(
            all(
                left in validation_ids and right in validation_ids
                for left, right, _ in trace_a["validation"]
            )
        )

    def test_training_without_validation_pool(self):
        audio, song_ids, train_pools, _ = _fixture()
        torch.manual_seed(31)
        initial = AudioOnlySiameseEncoder()
        initial_parameters = [parameter.detach().clone() for parameter in initial.parameters()]

        model_a, history_a, trace_a = train_audio_only_seed(
            audio, song_ids, train_pools, None, 31, epochs=2, batch_size=2
        )
        model_b, history_b, trace_b = train_audio_only_seed(
            audio, song_ids, train_pools, None, 31, epochs=2, batch_size=2
        )

        self.assertEqual(len(history_a), 2)
        self.assertTrue(all(np.isfinite(row["train_loss"]) for row in history_a))
        self.assertTrue(all(row["validation_loss"] is None for row in history_a))
        self.assertEqual(trace_a["validation"], [])
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(initial_parameters, model_a.parameters()))
        )
        self.assertEqual(history_a, history_b)
        self.assertEqual(trace_a, trace_b)
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(model_a.parameters(), model_b.parameters()))
        )

    def test_api_excludes_test_and_genre_inputs(self):
        parameters = inspect.signature(train_audio_only_seed).parameters
        self.assertEqual(
            list(parameters),
            [
                "audio_sequences",
                "song_ids",
                "train_pools",
                "validation_pools",
                "seed",
                "epochs",
                "batch_size",
                "learning_rate",
                "margin",
            ],
        )

    def test_rejects_non_string_song_ids_before_pair_sampling(self):
        audio = np.random.default_rng(9).normal(size=(10, 1024, 28)).astype(np.float32)
        song_ids = ["a", "b", 3, 4, 5, "v0", "v1", "v2", "v3", "v4"]
        train_pools = {
            "partition_ids": song_ids[:5],
            "positives": [("a", "b"), (3, 4)],
            "genre_groups": {},
            "random_pair_count": 8,
            "same_genre_pair_count": 0,
        }
        validation_pools = {
            "partition_ids": song_ids[5:],
            "positives": [("v0", "v1"), ("v2", "v3")],
            "genre_groups": {},
            "random_pair_count": 8,
            "same_genre_pair_count": 0,
        }
        with self.assertRaisesRegex(ValueError, "song_ids must contain only strings"):
            train_audio_only_seed(audio, song_ids, train_pools, validation_pools, 1, epochs=1)

    def test_rejects_invalid_audio(self):
        invalid_audio = [
            np.zeros((10, 1024, 27), dtype=np.float32),
            np.full((10, 1024, 28), np.nan, dtype=np.float32),
            np.full((10, 1024, 28), np.inf, dtype=np.float32),
            np.full((10, 1024, 28), 1e100, dtype=np.float64),
        ]
        _, song_ids, train_pools, validation_pools = _fixture()
        for bad_audio in invalid_audio:
            with self.subTest(shape=bad_audio.shape, dtype=bad_audio.dtype):
                with self.assertRaises(ValueError):
                    train_audio_only_seed(
                        bad_audio, song_ids, train_pools, validation_pools, 1, epochs=1
                    )

    def test_rejects_invalid_training_dimensions(self):
        cases = [("epochs", 0), ("epochs", 1.5), ("batch_size", -1), ("batch_size", True)]
        audio, song_ids, train_pools, validation_pools = _fixture()
        for parameter, value in cases:
            with self.subTest(parameter=parameter, value=value):
                with self.assertRaisesRegex(ValueError, parameter):
                    train_audio_only_seed(
                        audio,
                        song_ids,
                        train_pools,
                        validation_pools,
                        1,
                        **{parameter: value},
                    )

    def test_rejects_malformed_pools_and_ids(self):
        def duplicated_song_id(audio, ids, train, val):
            return audio, ids + [ids[0]], train, val

        def positive_outside_partition(audio, ids, train, val):
            train = {**train, "positives": train["positives"] + [(ids[5], ids[6])]}
            return audio, ids, train, val

        def missing_pool_key(audio, ids, train, val):
            return audio, ids, {"partition_ids": train["partition_ids"]}, val

        def overlapping_partitions(audio, ids, train, val):
            return audio, ids, train, {**val, "partition_ids": ids[4:]}

        def no_positives(audio, ids, train, val):
            return audio, ids, {**train, "positives": []}, val

        cases = [
            (duplicated_song_id, "unique"),
            (positive_outside_partition, "outside"),
            (missing_pool_key, "malformed"),
            (overlapping_partitions, "overlap"),
            (no_positives, "positive"),
        ]
        for mutate, message in cases:
            with self.subTest(case=message):
                args = mutate(*_fixture())
                with self.assertRaisesRegex(ValueError, message):
                    train_audio_only_seed(*args, seed=1, epochs=1)


if __name__ == "__main__":
    unittest.main()

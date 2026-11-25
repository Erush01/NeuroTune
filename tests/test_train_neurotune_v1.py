import inspect
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset
from src.model_v1 import NeuroTuneV1Encoder
from src.train_neurotune_v1 import train_neurotune_v1


def make_dataset(prefix, count=5, genre_width=3, modalities=("bpm", "lyrics", "genre")):
    ids = [f"{prefix}{index}" for index in range(count)]
    features = {
        "audio": [np.full((index + 1, 26), index / 10, dtype=np.float32) for index in range(count)],
    }
    if "bpm" in modalities:
        features["bpm"] = np.arange(count, dtype=np.float32).reshape(-1, 1)
    if "lyrics" in modalities:
        features["lyrics"] = np.zeros((count, 5000), dtype=np.float32)
    if "genre" in modalities:
        features["genre"] = np.eye(genre_width, dtype=np.float32)[np.arange(count) % genre_width]
    return NeuroTuneV1Dataset(ids, features, modalities=modalities)


def schedules(count=5, epochs=2):
    rows = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 4], [3, 2, 0]], dtype=np.int32)
    return np.tile(rows[None, :, :], (epochs, 1, 1)), rows[None, :, :].copy()


class TrainNeuroTuneV1Tests(unittest.TestCase):
    def setUp(self):
        self.train = make_dataset("tr")
        self.validation = make_dataset("va")
        self.train_schedule, self.validation_schedule = schedules()
        self.edges = [{"source": "va0", "target": "va1"}, {"source": "va2", "target": "va3"}]

    def run_training(self, path, **kwargs):
        options = dict(
            train_dataset=self.train,
            validation_dataset=self.validation,
            train_schedule=self.train_schedule,
            validation_schedule=self.validation_schedule,
            validation_positive_edges=self.edges,
            variant="A",
            seed=684,
            checkpoint_path=path,
            transform_manifest={"fit_ids_hash": "fit-hash"},
            provenance={"split_hash": "split-hash", "source_hash": "source-hash"},
            epochs=2,
            batch_size=2,
        )
        options.update(kwargs)
        return train_neurotune_v1(**options)

    def test_repeatable_updates_and_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = self.run_training(Path(temporary) / "first.pt")
            second = self.run_training(Path(temporary) / "second.pt")
        for key in ("train_triplet_loss", "validation_triplet_loss", "Recall@10", "NDCG@10", "MRR@10"):
            self.assertIn(key, first["history"][0])
            self.assertEqual(first["history"], second["history"])
        self.assertNotIn("MRR", first["history"][0])
        self.assertEqual(first["model"].state_dict().keys(), second["model"].state_dict().keys())
        for key, value in first["model"].state_dict().items():
            self.assertTrue(torch.equal(value, second["model"].state_dict()[key]))
        self.assertEqual(first["history"][0]["query_count"], 4)
        self.assertEqual(first["history"][0]["eligible_positive_edges"], 2)
        self.assertEqual(
            set(first["history"][0]),
            {"epoch", "train_triplet_loss", "validation_triplet_loss", "Recall@10",
             "NDCG@10", "MRR@10", "query_count", "eligible_positive_edges"},
        )
        self.assertNotIn("epoch_seconds", first["history"][0])

    def test_epoch_timing_is_opt_in_finite_and_positive(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_training(Path(temporary) / "timed.pt", record_epoch_timing=True)
        self.assertEqual(len(result["history"]), 2)
        for row in result["history"]:
            self.assertIn("epoch_seconds", row)
            self.assertTrue(np.isfinite(row["epoch_seconds"]))
            self.assertGreater(row["epoch_seconds"], 0)

    def test_schedule_provider_is_called_once_per_epoch_in_eval_no_grad_and_reports_telemetry(self):
        calls = []

        def provider(model, dataset, epoch_number):
            calls.append((model, dataset, epoch_number, model.training, torch.is_grad_enabled()))
            return SimpleNamespace(
                triplets=self.train_schedule[0].copy(),
                attempted_semi_hard_count=2,
                realized_uniform_fraction=0.75,
                realized_semi_hard_fraction=0.25,
                fallback_count=1,
                eligible_uniform_counts=np.array([2, 2, 2, 2], dtype=np.int32),
                eligible_semi_hard_counts=np.array([1, 0, 2, 0], dtype=np.int32),
            )

        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_training(
                Path(temporary) / "provider.pt", train_schedule=None,
                train_schedule_provider=provider,
            )
        self.assertEqual([call[2] for call in calls], [1, 2])
        self.assertTrue(all(call[1] is self.train for call in calls))
        self.assertTrue(all(call[3:] == (False, False) for call in calls))
        for record in result["history"]:
            self.assertTrue(0 <= record["active_triplet_fraction"] <= 1)
            self.assertGreaterEqual(record["mean_positive_cosine_distance"], 0)
            self.assertGreaterEqual(record["mean_negative_cosine_distance"], 0)
            self.assertTrue(0 <= record["margin_violation_fraction"] <= 1)
            self.assertEqual(record["sampler_attempted_semi_hard_count"], 2)
            self.assertEqual(record["sampler_realized_uniform_fraction"], 0.75)
            self.assertEqual(record["sampler_realized_semi_hard_fraction"], 0.25)
            self.assertEqual(record["sampler_fallback_count"], 1)
            self.assertEqual(record["sampler_mean_eligible_uniform_pool"], 2.0)
            self.assertEqual(record["sampler_mean_eligible_semi_hard_pool"], 0.75)

    def test_provider_schedule_must_be_nonempty_int32_and_in_bounds(self):
        invalid_schedules = (
            np.empty((0, 3), dtype=np.int32),
            np.zeros((2, 2), dtype=np.int32),
            np.zeros((1, 3), dtype=np.int64),
            np.array([[0, 1, len(self.train)]], dtype=np.int32),
        )
        for schedule in invalid_schedules:
            with self.subTest(schedule=schedule):
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaises(ValueError):
                        self.run_training(
                            Path(temporary) / "invalid.pt", train_schedule=None,
                            train_schedule_provider=lambda model, dataset, epoch: SimpleNamespace(
                                triplets=schedule,
                            ),
                        )

    def test_provider_exception_restores_training_mode(self):
        model_seen = []

        def provider(model, dataset, epoch_number):
            model_seen.append(model)
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            raise RuntimeError("provider failure")

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "provider failure"):
                self.run_training(
                    Path(temporary) / "provider-error.pt", train_schedule=None,
                    train_schedule_provider=provider,
                )
        self.assertTrue(model_seen[0].training)

    def test_model_is_updated_and_validation_metrics_are_no_grad(self):
        torch.manual_seed(684)
        initial = NeuroTuneV1Encoder("A", 3).state_dict()
        original_evaluator = __import__("src.train_neurotune_v1", fromlist=["evaluate_rankings"]).evaluate_rankings
        grad_states = []

        def evaluate(*args, **kwargs):
            grad_states.append(torch.is_grad_enabled())
            return original_evaluator(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch("src.train_neurotune_v1.evaluate_rankings", side_effect=evaluate):
            result = self.run_training(Path(temporary) / "model.pt")
        self.assertTrue(all(not enabled for enabled in grad_states))
        self.assertTrue(any(not torch.equal(result["model"].state_dict()[key], initial[key]) for key in initial))

    def test_selection_by_metric_then_loss_and_earliest_exact_tie(self):
        metric_values = iter([0.2, 0.8])
        loss_values = iter([0.5, 0.7])
        original_loss = __import__("src.train_neurotune_v1", fromlist=["triplet_margin_cosine_loss"]).triplet_margin_cosine_loss
        loss_call = [0]
        current_validation_offset = [0.0]

        def loss(*args, **kwargs):
            call = loss_call[0] % 4
            loss_call[0] += 1
            if call == 2:
                current_validation_offset[0] = next(loss_values)
            if call in (2, 3):
                return torch.tensor(current_validation_offset[0])
            return original_loss(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, \
             patch("src.train_neurotune_v1.evaluate_rankings") as evaluator, \
             patch("src.train_neurotune_v1.triplet_margin_cosine_loss", side_effect=loss):
            evaluator.side_effect = lambda *args, **kwargs: {
                "Recall@10": 0.0, "NDCG@10": next(metric_values), "MRR@10": 0.0,
                "query_count": 4, "eligible_positive_edges": 2,
            }
            result = self.run_training(Path(temporary) / "selected.pt")
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["best_metrics"]["NDCG@10"], 0.8)

        for ndcgs, losses, expected in [([0.5, 0.5], [0.4, 0.3], 2), ([0.5, 0.5], [0.3, 0.3], 1)]:
            metric_values = iter(ndcgs)
            loss_values = iter(losses)
            loss_call[0] = 0
            with tempfile.TemporaryDirectory() as temporary, \
                 patch("src.train_neurotune_v1.evaluate_rankings") as evaluator, \
                 patch("src.train_neurotune_v1.triplet_margin_cosine_loss", side_effect=loss):
                evaluator.side_effect = lambda *args, **kwargs: {
                    "Recall@10": 0.0, "NDCG@10": next(metric_values), "MRR@10": 0.0,
                    "query_count": 4, "eligible_positive_edges": 2,
                }
                result = self.run_training(Path(temporary) / "selected.pt")
            self.assertEqual(result["best_epoch"], expected)

    def test_checkpoint_contents_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            result = self.run_training(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["variant"], "A")
            self.assertEqual(saved["seed"], 684)
            self.assertEqual(saved["epoch"], result["best_epoch"])
            self.assertEqual(saved["config"]["epochs"], 2)
            self.assertEqual(saved["transform_manifest"], {"fit_ids_hash": "fit-hash"})
            self.assertEqual(saved["provenance"]["split_hash"], "split-hash")
            with self.assertRaises(FileExistsError):
                self.run_training(path)

    def test_one_shot_validation_edges_are_used_for_validation_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_training(
                Path(temporary) / "generator.pt",
                validation_positive_edges=iter(self.edges),
            )
        self.assertEqual(result["history"][0]["query_count"], 4)
        self.assertEqual(result["history"][0]["eligible_positive_edges"], 2)

    def test_rejects_invalid_splits_schedules_and_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unused.pt"
            with self.assertRaises(ValueError):
                self.run_training(path, validation_dataset=self.train)
            bad = self.train_schedule.astype(np.int64)
            with self.assertRaises(ValueError):
                self.run_training(path, train_schedule=bad)
            bad = self.train_schedule.copy()
            bad[0, 0, 0] = len(self.train)
            with self.assertRaises(ValueError):
                self.run_training(path, train_schedule=bad)
            with self.assertRaises(ValueError):
                self.run_training(path, validation_positive_edges=[{"source": "tr0", "target": "va1"}])
            with self.assertRaises(ValueError):
                self.run_training(path, validation_schedule=np.zeros((1, 0, 3), dtype=np.int32))

    def test_api_has_no_test_inputs(self):
        parameters = set(inspect.signature(train_neurotune_v1).parameters)
        self.assertFalse(any("test" in name.lower() or "relevance" in name.lower() for name in parameters))

    def test_all_variants_train_on_full_collator_dictionaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            for variant in "ABCD":
                result = self.run_training(
                    Path(temporary) / f"{variant}.pt",
                    variant=variant,
                    train_schedule=self.train_schedule[:1],
                    epochs=1,
                    batch_size=4,
                )
                self.assertEqual(result["model"].variant, variant)

    def test_d4_trains_and_validates_real_multi_window_batches(self):
        def d4_dataset(prefix):
            ids = [f"{prefix}{index}" for index in range(5)]
            windows = {}
            valid = np.ones((5, 3), dtype=np.bool_)
            for index, track_id in enumerate(ids):
                windows[track_id] = [
                    np.full((index + window + 1, 26), (index + window) / 10, dtype=np.float32)
                    if not (window == 1 and index % 2 == 0) else np.empty((0, 26), dtype=np.float32)
                    for window in range(3)
                ]
                if index % 2 == 0:
                    valid[index, 1] = False
            return NeuroTuneV1Dataset(ids, {
                "audio_windows": windows,
                "window_valid": valid,
                "bpm": np.arange(5, dtype=np.float32).reshape(-1, 1),
                "genre": np.eye(3, dtype=np.float32)[np.arange(5) % 3],
                "lyrics": np.zeros((5, 5000), dtype=np.float32),
            })

        train, validation = d4_dataset("d4tr"), d4_dataset("d4va")
        train_schedule, validation_schedule = schedules()
        with tempfile.TemporaryDirectory() as temporary:
            result = train_neurotune_v1(
                train, validation, train_schedule[:1], validation_schedule,
                [{"source": "d4va0", "target": "d4va1"}], "D", 684,
                Path(temporary) / "d4.pt", {"fit_ids_hash": "d4"}, {"split_hash": "d4"},
                epochs=1, batch_size=2,
            )
        self.assertTrue(all(np.isfinite(record[key]) for record in result["history"] for key in (
            "train_triplet_loss", "validation_triplet_loss", "Recall@10", "NDCG@10", "MRR@10")))
        self.assertEqual(result["model"].variant, "D")

    def test_a_and_b_train_with_only_active_modalities(self):
        with tempfile.TemporaryDirectory() as temporary:
            for variant, modalities in (("A", ()), ("B", ("bpm",))):
                self.train = make_dataset("tr", modalities=modalities)
                self.validation = make_dataset("va", modalities=modalities)
                expected_features = {"audio", *modalities}
                self.assertEqual(set(self.train.features), expected_features)
                self.assertEqual(set(self.validation.features), expected_features)
                result = self.run_training(
                    Path(temporary) / f"{variant}.pt",
                    variant=variant,
                    train_schedule=self.train_schedule[:1],
                    epochs=1,
                    batch_size=4,
                )
                self.assertEqual(result["model"].variant, variant)

    def test_rejects_missing_active_modalities_in_train_or_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unused.pt"
            for variant, modalities, missing_dataset in (
                ("B", ("bpm",), "train"),
                ("B", ("bpm",), "validation"),
            ):
                self.train = make_dataset("tr", modalities=modalities)
                self.validation = make_dataset("va", modalities=modalities)
                if missing_dataset == "train":
                    self.train = make_dataset("tr", modalities=())
                else:
                    self.validation = make_dataset("va", modalities=())
                with self.subTest(variant=variant, missing_dataset=missing_dataset):
                    with self.assertRaises(ValueError):
                        self.run_training(
                            path,
                            variant=variant,
                            train_schedule=self.train_schedule[:1],
                            epochs=1,
                            batch_size=4,
                        )

    def test_modality_dropout_masks_are_deterministic_and_training_only(self):
        original = NeuroTuneV1Encoder.forward_triplets
        large_schedule = np.tile(self.train_schedule[:1], (1, 100, 1))
        options = dict(
            variant="D", train_schedule=large_schedule, epochs=1,
            batch_size=400, bpm_dropout_probability=0.2,
            genre_dropout_probability=0.3, lyrics_dropout_probability=0.4,
            dropout_seed=12345,
        )

        def run_with_seed(path, seed):
            captured = []

            def capture(model, anchor, positive, negative, **kwargs):
                snapshot = {
                    key: {name: mask.clone() for name, mask in value.items()}
                    if value is not None else None
                    for key, value in kwargs.items()
                }
                captured.append((model.training, snapshot))
                return original(model, anchor, positive, negative, **kwargs)

            with patch.object(NeuroTuneV1Encoder, "forward_triplets", capture):
                self.run_training(Path(path), **{**options, "dropout_seed": seed})
            return captured

        with tempfile.TemporaryDirectory() as temporary:
            first = run_with_seed(Path(temporary) / "first.pt", 12345)
            same = run_with_seed(Path(temporary) / "same.pt", 12345)
            different = run_with_seed(Path(temporary) / "different.pt", 12346)
        self.assertEqual(len(first), 2)
        self.assertTrue(first[0][0])
        self.assertFalse(first[1][0])
        mask_keys = ("anchor_keep_masks", "positive_keep_masks", "negative_keep_masks")
        modalities = ("bpm", "genre", "lyrics")
        probabilities = {"bpm": 0.2, "genre": 0.3, "lyrics": 0.4}
        expected_generator = torch.Generator(device="cpu").manual_seed(options["dropout_seed"])
        for key in mask_keys:
            for modality in modalities:
                expected_mask = torch.rand(400, generator=expected_generator) >= probabilities[modality]
                self.assertTrue(torch.equal(first[0][1][key][modality], expected_mask))
                self.assertTrue(torch.equal(first[0][1][key][modality], same[0][1][key][modality]))
                self.assertIsNone(first[1][1].get(key))
        self.assertFalse(torch.equal(first[0][1]["anchor_keep_masks"]["bpm"],
                                     different[0][1]["anchor_keep_masks"]["bpm"]))

    def test_rejects_dropout_for_inactive_modality(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "bpm.*inactive|inactive.*bpm"):
                self.run_training(
                    Path(temporary) / "inactive-dropout.pt", variant="A",
                    bpm_dropout_probability=0.1, dropout_seed=684,
                )

    def test_rejects_dropout_seed_that_cannot_cover_all_epochs(self):
        maximum_torch_seed = 2**63 - 1
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, r"dropout_seed.*2\^63 - 1"):
                self.run_training(
                    Path(temporary) / "out-of-range-dropout-seed.pt",
                    train_schedule=self.train_schedule, epochs=2,
                    bpm_dropout_probability=0.1, dropout_seed=maximum_torch_seed,
                )

    def test_modality_dropout_records_counts_rates_and_validation_has_no_masks(self):
        calls = []
        original = NeuroTuneV1Encoder.forward_triplets

        def capture(model, anchor, positive, negative, **kwargs):
            calls.append((model.training, kwargs))
            return original(model, anchor, positive, negative, **kwargs)

        rows = np.tile(np.array([[0, 1, 2]], dtype=np.int32), (400, 1))
        schedule = rows[None, :, :]
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            NeuroTuneV1Encoder, "forward_triplets", capture
        ):
            result = self.run_training(
                Path(temporary) / "rates.pt", variant="D", train_schedule=schedule,
                epochs=1, batch_size=1200, bpm_dropout_probability=0.1,
                genre_dropout_probability=0.25, lyrics_dropout_probability=0.2,
                dropout_seed=684,
            )
        self.assertEqual(len(calls), 2)  # one training batch and one validation batch
        self.assertTrue(calls[0][0])
        self.assertTrue(all(calls[0][1][key] is not None for key in (
            "anchor_keep_masks", "positive_keep_masks", "negative_keep_masks")))
        self.assertFalse(calls[1][0])
        self.assertTrue(all(calls[1][1].get(key) is None for key in (
            "anchor_keep_masks", "positive_keep_masks", "negative_keep_masks")))
        record = result["history"][0]
        for modality, expected in (("bpm", 0.1), ("genre", 0.25), ("lyrics", 0.2)):
            attempted = record[f"{modality}_dropout_attempted_count"]
            dropped = record[f"{modality}_dropout_dropped_count"]
            rate = record[f"{modality}_dropout_realized_rate"]
            self.assertEqual(attempted, 1200)
            self.assertEqual(dropped, int((~calls[0][1]["anchor_keep_masks"][modality]).sum()
                                          + (~calls[0][1]["positive_keep_masks"][modality]).sum()
                                          + (~calls[0][1]["negative_keep_masks"][modality]).sum()))
            self.assertAlmostEqual(rate, dropped / attempted, places=12)
            self.assertLess(abs(rate - expected), 0.04)

    def test_dropout_disabled_history_and_input_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_training(Path(temporary) / "default.pt")
        self.assertEqual(set(result["history"][0]), {
            "epoch", "train_triplet_loss", "validation_triplet_loss", "Recall@10",
            "NDCG@10", "MRR@10", "query_count", "eligible_positive_edges",
        })
        with tempfile.TemporaryDirectory() as temporary:
            for kwargs in (
                {"bpm_dropout_probability": -0.1},
                {"genre_dropout_probability": 1.1},
                {"lyrics_dropout_probability": float("nan")},
                {"dropout_seed": -1},
                {"dropout_seed": True},
                {"bpm_dropout_probability": 0.1},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.run_training(Path(temporary) / f"bad-{len(kwargs)}-{str(kwargs)}.pt", **kwargs)


if __name__ == "__main__":
    unittest.main()

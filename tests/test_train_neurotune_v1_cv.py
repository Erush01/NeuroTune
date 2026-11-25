import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset
from src.triplets import sample_triplet_schedule
from src.train_neurotune_v1_cv import train_neurotune_v1_cv


class TrainNeuroTuneV1CVTests(unittest.TestCase):
    HASH = "a" * 64

    def setUp(self):
        self.ids = ["a", "b", "c", "d", "e", "f"]
        rng = np.random.default_rng(5)
        features = {
            "audio": [rng.normal(size=(2 + i % 2, 26)).astype(np.float32) for i in range(6)],
            "bpm": rng.normal(size=(6, 1)).astype(np.float32),
            "lyrics": np.zeros((6, 5000), dtype=np.float32),
            "genre": rng.normal(size=(6, 3)).astype(np.float32),
        }
        self.dataset = NeuroTuneV1Dataset(self.ids, features)
        self.train_edges = [("a", "b"), ("c", "d")]
        self.validation_edges = [("a", "c"), ("e", "f")]
        self.known = self.train_edges + self.validation_edges
        self.train_schedule = sample_triplet_schedule(
            self.ids, self.train_edges, 2, 31, known_positive_edges=self.known
        )
        self.validation_schedule = sample_triplet_schedule(
            self.ids, self.validation_edges, 1, 88, known_positive_edges=self.known
        )

    def run_train(self, path, *, train_schedule=None, validation_schedule=None, edges=None):
        return train_neurotune_v1_cv(
            self.dataset,
            self.train_edges if edges is None else edges[0],
            self.validation_edges if edges is None else edges[1],
            self.known,
            self.train_schedule if train_schedule is None else train_schedule,
            self.validation_schedule if validation_schedule is None else validation_schedule,
            31,
            path,
            {"fit_ids": self.ids},
            self.valid_provenance(),
            {"epochs": 2, "batch_size": 4, "fold": 1},
        )

    def valid_provenance(self):
        return {
            "split_sha256": "b" * 64,
            "source_sha256": {"labels": "c" * 64},
            "code_sha256": {"trainer": "d" * 64},
            "cache_sha256": "e" * 64,
        }

    def test_shared_node_pool_and_disjoint_edge_labels(self):
        self.assertEqual(set(self.dataset.track_ids), set(self.ids))
        self.assertEqual(set(self.train_edges) & set(self.validation_edges), set())
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_train(Path(directory) / "model.pt")
        self.assertEqual(len(result["history"]), 2)

    def test_validation_schedule_and_rankings_run_without_grad(self):
        seen = []
        original = torch.is_grad_enabled
        with mock.patch("src.train_neurotune_v1_cv.torch.is_grad_enabled", side_effect=lambda: (seen.append(original()), original())[1]):
            with tempfile.TemporaryDirectory() as directory:
                self.run_train(Path(directory) / "model.pt")
        self.assertIn(False, seen)

    def test_deterministic_updates_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.pt"
            second_path = Path(directory) / "second.pt"
            first = self.run_train(first_path)
            second = self.run_train(second_path)
            first_checkpoint = torch.load(first_path, map_location="cpu", weights_only=False)
            second_checkpoint = torch.load(second_path, map_location="cpu", weights_only=False)
        self.assertEqual(first["history"], second["history"])
        self.assertEqual(first["best_epoch"], second["best_epoch"])
        for key in first_checkpoint["model_state_dict"]:
            torch.testing.assert_close(first_checkpoint["model_state_dict"][key], second_checkpoint["model_state_dict"][key])
        self.assertEqual(first_checkpoint["fold"], 1)
        self.assertEqual(first_checkpoint["variant"], "C")
        self.assertEqual(first_checkpoint["provenance"], self.valid_provenance())
        self.assertEqual(first_checkpoint["artifact_hashes"]["split_sha256"], "b" * 64)
        self.assertEqual(first_checkpoint["artifact_hashes"]["source_sha256"], {"labels": "c" * 64})
        self.assertEqual(first_checkpoint["artifact_hashes"]["code_sha256"], {"trainer": "d" * 64})
        self.assertEqual(first_checkpoint["artifact_hashes"]["cache_sha256"], "e" * 64)

    def test_invalid_provenance_hashes_rejected_before_training_or_checkpoint(self):
        cases = []
        for key in ("split_sha256", "source_sha256", "code_sha256", "cache_sha256"):
            missing = self.valid_provenance()
            del missing[key]
            cases.append(missing)
        for key in ("split_sha256", "cache_sha256"):
            invalid = self.valid_provenance()
            invalid[key] = "not-a-sha256"
            cases.append(invalid)
        for key in ("source_sha256", "code_sha256"):
            for invalid_value in ({}, {"item": "bad"}, {1: self.HASH}):
                invalid = self.valid_provenance()
                invalid[key] = invalid_value
                cases.append(invalid)
        with tempfile.TemporaryDirectory() as directory:
            for number, provenance in enumerate(cases):
                path = Path(directory) / f"invalid-{number}.pt"
                with mock.patch("src.train_neurotune_v1_cv.NeuroTuneV1Encoder") as model:
                    with self.assertRaisesRegex(ValueError, "provenance"):
                        train_neurotune_v1_cv(
                            self.dataset, self.train_edges, self.validation_edges, self.known,
                            self.train_schedule, self.validation_schedule, 31, path,
                            {"fit_ids": self.ids}, provenance,
                            {"epochs": 2, "batch_size": 4},
                        )
                model.assert_not_called()
                self.assertFalse(path.exists())

    def test_checkpoint_publish_race_preserves_competing_file_and_cleans_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race.pt"
            stage_paths = []

            def competing_writer(staged, destination):
                staged = Path(staged)
                stage_paths.append(staged)
                self.assertTrue(staged.exists())
                Path(destination).write_bytes(b"competing checkpoint")
                raise FileExistsError(str(destination))

            with mock.patch("src.train_neurotune_v1_cv.os.link", side_effect=competing_writer):
                with self.assertRaises(FileExistsError):
                    self.run_train(path)
            self.assertEqual(path.read_bytes(), b"competing checkpoint")
            self.assertEqual(len(stage_paths), 1)
            self.assertFalse(stage_paths[0].exists())

    def test_selection_ties_use_validation_loss_then_earlier_epoch(self):
        # Deterministically force equal ranking metrics and equal loss.
        with mock.patch("src.train_neurotune_v1_cv.evaluate_neurotune_v1_retrieval",
                        side_effect=[{"NDCG@10": 0.5}, {"NDCG@10": 0.5}]), \
             mock.patch("src.train_neurotune_v1_cv._validation_loss", side_effect=[0.4, 0.4]):
            with tempfile.TemporaryDirectory() as directory:
                result = self.run_train(Path(directory) / "model.pt")
        self.assertEqual(result["best_epoch"], 1)

    def test_selection_equal_ndcg_prefers_lower_validation_loss(self):
        with mock.patch("src.train_neurotune_v1_cv.evaluate_neurotune_v1_retrieval",
                        side_effect=[{"NDCG@10": 0.5}, {"NDCG@10": 0.5}]), \
             mock.patch("src.train_neurotune_v1_cv._validation_loss", side_effect=[0.4, 0.3]):
            with tempfile.TemporaryDirectory() as directory:
                result = self.run_train(Path(directory) / "model.pt")
        self.assertEqual(result["best_epoch"], 2)

    def test_known_positive_neighbors_are_never_schedule_negatives(self):
        id_to_index = {track_id: index for index, track_id in enumerate(self.ids)}
        positives = {frozenset(edge) for edge in self.known}
        for schedule in (self.train_schedule, self.validation_schedule):
            for anchor, _, negative in schedule[0]:
                self.assertNotIn(frozenset((self.ids[int(anchor)], self.ids[int(negative)])), positives)

    def test_corrupt_edge_partition_or_schedule_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.run_train(Path(directory) / "bad.pt", edges=(self.train_edges, self.train_edges))
            bad_schedule = self.validation_schedule.copy()
            bad_schedule[0, 0, 1] = 5
            with self.assertRaises(ValueError):
                self.run_train(Path(directory) / "bad-schedule.pt", validation_schedule=bad_schedule)
            bad_negative = self.validation_schedule.copy()
            bad_negative[0, 0, 2] = 1
            with self.assertRaises(ValueError):
                self.run_train(Path(directory) / "bad-negative.pt", validation_schedule=bad_negative)

    def test_signature_has_no_test_inputs(self):
        parameters = set(inspect.signature(train_neurotune_v1_cv).parameters)
        self.assertNotIn("test_dataset", parameters)
        self.assertNotIn("test_ids", parameters)
        self.assertNotIn("test_edges", parameters)


if __name__ == "__main__":
    unittest.main()

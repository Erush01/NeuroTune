import csv
import hashlib
import json
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from scripts import run_neurotune_v1_abcd_cv as runner
from src.model_v1 import NeuroTuneV1Encoder


class RunnerContractTests(unittest.TestCase):
    def test_production_constants_are_fixed(self):
        self.assertEqual(runner.SEEDS, (684, 685, 686))
        self.assertEqual((runner.FOLDS, runner.EPOCHS, runner.BATCH_SIZE), ((0, 1, 2), 50, 64))

    def test_test_only_entrypoint_exists(self):
        self.assertTrue(callable(runner._run_synthetic))

    def _fixture(self, root):
        data = root / "data"
        for relative in runner.INPUTS.values():
            path = data / relative
            if relative.endswith("cv"):
                path.mkdir(parents=True, exist_ok=True)
                (path / "manifest.json").write_text("{}")
                (path / "known_positive_edges.csv").write_text("track_id_a,track_id_b\n")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
        folds = []
        for fold_id in runner.FOLDS:
            fold = {"fold": fold_id}
            for part, prefix in (("train", "tr"), ("validation", "va"), ("test", "te")):
                fold[part] = [f"{fold_id}-{prefix}0", f"{fold_id}-{prefix}1"]
                fold[part + "_edges"] = [(fold[part][0], fold[part][1])] if part != "test" else []
            fold["test_train_source_edges"] = [(fold["test"][0], fold["test"][1])]
            fold["test_official_source_edges"] = []
            folds.append(fold)
        loaded = {"folds": folds, "manifest": {"status": "complete", "seed": 684, "track_ids": [x for fold in folds for part in ("train", "validation", "test") for x in fold[part]], "known_positive_edge_count": 11},
                  "cache_ids": [x for fold in folds for part in ("train", "validation", "test") for x in fold[part]],
                  "catalog": {f"id-{i}": {} for i in range(20)},
                  "known_positive_edges": {("knownA", "knownB")}, "ab_test_source_edges": {("oldA", "oldB")}}
        (data / runner.INPUTS["cv_dir"] / "manifest.json").write_text(json.dumps(loaded["manifest"]))
        events = []
        stage_paths = []

        def feature_builder(_loaded, fold):
            events.append(("features", fold["fold"]))
            ids = sorted(fold["train"] + fold["validation"] + fold["test"])
            features = {
                "audio": [np.ones((2, 26), dtype=np.float32) * (i + 1) for i, _ in enumerate(ids)],
                "bpm": np.ones((len(ids), 1), dtype=np.float32),
                "genre": np.eye(2, dtype=np.float32)[np.arange(len(ids)) % 2],
            }
            return features, {"fit_ids": list(fold["train"]), "fold": fold["fold"]}

        def schedules(_loaded, fold, seed):
            return np.zeros((50, 1, 3), dtype=np.int32), np.zeros((1, 1, 3), dtype=np.int32)

        def trainer(train, validation, *_args, **kwargs):
            self.assertEqual(kwargs["epochs"], 50)
            self.assertEqual(kwargs["batch_size"], 64)
            self.assertEqual(set(train.features), {"audio", "bpm", "genre"})
            model = NeuroTuneV1Encoder("C", train.features["genre"].shape[1]).cpu()
            # Positional arguments: variant, seed, checkpoint path, transform manifest.
            checkpoint_path, transform, provenance = _args[5], _args[6], _args[7]
            stage_paths.append(checkpoint_path.parents[1])
            torch.save({"model_state_dict": model.state_dict(), "transform_manifest": transform,
                        "provenance": provenance, "variant": "C", "seed": provenance["seed"]}, checkpoint_path)
            events.append(("trained", checkpoint_path.name))
            history = [{"epoch": epoch, "train_triplet_loss": 0.1, "validation_triplet_loss": 0.2,
                        "Recall@10": 0.5, "NDCG@10": 0.4, "MRR@10": 0.3,
                        "query_count": 1, "eligible_positive_edges": 1} for epoch in range(1, 51)]
            return {"history": history, "best_epoch": 7, "best_metrics": {"NDCG@10": 0.4, "query_count": 1}}

        def evaluator(embeddings, ids, first, second):
            if first and isinstance(first[0], tuple) and first[0][0].startswith("te"):
                stage = stage_paths[-1]
                self.assertEqual(len(list((stage / "checkpoints").glob("*.pt"))), 9)
                self.assertEqual(len(list((stage / "triplet_schedules").glob("*.npz"))), 9)
                with (stage / "training_curves.csv").open() as stream:
                    self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 450)
            events.append(("evaluate", tuple(first), tuple(second)))
            return {"held_out_train_source": {"Recall@10": 0.25, "edge_count": len(first)},
                    "official_test_source": {"Recall@10": 0.75, "edge_count": len(second)}}

        return data, loaded, events, {"loader": lambda **_kwargs: loaded, "feature_builder": feature_builder,
                                      "schedule_builder": schedules, "trainer": trainer, "evaluator": evaluator}

    def test_synthetic_run_barrier_artifacts_and_source_separation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, _loaded, events, deps = self._fixture(root)
            # baseline embeddings are deterministic and deliberately tiny; fake evaluator avoids metrics work.
            deps["baseline_builder"] = lambda _features, ids: np.zeros((len(ids), 2), dtype=np.float32)
            report = runner._run_synthetic(data, root / "results", _dependencies=deps)
            output = root / "results/neurotune_v1_abcd_cv"
            self.assertEqual(len(report["runs"]), 9)
            self.assertEqual(report["counts"], {"model_observations": 9, "fold_level_baseline_observations": 3,
                "training_curve_rows": 450, "catalog_candidate_tracks": 20, "cache_valid_tracks": 18,
                "rejected_tracks": 2, "known_positive_edges": 1,
                "legacy_ab_test_source_exclusions": 1})
            self.assertEqual(len(list((output / "checkpoints").glob("*.pt"))), 9)
            self.assertEqual(len(list((output / "triplet_schedules").glob("*.npz"))), 9)
            self.assertEqual({p.name for p in (output / "checkpoints").glob("*.pt")},
                             {f"fold_{fold}_C_seed_{seed}.pt" for fold in runner.FOLDS for seed in runner.SEEDS})
            self.assertEqual({p.name for p in (output / "triplet_schedules").glob("*.npz")},
                             {f"fold_{fold}_seed_{seed}.npz" for fold in runner.FOLDS for seed in runner.SEEDS})
            self.assertEqual(report["protocol"], {"variant": "C", "modalities": ["audio", "bpm", "genre"],
                "folds": [0, 1, 2], "seeds": [684, 685, 686], "epochs": 50, "batch_size": 64,
                "margin": 0.2, "optimizer": "Adam", "learning_rate": 0.001, "device": "cpu", "split_seed": 684})
            expected_manifest_sha = hashlib.sha256((data / runner.INPUTS["cv_dir"] / "manifest.json").read_bytes()).hexdigest()
            self.assertTrue(all(r["provenance"]["split_manifest_sha256"] == expected_manifest_sha for r in report["runs"]))
            self.assertTrue(all(r["provenance"]["data_source_identities"]["split_seed"] == 684 for r in report["runs"]))
            with (output / "training_curves.csv").open() as stream:
                self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 450)
            self.assertTrue((output / "experiment.json").is_file())
            self.assertTrue(all("held_out_train_source" in run["test"] and "official_test_source" in run["test"] for run in report["runs"]))
            self.assertEqual(sum(event[0] == "evaluate" for event in events), 12)
            self.assertEqual(report["folds"][0]["track_counts"], {"train": 2, "validation": 2, "test": 2})
            self.assertEqual(report["folds"][0]["source_edge_counts"], {
                "train_source_train": 1, "train_source_validation": 1,
                "train_source_test": 1, "official_test_source_test": 0})
            for run in report["runs"]:
                checkpoint = output / run["checkpoint_path"]
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), run["checkpoint_sha256"])
                schedule = output / run["schedule_path"]
                self.assertEqual(hashlib.sha256(schedule.read_bytes()).hexdigest(), run["schedule_sha256"])
                with np.load(schedule) as archive:
                    self.assertEqual(set(archive.files), {"train_ids", "validation_ids", "train_schedule", "validation_schedule"})
            experiment = json.loads((output / "experiment.json").read_text())
            self.assertEqual(experiment["hashes"]["checkpoints"], {r["checkpoint_path"]: r["checkpoint_sha256"] for r in report["runs"]})
            last_train = max(j for j, e in enumerate(events) if e[0] == "trained")
            test_feature_positions = [i for i, event in enumerate(events) if event[0] == "features" and i > last_train]
            self.assertEqual(len(test_feature_positions), 3)
            allowed = {"Recall@10", "HitRate@10", "MRR@10", "NDCG@10"}
            self.assertTrue(all(set(metrics) <= allowed for metrics in report["aggregate_model_test"].values()))
            self.assertTrue(all(set(metrics) <= allowed for metrics in report["aggregate_baseline_test"].values()))
            for record in report["runs"]:
                self.assertEqual(record["provenance"]["fold"], record["fold"])
                self.assertEqual(record["provenance"]["seed"], record["seed"])
                self.assertEqual(record["provenance"]["variant"], "C")
                self.assertEqual(record["provenance"]["transform_sha256"], record["transform_sha256"])
                self.assertEqual(record["provenance"]["schedule_sha256"], record["schedule_sha256"])
                self.assertEqual(set(record["provenance"]["code_sha256"]), set(runner.CODE_PATHS))
                self.assertIn("split_manifest_sha256", record["provenance"])
                self.assertEqual(record["test_source_counts"]["held_out_train_source_edges"], 1)
                checkpoint = torch.load(output / record["checkpoint_path"], weights_only=False)
                self.assertEqual(checkpoint["provenance"], record["provenance"])
                self.assertEqual(checkpoint["variant"], "C")
                self.assertEqual(checkpoint["seed"], record["seed"])

    def test_input_mutation_during_training_aborts_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, _, _, deps = self._fixture(root)
            original = deps["trainer"]
            mutated = False
            def mutate(*args, **kwargs):
                nonlocal mutated
                if not mutated:
                    (data / runner.INPUTS["catalog_path"]).write_text("mutated during training")
                    mutated = True
                return original(*args, **kwargs)
            deps["trainer"] = mutate
            with self.assertRaisesRegex(RuntimeError, "changed during run"):
                runner._run_synthetic(data, root / "results", _dependencies=deps)
            self.assertFalse((root / "results/neurotune_v1_abcd_cv").exists())

    def test_manifest_byte_mutation_with_same_json_aborts_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, _, _, deps = self._fixture(root)
            manifest_path = data / runner.INPUTS["cv_dir"] / "manifest.json"
            manifest_value = json.loads(manifest_path.read_text())
            original_trainer = deps["trainer"]
            mutated = False

            def mutate_manifest(*args, **kwargs):
                nonlocal mutated
                if not mutated:
                    manifest_path.write_text(json.dumps(manifest_value, indent=2))
                    self.assertEqual(json.loads(manifest_path.read_text()), manifest_value)
                    mutated = True
                return original_trainer(*args, **kwargs)

            deps["trainer"] = mutate_manifest
            real_snapshot = runner._hash_snapshot
            frozen_snapshot = []
            def snapshot_without_manifest_signal(paths):
                if not frozen_snapshot:
                    frozen_snapshot.append(real_snapshot(paths))
                return frozen_snapshot[0]

            with mock.patch.object(runner, "_hash_snapshot", side_effect=snapshot_without_manifest_signal):
                with self.assertRaisesRegex(RuntimeError, "manifest bytes changed"):
                    runner._run_synthetic(data, root / "results", _dependencies=deps)
            self.assertFalse((root / "results/neurotune_v1_abcd_cv").exists())

    def test_late_target_creation_and_publish_failure_preserve_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, _, _, deps = self._fixture(root)
            original = deps["trainer"]
            created = False
            final = root / "results/neurotune_v1_abcd_cv"
            def create_late(*args, **kwargs):
                nonlocal created
                result = original(*args, **kwargs)
                if not created:
                    final.mkdir(parents=True)
                    (final / "marker").write_text("preserve")
                    created = True
                return result
            deps["trainer"] = create_late
            with self.assertRaises(FileExistsError):
                runner._run_synthetic(data, root / "results", _dependencies=deps)
            self.assertEqual((final / "marker").read_text(), "preserve")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, _, _, deps = self._fixture(root)
            with mock.patch.object(runner.os, "link", side_effect=OSError("injected link failure")):
                with self.assertRaisesRegex(OSError, "injected link failure"):
                    runner._run_synthetic(data, root / "results", _dependencies=deps)
            self.assertFalse((root / "results/neurotune_v1_abcd_cv").exists())
            self.assertFalse(list((root / "results").glob(".neurotune_v1_abcd_cv.stage-*")))

    def test_default_results_path_is_root_anchored(self):
        with mock.patch.object(runner, "_run") as run:
            runner.run_experiment()
            self.assertEqual(run.call_args.kwargs["results_root"], runner.ROOT / "results")

    def test_total_timer_starts_before_input_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, loaded, events, deps = self._fixture(root)
            real_monotonic = time.monotonic
            def clock():
                events.append(("clock",))
                return real_monotonic()
            def loader(**_kwargs):
                events.append(("loader",))
                return loaded
            deps["loader"] = loader
            deps["baseline_builder"] = lambda _features, ids: np.zeros((len(ids), 2), dtype=np.float32)
            with mock.patch.object(runner.time, "monotonic", side_effect=clock):
                runner._run_synthetic(data, root / "results", _dependencies=deps)
            self.assertLess(next(i for i, event in enumerate(events) if event[0] == "clock"),
                            next(i for i, event in enumerate(events) if event[0] == "loader"))

    def test_existing_namespace_refused_and_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, _, _, deps = self._fixture(root)
            final = root / "results/neurotune_v1_abcd_cv"
            final.mkdir(parents=True)
            marker = final / "keep"
            marker.write_text("prior")
            with self.assertRaises(FileExistsError):
                runner._run_synthetic(data, root / "results", _dependencies=deps)
            self.assertEqual(marker.read_text(), "prior")
            marker.unlink(); final.rmdir()
            def bad_trainer(*_args, **_kwargs):
                raise RuntimeError("synthetic failure")
            deps["trainer"] = bad_trainer
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                runner._run_synthetic(data, root / "results", _dependencies=deps)
            self.assertFalse(final.exists())
            self.assertFalse(list((root / "results").glob(".neurotune_v1_abcd_cv.stage-*")))


if __name__ == "__main__":
    unittest.main()

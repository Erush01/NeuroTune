"""Synthetic safety tests for the validation-only full-modal runner."""

import importlib
import inspect
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from src.evaluate import build_test_relevance


class FullModalAblationRunnerTests(unittest.TestCase):
    def test_runner_module_is_available(self):
        runner = importlib.import_module("scripts.run_neurotune_v1_full_modal_ablation")
        self.assertEqual(runner.FOLDS, (0, 1, 2))
        self.assertEqual(runner.SEEDS, (684, 685, 686))
        self.assertEqual(runner.EPOCHS, 50)
        self.assertEqual(runner.BATCH_SIZE, 64)
        self.assertEqual(runner.LEARNING_RATE, 0.001)
        self.assertEqual(runner.MARGIN, 0.2)
        self.assertEqual(runner.MODALITIES["A"], ())
        self.assertEqual(runner.MODALITIES["B"], ("bpm",))
        self.assertEqual(runner.MODALITIES["D"], ("bpm", "lyrics", "genre"))
        self.assertIn("src/data/neurotune_v1_abcd_cv.py", runner.CODE_PATHS)
        self.assertIn("src/evaluate.py", runner.CODE_PATHS)
        self.assertIn("src/retrieve.py", runner.CODE_PATHS)
        self.assertEqual(set(inspect.signature(runner.load_abcd_cv_training).parameters),
                         set(runner.INPUTS) - {"mxm_path"})
        self.assertTrue(callable(runner._run))
        self.assertTrue(callable(runner.build_abcd_cv_development_features))
        self.assertFalse(hasattr(runner, "build_abcd_cv_fold_features"))
        self.assertNotIn("neurotune_v1_abcd_cv/experiment.json", inspect.getsource(runner._run))

    def setUp(self):
        self.runner = importlib.import_module("scripts.run_neurotune_v1_full_modal_ablation")
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.results = self.base / "results"
        for rel in self.runner.INPUTS.values():
            p = self.data / rel
            if rel.endswith("mxm_counts"):
                p.mkdir(parents=True)
                (p / "source.txt").write_text("fixture mxm\n")
            elif rel.endswith("/cv"):
                p.mkdir(parents=True)
                (p / "manifest.json").write_text('{"fixture":true}\n')
                (p / "known_positive_edges.csv").write_text("source,target\nfixture-a,fixture-b\n")
                for fold_id in (0, 1, 2):
                    (p / f"fold_{fold_id}_edges.csv").write_text("source,target\nfixture-a,fixture-b\n")
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("fixture input\n")
        self.croot = self.results / "neurotune_v1_abcd_cv"
        (self.croot / "checkpoints").mkdir(parents=True)
        (self.croot / "triplet_schedules").mkdir()
        # A malformed legacy C report is intentionally present and must never be read.
        (self.croot / "experiment.json").write_text("INVALID SENTINEL: do not parse\n")
        self.folds = []
        for fold_id in self.runner.FOLDS:
            train = [f"f{fold_id}_train{i}" for i in range(3)]
            validation = [f"f{fold_id}_validation{i}" for i in range(3)]
            test = [f"SECRET_TEST_{fold_id}_{i}" for i in range(2)]
            self.folds.append({"fold": fold_id, "train": train, "validation": validation, "test": test,
                               "train_edges": [(train[0], train[1])],
                               "validation_edges": [(validation[0], validation[1])],
                               "test_train_source_edges": [(test[0], test[1])],
                               "test_official_source_edges": [(test[0], test[1])]})
        self.loaded = {"folds": self.folds, "catalog": {}, "known_positive_edges": [], "manifest": {"seed": 1},
                       "cache_path": str(self.data / self.runner.INPUTS["cache_path"])}
        self.inputs_hashes = {k: self.runner._hash_path(self.data / v) for k, v in self.runner.INPUTS.items()
                              if k != "mxm_path"}
        self.manifest_hash = self.runner.sha256_file(self.data / self.runner.INPUTS["cv_dir"] / "manifest.json")
        self.transforms = {}
        self.features = {}
        self.curve_metrics = {"Recall@10": 0.25, "HitRate@10": 0.6, "NDCG@10": 0.5, "MRR@10": 0.4,
                              "query_count": 2, "eligible_positive_edges": 1}
        for fold in self.folds:
            ids = sorted(fold["train"] + fold["validation"])
            source_hashes = {"audio_cache": self.runner._hash_path(self.data / self.runner.INPUTS["cache_path"]),
                             "split_manifest": "a" * 64}
            transform = {"fit_ids_hash": self.runner._digest(sorted(fold["train"])),
                         "audio_mean": [0.0] * 26, "audio_std": [1.0] * 26, "audio_fit_rows": 3,
                         "bpm_mean": 100.0, "bpm_std": 10.0, "genre_vocabulary": ["g"],
                         "split_hash": "a" * 64, "source_sha256": source_hashes,
                         "code_sha256": {p: self.runner.sha256_file(self.runner.ROOT / p) for p in
                                         ("src/features_neurotune_v1.py", "src/features.py")}}
            self.transforms[fold["fold"]] = transform
            self.features[fold["fold"]] = {
                "audio": [np.ones((1, 26), dtype=np.float32) * (i + 1) for i in range(6)],
                "bpm": np.ones((6, 1), dtype=np.float32),
                "genre": np.ones((6, 1), dtype=np.float32),
                "lyrics": np.ones((6, 5000), dtype=np.float32),
            }
        for fold in self.folds:
            f = fold["fold"]
            for seed in self.runner.SEEDS:
                schedule_path = self.croot / f"triplet_schedules/fold_{f}_seed_{seed}.npz"
                np.savez_compressed(schedule_path,
                    train_ids=np.asarray(fold["train"], dtype="U"), validation_ids=np.asarray(fold["validation"], dtype="U"),
                    train_schedule=np.zeros((50, 1, 3), dtype=np.int32),
                    validation_schedule=np.zeros((1, 1, 3), dtype=np.int32))
                schedule_hash = self.runner.sha256_file(schedule_path)
                transform = self.transforms[f]
                metrics = dict(self.curve_metrics)
                checkpoint = {"variant": "C", "seed": seed, "epoch": 2, "selection_metrics": metrics,
                    "config": {"optimizer": "Adam", "learning_rate": 0.001, "batch_size": 64,
                               "epochs": 50, "margin": 0.2, "device": "cpu"},
                    "transform_manifest": transform,
                    "provenance": {"fold": f, "seed": seed, "variant": "C",
                        "input_sha256": self.inputs_hashes, "split_manifest_sha256": self.manifest_hash,
                        "transform_sha256": self.runner._digest(transform), "schedule_sha256": schedule_hash}}
                torch.save(checkpoint, self.croot / f"checkpoints/fold_{f}_C_seed_{seed}.pt")
        with (self.croot / "training_curves.csv").open("w", newline="") as stream:
            fields = ("fold", "variant", "seed", "epoch", "train_triplet_loss", "validation_triplet_loss",
                      *(key for key in self.curve_metrics if key != "HitRate@10"))
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for fold in self.folds:
                for seed in self.runner.SEEDS:
                    for epoch in range(1, 51):
                        writer.writerow({"fold": fold["fold"], "variant": "C", "seed": seed, "epoch": epoch,
                            "train_triplet_loss": 1.0, "validation_triplet_loss": 0.8,
                            **{k: v for k, v in self.curve_metrics.items() if k != "HitRate@10"}})
        self.c_artifact_hashes = {p.relative_to(self.croot).as_posix(): self.runner.sha256_file(p)
                                  for p in self.croot.rglob("*") if p.is_file() and p.name != "experiment.json"}
        self.calls = {"features": [], "datasets": [], "baseline": [], "evaluation": [], "training": []}

    def tearDown(self):
        self.tmp.cleanup()

    def dependencies(self):
        runner = self.runner
        def feature_builder(loaded, fold, lyrics):
            self.assertEqual(set(loaded), {"manifest", "catalog", "cache_path"})
            self.assertEqual(set(fold), {"fold", "train", "validation"})
            self.assertNotIn("SECRET_TEST", repr((loaded, fold)))
            self.calls["features"].append((fold["fold"], tuple(fold["train"] + fold["validation"])))
            return self.features[fold["fold"]], self.transforms[fold["fold"]]
        def baseline_builder(features, ids, modalities):
            self.assertFalse(any("SECRET_TEST" in i for i in ids))
            self.assertEqual(set(features), set(modalities))
            self.calls["baseline"].append(tuple(modalities))
            return np.ones((len(ids), 2), dtype=np.float32)
        def evaluator(embeddings, ids, edges):
            self.assertFalse(any("SECRET_TEST" in i for i in ids))
            self.assertFalse(any("SECRET_TEST" in x for edge in edges for x in edge))
            self.calls["evaluation"].append((tuple(ids), tuple(edges)))
            return {"Recall@10": 0.2, "HitRate@10": 0.5, "NDCG@10": 0.3, "MRR@10": 0.25,
                    "query_count": 2, "eligible_positive_edges": 1}
        class RecordingDataset(runner.NeuroTuneV1Dataset):
            def __init__(inner, ids, features, modalities=None):
                self.assertFalse(any("SECRET_TEST" in i for i in ids))
                self.calls["datasets"].append((tuple(modalities), set(features)))
                super().__init__(ids, features, modalities)
        def trainer(train, validation, train_sched, val_sched, edges, variant, seed, path,
                    transform, provenance, *, epochs, batch_size):
            self.assertEqual((epochs, batch_size), (50, 64))
            self.assertIn(variant, ("A", "B", "D"))
            expected = {"A": {"audio"}, "B": {"audio", "bpm"}, "D": {"audio", "bpm", "lyrics", "genre"}}[variant]
            self.assertEqual(set(train.features), expected)
            self.assertEqual(set(validation.features), expected)
            self.assertFalse(any("SECRET_TEST" in x for edge in edges for x in edge.values()))
            relevance = build_test_relevance(edges, validation.track_ids)
            self.assertTrue(relevance)
            self.assertFalse(any("SECRET_TEST" in x for x in relevance))
            self.calls["training"].append((variant, seed, tuple(train.modalities)))
            self.assertEqual(train_sched.dtype, np.int32)
            self.assertEqual(val_sched.dtype, np.int32)
            self.assertEqual(provenance["mxm"]["mapping_sha256"], "c" * 64)
            self.assertIn("known_positive_edges_sha256_exclusion_only", provenance)
            self.assertIn("official_test_edge_source_sha256_exclusion_only", provenance)
            self.assertIn("legacy_test_edge_source_sha256_exclusion_only", provenance)
            self.assertEqual(provenance["known_positive_edges_sha256_exclusion_only"],
                             self.runner.sha256_file(self.data / self.runner.INPUTS["cv_dir"] / "known_positive_edges.csv"))
            self.assertEqual(provenance["official_test_edge_source_sha256_exclusion_only"],
                             self.runner.sha256_file(self.data / self.runner.INPUTS["test_edges_path"]))
            self.assertFalse(provenance["test_source_edges_used_as_targets"])
            torch.save({"fixture": True}, path)
            history = [{"epoch": ep, "train_triplet_loss": 1.0, "validation_triplet_loss": 0.8,
                        **self.curve_metrics} for ep in range(1, 51)]
            return {"history": history, "best_epoch": 2, "best_metrics": dict(self.curve_metrics)}
        def mxm_builder(loaded, path):
            self.assertEqual(set(loaded), {"manifest", "catalog", "cache_path"})
            docs = {track: {} for fold in self.folds for track in fold["train"] + fold["validation"]}
            return (docs,
                    {"source_sha256": {"mxm": "b" * 64}, "directory_sha256": "b" * 64, "mapping_sha256": "c" * 64})
        def loader(*, audit_path, catalog_path, train_edges_path, test_edges_path, cache_path,
                   cache_metadata_path, rejection_path, ab_catalog_path, ab_split_path,
                   ab_test_edges_path, cv_dir):
            received = {k: value for k, value in locals().items() if k != "self"}
            self.assertEqual(set(received), set(self.runner.INPUTS) - {"mxm_path"})
            self.assertTrue(all(value == self.data / self.runner.INPUTS[key] for key, value in received.items()))
            return self.loaded
        return {"loader": loader,
                "mxm_builder": mxm_builder,
                "feature_builder": feature_builder, "baseline_builder": baseline_builder,
                "evaluator": evaluator, "trainer": trainer, "Dataset": RecordingDataset}

    def run_fixture(self, deps=None):
        deps = deps or self.dependencies()
        with mock.patch.object(self.runner, "NeuroTuneV1Dataset", deps.pop("Dataset", self.runner.NeuroTuneV1Dataset)):
            return self.runner._run(self.data, self.results, _test_only=True, _dependencies=deps)

    def test_end_to_end_validation_only_run_and_artifacts(self):
        report = self.run_fixture()
        out = self.results / "neurotune_v1_full_modal_ablation"
        self.assertEqual(len(report["model_validation"]), 36)
        self.assertEqual(len(report["cosine_validation"]), 12)
        self.assertEqual(len(report["hashes"]["schedule_copies"]), 9)
        with (out / "training_curves.csv").open() as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 1800)
        self.assertEqual(len(list((out / "checkpoints").glob("*.pt"))), 27)
        self.assertTrue(report["test_evaluation_deferred"])
        self.assertEqual(report["protocol"]["modalities"]["A"], ())
        self.assertEqual(report["protocol"]["modalities"]["B"], ("bpm",))
        self.assertEqual(report["protocol"]["modalities"]["D"], ("bpm", "lyrics", "genre"))
        self.assertEqual((report["protocol"]["epochs"], report["protocol"]["batch_size"],
                          report["protocol"]["learning_rate"], report["protocol"]["triplet_margin"],
                          report["protocol"]["optimizer"]), (50, 64, 0.001, 0.2, "Adam"))
        after_c = {p.relative_to(self.croot).as_posix(): self.runner.sha256_file(p)
                   for p in self.croot.rglob("*") if p.is_file() and p.name != "experiment.json"}
        self.assertEqual(after_c, self.c_artifact_hashes)
        self.assertEqual(len(self.calls["features"]), 3)
        self.assertEqual(len(self.calls["training"]), 27)
        self.assertTrue(all(keys == {"audio"} for modalities, keys in self.calls["datasets"] if not modalities))
        self.assertTrue(all(keys == {"audio", "bpm"} for modalities, keys in self.calls["datasets"] if modalities == ("bpm",)))
        self.assertTrue(all(keys == {"audio", "bpm", "lyrics", "genre"}
                            for modalities, keys in self.calls["datasets"] if modalities == ("bpm", "lyrics", "genre")))
        self.assertEqual(set(self.calls["baseline"]), {("audio",), ("audio", "bpm"),
            ("audio", "bpm", "genre"), ("audio", "bpm", "lyrics", "genre")})
        self.assertFalse(any("SECRET_TEST" in x for x in json.dumps(report)))
        forbidden_fields = {"test_ids", "test_count", "test_metrics", "official_test_metrics"}
        def keys(value):
            if isinstance(value, dict):
                yield from value.keys()
                for nested in value.values(): yield from keys(nested)
            elif isinstance(value, list):
                for nested in value: yield from keys(nested)
        self.assertFalse(forbidden_fields & set(keys(report)))
        self.assertTrue(all("provenance" not in row for row in report["model_validation"][:9]))
        self.assertEqual(len(report["hashes"]["new_checkpoints"]), 27)
        self.assertEqual(report["counts"], {"model_validation_observations": 36, "new_training_runs": 27,
            "read_only_c_references": 9, "cosine_validation_observations": 12,
            "training_curve_rows": 1800, "schedule_copies": 9, "new_checkpoints": 27})
        self.assertIn("c_training_curves", report["hashes"])
        self.assertEqual(set(report["aggregate_model_validation"]["C"]),
                         {"Recall@10", "HitRate@10", "MRR@10", "NDCG@10"})
        self.assertEqual(set(report["aggregate_cosine_validation"]["C"]),
                         {"Recall@10", "HitRate@10", "MRR@10", "NDCG@10"})
        self.assertTrue(all(caveat for caveat in report["caveats"]))
        self.assertIn("known_positive_edges_sha256", report["negative_exclusion_provenance"])
        self.assertEqual(report["negative_exclusion_provenance"]["known_positive_edges_sha256"],
                         self.runner.sha256_file(self.data / self.runner.INPUTS["cv_dir"] / "known_positive_edges.csv"))
        self.assertEqual(report["negative_exclusion_provenance"]["official_test_edge_source_used_as_targets"], False)
        self.assertEqual(report["hashes"]["cv_edge_artifacts_directory"],
                         self.runner._hash_path(self.data / self.runner.INPUTS["cv_dir"]))
        self.assertEqual(set(report["hashes"]["fold_edge_csv_sha256"]),
                         {f"fold_{f}_edges.csv" for f in self.runner.FOLDS})
        self.assertEqual(report["model_validation"][0]["path"],
                         "neurotune_v1_abcd_cv/checkpoints/fold_0_C_seed_684.pt")
        c_checkpoint = torch.load(self.croot / "checkpoints/fold_0_C_seed_684.pt", map_location="cpu", weights_only=False)
        self.assertEqual(report["model_validation"][0]["transform_sha256"],
                         c_checkpoint["provenance"]["transform_sha256"])
        self.assertEqual(report["model_validation"][0]["active_transform_sha256"],
                         self.runner._digest({k: c_checkpoint["transform_manifest"].get(k)
                                              for k in self.runner.ACTIVE_TRANSFORM_FIELDS}))
        self.assertIn("aggregate_cosine_validation", report)
        with (out / "training_curves.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(float(next(r["NDCG@10"] for r in rows if r["variant"] == "C")), 0.5)
        for key, source_rel in (("schedule_0_684", "triplet_schedules/fold_0_seed_684.npz"),):
            self.assertEqual(report["hashes"]["schedule_copies"]["0/684"],
                             self.runner.sha256_file(out / source_rel))
        # C's sentinel report remains byte-identical and was not needed by the run.
        self.assertEqual((self.croot / "experiment.json").read_text(), "INVALID SENTINEL: do not parse\n")

    def test_invalid_c_curves_and_parity_are_rejected(self):
        curve_path = self.croot / "training_curves.csv"
        original = curve_path.read_bytes()
        curve_path.write_text("fold,variant,seed,epoch,train_triplet_loss,validation_triplet_loss\n0,C,684,1,NaN,1\n")
        with self.assertRaises(ValueError):
            self.run_fixture()
        curve_path.write_bytes(original)
        # Metric selected by C must agree with the corresponding validation curve row.
        with curve_path.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[1]["NDCG@10"] = "0.1"
        with curve_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "selection metric"):
            self.run_fixture()
        curve_path.write_bytes(original)
        cp = self.croot / "checkpoints/fold_0_C_seed_684.pt"
        check = torch.load(cp, map_location="cpu", weights_only=False); check["config"]["device"] = "cuda"; torch.save(check, cp)
        with self.assertRaises(ValueError):
            self.run_fixture()
        check["config"]["device"] = "cpu"
        check["transform_manifest"]["bpm_mean"] = -99.0
        check["provenance"]["transform_sha256"] = self.runner._digest(check["transform_manifest"])
        torch.save(check, cp)
        with self.assertRaisesRegex(ValueError, "active transform"):
            self.run_fixture()

    def test_c_schedule_hash_mismatch_is_rejected(self):
        cp = self.croot / "checkpoints/fold_0_C_seed_684.pt"
        checkpoint = torch.load(cp, map_location="cpu", weights_only=False)
        checkpoint["provenance"]["schedule_sha256"] = "0" * 64
        torch.save(checkpoint, cp)
        with self.assertRaisesRegex(ValueError, "schedule hash"):
            self.run_fixture()

    def test_required_c_selection_metrics_are_rejected_when_missing(self):
        cp = self.croot / "checkpoints/fold_0_C_seed_684.pt"
        checkpoint = torch.load(cp, map_location="cpu", weights_only=False)
        del checkpoint["selection_metrics"]["HitRate@10"]
        torch.save(checkpoint, cp)
        with self.assertRaisesRegex(ValueError, "selection_metrics"):
            self.run_fixture()
        checkpoint["selection_metrics"]["HitRate@10"] = float("nan")
        torch.save(checkpoint, cp)
        with self.assertRaisesRegex(ValueError, "selection_metrics"):
            self.run_fixture()

    def test_new_checkpoint_best_metrics_are_validated_before_final_staging(self):
        deps = self.dependencies()
        trainer = deps["trainer"]
        def bad_metrics(*args, **kwargs):
            result = trainer(*args, **kwargs)
            result["best_metrics"].pop("HitRate@10")
            return result
        deps["trainer"] = bad_metrics
        with self.assertRaisesRegex(ValueError, "best_metrics"):
            self.run_fixture(deps)

    def test_new_checkpoint_nonfinite_best_metric_is_rejected(self):
        deps = self.dependencies()
        trainer = deps["trainer"]
        def nonfinite_metrics(*args, **kwargs):
            result = trainer(*args, **kwargs)
            result["best_metrics"]["MRR@10"] = float("nan")
            return result
        deps["trainer"] = nonfinite_metrics
        with self.assertRaisesRegex(ValueError, "best_metrics"):
            self.run_fixture(deps)

    def test_c_reference_paths_follow_temporary_results_root(self):
        report = self.run_fixture()
        self.assertTrue(report["model_validation"][0]["path"].startswith("neurotune_v1_abcd_cv/"))

    def test_c_curves_and_schedules_are_consumed_from_hashed_byte_snapshots(self):
        runner = self.runner
        original_curve_reader = runner._read_c_curves
        original_np_load = runner.np.load
        expected_curve_hash = runner.sha256_file(self.croot / "training_curves.csv")
        schedule_load_count = 0
        def curve_reader(payload):
            self.assertIsInstance(payload, bytes)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_curve_hash)
            return original_curve_reader(payload)
        def np_load(source, *args, **kwargs):
            nonlocal schedule_load_count
            self.assertIsInstance(source, io.BytesIO)
            schedule_load_count += 1
            return original_np_load(source, *args, **kwargs)
        with mock.patch.object(runner, "_read_c_curves", side_effect=curve_reader), \
                mock.patch.object(runner.np, "load", side_effect=np_load):
            report = self.run_fixture()
        self.assertEqual(len(report["hashes"]["schedule_copies"]), 9)
        self.assertEqual(schedule_load_count, 9)

    def test_schedule_ids_hash_and_no_overwrite(self):
        sp = self.croot / "triplet_schedules/fold_0_seed_684.npz"
        np.savez_compressed(sp, train_ids=np.asarray(["wrong"], dtype="U"),
            validation_ids=np.asarray(self.folds[0]["validation"], dtype="U"),
            train_schedule=np.zeros((50, 1, 3), dtype=np.int32), validation_schedule=np.zeros((1, 1, 3), dtype=np.int32))
        cp = self.croot / "checkpoints/fold_0_C_seed_684.pt"
        checkpoint = torch.load(cp, map_location="cpu", weights_only=False)
        checkpoint["provenance"]["schedule_sha256"] = self.runner.sha256_file(sp)
        torch.save(checkpoint, cp)
        with self.assertRaisesRegex(ValueError, "IDs"):
            self.run_fixture()
        (self.results / "neurotune_v1_full_modal_ablation").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.run_fixture()

    def test_input_and_c_mutations_are_detected_and_stage_is_removed(self):
        deps = self.dependencies()
        original_trainer = deps["trainer"]
        calls = 0
        def changing_trainer(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original_trainer(*args, **kwargs)
            if calls == 1:
                (self.data / self.runner.INPUTS["catalog_path"]).write_text("mutated\n")
            return result
        deps["trainer"] = changing_trainer
        with self.assertRaisesRegex(RuntimeError, "protected inputs"):
            self.run_fixture(deps)
        self.assertFalse((self.results / "neurotune_v1_full_modal_ablation").exists())

    def test_c_artifact_mutation_during_training_is_detected(self):
        deps = self.dependencies()
        original_trainer = deps["trainer"]
        def changing_trainer(*args, **kwargs):
            result = original_trainer(*args, **kwargs)
            cp = self.croot / "checkpoints/fold_0_C_seed_684.pt"
            cp.write_bytes(cp.read_bytes() + b"changed")
            return result
        deps["trainer"] = changing_trainer
        with self.assertRaisesRegex(RuntimeError, "protected inputs"):
            self.run_fixture(deps)
        self.assertFalse((self.results / "neurotune_v1_full_modal_ablation").exists())

    def test_publication_rolls_back_only_new_files(self):
        original_link = self.runner.os.link
        calls = 0
        def fail_during_publish(src, dst):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("synthetic link failure")
            return original_link(src, dst)
        with mock.patch.object(self.runner.os, "link", side_effect=fail_during_publish):
            with self.assertRaisesRegex(OSError, "synthetic link failure"):
                self.run_fixture()
        self.assertFalse((self.results / "neurotune_v1_full_modal_ablation").exists())
        self.assertTrue((self.croot / "experiment.json").is_file())


if __name__ == "__main__":
    unittest.main()

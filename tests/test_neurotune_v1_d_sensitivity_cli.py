"""Focused safety tests for the D validation-only sensitivity runner."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


class DSensitivityRunnerTests(unittest.TestCase):
    def test_runner_has_fixed_validation_protocol(self):
        runner = importlib.import_module("scripts.evaluate_neurotune_v1_d_sensitivity")
        self.assertEqual(runner.FOLDS, (0, 1, 2))
        self.assertEqual(runner.SEEDS, (684, 685, 686))
        self.assertEqual(runner.CONDITIONS, ("original", "audio", "bpm", "genre", "lyrics"))
        self.assertTrue(callable(runner._run))
        self.assertTrue(callable(runner._evaluate_checkpoint))

    def test_parity_failure_is_rejected(self):
        runner = importlib.import_module("scripts.evaluate_neurotune_v1_d_sensitivity")
        measured = {"Recall@10": 0.5, "HitRate@10": 0.5, "MRR@10": 0.5,
                    "NDCG@10": 0.5, "query_count": 2, "eligible_positive_edges": 1}
        expected = dict(measured)
        expected["query_count"] = 3
        with self.assertRaisesRegex(ValueError, "parity"):
            runner._assert_original_parity(measured, expected)

    def _fixture(self):
        runner = importlib.import_module("scripts.evaluate_neurotune_v1_d_sensitivity")
        temp = tempfile.TemporaryDirectory()
        base = Path(temp.name)
        data, results = base / "data", base / "results"
        for relative in runner.INPUTS.values():
            path = data / relative
            if relative.endswith("mxm_counts"):
                path.mkdir(parents=True)
                (path / "source.txt").write_text("mxm fixture")
            elif relative.endswith("/cv"):
                path.mkdir(parents=True)
                (path / "manifest.json").write_text('{ "fixture" : true }\n')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture input\n")
        folds = []
        for fold_id in runner.FOLDS:
            train = [f"f{fold_id}-train-{i}" for i in range(2)]
            validation = [f"f{fold_id}-validation-{i}" for i in range(3)]
            folds.append({"fold": fold_id, "train": train, "validation": validation,
                          "test": [f"SECRET_TEST_{fold_id}"],
                          "validation_edges": [(validation[0], validation[1])]})
        loaded = {"folds": folds, "manifest": {"fixture": True}, "catalog": {},
                  "cache_path": str(data / runner.INPUTS["cache_path"])}
        input_hashes = {key: runner._hash_path(data / relative) for key, relative in runner.INPUTS.items()
                        if key not in ("mxm_path", "test_edges_path", "ab_test_edges_path", "cv_dir")}
        source_hashes = {key: runner._hash_path(data / relative) for key, relative in runner.INPUTS.items()
                         if key not in ("mxm_path", "test_edges_path", "ab_test_edges_path", "cv_dir")}
        raw_split_hash = runner.sha256_file(data / runner.INPUTS["cv_dir"] / "manifest.json")
        split_hash = runner._digest(loaded["manifest"])
        self.assertNotEqual(split_hash, raw_split_hash)
        feature_code = {key: runner.sha256_file(runner.ROOT / key)
                        for key in ("src/features_neurotune_v1.py", "src/features.py")}
        feature_ids = []
        transform_by_fold = {}
        features_by_fold = {}
        for fold in folds:
            all_ids = sorted(fold["train"] + fold["validation"])
            transform = {"fit_ids_hash": "a" * 64, "audio_mean": [0.0] * 26,
                         "audio_std": [1.0] * 26, "audio_fit_rows": 2, "bpm_mean": 100.0,
                         "bpm_std": 10.0, "genre_vocabulary": ["rock"], "split_hash": split_hash,
                         "source_sha256": {"audio_cache": input_hashes["cache_path"], "split_manifest": split_hash},
                         "code_sha256": feature_code}
            transform_by_fold[fold["fold"]] = transform
            features_by_fold[fold["fold"]] = {
                "audio": [np.full((1 + i % 2, 26), i + 1, dtype=np.float32) for i in range(5)],
                "bpm": np.ones((5, 1), dtype=np.float32), "genre": np.ones((5, 1), dtype=np.float32),
                "lyrics": np.ones((5, 5000), dtype=np.float32),
            }
            self.assertEqual(len(all_ids), 5)
        p0 = results / "neurotune_v1_full_modal_ablation"
        (p0 / "checkpoints").mkdir(parents=True)
        (p0 / "triplet_schedules").mkdir()
        model = runner.NeuroTuneV1Encoder("D", 1)
        torch.manual_seed(52)
        state = model.state_dict()
        references = []
        from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
        from src.evaluate_neurotune_v1 import evaluate_neurotune_v1_retrieval
        for fold in folds:
            f = fold["fold"]
            ids = sorted(fold["validation"])
            feature_rows = {key: ([value[sorted(fold["train"] + fold["validation"]).index(i)] for i in ids]
                                   if isinstance(value, list) else value[[sorted(fold["train"] + fold["validation"]).index(i)
                                                                          for i in ids]])
                            for key, value in features_by_fold[f].items()}
            dataset = NeuroTuneV1Dataset(ids, feature_rows, modalities=("bpm", "genre", "lyrics"))
            batch = neurotune_v1_collate([dataset[i] for i in range(len(dataset))])
            model.load_state_dict(state); model.eval()
            with torch.no_grad():
                embeddings = model(**{key: batch[key] for key in ("audio", "mask", "bpm", "genre", "lyrics")}).numpy()
            metrics = evaluate_neurotune_v1_retrieval(embeddings, ids, fold["validation_edges"])
            validation = {key: metrics[key] for key in runner.METRICS}
            validation.update(query_count=metrics["query_count"],
                              eligible_positive_edges=metrics["unique_undirected_edge_count"])
            transform = transform_by_fold[f]
            for seed in runner.SEEDS:
                key = f"{f}/{seed}"
                schedule_path = p0 / f"triplet_schedules/fold_{f}_seed_{seed}.npz"
                np.savez_compressed(schedule_path, validation_ids=np.asarray(fold["validation"], dtype="U"))
                schedule_hash = runner.sha256_file(schedule_path)
                checkpoint = {"model_state_dict": state, "variant": "D", "fold": f, "seed": seed,
                              "epoch": 2, "transform_manifest": transform,
                              "selection_metrics": validation,
                              "provenance": {"variant": "D", "fold": f, "seed": seed,
                                             "transform_sha256": runner._digest(transform),
                                             "schedule_sha256": schedule_hash, "split_manifest_sha256": raw_split_hash,
                                             "input_sha256": input_hashes}}
                cp = p0 / f"checkpoints/fold_{f}_D_seed_{seed}.pt"
                torch.save(checkpoint, cp)
                references.append({"fold": f, "seed": seed, "variant": "D", "selected_epoch": 2,
                                  "checkpoint_path": cp.relative_to(p0).as_posix(),
                                  "checkpoint_sha256": runner.sha256_file(cp), "schedule_sha256": schedule_hash,
                                  "transform_sha256": runner._digest(transform), "validation": validation})
        (p0 / "experiment.json").write_text(json.dumps({"model_validation": references,
            "hashes": {"split_manifest": raw_split_hash, "data_sources": source_hashes}}) + "\n")
        calls = []

        def loader(**kwargs):
            return loaded

        def mxm_builder(*args):
            return {}, {"mapping_sha256": "d" * 64}

        def feature_builder(loaded_input, fold_input, docs):
            self.assertEqual(set(fold_input), {"fold", "train", "validation"})
            self.assertNotIn("SECRET_TEST", repr(fold_input))
            self.assertEqual(set(fold_input["train"] + fold_input["validation"]),
                             set(fold_input["train"]) | set(fold_input["validation"]))
            calls.append(tuple(sorted(fold_input["train"] + fold_input["validation"])))
            return features_by_fold[fold_input["fold"]], transform_by_fold[fold_input["fold"]]

        deps = {"loader": loader, "mxm_builder": mxm_builder, "feature_builder": feature_builder}
        return temp, runner, data, results, deps, calls

    def test_synthetic_run_uses_validation_only_and_writes_45_records(self):
        temp, runner, data, results, deps, calls = self._fixture()
        self.addCleanup(temp.cleanup)
        p0 = results / "neurotune_v1_full_modal_ablation"
        p0_report = json.loads((p0 / "experiment.json").read_text())
        self.assertEqual(p0_report["model_validation"][0]["checkpoint_path"],
                         "checkpoints/fold_0_D_seed_684.pt")
        report = runner._run(data, results, _test_only=True, _dependencies=deps)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(report["records"]), 45)
        self.assertEqual(report["counts"]["condition_records"], 45)
        self.assertTrue(report["protocol"]["test_evaluation_deferred"])
        self.assertEqual(len(report["checkpoint_records"]), 9)
        output = results / runner.OUTPUT_NAME
        self.assertEqual({path.name for path in output.iterdir()}, {"experiment.json", "condition_records.csv"})

    def test_checkpoint_reference_mismatch_aborts_without_publication(self):
        temp, runner, data, results, deps, calls = self._fixture()
        self.addCleanup(temp.cleanup)
        report_path = results / "neurotune_v1_full_modal_ablation" / "experiment.json"
        report = json.loads(report_path.read_text())
        report["model_validation"][0]["checkpoint_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "checkpoint.*mismatch"):
            runner._run(data, results, _test_only=True, _dependencies=deps)
        self.assertFalse((results / runner.OUTPUT_NAME).exists())

    def test_reference_checkpoint_path_mismatch_aborts(self):
        temp, runner, data, results, deps, calls = self._fixture()
        self.addCleanup(temp.cleanup)
        report_path = results / "neurotune_v1_full_modal_ablation" / "experiment.json"
        report = json.loads(report_path.read_text())
        report["model_validation"][0]["checkpoint_path"] = "checkpoints/not-the-loaded-file.pt"
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "path"):
            runner._run(data, results, _test_only=True, _dependencies=deps)
        self.assertFalse((results / runner.OUTPUT_NAME).exists())

    def test_checkpoint_selection_metric_mismatch_aborts(self):
        temp, runner, data, results, deps, calls = self._fixture()
        self.addCleanup(temp.cleanup)
        checkpoint_path = results / "neurotune_v1_full_modal_ablation" / "checkpoints/fold_0_D_seed_684.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["selection_metrics"]["Recall@10"] += 0.01
        torch.save(checkpoint, checkpoint_path)
        # The reference digest still describes the original snapshot, so update it to exercise
        # selection-metric parity rather than the earlier checkpoint-byte guard.
        report_path = results / "neurotune_v1_full_modal_ablation" / "experiment.json"
        report = json.loads(report_path.read_text())
        report["model_validation"][0]["checkpoint_sha256"] = runner.sha256_file(checkpoint_path)
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "selection metric"):
            runner._run(data, results, _test_only=True, _dependencies=deps)
        self.assertFalse((results / runner.OUTPUT_NAME).exists())

    def test_transform_fit_id_mismatch_aborts(self):
        temp, runner, data, results, deps, calls = self._fixture()
        self.addCleanup(temp.cleanup)
        original_builder = deps["feature_builder"]

        def mismatched_builder(*args):
            features, transform = original_builder(*args)
            transform = dict(transform, fit_ids_hash="e" * 64)
            return features, transform

        deps["feature_builder"] = mismatched_builder
        with self.assertRaisesRegex(ValueError, "transform"):
            runner._run(data, results, _test_only=True, _dependencies=deps)
        self.assertFalse((results / runner.OUTPUT_NAME).exists())

    def test_late_protected_input_mutation_aborts_without_publication(self):
        temp, runner, data, results, deps, calls = self._fixture()
        self.addCleanup(temp.cleanup)
        original_builder = deps["feature_builder"]
        cache_path = data / runner.INPUTS["cache_path"]

        def mutating_builder(*args):
            result = original_builder(*args)
            if len(calls) == 3:
                cache_path.write_text("changed after protected snapshot")
            return result

        deps["feature_builder"] = mutating_builder
        with self.assertRaisesRegex(RuntimeError, "protected inputs"):
            runner._run(data, results, _test_only=True, _dependencies=deps)
        self.assertFalse((results / runner.OUTPUT_NAME).exists())

    def test_original_records_use_parity_checked_direct_endpoint(self):
        temp, runner, data, results, deps, calls = self._fixture()
        self.addCleanup(temp.cleanup)
        actual_helper = runner.measure_behavior_change

        def inconsistent_helper(original, masked, *args, **kwargs):
            result = actual_helper(original, masked, *args, **kwargs)
            if np.array_equal(original, masked):
                result["retrieval_metrics"]["Recall@10"] = 0.123456
                result["retrieval_metrics"]["query_count"] += 17
                result["retrieval_metrics"]["unique_undirected_edge_count"] += 23
            return result

        with mock.patch.object(runner, "measure_behavior_change", side_effect=inconsistent_helper):
            report = runner._run(data, results, _test_only=True, _dependencies=deps)
        original = next(row for row in report["records"] if row["condition"] == "original")
        p0 = json.loads((results / "neurotune_v1_full_modal_ablation" / "experiment.json").read_text())
        expected = p0["model_validation"][0]["validation"]
        self.assertEqual(original["Recall@10"], expected["Recall@10"])
        self.assertEqual(original["query_count"], expected["query_count"])
        self.assertEqual(original["eligible_positive_edges"], expected["eligible_positive_edges"])

    def test_output_collision_is_rejected_before_loader(self):
        temp, runner, data, results, deps, calls = self._fixture()
        self.addCleanup(temp.cleanup)
        final = results / runner.OUTPUT_NAME
        final.mkdir()
        deps["loader"] = mock.Mock(side_effect=AssertionError("loader must not run"))
        with self.assertRaises(FileExistsError):
            runner._run(data, results, _test_only=True, _dependencies=deps)
        deps["loader"].assert_not_called()

    def test_staged_publication_rolls_back_only_new_files(self):
        runner = importlib.import_module("scripts.evaluate_neurotune_v1_d_sensitivity")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            stage, final = base / "stage", base / "new-output"
            (stage / "nested").mkdir(parents=True)
            (stage / "first.json").write_text("first")
            (stage / "nested" / "second.json").write_text("second")
            real_link = runner.os.link
            calls = 0

            def fail_second(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publication failure")
                return real_link(source, destination)

            with mock.patch.object(runner.os, "link", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected"):
                    runner._atomic_publish(stage, final)
            self.assertFalse(final.exists())


if __name__ == "__main__":
    unittest.main()

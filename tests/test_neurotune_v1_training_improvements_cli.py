import csv
import json
import tempfile
import unittest
import gc
import weakref
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts import run_neurotune_v1_training_improvements as runner


class TrainingImprovementsRunnerTests(unittest.TestCase):
    def _d0_fixture(self):
        transform = {"fit_ids_hash": "train", "source_sha256": {"audio_cache": "a", "split_manifest": "s"}}
        metrics = {"Recall@10": 0.1, "HitRate@10": 0.2, "MRR@10": 0.3, "NDCG@10": 0.4,
                   "query_count": 3, "eligible_positive_edges": 2}
        fold = {"fold": 0, "train": ["a", "b", "c"], "validation": ["u", "v", "w"]}
        checkpoint = {"fold": 0, "seed": 684, "variant": "D", "config": {"device": "cpu", "epochs": 50,
            "batch_size": 64, "optimizer": "Adam", "learning_rate": 0.001, "margin": 0.2},
            "transform_manifest": transform, "selection_metrics": metrics,
            "provenance": {"fold": 0, "seed": 684, "variant": "D", "transform_sha256": runner._digest(transform),
                "input_sha256": {"input": "hash"}, "split_manifest_sha256": "manifest"}}
        schedule = {"train_ids": np.asarray(fold["train"]), "validation_ids": np.asarray(fold["validation"])}
        record = {"fold": 0, "seed": 684, "variant": "D", "checkpoint_path": "checkpoints/fold_0_D_seed_684.pt",
            "checkpoint_sha256": "checkpoint", "schedule_sha256": "schedule",
            "transform_sha256": runner._digest(transform), "validation": metrics}
        args = (record, checkpoint, schedule, fold, 684, "checkpoint", "schedule", "manifest", {"input": "hash"})
        return args

    def test_d0_hash_config_id_and_transform_mismatches_are_rejected(self):
        base = self._d0_fixture()
        runner._verify_d0_record(*base)
        for changed in ("hash", "config", "ids", "transform"):
            args = list(self._d0_fixture())
            if changed == "hash":
                args[0] = {**args[0], "checkpoint_sha256": "bad"}
            elif changed == "config":
                args[1]["config"]["batch_size"] = 32
            elif changed == "ids":
                args[2]["train_ids"] = np.asarray(["outside", "b", "c"])
            else:
                args[0] = {**args[0], "transform_sha256": "bad"}
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner._verify_d0_record(*args)

    def test_namespace_collision_precedes_input_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "neurotune_v1_training_improvements").mkdir()
            with mock.patch.object(runner, "_hash_path", side_effect=AssertionError("heavy I/O")):
                with self.assertRaises(FileExistsError):
                    runner._run(root / "missing", root, _test_only=True, _dependencies={})

    def test_schedule_digest_covers_triplets_and_kinds(self):
        triplets = np.asarray([[0, 1, 2]], dtype=np.int32)
        codes = np.asarray([0], dtype=np.uint8)
        digest = runner._schedule_digest(triplets, codes)
        self.assertEqual(len(digest), 64)
        self.assertNotEqual(digest, runner._schedule_digest(triplets, np.asarray([1], dtype=np.uint8)))

    def test_provider_warmup_refresh_and_train_only_mining(self):
        ids = ["a", "b", "c", "d"]
        uniform = np.asarray([[[0, 1, 2]], [[1, 0, 3]], [[2, 3, 0]]], dtype=np.int32)
        seen = []

        class Dataset:
            track_ids = ids

        class Model:
            active_modalities = ()

        class Sampler:
            def __init__(self, candidates, edges, embeddings, **kwargs):
                seen.append((tuple(candidates), tuple(edges), tuple(kwargs["known_positive_edges"])))

            def sample(self, seed):
                return Sample(seed)

        class Sample:
            def __init__(self, seed):
                self.triplets = np.asarray([[0, 1, 2]], dtype=np.int32)
                self.sampling_kinds = ("uniform",)
                self.fallback_count = 0

        class TensorDataset:
            track_ids = ids

        dataset = TensorDataset()
        provider = runner._make_schedule_provider(
            ids, uniform, [("a", "b")], [("a", "b"), ("a", "c"), ("b", "d")],
            lambda candidates, edges, embeddings, **kwargs: Sampler(candidates, edges, embeddings, **kwargs),
        )
        with mock.patch.object(runner, "_model_embeddings", return_value=np.eye(4, dtype=np.float32)) as embeddings:
            for epoch in range(1, 15):
                sample = provider(Model(), dataset, epoch)
                self.assertEqual(sample.triplets.dtype, np.int32)
                if epoch <= 3:
                    np.testing.assert_array_equal(sample.triplets, uniform[epoch - 1])
        self.assertEqual(len(seen), 3)  # epochs 4, 9, 14
        self.assertEqual(embeddings.call_count, 3)
        self.assertTrue(all(entry[0] == tuple(ids) for entry in seen))
        self.assertTrue(all(set(edge).issubset(ids) for entry in seen for edge in entry[1] + entry[2]))

    def test_old_sampler_is_released_before_refresh_constructs_next_snapshot(self):
        ids = ["a", "b", "c", "d"]
        schedule = np.asarray([[[0, 1, 2]]] * 3, dtype=np.int32)
        prior = []

        class Dataset:
            track_ids = ids

        class Model:
            active_modalities = ()

        class Sample:
            triplets = np.asarray([[0, 1, 2]], dtype=np.int32)
            sampling_kinds = ("semi_hard",)

        class Sampler:
            def __init__(self, *args, **kwargs):
                gc.collect()
                if prior:
                    self_outer.assertIsNone(prior[-1](), "previous dense sampler is still retained at refresh")
                prior.append(weakref.ref(self))

            def sample(self, seed):
                return Sample()

        self_outer = self
        provider = runner._make_schedule_provider(ids, schedule, [("a", "b")], [("a", "b")], Sampler)
        with mock.patch.object(runner, "_model_embeddings", return_value=np.eye(4, dtype=np.float32)):
            for epoch in range(1, 10):
                provider(Model(), Dataset(), epoch)
        self.assertEqual(len(prior), 2)  # refreshes at epochs 4 and 9

    def test_report_is_validation_only_and_has_paired_rows(self):
        report = runner._build_report(
            scope="pilot", d1_records=[{"fold": 0, "variant": "D", "seed": 684}],
            d0_records=[{"fold": 0, "variant": "D", "seed": 684}],
            baselines=[{"fold": 0, "variant": "D", "validation": {"NDCG@10": 0.1}}],
        )
        self.assertEqual(report["evaluation_scope"], "validation_only_development")
        self.assertFalse(report["independent_test_completed"])
        self.assertNotIn("test_metrics", report)
        self.assertEqual(report["run_scope"], "pilot")

    def test_cli_pilot_namespace_is_separate(self):
        with mock.patch.object(runner, "_run") as run:
            with mock.patch("sys.argv", ["runner", "--pilot", "--data-root", "/data", "--results-root", "/results"]):
                runner.main()
        self.assertEqual(run.call_args.args, (Path("/data"), Path("/results")))
        self.assertTrue(run.call_args.kwargs["pilot"])

    def test_uniform_sample_reports_actual_known_positive_pool_sizes(self):
        ids = ["a", "b", "c", "d", "e"]
        triplets = np.asarray([[0, 1, 3], [1, 0, 4]], dtype=np.int32)
        sample = runner._uniform_sample(triplets, ids, [("a", "b"), ("a", "c"), ("b", "d")])
        # a excludes a,b,c; b excludes a,b,d.
        np.testing.assert_array_equal(sample.eligible_uniform_counts, [2, 2])
        np.testing.assert_array_equal(sample.triplets, triplets)
        with self.assertRaisesRegex(ValueError, "outside train IDs"):
            runner._uniform_sample(np.asarray([[9, 0, 1]], dtype=np.int32), ids, [])

    def test_schedule_archive_uses_uint8_codes_and_compact_diagnostics(self):
        rows = np.asarray([[0, 1, 2], [1, 0, 2]], dtype=np.int32)
        codes = np.asarray([[0, 1]], dtype=np.uint8)
        mapping = {"uniform": 0, "semi_hard": 1, "fallback_uniform": 2}
        self.assertEqual(len(runner._schedule_digest(rows, codes[0], mapping)), 64)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "schedule.npz"
            np.savez_compressed(archive, train_schedule=np.asarray([rows, rows]), sampling_type_codes=codes,
                                sampling_kind_code_mapping=np.asarray(json.dumps(mapping)))
            with np.load(archive, allow_pickle=False) as saved:
                self.assertEqual(saved["sampling_type_codes"].dtype, np.uint8)
                self.assertEqual(json.loads(str(saved["sampling_kind_code_mapping"])), mapping)

    def test_pilot_protocol_lists_only_executed_fold(self):
        report = runner._build_report("pilot", [], [], [], folds=[0])
        self.assertEqual(report["run_scope"], "pilot")
        self.assertEqual(report["protocol"]["folds"], [0])

    def test_rng_and_miner_versions_are_explicit_in_report(self):
        report = runner._build_report("pilot", [], [], [], folds=[0])
        self.assertEqual(report["protocol"]["python_random_version"], 2)
        self.assertEqual(report["protocol"]["epoch_seed_derivation"], "seed*1000003 + fold*1009 + epoch")
        self.assertEqual(report["protocol"]["miner_version"], "strict_cosine_band_v1")

    def test_runner_rejects_mutated_frozen_protocol_config(self):
        config = {
            "variant": "D", "folds": [0, 1, 2], "seeds": [684, 685, 686], "epochs": 50,
            "batch_size": 64, "optimizer": "Adam", "learning_rate": 0.001, "margin": 0.2,
            "device": "cpu", "negative_strategy": "mixed_semi_hard", "semi_hard_fraction": 0.25,
            "uniform_warmup_epochs": 3, "refresh_interval": 5,
            "rng": {"implementation": "python.random.Random", "version": 2,
                    "epoch_seed_derivation": "seed*1000003 + fold*1009 + epoch"},
            "miner_version": "strict_cosine_band_v1",
            "sampling_kind_codes": {"uniform": 0, "semi_hard": 1, "fallback_uniform": 2},
            "pilot": {"folds": [0], "seeds": [684, 685, 686],
                      "output_namespace": "neurotune_v1_training_improvements_pilot"},
            "full": {"folds": [0, 1, 2], "seeds": [684, 685, 686],
                     "output_namespace": "neurotune_v1_training_improvements"},
            "evaluation_scope": "validation_only_development",
        }
        config["learning_rate"] = 0.01
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "neurotune_v1_d1_semi_hard.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner._load_protocol_config(path)

    def test_protocol_config_rejects_bad_types_and_implementation_declarations(self):
        for name, mutate in (
            ("epoch_type", lambda config: config.update(epochs="50")),
            ("fold_type", lambda config: config.update(folds=[False, 1, 2])),
            ("pilot_shape", lambda config: config["pilot"].update(folds="0")),
            ("rng_implementation", lambda config: config["rng"].update(implementation="random.Random")),
            ("miner_version", lambda config: config.update(miner_version="other_miner")),
            ("type_code", lambda config: config["sampling_kind_codes"].update(semi_hard=3)),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                config = json.loads(json.dumps(runner.PROTOCOL_CONFIG))
                mutate(config)
                path = Path(tmp) / "malformed.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner._load_protocol_config(path, expected=config)

    def test_unsafe_output_namespace_rejected_before_run_input_access(self):
        config = json.loads(json.dumps(runner.PROTOCOL_CONFIG))
        config["pilot"]["output_namespace"] = "../outside"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "malformed.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output_namespace"):
                runner._load_protocol_config(path, expected=config)
            with mock.patch.object(runner, "CONFIG_PATH", path), \
                 mock.patch.object(runner, "_hash_path", side_effect=AssertionError("input access")):
                with self.assertRaisesRegex(ValueError, "output_namespace"):
                    runner._run(Path(tmp) / "missing-data", Path(tmp) / "results", pilot=True)
            self.assertFalse((Path(tmp) / "outside").exists())

    def _exercise_injected_pilot_run(self, fail_link=False):
        with tempfile.TemporaryDirectory() as tmp:
            root, results = Path(tmp) / "data", Path(tmp) / "results"
            root.mkdir(); results.mkdir()
            input_paths = {key: root / value for key, value in runner.INPUTS.items()}
            for key, path in input_paths.items():
                if key == "cv_dir":
                    path.mkdir(parents=True)
                    (path / "manifest.json").write_text("{}", encoding="utf-8")
                elif key == "mxm_path":
                    path.mkdir(parents=True)
                    (path / "source").write_text("mxm", encoding="utf-8")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(key.encode())
            d0 = results / "neurotune_v1_full_modal_ablation"
            (d0 / "checkpoints").mkdir(parents=True); (d0 / "triplet_schedules").mkdir()
            train_ids, validation_ids = ["z", "a", "c"], ["v", "u", "w"]
            train_edges, validation_edges = [("z", "a")], [("v", "u")]
            ids_sorted = sorted(train_ids + validation_ids)
            transform = {key: None for key in runner.ACTIVE_TRANSFORM_FIELDS}
            transform.update({"fit_ids_hash": "fit", "source_sha256": {
                "audio_cache": runner._hash_path(input_paths["cache_path"]),
                "split_manifest": runner._hash_path(input_paths["cv_dir"] / "manifest.json")}})
            metrics = {"Recall@10": 0.1, "HitRate@10": 0.2, "MRR@10": 0.3, "NDCG@10": 0.4,
                       "query_count": 2, "eligible_positive_edges": 1}
            input_hashes = {k: runner._hash_path(input_paths[k]) for k in runner.INPUTS
                            if k not in {"mxm_path", "test_edges_path", "ab_test_edges_path", "cv_dir"}}
            d0records, baseline_rows, protected_bytes = [], [], {}
            train_schedule = np.tile(np.asarray([[[0, 1, 2], [1, 0, 2]]], dtype=np.int32), (4, 1, 1))
            validation_schedule = np.asarray([[[0, 1, 2], [1, 0, 2]]], dtype=np.int32)
            for fold_no, seed in ((0, 684), (1, 684), (2, 684)):
                checkpoint_path = d0 / f"checkpoints/fold_{fold_no}_D_seed_{seed}.pt"
                schedule_path = d0 / f"triplet_schedules/fold_{fold_no}_seed_{seed}.npz"
                checkpoint = {"fold": fold_no, "seed": seed, "variant": "D", "epoch": 1,
                    "config": {"device": "cpu", "epochs": 4, "batch_size": 64, "optimizer": "Adam",
                               "learning_rate": 0.001, "margin": 0.2},
                    "transform_manifest": transform, "selection_metrics": metrics,
                    "provenance": {"fold": fold_no, "seed": seed, "variant": "D",
                        "transform_sha256": runner._digest(transform), "input_sha256": input_hashes,
                        "split_manifest_sha256": runner._hash_path(input_paths["cv_dir"] / "manifest.json")}}
                torch.save(checkpoint, checkpoint_path)
                np.savez_compressed(schedule_path, train_ids=np.asarray(train_ids), validation_ids=np.asarray(validation_ids),
                    train_schedule=train_schedule, validation_schedule=validation_schedule)
                cp_hash, sch_hash = runner.sha256_file(checkpoint_path), runner.sha256_file(schedule_path)
                d0records.append({"fold": fold_no, "seed": seed, "variant": "D", "selected_epoch": 1,
                    "checkpoint_path": f"checkpoints/fold_{fold_no}_D_seed_{seed}.pt", "checkpoint_sha256": cp_hash,
                    "schedule_sha256": sch_hash, "transform_sha256": runner._digest(transform), "validation": metrics})
                protected_bytes[checkpoint_path] = checkpoint_path.read_bytes()
                protected_bytes[schedule_path] = schedule_path.read_bytes()
                baseline_rows.append({"fold": fold_no, "variant": "D", "validation": metrics})
            d0_report_path = d0 / "experiment.json"
            d0_report_path.write_text(json.dumps({"model_validation": d0records, "cosine_validation": baseline_rows,
                "hashes": {"code": {"old": "historical"}}}), encoding="utf-8")
            protected_bytes[d0_report_path] = d0_report_path.read_bytes()
            fold = {"fold": 0, "train": train_ids, "validation": validation_ids, "test": ["test-only"],
                    "train_edges": train_edges, "validation_edges": validation_edges}
            known = train_edges + [("z", "c"), validation_edges[0]]
            loaded = {"folds": [fold, {**fold, "fold": 1}, {**fold, "fold": 2}],
                "catalog": {track_id: {"track_id": track_id} for track_id in ids_sorted + ["test-only"]},
                "manifest": {}, "cache_path": input_paths["cache_path"], "known_positive_edges": known}
            calls, sampler_inputs, captured = {"mxm": [], "features": []}, [], {}

            def loader(**kwargs):
                return loaded

            def mxm_builder(fold_loaded, mxm_path):
                calls["mxm"].append(set(fold_loaded["catalog"]))
                return {track_id: np.zeros(2, dtype=np.float32) for track_id in fold_loaded["catalog"]}, {"sources": {}}

            def feature_builder(feature_loaded, fold_arg, lyrics):
                feature_ids = sorted(fold_arg["train"] + fold_arg["validation"])
                self.assertEqual(set(feature_loaded["catalog"]), set(feature_ids))
                calls["features"].append(set(feature_ids))
                return ({"audio": [np.full((1, 26), i + 1, dtype=np.float32) for i, _ in enumerate(feature_ids)],
                         "bpm": np.zeros((6, 1), dtype=np.float32), "lyrics": np.zeros((6, 5000), dtype=np.float32),
                         "genre": np.zeros((6, 1), dtype=np.float32)}, transform)

            def baseline_builder(features, ids, modalities):
                return np.zeros((len(ids), 1), dtype=np.float32)

            def evaluator(embedding, ids, edges):
                return metrics

            class Sampler:
                def __init__(self, ids, positives, embeddings, **kwargs):
                    sampler_inputs.append((tuple(ids), tuple(positives), tuple(kwargs["known_positive_edges"])))

                def sample(self, seed):
                    return runner._uniform_sample(np.asarray([[0, 1, 2], [1, 0, 2]], dtype=np.int32), train_ids, known)

            def trainer(train_ds, val_ds, static_schedule, val_schedule, edges, variant, seed, path,
                        transform_manifest, provenance, **kwargs):
                self.assertEqual(variant, "D")
                self.assertTrue(kwargs["record_epoch_timing"])
                self.assertEqual(list(train_ds.track_ids), train_ids)
                # Feature values must correspond to sorted feature IDs, not train+validation concatenation.
                id_to_row = {track_id: row + 1 for row, track_id in enumerate(ids_sorted)}
                np.testing.assert_array_equal([row[0, 0] for row in train_ds.features["audio"]],
                                              [id_to_row[x] for x in train_ids])
                np.testing.assert_array_equal(val_schedule, validation_schedule)
                captured.setdefault("validation_schedule", val_schedule.copy())
                for epoch in range(1, 5):
                    kwargs["train_schedule_provider"](object(), train_ds, epoch)
                torch.save({"provenance": provenance}, path)
                return {"history": [{"epoch": epoch, "train_triplet_loss": 1.0, "epoch_seconds": 0.001}
                                     for epoch in range(1, 5)],
                        "best_epoch": 1, "best_metrics": metrics}

            deps = {"loader": loader, "mxm_builder": mxm_builder, "feature_builder": feature_builder,
                "baseline_builder": baseline_builder, "evaluator": evaluator, "trainer": trainer,
                "sampler_factory": Sampler}
            patched = (mock.patch.object(runner, "FOLDS", (0, 1, 2)), mock.patch.object(runner, "SEEDS", (684,)),
                       mock.patch.object(runner, "EPOCHS", 4), mock.patch.object(runner, "_model_embeddings", return_value=np.eye(3, dtype=np.float32)))
            with patched[0], patched[1], patched[2], patched[3]:
                if fail_link:
                    original_link = runner.os.link
                    calls_to_link = []
                    def fail_second_link(*args, **kwargs):
                        calls_to_link.append(args)
                        if len(calls_to_link) == 2:
                            raise OSError("injected publication failure")
                        return original_link(*args, **kwargs)
                    with mock.patch.object(runner.os, "link", side_effect=fail_second_link):
                        with self.assertRaisesRegex(OSError, "injected publication failure"):
                            runner._run(root, results, pilot=True, _test_only=True, _dependencies=deps)
                    self.assertEqual(len(calls_to_link), 2)
                    self.assertFalse((results / "neurotune_v1_training_improvements_pilot").exists())
                    self.assertFalse(list(results.glob(".neurotune_v1_training_improvements_pilot.stage-*")))
                    for path, raw in protected_bytes.items():
                        self.assertEqual(path.read_bytes(), raw)
                    return
                report = runner._run(root, results, pilot=True, _test_only=True, _dependencies=deps)
            self.assertTrue(all(ids == set(ids_sorted) for ids in calls["mxm"] + calls["features"]))
            self.assertTrue(all("test-only" not in ids for ids in calls["mxm"] + calls["features"]))
            self.assertTrue(any(("z", "c") in entry[2] for entry in sampler_inputs))
            self.assertTrue(all(("z", "c") not in entry[1] for entry in sampler_inputs))
            self.assertEqual(report["protocol"]["folds"], [0])
            self.assertEqual(report["evaluation_scope"], "validation_only_development")
            self.assertFalse(report["independent_test_completed"])
            self.assertEqual(len(report["model_validation"]), 2)
            self.assertEqual(len(report["cosine_validation"]), 1)
            output = results / "neurotune_v1_training_improvements_pilot"
            config = json.loads((output / "d1_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["frozen_config"], runner.PROTOCOL_CONFIG)
            self.assertEqual(config["run_scope"], "pilot")
            self.assertEqual(config["actual_folds"], [0])
            self.assertEqual(config["frozen_config"]["rng"]["version"], 2)
            self.assertEqual(config["frozen_config"]["rng"]["epoch_seed_derivation"], "seed*1000003 + fold*1009 + epoch")
            self.assertEqual(config["frozen_config"]["miner_version"], "strict_cosine_band_v1")
            self.assertEqual(report["rng_mining_provenance"]["miner_version"], "strict_cosine_band_v1")
            self.assertEqual(report["hashes"]["d1_config_sha256"], runner.sha256_file(runner.CONFIG_PATH))
            with np.load(output / "triplet_schedules/fold_0_seed_684.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["sampling_type_codes"].dtype, np.uint8)
                self.assertEqual(archive["sampling_type_codes"].shape, (4, 2))
                mapping = json.loads(str(archive["sampling_kind_code_mapping"]))
                digest = runner._schedule_digest(archive["train_schedule"][0], archive["sampling_type_codes"][0], mapping)
                self.assertEqual(digest, str(archive["schedule_sha256"][0]))
            with (output / "mining_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                self.assertIn("uniform_count", columns)
                self.assertNotIn("sampling_kinds", columns)
                rows = list(reader)
                self.assertEqual(rows[0]["schedule_source"], "verified_d0_uniform_schedule")
                self.assertEqual(rows[0]["epoch_seed"], "")
                self.assertEqual(rows[0]["uniform_count"], "2")
                self.assertEqual(rows[0]["uniform_fraction"], "1.0")
                self.assertEqual(rows[3]["miner_version"], "strict_cosine_band_v1")
                self.assertEqual(int(rows[3]["epoch_seed"]), 684 * 1000003 + 4)
            with (output / "training_curves.csv").open(newline="", encoding="utf-8") as stream:
                curve_reader = csv.DictReader(stream)
                self.assertIn("epoch_seconds", curve_reader.fieldnames)
                self.assertTrue(all(float(row["epoch_seconds"]) > 0 for row in curve_reader))
            for path, raw in protected_bytes.items():
                self.assertEqual(path.read_bytes(), raw)

    def test_successful_injected_pilot_run_isolated_and_transactional(self):
        self._exercise_injected_pilot_run()

    def test_publication_failure_rolls_back_partial_namespace(self):
        self._exercise_injected_pilot_run(fail_link=True)


if __name__ == "__main__":
    unittest.main()

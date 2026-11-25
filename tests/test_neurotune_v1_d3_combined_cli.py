import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts import run_neurotune_v1_d3_combined as runner


EXPECTED_CONFIG = {
    "variant": "D",
    "folds": [0, 1, 2],
    "seeds": [684, 685, 686],
    "epochs": 50,
    "batch_size": 64,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "margin": 0.2,
    "device": "cpu",
    "trainer_seed_derivation": "seed",
    "sampler": {
        "strategy": "mixed_semi_hard",
        "semi_hard_fraction": 0.25,
        "uniform_warmup_epochs": 3,
        "refresh_interval": 5,
        "miner_version": "strict_cosine_band_v1",
        "rng": {
            "implementation": "python.random.Random",
            "version": 2,
            "epoch_seed_derivation": "seed*1000003 + fold*1009 + epoch",
        },
        "sampling_kind_codes": {"uniform": 0, "semi_hard": 1, "fallback_uniform": 2},
    },
    "dropout": {
        "probabilities": {"audio": 0.0, "bpm": 0.1, "genre": 0.25, "lyrics": 0.2},
        "placement": "post_branch_activation_before_fusion",
        "scaling": "none",
        "scope": "training_only",
        "rng": {
            "implementation": "torch.Generator(device='cpu')",
            "fold_seed_derivation": "seed * 1000003 + fold * 1009",
            "epoch_seed_derivation": "dropout_seed + epoch_index_zero_based",
            "consumption": "deterministic_triplet_batch_order; anchor_positive_negative; bpm_genre_lyrics",
        },
    },
    "initialization": "independent_per_fold_seed",
    "reference_namespaces": {
        "D0": "neurotune_v1_full_modal_ablation",
        "D1": "neurotune_v1_training_improvements",
        "D2": "neurotune_v1_d2_modality_dropout",
    },
    "output_namespace": "neurotune_v1_d3_combined",
    "evaluation_scope": "validation_only_development",
}


class NeuroTuneV1D3CombinedConfigTests(unittest.TestCase):
    def test_frozen_config_matches_combined_d1_d2_protocol(self):
        config = runner._load_config()
        self.assertEqual(config, EXPECTED_CONFIG)
        self.assertIs(runner._validate_config(config), True)

        self.assertEqual(config["sampler"]["rng"]["epoch_seed_derivation"],
                         "seed*1000003 + fold*1009 + epoch")
        self.assertEqual(config["dropout"]["rng"]["fold_seed_derivation"],
                         "seed * 1000003 + fold * 1009")
        self.assertEqual(config["dropout"]["rng"]["epoch_seed_derivation"],
                         "dropout_seed + epoch_index_zero_based")
        self.assertEqual(config["evaluation_scope"], "validation_only_development")
        self.assertEqual(config["output_namespace"], "neurotune_v1_d3_combined")
        self.assertEqual(config["trainer_seed_derivation"], "seed")

    def test_config_file_is_valid_json(self):
        config_path = Path(runner.ROOT) / "config/neurotune_v1_d3_combined.json"
        with config_path.open(encoding="utf-8") as config_file:
            self.assertEqual(json.load(config_file), EXPECTED_CONFIG)

    def test_validator_rejects_changed_values_and_types(self):
        mutations = (
            ("epochs", True),
            ("learning_rate", True),
            ("margin", 0.3),
            ("folds", [0, 1]),
            ("device", "cuda"),
            ("output_namespace", "../unsafe"),
            ("evaluation_scope", "test_set"),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(EXPECTED_CONFIG)
                config[key] = value
                with self.assertRaises(ValueError):
                    runner._validate_config(config)


class NeuroTuneV1D3ReferenceTests(unittest.TestCase):
    folds, seeds = (0, 1, 2), (684, 685, 686)

    def setUp(self):
        if self._testMethodName not in {"test_good_d0_d1_d2_references_validate",
                                       "test_rejects_missing_duplicate_bad_hash_unsafe_path_and_count_mismatch",
                                       "test_rejects_missing_referenced_artifact",
                                       "test_rejects_missing_d1_or_d2_output_manifest_entry",
                                       "test_rejects_boolean_fold_ids_in_reference_rows",
                                       "test_four_epoch_real_trainer_smoke"}:
            return
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.d0 = self.results / "neurotune_v1_full_modal_ablation"
        self.d1 = self.results / "neurotune_v1_training_improvements"
        self.d2 = self.results / "neurotune_v1_d2_modality_dropout"
        self.d1_config = json.loads((runner.ROOT / "config/neurotune_v1_d1_semi_hard.json").read_text())
        self.d2_config_bytes = (runner.ROOT / "config/neurotune_v1_d2_modality_dropout.json").read_bytes()
        self.d1.mkdir(parents=True)
        self.d2.mkdir(parents=True)
        (self.d1 / "d1_config.json").write_text(json.dumps({"frozen_config": self.d1_config, "run_scope": "full",
            "actual_folds": list(self.folds), "actual_seeds": list(self.seeds)}))
        (self.d2 / "d2_config.json").write_bytes(self.d2_config_bytes)
        for ns in (self.d0, self.d1, self.d2):
            (ns / "checkpoints").mkdir(parents=True, exist_ok=True)
            (ns / "triplet_schedules").mkdir(parents=True, exist_ok=True)
        for filename in ("training_curves.csv", "mining_diagnostics.csv", "validation_metrics.csv"):
            (self.d1 / filename).write_text("fixture\n")
        for filename in ("training_curves.csv", "modality_dropout_diagnostics.csv"):
            (self.d2 / filename).write_text("fixture\n")

        self.metrics = {"Recall@10": .2, "HitRate@10": .3, "MRR@10": .1, "NDCG@10": .15,
                        "query_count": 2, "eligible_positive_edges": 1}
        self.d0_models, self.d1_models, self.d2_models = [], [], []
        d0_cosines, d0_copies, d1_rows, d1_copies, d2_rows, d2_outputs = [], {}, [], {}, [], {}
        for fold in self.folds:
            d0_cosines.append({"fold": fold, "variant": "D", "validation": dict(self.metrics)})
            for seed in self.seeds:
                key = f"{fold}/{seed}"
                d0sch = self.d0 / f"triplet_schedules/fold_{fold}_seed_{seed}.npz"
                train_ids = np.asarray(["a", "b", "c", "d"])
                validation_ids = np.asarray(["v0", "v1", "v2"])
                warmup = np.tile(np.asarray([[[0, 1, 3], [1, 2, 3]]], dtype=np.int32), (50, 1, 1))
                validation_schedule = np.asarray([[[0, 1, 2], [1, 0, 2]]], dtype=np.int32)
                np.savez_compressed(d0sch, train_ids=train_ids, validation_ids=validation_ids,
                    train_schedule=warmup, validation_schedule=validation_schedule)
                transform = {"fit_ids_hash": "fit", "audio_mean": [0.0], "audio_std": [1.0],
                    "audio_fit_rows": 4, "bpm_mean": 0.0, "bpm_std": 1.0,
                    "genre_vocabulary": ["x", "y"], "split_hash": "split",
                    "source_sha256": {"audio_cache": "cache", "split_manifest": "manifest"}}
                d0cp = self.d0 / f"checkpoints/fold_{fold}_D_seed_{seed}.pt"
                torch.save({"transform_manifest": transform}, d0cp)
                dh, ch = self.sha(d0sch), self.sha(d0cp)
                d0_copies[key] = dh
                self.d0_models.append(self.record(fold, seed, "checkpoints/fold_%d_D_seed_%d.pt" % (fold, seed), ch, dh))

                d1sch = self.d1 / f"triplet_schedules/fold_{fold}_seed_{seed}.npz"
                codes = np.zeros((50, 1), dtype=np.uint8)
                from scripts.run_neurotune_v1_training_improvements import _schedule_digest
                digests = [_schedule_digest(np.zeros((1, 3), dtype=np.int32), code) for code in codes]
                np.savez_compressed(d1sch, train_schedule=np.zeros((50, 1, 3), dtype=np.int32),
                                    sampling_type_codes=codes, schedule_sha256=np.asarray(digests))
                d1cp = self.d1 / f"checkpoints/fold_{fold}_D_seed_{seed}.pt"; d1cp.write_bytes(f"d1-{key}".encode())
                dh, ch = self.sha(d1sch), self.sha(d1cp)
                digest = hashlib.sha256(json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                d1row = self.record(fold, seed, f"checkpoints/fold_{fold}_D_seed_{seed}.pt", ch, digest)
                d1row["schedule_file_sha256"] = dh
                self.d1_models.append(d1row); d1_copies[key] = {"archive_sha256": dh, "schedule_sha256": digest}

                d2sch = self.d2 / f"triplet_schedules/fold_{fold}_seed_{seed}.npz"
                d2sch.write_bytes(d0sch.read_bytes())
                d2cp = self.d2 / f"checkpoints/fold_{fold}_D_seed_{seed}.pt"; d2cp.write_bytes(f"d2-{key}".encode())
                d2row = self.record(fold, seed, f"checkpoints/fold_{fold}_D_seed_{seed}.pt", self.sha(d2cp), self.sha(d2sch))
                d2_rows.append(d2row)
                d2_outputs[f"checkpoints/fold_{fold}_D_seed_{seed}.pt"] = d2row["checkpoint_sha256"]
                d2_outputs[f"triplet_schedules/fold_{fold}_seed_{seed}.npz"] = self.sha(d2sch)
        d2_outputs["d2_config.json"] = self.sha(self.d2 / "d2_config.json")

        d0_protocol = {"folds": list(self.folds), "seeds": list(self.seeds), "epochs": 50, "batch_size": 64,
            "optimizer": "Adam", "learning_rate": .001, "triplet_margin": .2, "device": "cpu",
            "trained_variants": ["A", "B", "D"], "modalities": {"A": [], "B": ["bpm"],
            "C": ["bpm", "genre"], "D": ["bpm", "lyrics", "genre"]}}
        self.write_report(self.d0 / "experiment.json", {"status": "complete", "protocol": d0_protocol,
            "model_validation": self.d0_models, "cosine_validation": d0_cosines,
            "hashes": {"schedule_copies": d0_copies}})
        d1_protocol = {"folds": list(self.folds), "seeds": list(self.seeds), "epochs": 50, "batch_size": 64,
            "learning_rate": .001, "triplet_margin": .2, "optimizer": "Adam", "device": "cpu",
            "trained_variants": ["D"], "negative_strategy": "mixed_semi_hard", "modalities": {"D": ["bpm", "lyrics", "genre"]},
            "semi_hard_fraction": .25, "uniform_warmup_epochs": 3, "refresh_interval": 5,
            "python_random_version": 2, "epoch_seed_derivation": "seed*1000003 + fold*1009 + epoch",
            "miner_version": "strict_cosine_band_v1", "selection": "validation NDCG@10, then validation loss"}
        self.write_report(self.d1 / "experiment.json", {"status": "complete", "run_scope": "full",
            "evaluation_scope": "validation_only_development", "independent_test_completed": False,
            "protocol": d1_protocol, "model_validation": self.d0_models + self.d1_models,
            "cosine_validation": d0_cosines,
            "hashes": {"d1_config_sha256": self.sha(runner.ROOT / "config/neurotune_v1_d1_semi_hard.json"),
                "d0_experiment_sha256": self.sha(self.d0 / "experiment.json"),
                "d0_checkpoints": {k: self.d0_models[i]["checkpoint_sha256"] for i, k in enumerate(d0_copies)},
                "d0_schedules": d0_copies, "outputs": {**d1_copies,
                    "checkpoints": {r["checkpoint_path"]: r["checkpoint_sha256"] for r in self.d1_models},
                    "files": {p.relative_to(self.d1).as_posix(): self.sha(p)
                              for p in sorted(self.d1.rglob("*")) if p.is_file()}}}})
        d2_protocol = {**json.loads(self.d2_config_bytes), "dropout_seed_derivation":
            "seed * 1000003 + fold * 1009; trainer epoch stream uses dropout_seed + epoch_index_zero_based"}
        self.write_report(self.d2 / "experiment.json", {"status": "complete", "protocol": d2_protocol, "records": d2_rows,
            "test_evaluation_deferred": True, "test_features_constructed": False,
            "test_embeddings_constructed": False, "test_relevance_constructed": False, "test_metrics_computed": False,
            "hashes": {"config": self.sha(self.d2 / "d2_config.json"),
                "d0_experiment": self.sha(self.d0 / "experiment.json"), "d0_schedules": d0_copies,
                "schedule_copies": d0_copies, "output_files": {**d2_outputs,
                    **{p.relative_to(self.d2).as_posix(): self.sha(p)
                       for p in (self.d2 / "training_curves.csv", self.d2 / "modality_dropout_diagnostics.csv")}}}})
        self.report_bytes = {name: (root / "experiment.json").read_bytes()
                             for name, root in (("D0", self.d0), ("D1", self.d1), ("D2", self.d2))}

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def record(self, fold, seed, path, checkpoint_hash, schedule_hash):
        return {"fold": fold, "seed": seed, "variant": "D", "selected_epoch": 7,
            "validation": dict(self.metrics), "checkpoint_path": path,
            "checkpoint_sha256": checkpoint_hash, "schedule_sha256": schedule_hash}

    @staticmethod
    def write_report(path, report):
        path.write_text(json.dumps(report, allow_nan=True))

    def load_report(self, namespace):
        path = {"D0": self.d0, "D1": self.d1, "D2": self.d2}[namespace] / "experiment.json"
        return json.loads(path.read_text())

    def save_report(self, namespace, report):
        path = {"D0": self.d0, "D1": self.d1, "D2": self.d2}[namespace] / "experiment.json"
        self.write_report(path, report)

    def restore_reports(self):
        for name, root in (("D0", self.d0), ("D1", self.d1), ("D2", self.d2)):
            (root / "experiment.json").write_bytes(self.report_bytes[name])

    def tearDown(self):
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def test_good_d0_d1_d2_references_validate(self):
        d1_report = self.load_report("D1")
        self.assertEqual(d1_report["hashes"]["d1_config_sha256"],
                         self.sha(runner.ROOT / "config/neurotune_v1_d1_semi_hard.json"))
        validated = runner._validate_references(self.results)
        self.assertEqual(set(validated["models"]["D0"]), {(f, s) for f in self.folds for s in self.seeds})
        self.assertEqual(set(validated["models"]["D1"]), {(f, s) for f in self.folds for s in self.seeds})
        self.assertEqual(set(validated["models"]["D2"]), {(f, s) for f in self.folds for s in self.seeds})
        self.assertEqual(set(validated["cosine"]), set(self.folds))

    def test_rejects_missing_duplicate_bad_hash_unsafe_path_and_count_mismatch(self):
        mutations = (
            ("D0", lambda r: r["model_validation"].pop()),
            ("D0", lambda r: r["model_validation"].append(copy.deepcopy(r["model_validation"][0]))),
            ("D0", lambda r: r["cosine_validation"].append(copy.deepcopy(r["cosine_validation"][0]))),
            ("D0", lambda r: r["model_validation"][0].update(checkpoint_path="../escape")),
            ("D0", lambda r: r["model_validation"][0].update(selected_epoch=True)),
            ("D0", lambda r: r["model_validation"][0]["validation"].update(**{"NDCG@10": float("nan")})),
            ("D1", lambda r: r["model_validation"].append(copy.deepcopy(r["model_validation"][-1]))),
            ("D1", lambda r: r["model_validation"][-1].update(schedule_file_sha256="f" * 64)),
            ("D1", lambda r: r["hashes"].update(d1_config_sha256="0" * 64)),
            ("D2", lambda r: r["records"][0].update(checkpoint_sha256="0" * 64)),
            ("D2", lambda r: r["records"][0].update(checkpoint_path="../escape")),
            ("D2", lambda r: r["records"].pop()),
            ("D2", lambda r: r["records"][0]["validation"].update(query_count=99)),
        )
        for namespace, mutate in mutations:
            with self.subTest(namespace=namespace, mutate=mutate):
                self.restore_reports()
                report = self.load_report(namespace)
                mutate(report)
                self.save_report(namespace, report)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    runner._validate_references(self.results)

    def test_rejects_boolean_fold_ids_in_reference_rows(self):
        mutations = (
            ("D0", lambda r: r["model_validation"][0].update(fold=True)),
            ("D1", lambda r: r["model_validation"][0].update(fold=True)),
            ("D1", lambda r: r["model_validation"][-1].update(fold=True)),
            ("D2", lambda r: r["records"][0].update(fold=True)),
            ("D0", lambda r: r["cosine_validation"][0].update(fold=False)),
        )
        for namespace, mutate in mutations:
            with self.subTest(namespace=namespace, mutate=mutate):
                self.restore_reports()
                report = self.load_report(namespace)
                mutate(report)
                self.save_report(namespace, report)
                if namespace == "D0":
                    d0_hash = self.sha(self.d0 / "experiment.json")
                    for dependent, key in (("D1", "d0_experiment_sha256"), ("D2", "d0_experiment")):
                        dependent_report = self.load_report(dependent)
                        dependent_report["hashes"][key] = d0_hash
                        self.save_report(dependent, dependent_report)
                with self.assertRaises(ValueError):
                    runner._validate_references(self.results)

    def test_rejects_missing_referenced_artifact(self):
        (self.d1 / "triplet_schedules/fold_2_seed_686.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            runner._validate_references(self.results)

    def test_rejects_missing_d1_or_d2_output_manifest_entry(self):
        for namespace, relative in (("D1", "mining_diagnostics.csv"),
                                    ("D2", "modality_dropout_diagnostics.csv")):
            with self.subTest(namespace=namespace):
                self.restore_reports()
                report = self.load_report(namespace)
                manifest = report["hashes"]["outputs"]["files"] if namespace == "D1" else report["hashes"]["output_files"]
                self.assertTrue((self.d1 if namespace == "D1" else self.d2).joinpath(relative).is_file())
                del manifest[relative]
                self.save_report(namespace, report)
                with self.assertRaises(ValueError):
                    runner._validate_references(self.results)

    def test_validator_rejects_changed_sampler_and_dropout_values_or_types(self):
        mutations = (
            (("sampler", "semi_hard_fraction"), True),
            (("sampler", "semi_hard_fraction"), 0.5),
            (("sampler", "refresh_interval"), 4),
            (("sampler", "miner_version"), "other"),
            (("sampler", "rng", "version"), True),
            (("sampler", "sampling_kind_codes", "uniform"), False),
            (("dropout", "probabilities", "audio"), False),
            (("dropout", "probabilities", "genre"), 0.2),
            (("dropout", "placement"), "after_fusion"),
            (("dropout", "scope"), "all_forwards"),
            (("dropout", "rng", "implementation"), "python.random.Random"),
        )
        for path, value in mutations:
            with self.subTest(path=path, value=value):
                config = copy.deepcopy(EXPECTED_CONFIG)
                target = config
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    runner._validate_config(config)

    def test_validator_rejects_missing_or_extra_protocol_fields(self):
        for change in ("missing", "extra"):
            with self.subTest(change=change):
                config = copy.deepcopy(EXPECTED_CONFIG)
                if change == "missing":
                    del config["device"]
                else:
                    config["unfrozen_option"] = True
                with self.assertRaises(ValueError):
                    runner._validate_config(config)

    def test_four_epoch_real_trainer_smoke(self):
        import csv
        from src.model_v1 import NeuroTuneV1Encoder
        from src.train_neurotune_v1 import train_neurotune_v1

        self.restore_reports()
        data = self.root / "data"
        for relative in runner.INPUTS.values():
            target = data / relative
            if relative == runner.INPUTS["mxm_path"]:
                target.mkdir(parents=True, exist_ok=True)
                (target / "fixture.txt").write_text("mxm source fixture")
            elif relative == runner.INPUTS["cv_dir"]:
                target.mkdir(parents=True, exist_ok=True)
                (target / "manifest.json").write_text("{}")
                (target / "known_positive_edges.csv").write_text("source,target\n")
                (target / "fold_0_edges.csv").write_text("fixture\n")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("fixture\n")
        train_ids, val_ids = ["a", "b", "c", "d"], ["v0", "v1", "v2"]
        positive_train, positive_val = [("a", "b"), ("b", "c")], [("v0", "v1")]
        fold = {"fold": 0, "train": train_ids, "validation": val_ids,
            "train_edges": positive_train, "validation_edges": positive_val}
        loaded = {"folds": [fold], "catalog": {x: {"track_id": x} for x in train_ids + val_ids + ["heldout"]},
            "manifest": {}, "cache_path": data / runner.INPUTS["cache_path"],
            "known_positive_edges": positive_train + positive_val + [("a", "c")]}
        transform = {"fit_ids_hash": "fit", "audio_mean": [0.0], "audio_std": [1.0],
            "audio_fit_rows": 4, "bpm_mean": 0.0, "bpm_std": 1.0,
            "genre_vocabulary": ["x", "y"], "split_hash": "split",
            "source_sha256": {"audio_cache": "cache", "split_manifest": "manifest"}}
        feature_ids_seen = []

        def mxm_builder(fold_loaded, _path):
            self.assertNotIn("heldout", fold_loaded["catalog"])
            return {key: {} for key in fold_loaded["catalog"]}, {"fixture": True}

        def feature_builder(fold_loaded, _fold, _lyrics):
            ids = sorted(fold_loaded["catalog"])
            feature_ids_seen.extend(ids)
            count = len(ids)
            features = {"audio": [np.ones((3, 26), dtype=np.float32) * (i + 1) for i in range(count)],
                "bpm": np.arange(count, dtype=np.float32).reshape(-1, 1),
                "genre": np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (count, 1)),
                "lyrics": np.zeros((count, 5000), dtype=np.float32)}
            return features, transform

        def loader(**_kwargs):
            return loaded

        trainer_calls, forward_modes, triplet_forward_calls = [], [], []
        real_forward = NeuroTuneV1Encoder.forward
        real_triplet_forward = NeuroTuneV1Encoder.forward_triplets

        def observed_forward(model, *args, **kwargs):
            forward_modes.append((model.training, torch.is_grad_enabled()))
            return real_forward(model, *args, **kwargs)

        def observed_triplet_forward(model, *args, **kwargs):
            triplet_forward_calls.append((model.training, set(kwargs)))
            return real_triplet_forward(model, *args, **kwargs)

        def trainer_spy(*args, **kwargs):
            trainer_calls.append({"seed": args[6], "train_schedule": args[2],
                "provider": kwargs.get("train_schedule_provider"), "dropout_seed": kwargs.get("dropout_seed"),
                "epochs": kwargs.get("epochs"), "batch_size": kwargs.get("batch_size")})
            return train_neurotune_v1(*args, **kwargs)

        with mock.patch.object(NeuroTuneV1Encoder, "forward", observed_forward), \
             mock.patch.object(NeuroTuneV1Encoder, "forward_triplets", observed_triplet_forward):
            report = runner._run(data_root=data, results_root=self.results, _test_only=True,
                _dependencies={"loader": loader, "mxm_builder": mxm_builder,
                    "feature_builder": feature_builder, "trainer": trainer_spy},
                _protocol={"folds": (0,), "seeds": (684,), "epochs": 4, "batch_size": 2})
        self.assertEqual(feature_ids_seen, sorted(train_ids + val_ids))
        self.assertEqual(len(trainer_calls), 1)
        call = trainer_calls[0]
        self.assertEqual(call["seed"], 684)
        self.assertIsNone(call["train_schedule"])
        self.assertEqual(call["dropout_seed"], 684 * 1000003)
        self.assertEqual(call["epochs"], 4)
        self.assertEqual(call["batch_size"], 2)
        self.assertEqual(len(call["provider"].state["history"]), 4)
        self.assertEqual([item["schedule_source"] for item in call["provider"].state["history"][:3]],
                         ["verified_d0_uniform_schedule"] * 3)
        self.assertEqual(call["provider"].state["history"][3]["schedule_source"], "strict_cosine_band_v1")
        self.assertIn((False, False), forward_modes)  # Mining embeddings/validation are eval + no-grad.
        self.assertTrue(any(training and {"anchor_keep_masks", "positive_keep_masks", "negative_keep_masks"} <= keys
                            for training, keys in triplet_forward_calls))
        self.assertTrue(any(not training and not keys for training, keys in triplet_forward_calls))
        self.assertEqual(report["counts"]["training_curve_rows"], 4)
        self.assertEqual(report["counts"]["mining_diagnostic_rows"], 4)
        self.assertEqual(report["counts"]["modality_dropout_diagnostic_rows"], 4)
        self.assertFalse(report["test_features_constructed"])
        self.assertFalse(report["test_embeddings_constructed"])
        self.assertFalse(report["test_relevance_constructed"])
        self.assertFalse(report["test_metrics_computed"])
        self.assertIn("references", report["hashes"])
        self.assertIn("protected_before", report["hashes"])
        out = self.results / "neurotune_v1_d3_combined"
        self.assertTrue((out / "experiment.json").is_file())
        with np.load(out / "triplet_schedules/fold_0_seed_684.npz", allow_pickle=False) as archive:
            self.assertEqual(len(archive["epoch_offsets"]), 5)
            self.assertEqual(archive["train_triplets"].shape[0], len(archive["sampling_type_codes"]))
            self.assertEqual(len(archive["schedule_sha256"]), 4)
        with (out / "training_curves.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)

    def test_runner_refuses_existing_namespace_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            result_root = Path(temp)
            target = result_root / "neurotune_v1_d3_combined"
            target.mkdir()
            sentinel = target / "keep.txt"
            sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                runner._run(results_root=result_root, _test_only=True)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_production_dependencies_and_protocol_overrides_are_forbidden(self):
        with self.assertRaises(ValueError):
            runner._run(_dependencies={"trainer": object()})
        with self.assertRaises(ValueError):
            runner._run(_protocol={"epochs": 4})

    def test_cli_exposes_data_and_results_roots(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "-m", "scripts.run_neurotune_v1_d3_combined", "--help"],
                                cwd=runner.ROOT, check=True, capture_output=True, text=True)
        self.assertIn("--data-root", result.stdout)
        self.assertIn("--results-root", result.stdout)


if __name__ == "__main__":
    unittest.main()

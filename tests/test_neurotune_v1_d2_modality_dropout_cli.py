"""Safety and protocol tests for the isolated D2 runner."""

import hashlib
import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class D2RunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = importlib.import_module("scripts.run_neurotune_v1_modality_dropout")
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.data, self.results = self.base / "data", self.base / "results"
        for rel in self.runner.INPUTS.values():
            p = self.data / rel
            if rel.endswith("mxm_counts"):
                p.mkdir(parents=True)
                (p / "source.txt").write_text("mxm\n")
            elif rel.endswith("/cv"):
                p.mkdir(parents=True)
                (p / "manifest.json").write_text('{"fixture":true}\n')
                (p / "known_positive_edges.csv").write_text("source,target\na,b\n")
                (p / "fold_0_edges.csv").write_text("source,target\na,b\n")
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("fixture\n")
        self.d0 = self.results / "neurotune_v1_full_modal_ablation"
        (self.d0 / "triplet_schedules").mkdir(parents=True)
        self.fold = {"fold": 0, "train": ["tr0", "tr1", "tr2"],
                     "validation": ["va0", "va1", "va2"],
                     "validation_edges": [("va0", "va1")]}
        self.loaded = {"folds": [self.fold], "manifest": {"fixture": True},
                       "catalog": {}, "cache_path": str(self.data / "dummy"), "known_positive_edges": []}
        self.schedule_bytes = {}
        for seed in (684,):
            p = self.d0 / "triplet_schedules/fold_0_seed_684.npz"
            np.savez_compressed(p, train_ids=np.asarray(self.fold["train"], dtype="U"),
                validation_ids=np.asarray(self.fold["validation"], dtype="U"),
                train_schedule=np.zeros((2, 2, 3), dtype=np.int32),
                validation_schedule=np.zeros((1, 1, 3), dtype=np.int32))
            self.schedule_bytes[(0, seed)] = p.read_bytes()
        (self.d0 / "checkpoints").mkdir()
        self.d0_checkpoint = self.d0 / "checkpoints/fold_0_D_seed_684.pt"
        torch.save({"fixture": "d0"}, self.d0_checkpoint)
        self.write_d0_report()

    @staticmethod
    def metrics():
        return {"Recall@10": .2, "HitRate@10": .3, "MRR@10": .1, "NDCG@10": .15,
                "query_count": 2, "eligible_positive_edges": 1}

    def write_d0_report(self, *, schedule_hash=None, checkpoint_hash=None, checkpoint_path=None,
                        duplicate_cosine=False, selected_epoch=2, validation=None):
        schedule_hash = schedule_hash or hashlib.sha256(self.schedule_bytes[(0, 684)]).hexdigest()
        record = {"fold": 0, "variant": "D", "seed": 684, "selected_epoch": selected_epoch,
                  "validation": validation if validation is not None else self.metrics(), "checkpoint_path": checkpoint_path or
                  "checkpoints/fold_0_D_seed_684.pt",
                  "checkpoint_sha256": checkpoint_hash or hashlib.sha256(self.d0_checkpoint.read_bytes()).hexdigest(),
                  "schedule_sha256": schedule_hash}
        cosine = {"fold": 0, "variant": "D", "validation": self.metrics()}
        report = {"status": "complete", "protocol": {"folds": [0, 1, 2], "seeds": [684, 685, 686],
            "epochs": 50, "batch_size": 64, "optimizer": "Adam", "learning_rate": .001,
            "triplet_margin": .2, "device": "cpu", "trained_variants": ["A", "B", "D"],
            "modalities": {"A": [], "B": ["bpm"], "C": ["bpm", "genre"],
                           "D": ["bpm", "lyrics", "genre"]}, "extra_report_metadata": "ignored"},
            "model_validation": [record],
            "cosine_validation": [cosine] * (2 if duplicate_cosine else 1),
            "hashes": {"schedule_copies": {"0/684": hashlib.sha256(self.schedule_bytes[(0, 684)]).hexdigest()}}}
        (self.d0 / "experiment.json").write_text(json.dumps(report))

    def tearDown(self):
        self.tmp.cleanup()

    def deps(self, failure=False):
        calls = {"features": 0, "trainer": []}
        def loader(**kwargs): return self.loaded
        def mxm_builder(loaded, path): return ({}, {"mapping_sha256": "a" * 64})
        def features(loaded, fold, lyrics):
            self.assertEqual(set(fold), {"fold", "train", "validation"})
            calls["features"] += 1
            return ({"audio": [np.ones((1, 26), np.float32)] * 6,
                     "bpm": np.ones((6, 1), np.float32), "genre": np.ones((6, 1), np.float32),
                     "lyrics": np.ones((6, 5000), np.float32)}, {"fit_ids_hash": "fixture"})
        def trainer(train, val, ts, vs, edges, variant, seed, path, transform, provenance, **kwargs):
            self.assertEqual(variant, "D")
            self.assertEqual(kwargs["epochs"], 2)
            self.assertEqual(kwargs["batch_size"], 2)
            self.assertEqual((kwargs["bpm_dropout_probability"], kwargs["genre_dropout_probability"],
                              kwargs["lyrics_dropout_probability"]), (.1, .25, .2))
            self.assertEqual(kwargs["dropout_seed"], seed * 1000003 + 0 * 1009)
            self.assertEqual(ts.shape, (2, 2, 3))
            self.assertFalse(any("test" in str(x).lower() for x in train.track_ids + val.track_ids))
            calls["trainer"].append((variant, seed, kwargs["dropout_seed"]))
            if failure: raise RuntimeError("injected training failure")
            torch.save({"ok": True}, path)
            return {"history": [{"epoch": i, "train_triplet_loss": .7,
                                  "validation_triplet_loss": .6,
                                  "validation_NDCG@10": .15,
                                  "bpm_dropout_attempted_count": 12, "bpm_dropout_dropped_count": 1,
                                  "genre_dropout_attempted_count": 12, "genre_dropout_dropped_count": 3,
                                  "lyrics_dropout_attempted_count": 12, "lyrics_dropout_dropped_count": 2}
                                 for i in (1, 2)], "best_epoch": 2, "best_metrics": self.metrics()}
        return {"loader": loader, "mxm_builder": mxm_builder, "feature_builder": features,
                "trainer": trainer}, calls

    def test_frozen_config_namespace_and_config_hash(self):
        config = json.loads((self.runner.ROOT / self.runner.CONFIG_PATH).read_text())
        self.assertEqual(config["output_namespace"], "neurotune_v1_d2_modality_dropout")
        self.assertEqual(config["dropout_probabilities"], {"audio": 0.0, "bpm": .1, "genre": .25, "lyrics": .2})
        self.assertEqual(config["folds"], [0, 1, 2])
        self.assertEqual(config["dropout_rng"]["fold_seed_derivation"], "seed * 1000003 + fold * 1009")
        self.assertIn("scripts/run_neurotune_v1_full_modal_ablation.py", self.runner.CODE_PATHS)
        self.assertTrue(self.runner._validate_config(config))
        config["epochs"] = 1
        with self.assertRaises(ValueError): self.runner._validate_config(config)
        config = json.loads((self.runner.ROOT / self.runner.CONFIG_PATH).read_text())
        config["dropout_probabilities"]["audio"] = False
        with self.assertRaises(ValueError): self.runner._validate_config(config)
        config = json.loads((self.runner.ROOT / self.runner.CONFIG_PATH).read_text())
        config["folds"][1] = True
        with self.assertRaises(ValueError): self.runner._validate_config(config)

    def test_rejects_mismatched_d0_schedule_reference_hash(self):
        self.write_d0_report(schedule_hash="0" * 64)
        deps, _ = self.deps()
        with self.assertRaisesRegex(ValueError, "schedule"):
            self.runner._run(self.data, self.results, _test_only=True, _dependencies=deps,
                _protocol={"folds": (0,), "seeds": (684,), "epochs": 2, "batch_size": 2})

    def test_rejects_bad_d0_checkpoint_hash_or_path(self):
        for overrides in ({"checkpoint_hash": "0" * 64}, {"checkpoint_path": "bad/path.pt"}):
            with self.subTest(overrides=overrides):
                self.write_d0_report(**overrides)
                deps, _ = self.deps()
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    self.runner._run(self.data, self.results, _test_only=True, _dependencies=deps,
                        _protocol={"folds": (0,), "seeds": (684,), "epochs": 2, "batch_size": 2})

    def test_rejects_duplicate_d0_cosine_reference(self):
        self.write_d0_report(duplicate_cosine=True)
        deps, _ = self.deps()
        with self.assertRaisesRegex(ValueError, "cosine"):
            self.runner._run(self.data, self.results, _test_only=True, _dependencies=deps,
                _protocol={"folds": (0,), "seeds": (684,), "epochs": 2, "batch_size": 2})

    def test_rejects_invalid_d0_selected_epoch_before_training(self):
        for selected_epoch in (True, 0, 51, None):
            with self.subTest(selected_epoch=selected_epoch):
                self.write_d0_report(selected_epoch=selected_epoch)
                deps, calls = self.deps()
                try:
                    with self.assertRaises(ValueError):
                        self.runner._run(self.data, self.results, _test_only=True, _dependencies=deps,
                            _protocol={"folds": (0,), "seeds": (684,), "epochs": 2, "batch_size": 2})
                    self.assertEqual(calls["trainer"], [])
                finally:
                    final = self.results / "neurotune_v1_d2_modality_dropout"
                    if final.exists(): shutil.rmtree(final)

    def test_validation_only_runner_copies_schedules_and_records_hashes(self):
        deps, calls = self.deps()
        report = self.runner._run(self.data, self.results, _test_only=True,
            _dependencies=deps, _protocol={"folds": (0,), "seeds": (684,), "epochs": 2, "batch_size": 2})
        out = self.results / "neurotune_v1_d2_modality_dropout"
        self.assertEqual(calls["features"], 1)
        self.assertEqual(calls["trainer"], [("D", 684, 684 * 1000003)])
        self.assertEqual((out / "triplet_schedules/fold_0_seed_684.npz").read_bytes(), self.schedule_bytes[(0, 684)])
        self.assertEqual(report["records"][0]["selected_epoch"], 2)
        ref = report["records"][0]["d0_model_reference"]
        self.assertEqual(ref["path"], "neurotune_v1_full_modal_ablation/checkpoints/fold_0_D_seed_684.pt")
        self.assertEqual(ref["sha256"], hashlib.sha256(self.d0_checkpoint.read_bytes()).hexdigest())
        self.assertIn("output_files", report["hashes"])
        self.assertEqual(report["test_evaluation_deferred"], True)
        with (out / "modality_dropout_diagnostics.csv").open() as stream:
            import csv
            diag = next(csv.DictReader(stream))
        self.assertEqual((diag["bpm_attempted"], diag["bpm_dropped"], diag["bpm_realized_rate"]), ("12", "1", str(1 / 12)))
        self.assertFalse(report["test_features_constructed"])
        self.assertFalse(report["test_embeddings_constructed"])
        self.assertFalse(report["test_relevance_constructed"])
        self.assertFalse(report["test_metrics_computed"])
        self.assertEqual(hashlib.sha256((out / "d2_config.json").read_bytes()).hexdigest(), report["hashes"]["config"])

    def test_existing_namespace_refused(self):
        final = self.results / "neurotune_v1_d2_modality_dropout"
        final.mkdir(parents=True)
        with self.assertRaises(FileExistsError): self.runner._run(self.data, self.results, _test_only=True)

    def test_failure_cleans_staging_and_publishes_nothing(self):
        deps, _ = self.deps(failure=True)
        with self.assertRaisesRegex(RuntimeError, "injected training failure"):
            self.runner._run(self.data, self.results, _test_only=True, _dependencies=deps,
                _protocol={"folds": (0,), "seeds": (684,), "epochs": 2, "batch_size": 2})
        self.assertFalse((self.results / "neurotune_v1_d2_modality_dropout").exists())
        self.assertFalse(list(self.results.glob(".neurotune_v1_d2_modality_dropout.stage-*")))

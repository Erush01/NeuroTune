import hashlib
import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src import neurotune_v1_triplet_diagnostics as diagnostics
from src.model_v1 import NeuroTuneV1Encoder


class TripletSamplingTests(unittest.TestCase):
    def test_epoch_sampling_is_deterministic_exact_and_pairable(self):
        schedule = np.arange(50 * 400 * 3, dtype=np.int32).reshape(50, 400, 3)
        first = diagnostics._sample_train_rows(schedule, fold=2, seed=685)
        second = diagnostics._sample_train_rows(schedule, fold=2, seed=685)
        self.assertTrue(np.array_equal(first["selected_indices"], second["selected_indices"]))
        self.assertEqual(first["selected_indices"].shape, (10_000, 2))
        self.assertEqual(first["selected_rows"], 10_000)
        self.assertEqual(first["total_rows"], 20_000)
        self.assertEqual(first["indices_sha256"], hashlib.sha256(
            first["selected_indices"].astype("<i8").tobytes()).hexdigest())
        for epoch in range(50):
            rows = first["selected_indices"][epoch * 200:(epoch + 1) * 200, 1]
            self.assertEqual(len(set(rows)), 200)

    def test_triplet_statistics_use_margin_and_named_sampled_negative(self):
        stats = diagnostics._triplet_statistics(
            np.asarray([0.2, 0.5, 0.8]), np.asarray([0.5, 0.6, 0.7]), margin=0.2)
        self.assertEqual(stats["triplet_count"], 3)
        self.assertEqual(stats["active_count"], 2)
        self.assertEqual(stats["easy_count"], 1)
        self.assertEqual(stats["semi_hard_count"], 1)
        self.assertEqual(stats["hard_count"], 1)
        self.assertAlmostEqual(stats["active_rate"], 2 / 3)
        with self.assertRaises(ValueError):
            diagnostics._triplet_statistics(np.array([]), np.array([]))

    def test_pair_sampling_shares_deterministic_unique_pairs(self):
        first = diagnostics._sample_track_pairs(["a", "b", "c", "d"], seed=123, count=5)
        again = diagnostics._sample_track_pairs(["a", "b", "c", "d"], seed=123, count=5)
        self.assertEqual(first, again)
        self.assertEqual(len(first["pairs"]), 5)
        self.assertEqual(len({tuple(pair) for pair in first["pairs"]}), 5)
        self.assertTrue(all(a < b for a, b in first["pairs"]))

    def test_summary_metadata_selects_each_models_own_checkpoint(self):
        references = {
            0: {(0, 684): {"checkpoint_path": "d0.pt", "selected_epoch": 7}},
            1: {(0, 684): {"checkpoint_path": "d4.pt", "selected_epoch": 19}},
        }
        self.assertEqual(diagnostics._summary_checkpoint_metadata("D0", references, (0, 684)),
                         {"checkpoint_path": "d0.pt", "checkpoint_epoch": 7})
        self.assertEqual(diagnostics._summary_checkpoint_metadata("D4", references, (0, 684)),
                         {"checkpoint_path": "d4.pt", "checkpoint_epoch": 19})

    def test_non_audio_transform_mismatch_is_rejected(self):
        base = {"fit_ids_hash": "a", "bpm_mean": [1], "bpm_std": [2], "genre_vocabulary": ["x"],
                "idf": [3], "split_hash": "b", "source_sha256": {"lyrics": "c", "audio_cache": "d"}}
        multi = {**base, "source_sha256": {"lyrics": "c", "audio_cache": "e", "d4_audio_cache": "f"},
                 "audio_mean": [99]}
        self.assertIsNone(diagnostics._assert_matching_non_audio_transforms(base, multi))
        for field, bad in (("idf", [4]), ("fit_ids_hash", "different")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostics._assert_matching_non_audio_transforms(base, {**multi, field: bad})

    def test_checkpoint_transform_manifest_and_provenance_hashes_are_required(self):
        manifest = {"split_hash": "synthetic"}
        transform_hash = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        reference = {"transform_sha256": transform_hash, "selected_epoch": 3}
        checkpoint = {"model_state_dict": {}, "transform_manifest": manifest, "variant": "D", "seed": 684,
                      "epoch": 3, "config": {"margin": 0.2}, "provenance": {"fold": 0, "seed": 684,
                      "variant": "D", "schedule_sha256": "schedule", "transform_sha256": transform_hash}}
        diagnostics._validate_checkpoint_identity(checkpoint, reference, 0, 684, "schedule")
        d4_checkpoint = {**checkpoint, "provenance": {k: v for k, v in checkpoint["provenance"].items()
                                                        if k != "transform_sha256"}}
        diagnostics._validate_checkpoint_identity(d4_checkpoint, reference, 0, 684, "schedule", model="D4")
        with self.assertRaises(ValueError):
            diagnostics._validate_checkpoint_identity(d4_checkpoint, reference, 0, 684, "schedule", model="D0")
        bad_manifest = {**checkpoint, "transform_manifest": {"split_hash": "changed"}}
        with self.assertRaises(ValueError):
            diagnostics._validate_checkpoint_identity(bad_manifest, reference, 0, 684, "schedule", model="D4")
        bad_provenance = {**checkpoint, "provenance": {**checkpoint["provenance"], "transform_sha256": "0" * 64}}
        with self.assertRaises(ValueError):
            diagnostics._validate_checkpoint_identity(bad_provenance, reference, 0, 684, "schedule", model="D4")

    def test_only_d0_features_code_hash_drift_is_allowed(self):
        frozen = {"audio_mean": [1.], "audio_std": [2.], "audio_fit_rows": 10,
                  "bpm_mean": 3., "bpm_std": 4., "idf": [5.], "genre_vocabulary": ["g"],
                  "fit_ids_hash": "fit", "split_hash": "split",
                  "source_sha256": {"audio_cache": "a", "lyrics": "l"},
                  "code_sha256": {"src/features_neurotune_v1.py": "a" * 64, "src/features.py": "b" * 64}}
        current = copy.deepcopy(frozen)
        current["code_sha256"]["src/features_neurotune_v1.py"] = "c" * 64
        reference_hash = hashlib.sha256(json.dumps(
            frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        allowed = diagnostics._validate_current_transform("D0", current, frozen, reference_hash)
        self.assertEqual(allowed["allowance_field"], "code_sha256.src/features_neurotune_v1.py")
        self.assertEqual(allowed["old_code_sha256"], "a" * 64)
        self.assertEqual(allowed["current_code_sha256"], "c" * 64)
        with self.assertRaises(ValueError):
            diagnostics._validate_current_transform("D4", current, frozen, reference_hash)
        for field, changed in (("audio_mean", [8.]), ("bpm_mean", 9.), ("idf", [7.]),
                               ("split_hash", "other"), ("fit_ids_hash", "other"),
                               ("source_sha256", {"audio_cache": "changed", "lyrics": "l"})):
            bad = copy.deepcopy(current)
            bad[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostics._validate_current_transform("D0", bad, frozen, reference_hash)
        bad_code = copy.deepcopy(current)
        bad_code["code_sha256"]["src/features.py"] = "d" * 64
        with self.assertRaises(ValueError):
            diagnostics._validate_current_transform("D0", bad_code, frozen, reference_hash)


class SyntheticRunTests(unittest.TestCase):
    @staticmethod
    def _features(ids, multi_window):
        n = len(ids)
        features = {"bpm": np.ones((n, 1), np.float32), "lyrics": np.ones((n, 5000), np.float32),
                    "genre": np.tile(np.asarray([[1, 0]], np.float32), (n, 1))}
        if multi_window:
            features["audio_windows"] = {track: [np.zeros((1, 26), np.float32) for _ in range(3)] for track in ids}
            features["window_valid"] = np.ones((n, 3), dtype=np.bool_)
        else:
            features["audio"] = [np.zeros((1, 26), np.float32) for _ in ids]
        return features

    def test_synthetic_run_pairs_schedules_isolates_test_and_publishes_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            results = Path(td) / "results"
            for key, rel in diagnostics.INPUTS.items():
                path = root / rel
                if key in {"cv_dir", "mxm_path"}:
                    path.mkdir(parents=True, exist_ok=True)
                    if key == "cv_dir":
                        (path / "manifest.json").write_text("synthetic manifest\n", encoding="utf-8")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("synthetic\n", encoding="utf-8")
            d0root = results / "neurotune_v1_full_modal_ablation"
            d4root = results / "neurotune_v1_d4_multi_window_audio"
            schedroot = results / "neurotune_v1_abcd_cv" / "triplet_schedules"
            d4schedroot = d4root / "triplet_schedules"
            (d0root / "checkpoints").mkdir(parents=True)
            (d4root / "checkpoints").mkdir(parents=True)
            schedroot.mkdir(parents=True)
            d4schedroot.mkdir(parents=True)
            references = {0: [], 1: []}
            copied_hashes = {}
            fold_rows = []
            transform0 = {"fit_ids_hash": "fit", "bpm_mean": [1.], "bpm_std": [1.],
                          "genre_vocabulary": ["g0", "g1"], "idf": [1.], "split_hash": "split",
                          "source_sha256": {"lyrics": "l", "audio_cache": "a"},
                          "code_sha256": {"src/features_neurotune_v1.py": "a" * 64,
                                          "src/features.py": "b" * 64}}
            current_transform0 = copy.deepcopy(transform0)
            current_transform0["code_sha256"]["src/features_neurotune_v1.py"] = "c" * 64
            transform4 = {**transform0, "source_sha256": {"lyrics": "l", "audio_cache": "different",
                                                             "d4_audio_cache": "d4"},
                          "audio_mean": [9.]}
            transform_hashes = [hashlib.sha256(json.dumps(t, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                                for t in (transform0, transform4)]
            for fold in diagnostics.FOLDS:
                train = [f"tr{fold}_{i}" for i in range(150)]
                validation = [f"va{fold}_{i}" for i in range(150)]
                test = [f"te{fold}_{i}" for i in range(2)]
                fold_rows.append({"fold": fold, "train": train, "validation": validation, "test": test})
                for seed in diagnostics.SEEDS:
                    schedule_path = schedroot / f"fold_{fold}_seed_{seed}.npz"
                    rng = np.random.default_rng(fold * 1000 + seed)
                    train_schedule = rng.integers(0, 150, size=(50, 200, 3), dtype=np.int32)
                    validation_schedule = rng.integers(0, 150, size=(1, 1500, 3), dtype=np.int32)
                    np.savez(schedule_path, train_ids=np.asarray(train), validation_ids=np.asarray(validation),
                             train_schedule=train_schedule, validation_schedule=validation_schedule)
                    (d4schedroot / schedule_path.name).write_bytes(schedule_path.read_bytes())
                    schedule_hash = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
                    copied_hashes[f"{fold}/{seed}"] = schedule_hash
                    for model_index, (model_name, base) in enumerate((("D0", d0root), ("D4", d4root))):
                        checkpoint_rel = f"checkpoints/fold_{fold}_D_seed_{seed}.pt"
                        checkpoint_path = base / checkpoint_rel
                        model_transform = (transform0, transform4)[model_index]
                        checkpoint = {"model_state_dict": NeuroTuneV1Encoder("D", 2).state_dict(),
                                      "variant": "D", "seed": seed, "epoch": 4 + model_index,
                                      "config": {"margin": 0.2}, "transform_manifest": model_transform,
                                      "provenance": {"fold": fold, "seed": seed, "variant": "D",
                                      "schedule_sha256": schedule_hash,
                                      **({"transform_sha256": transform_hashes[model_index]} if model_index == 0 else {})}}
                        torch.save(checkpoint, checkpoint_path)
                        references[model_index].append({"fold": fold, "seed": seed, "variant": "D",
                            "selected_epoch": 4 + model_index, "checkpoint_path": checkpoint_rel,
                            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                            "schedule_sha256": schedule_hash, "transform_sha256": transform_hashes[model_index]})
            d0_data_keys = ("audit_path", "catalog_path", "train_edges_path", "cache_path", "cache_metadata_path",
                            "rejection_path", "ab_catalog_path", "ab_split_path")
            d0_report = {"status": "complete", "model_validation": references[0],
                         "hashes": {"data_sources": {key: diagnostics._hash_path(root / diagnostics.INPUTS[key])
                                                      for key in d0_data_keys},
                                    "split_manifest": diagnostics._hash_path(root / diagnostics.INPUTS["cv_dir"] / "manifest.json")}}
            d0_report_path = d0root / "experiment.json"
            d0_report_path.write_text(json.dumps(d0_report), encoding="utf-8")
            d4_report_path = d4root / "experiment.json"
            protected_before = {"d0_report": diagnostics._hash_path(d0_report_path)}
            protected_before.update({f"input:{key}": diagnostics._hash_path(root / rel)
                                     for key, rel in diagnostics.INPUTS.items()})
            protected_before.update({f"checkpoint:{fold}/{seed}": diagnostics._hash_path(
                d0root / "checkpoints" / f"fold_{fold}_D_seed_{seed}.pt")
                for fold in diagnostics.FOLDS for seed in diagnostics.SEEDS})
            protected_before.update({f"schedule:{fold}/{seed}": copied_hashes[f"{fold}/{seed}"]
                                     for fold in diagnostics.FOLDS for seed in diagnostics.SEEDS})
            d4_report = {"status": "complete",
                "evaluation_scope": "validation_only_development", "independent_test_artifacts_created": False,
                "model_validation": references[1], "hashes": {"schedule_copies": copied_hashes,
                                                                  "protected_before": protected_before}}
            d4_report_path.write_text(json.dumps(d4_report), encoding="utf-8")

            def loader(**_kwargs):
                return {"manifest": {}, "catalog": {}, "cache_path": root / "cache.h5", "folds": fold_rows}

            def mxm_builder(*_args):
                return {}, {"synthetic": True}

            seen_feature_ids, seen_embedding_ids = [], []
            def feature_builder(_loaded, fold, _lyrics):
                self.assertEqual(set(fold), {"fold", "train", "validation"})
                ids = sorted(fold["train"] + fold["validation"])
                self.assertTrue(set(ids).isdisjoint(fold_rows[fold["fold"]]["test"]))
                seen_feature_ids.extend(ids)
                return self._features(ids, False), current_transform0

            def d4_builder(_loaded, fold, _lyrics, _cache):
                self.assertEqual(set(fold), {"fold", "train", "validation"})
                ids = sorted(fold["train"] + fold["validation"])
                self.assertTrue(set(ids).isdisjoint(fold_rows[fold["fold"]]["test"]))
                seen_feature_ids.extend(ids)
                return self._features(ids, True), transform4

            def embedder(_model, dataset):
                self.assertTrue(all(not track.startswith("te") for track in dataset.track_ids))
                seen_embedding_ids.extend(dataset.track_ids)
                values = np.zeros((len(dataset), 64), dtype=np.float32)
                for i in range(len(dataset)):
                    values[i, i % 64] = 1.0
                return values

            deps = {"loader": loader, "mxm_builder": mxm_builder, "d0_feature_builder": feature_builder,
                    "d4_feature_builder": d4_builder, "embedder": embedder}
            d4_report_path.write_text(json.dumps({**d4_report, "evaluation_scope": "test"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                diagnostics._run(root, results, _test_only=True, _dependencies=deps)
            mismatched_d4 = {**d4_report, "hashes": {**d4_report["hashes"],
                "protected_before": {**protected_before, "input:audit_path": "0" * 64}}}
            d4_report_path.write_text(json.dumps(mismatched_d4), encoding="utf-8")
            with self.assertRaises(ValueError):
                diagnostics._run(root, results, _test_only=True, _dependencies=deps)
            d4_report_path.write_text(json.dumps(d4_report), encoding="utf-8")
            bad_d0_report = {**d0_report, "hashes": {**d0_report["hashes"],
                "data_sources": {**d0_report["hashes"]["data_sources"], "catalog_path": "0" * 64}}}
            d0_report_path.write_text(json.dumps(bad_d0_report), encoding="utf-8")
            with self.assertRaises(ValueError):
                diagnostics._run(root, results, _test_only=True, _dependencies=deps)
            d0_report_path.write_text(json.dumps(d0_report), encoding="utf-8")
            d4_copy_path = d4schedroot / "fold_0_seed_684.npz"
            source_copy_bytes = d4_copy_path.read_bytes()
            d4_copy_path.write_bytes(source_copy_bytes + b"changed")
            with self.assertRaises(ValueError):
                diagnostics._run(root, results, _test_only=True, _dependencies=deps)
            d4_copy_path.write_bytes(source_copy_bytes)
            report = diagnostics._run(root, results, _test_only=True, _dependencies=deps)
            output = results / "neurotune_v1_d4_triplet_diagnostics"
            with (output / "triplet_summary_by_fold_seed.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 36)
            self.assertEqual(len(seen_feature_ids), 3 * 300 * 2)
            self.assertEqual(len(seen_embedding_ids), 3 * 3 * 300 * 2)
            for row in rows:
                model_index = 0 if row["model"] == "D0" else 1
                ref = references[model_index][int(row["fold"]) * 3 + diagnostics.SEEDS.index(int(row["seed"]))]
                self.assertEqual(row["checkpoint_path"], ref["checkpoint_path"])
                self.assertEqual(int(row["checkpoint_epoch"]), ref["selected_epoch"])
                self.assertEqual(row["schedule_sha256"], ref["schedule_sha256"])
                self.assertEqual(row["schedule_source_sha256"], row["schedule_copy_sha256"])
            self.assertEqual(report["status"], "complete")
            provenance_by_model = {row["model"]: row for row in report["transform_provenance"]
                                   if row["fold"] == 0 and row["seed"] == 684}
            self.assertEqual(provenance_by_model["D0"]["allowance_field"],
                             "code_sha256.src/features_neurotune_v1.py")
            self.assertEqual(provenance_by_model["D0"]["old_code_sha256"], "a" * 64)
            self.assertEqual(provenance_by_model["D0"]["current_code_sha256"], "c" * 64)
            self.assertIsNone(provenance_by_model["D4"]["allowance_field"])
            analysis = (output / "analysis.md").read_text(encoding="utf-8")
            for text in ("Saved D0 schedules are reused as-is",
                         "official/legacy test-source positives may remain exclusion-only",
                         "does not rebuild schedules, create test targets, or change that policy",
                         "`sampled_negative`"):
                self.assertIn(text, report["caveat"])
                self.assertIn(text, analysis)
            with self.assertRaises(FileExistsError):
                diagnostics._run(root, results, _test_only=True, _dependencies=deps)


if __name__ == "__main__":
    unittest.main()

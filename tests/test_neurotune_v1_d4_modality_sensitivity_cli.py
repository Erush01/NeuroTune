import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.evaluate_d4_modality_sensitivity import (
    _assert_original_parity, _mask_batch, _paired_rows, _validate_checkpoint_transform_identity,
    _validate_d0_transform_manifest,
)
from src import evaluate_d4_modality_sensitivity as runner
from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
from src.evaluate_d_modality_sensitivity import measure_behavior_change
from src.model_v1 import NeuroTuneV1Encoder


class D4SensitivityContractTests(unittest.TestCase):
    def test_audio_mask_zeros_values_only_and_never_mutates_source(self):
        batch = {
            "audio_windows": torch.tensor([[[[1.0], [2.0]], [[3.0], [0.0]]]]),
            "mask": torch.tensor([[[True, True], [True, False]]]),
            "window_mask": torch.tensor([[True, True]]),
            "lengths": torch.tensor([[2, 1]]),
            "track_ids": ["v1"], "bpm": torch.tensor([[1.0]]),
            "genre": torch.tensor([[2.0]]), "lyrics": torch.tensor([[3.0]]),
        }
        before = copy.deepcopy(batch)
        masked = _mask_batch(batch, "audio")
        self.assertEqual(set(masked), set(batch))
        self.assertTrue(torch.equal(masked["audio_windows"], torch.zeros_like(batch["audio_windows"])))
        for key in ("mask", "window_mask", "lengths", "bpm", "genre", "lyrics"):
            self.assertTrue(torch.equal(masked[key], batch[key]), key)
        self.assertEqual(masked["track_ids"], batch["track_ids"])
        for key in batch:
            if torch.is_tensor(batch[key]):
                self.assertTrue(torch.equal(batch[key], before[key]), key)

    def test_non_audio_mask_only_zeros_selected_tensor(self):
        batch = {"audio_windows": torch.ones((1, 3, 2, 26)),
                 "mask": torch.ones((1, 3, 2), dtype=torch.bool),
                 "window_mask": torch.ones((1, 3), dtype=torch.bool),
                 "lengths": torch.ones((1, 3), dtype=torch.long),
                 "track_ids": ["v1"], "bpm": torch.ones((1, 1)),
                 "genre": torch.ones((1, 2)), "lyrics": torch.ones((1, 4))}
        masked = _mask_batch(batch, "genre")
        self.assertEqual(float(masked["genre"].sum()), 0.0)
        for key in ("audio_windows", "mask", "window_mask", "lengths", "bpm", "lyrics"):
            self.assertTrue(torch.equal(masked[key], batch[key]), key)

    def test_parity_requires_four_metrics_and_both_counts(self):
        expected = {"Recall@10": .1, "HitRate@10": .2, "MRR@10": .3,
                    "NDCG@10": .4, "query_count": 8, "eligible_positive_edges": 9}
        _assert_original_parity(expected, expected)
        for field in expected:
            changed = dict(expected, **{field: expected[field] + 1})
            with self.subTest(field=field), self.assertRaises(ValueError):
                _assert_original_parity(changed, expected)

    def test_pairing_requires_exact_fold_seed_conditions_and_computes_deltas(self):
        d0 = [{"fold": fold, "seed": seed, "condition": condition, "delta_ndcg": .4,
               "mean_top_10_overlap": .8, "mean_embedding_cosine_shift": .1}
              for fold in (0, 1, 2) for seed in (684, 685, 686)
              for condition in ("audio_zero", "bpm_zero", "genre_zero", "lyrics_zero")]
        d4 = [{**row, "delta_ndcg": .3} for row in d0]
        rows = _paired_rows(d0, d4)
        self.assertEqual(len(rows), 36)
        self.assertAlmostEqual(rows[0]["delta_ndcg_D4_minus_D0"], -.1)
        self.assertAlmostEqual(rows[0]["relative_delta_ndcg_change_percent"], -25.0)
        with self.assertRaises(ValueError):
            _paired_rows(d0, [*d4[:-1], {**d4[-1], "seed": 685}])

    def test_relative_change_is_null_at_zero_d0_delta(self):
        rows = [{"fold": f, "seed": s, "condition": c, "delta_ndcg": 0.0,
                 "mean_top_10_overlap": .5, "mean_embedding_cosine_shift": .2}
                for f in (0, 1, 2) for s in (684, 685, 686)
                for c in ("audio_zero", "bpm_zero", "genre_zero", "lyrics_zero")]
        paired = _paired_rows(rows, [{**row, "delta_ndcg": .1} for row in rows])
        self.assertIsNone(paired[0]["relative_delta_ndcg_change_percent"])

    def test_d0_current_transform_allows_only_the_d4_added_feature_code_hash_drift(self):
        frozen = {"audio_mean": [1.0], "audio_std": [2.0], "audio_fit_rows": 10,
                  "idf": [3.0], "fit_ids_hash": "a" * 64, "split_hash": "b" * 64,
                  "source_sha256": {"audio_cache": "c" * 64},
                  "code_sha256": {"src/features_neurotune_v1.py": "d" * 64,
                                  "src/features.py": "e" * 64}}
        current = copy.deepcopy(frozen)
        current["code_sha256"]["src/features_neurotune_v1.py"] = "f" * 64
        drift = _validate_d0_transform_manifest(current, frozen)
        self.assertEqual(drift["checkpoint_sha256"], "d" * 64)
        self.assertEqual(drift["current_sha256"], "f" * 64)
        mutations = (
            lambda value: value.update(audio_mean=[9.0]),
            lambda value: value.update(idf=[9.0]),
            lambda value: value["source_sha256"].update(audio_cache="9" * 64),
            lambda value: value.update(split_hash="9" * 64),
            lambda value: value["code_sha256"].update({"src/features.py": "9" * 64}),
        )
        for mutate in mutations:
            changed = copy.deepcopy(current)
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _validate_d0_transform_manifest(changed, frozen)

    def test_d4_checkpoint_may_omit_transform_provenance_but_manifest_stays_bound(self):
        manifest = {"stats": [1.0], "code_sha256": {"src/features.py": "a" * 64}}
        digest = runner._digest(manifest)
        reference = {"transform_sha256": digest}
        checkpoint = {"transform_manifest": copy.deepcopy(manifest), "provenance": {}}
        self.assertTrue(_validate_checkpoint_transform_identity(checkpoint, "D4", reference, manifest))
        bad_manifest = copy.deepcopy(checkpoint)
        bad_manifest["transform_manifest"]["stats"] = [2.0]
        with self.assertRaises(ValueError):
            _validate_checkpoint_transform_identity(bad_manifest, "D4", reference, manifest)
        wrong_provenance = copy.deepcopy(checkpoint)
        wrong_provenance["provenance"]["transform_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            _validate_checkpoint_transform_identity(wrong_provenance, "D4", reference, manifest)
        with self.assertRaises(ValueError):
            _validate_checkpoint_transform_identity(checkpoint, "D0", reference, manifest)

    def test_synthetic_paired_run_is_validation_only_and_publishes_exact_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data_root, results_root = base / "data", base / "results"
            inputs = {**runner.D0_INPUTS, **runner.D4_INPUTS}
            paths = {key: data_root / value for key, value in inputs.items()}
            for path in set(paths.values()):
                if path.suffix == "" or path.name in {"cv", "mxm_counts"}:
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("synthetic-input\n", encoding="utf-8")

            folds = []
            features_by_fold = {}
            transform_by_fold = {}
            d4_transform_by_fold = {}
            edges_by_fold = {}
            for fold in runner.FOLDS:
                train_ids = [f"tr{fold}a", f"tr{fold}b"]
                validation_ids = [f"va{fold}a", f"va{fold}b"]
                test_ids = [f"te{fold}"]
                edges = [(validation_ids[0], validation_ids[1])]
                folds.append({"fold": fold, "train": train_ids, "validation": validation_ids,
                              "test": test_ids, "validation_edges": edges, "train_edges": []})
                ids = sorted(train_ids + validation_ids)
                features_by_fold[fold] = {
                    "audio": [np.full((1, 26), .1 * i + .1, dtype=np.float32) for i, _ in enumerate(ids)],
                    "bpm": np.asarray([[.1 * i] for i in range(len(ids))], dtype=np.float32),
                    "genre": np.asarray([[float(i % 2), float((i + 1) % 2)] for i in range(len(ids))], dtype=np.float32),
                    "lyrics": np.zeros((len(ids), 5000), dtype=np.float32),
                }
                features_by_fold[fold]["lyrics"][:, fold] = 1.0
                common_transform = {"fit_ids_hash": runner._digest(train_ids), "audio_mean": [0.0],
                                    "audio_std": [1.0], "audio_fit_rows": len(train_ids), "bpm_mean": 0.0,
                                    "bpm_std": 1.0, "genre_vocabulary": ["g0", "g1"],
                                    "split_hash": "e" * 64,
                                    "code_sha256": {"src/features_neurotune_v1.py": "d" * 64,
                                                    "src/features.py": "e" * 64},
                                    "source_sha256": {"audio_cache": "a" * 64,
                                                       "catalog_bpm_genre": "c" * 64,
                                                       "lyrics": "d" * 64, "split_manifest": "e" * 64}}
                transform_by_fold[fold] = common_transform
                d4_sources = {k: v for k, v in common_transform["source_sha256"].items() if k != "audio_cache"}
                d4_transform_by_fold[fold] = {**common_transform, "audio_mean": [1.0],
                    "audio_std": [2.0], "audio_fit_rows": len(train_ids),
                    "source_sha256": {**d4_sources, "d4_audio_cache": "b" * 64}}
                edges_by_fold[fold] = edges

            torch.manual_seed(17)
            model = NeuroTuneV1Encoder("D", 2)
            state = model.state_dict()
            schedule_hashes, d0_checkpoint_hashes, d4_checkpoint_hashes = {}, {}, {}
            schedule_root = results_root / "neurotune_v1_abcd_cv" / "triplet_schedules"
            d0_root = results_root / "neurotune_v1_full_modal_ablation"
            d4_root = results_root / "neurotune_v1_d4_multi_window_audio"
            for namespace in runner.PROTECTED_RESULT_NAMESPACES:
                (results_root / namespace).mkdir(parents=True, exist_ok=True)
            for target in (d0_root / "checkpoints", d4_root / "checkpoints", schedule_root):
                target.mkdir(parents=True, exist_ok=True)
            references0, references4 = [], []
            schedule_copy_hashes = {}

            def expected_validation(fold, is_d4):
                ids = folds[fold]["validation"]
                if is_d4:
                    features = {"audio_windows": {track: [np.full((1, 26), .2, dtype=np.float32)
                                                             for _ in range(3)] for track in ids},
                                "window_valid": np.ones((len(ids), 3), dtype=np.bool_),
                                **{k: v[2:] for k, v in features_by_fold[fold].items() if k != "audio"}}
                else:
                    features = {"audio": features_by_fold[fold]["audio"][2:],
                                **{k: v[2:] for k, v in features_by_fold[fold].items() if k != "audio"}}
                dataset = NeuroTuneV1Dataset(ids, features)
                batch = neurotune_v1_collate([dataset[i] for i in range(len(dataset))])
                instance = NeuroTuneV1Encoder("D", 2)
                instance.load_state_dict(state)
                instance.eval()
                embedding = runner._embed(instance, batch, is_d4)
                metrics = measure_behavior_change(embedding, embedding, ids, edges_by_fold[fold])
                retrieval = metrics["retrieval_metrics"]
                return {**{key: retrieval[key] for key in runner.METRICS},
                        "query_count": retrieval["query_count"],
                        "eligible_positive_edges": metrics["unique_undirected_edge_count"]}

            for fold in runner.FOLDS:
                val = expected_validation(fold, False)
                val4 = expected_validation(fold, True)
                for seed in runner.SEEDS:
                    schedule = schedule_root / f"fold_{fold}_seed_{seed}.npz"
                    np.savez(schedule, train_ids=np.asarray(folds[fold]["train"]),
                             validation_ids=np.asarray(folds[fold]["validation"]),
                             train_schedule=np.zeros((1, 1, 3), dtype=np.int32),
                             validation_schedule=np.zeros((1, 1, 3), dtype=np.int32))
                    schedule_hash = hashlib.sha256(schedule.read_bytes()).hexdigest()
                    schedule_hashes[(fold, seed)] = schedule_hash
                    schedule_copy = d4_root / "triplet_schedules" / schedule.name
                    schedule_copy.parent.mkdir(parents=True, exist_ok=True)
                    schedule_copy.write_bytes(schedule.read_bytes())
                    schedule_copy_hashes[f"{fold}/{seed}"] = hashlib.sha256(schedule_copy.read_bytes()).hexdigest()
                    for variant, root, transform, expected, references, cp_hashes in (
                            ("D0", d0_root, transform_by_fold[fold], val, references0, d0_checkpoint_hashes),
                            ("D4", d4_root, d4_transform_by_fold[fold], val4, references4, d4_checkpoint_hashes)):
                        buffer = io.BytesIO()
                        torch.save({"model_state_dict": state, "epoch": 1, "variant": "D", "fold": fold,
                                    "seed": seed, "transform_manifest": transform,
                                    "selection_metrics": expected,
                                    "provenance": {"fold": fold, "seed": seed, "variant": "D",
                                                   "schedule_sha256": schedule_hash,
                                                   "transform_sha256": runner._digest(transform)}}, buffer)
                        cp = root / "checkpoints" / f"fold_{fold}_D_seed_{seed}.pt"
                        cp.write_bytes(buffer.getvalue())
                        cp_hash = hashlib.sha256(cp.read_bytes()).hexdigest()
                        cp_hashes[(fold, seed)] = cp_hash
                        references.append({"fold": fold, "seed": seed, "variant": "D", "selected_epoch": 1,
                                           "validation": expected,
                                           "checkpoint_path": f"checkpoints/fold_{fold}_D_seed_{seed}.pt",
                                           "checkpoint_sha256": cp_hash, "schedule_sha256": schedule_hash,
                                           "transform_sha256": runner._digest(transform)})

            before_inputs = {f"input:{key}": runner._hash_path(path) for key, path in paths.items()}
            d0_sources = {key: before_inputs[f"input:{key}"] for key in runner.D0_INPUTS
                          if key not in ("test_edges_path", "ab_test_edges_path", "mxm_path", "cv_dir")}
            d0_report = {"status": "complete", "hashes": {"data_sources": d0_sources},
                         "model_validation": references0}
            d0_report_path = d0_root / "experiment.json"
            d0_report_path.write_text(json.dumps(d0_report), encoding="utf-8")
            protected_before = {**before_inputs, "d0_report": runner._hash_path(d0_report_path)}
            protected_before.update({f"checkpoint:{f}/{s}": d0_checkpoint_hashes[(f, s)]
                                     for f in runner.FOLDS for s in runner.SEEDS})
            protected_before.update({f"schedule:{f}/{s}": schedule_hashes[(f, s)]
                                     for f in runner.FOLDS for s in runner.SEEDS})
            d4_report = {"status": "complete", "evaluation_scope": "validation_only_development",
                         "independent_test_artifacts_created": False,
                         "hashes": {"protected_before": protected_before,
                                    "schedule_copies": schedule_copy_hashes},
                         "model_validation": references4}
            (d4_root / "experiment.json").write_text(json.dumps(d4_report), encoding="utf-8")

            builder_ids, eval_calls = [], []
            def loader(**_kwargs):
                return {"manifest": {}, "catalog": {}, "cache_path": paths["cache_path"], "folds": folds}
            def mxm_builder(*_args):
                return {}, {"synthetic": True}
            def d0_builder(_loaded, fold, _lyrics):
                self.assertEqual(set(fold), {"fold", "train", "validation"})
                ids = set(fold["train"] + fold["validation"])
                self.assertTrue(ids.isdisjoint(f"te{fold['fold']}" for _ in range(1)))
                builder_ids.append(ids)
                return features_by_fold[fold["fold"]], transform_by_fold[fold["fold"]]
            def d4_builder(_loaded, fold, _lyrics, _cache):
                self.assertEqual(set(fold), {"fold", "train", "validation"})
                ids = set(fold["train"] + fold["validation"])
                self.assertTrue(ids.isdisjoint({f"te{fold['fold']}"}))
                builder_ids.append(ids)
                ordered = sorted(ids)
                return {"audio_windows": {track: [np.full((1, 26), .2, dtype=np.float32)
                                                     for _ in range(3)] for track in ordered},
                        "window_valid": np.ones((len(ordered), 3), dtype=np.bool_),
                        **{k: v for k, v in features_by_fold[fold["fold"]].items() if k != "audio"}}, \
                       d4_transform_by_fold[fold["fold"]]
            original_evaluate = runner._evaluate
            def traced_evaluate(model_arg, feature_arg, ids, edges, is_d4, expected):
                eval_calls.append((is_d4, tuple(ids), tuple(edges)))
                return original_evaluate(model_arg, feature_arg, ids, edges, is_d4, expected)
            with mock.patch.object(runner, "_evaluate", side_effect=traced_evaluate):
                damaged_copy = d4_root / "triplet_schedules" / "fold_0_seed_684.npz"
                original_copy_bytes = damaged_copy.read_bytes()
                with np.load(io.BytesIO(original_copy_bytes), allow_pickle=False) as archive:
                    altered = {key: archive[key].copy() for key in archive.files}
                altered["train_schedule"][0, 0, 0] = 1
                np.savez(damaged_copy, **altered)
                with self.assertRaises(ValueError):
                    runner._run(data_root, results_root, _test_only=True,
                        _dependencies={"loader": loader, "mxm_builder": mxm_builder,
                                       "d0_builder": d0_builder, "d4_builder": d4_builder})
                damaged_copy.write_bytes(original_copy_bytes)
                builder_ids.clear()
                eval_calls.clear()
                report = runner._run(data_root, results_root, _test_only=True,
                    _dependencies={"loader": loader, "mxm_builder": mxm_builder,
                                   "d0_builder": d0_builder, "d4_builder": d4_builder})
            self.assertEqual(len(builder_ids), 6)
            self.assertEqual(len(eval_calls), 18)
            for fold in runner.FOLDS:
                paired_calls = [call for call in eval_calls if set(call[1]) == set(folds[fold]["validation"])]
                self.assertEqual(len(paired_calls), 6)
                self.assertEqual(len({call[1] for call in paired_calls}), 1)
                self.assertEqual(len({call[2] for call in paired_calls}), 1)
            output = results_root / runner.OUTPUT_NAME
            self.assertEqual(report["counts"]["D0_condition_records"], 45)
            self.assertEqual(report["counts"]["D4_condition_records"], 45)
            self.assertEqual(report["counts"]["paired_modality_rows"], 36)
            protected = report["hashes"]["protected_result_namespaces"]
            self.assertEqual(set(protected), set(runner.PROTECTED_RESULT_NAMESPACES))
            self.assertTrue(all(value["before"] == value["after"] for value in protected.values()))
            self.assertIn("inherited negative-candidate exclusion policy", report["inherited_training_boundary"])
            self.assertIn("created no test targets", report["inherited_training_boundary"])
            self.assertIn("did not change that policy", (output / "README.md").read_text(encoding="utf-8"))
            self.assertEqual(len((output / "condition_records.csv").read_text().splitlines()), 91)
            self.assertEqual(len((output / "paired_d0_d4_comparison.csv").read_text().splitlines()), 37)
            with self.assertRaises(FileExistsError):
                runner._run(data_root, results_root, _test_only=True, _dependencies={"loader": loader})


if __name__ == "__main__":
    unittest.main()

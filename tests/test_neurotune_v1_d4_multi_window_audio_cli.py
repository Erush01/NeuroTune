import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scripts import run_neurotune_v1_d4_multi_window_audio as runner


class D4ConfigTests(unittest.TestCase):
    def test_checked_in_config_is_frozen_protocol(self):
        self.assertTrue(runner._validate_config(runner._load_config()))
        config = runner._load_config()
        self.assertEqual(config["folds"], [0, 1, 2])
        self.assertEqual(config["seeds"], [684, 685, 686])
        self.assertEqual(config["epochs"], 50)
        self.assertEqual(config["batch_size"], 64)
        self.assertEqual(config["optimizer"], "Adam")
        self.assertEqual(config["learning_rate"], 0.001)
        self.assertEqual(config["margin"], 0.2)
        self.assertEqual(config["audio_windows"], ["start", "middle", "end"])
        self.assertEqual(config["window_formula"],
                         "start=0; middle=max(0,(duration-30)/2); end=max(0,duration-30)")
        self.assertEqual(config["window_interval"], "[start,start+30)")
        self.assertEqual(config["channel_count"], 26)
        self.assertEqual(config["feature_cache_dtype"], "float32")
        self.assertEqual(config["feature_cache_normalization"], "raw/unnormalized; deferred until train split")
        self.assertEqual(config["pooling"], "mean_valid_raw_projected_64d_before_fusion")
        self.assertEqual(config["output_namespaces"], {
            "pilot": "neurotune_v1_d4_multi_window_audio_pilot",
            "full": "neurotune_v1_d4_multi_window_audio",
        })

    def test_rejects_any_protocol_drift_extra_field_or_boolean(self):
        config = runner._load_config()
        for mutate in (
            lambda c: c.update(epochs=True),
            lambda c: c.update(epochs=49),
            lambda c: c.update(unapproved=True),
            lambda c: c["dropout"]["probabilities"].update(audio=True),
        ):
            changed = copy.deepcopy(config)
            mutate(changed)
            with self.assertRaises(ValueError):
                runner._validate_config(changed)


class D4RunnerContractTests(unittest.TestCase):
    def test_scopes_are_fixed_and_distinct(self):
        self.assertEqual(runner._scope(True), ((0, 684),))
        self.assertEqual(runner._scope(False), tuple((f, s) for f in (0, 1, 2) for s in (684, 685, 686)))

    def test_schedule_hash_and_exact_fold_ids_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.npz"
            np.savez(path, train_ids=np.asarray(["t1", "t2"]), validation_ids=np.asarray(["v1"]),
                     train_schedule=np.zeros((50, 1, 3), dtype=np.int32),
                     validation_schedule=np.zeros((1, 1, 3), dtype=np.int32))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fold = {"train": ["t1", "t2"], "validation": ["v1"]}
            schedule, raw = runner._schedule(path, fold, (0, 684), digest)
            self.assertEqual(len(raw), path.stat().st_size)
            self.assertEqual(schedule["train_schedule"].shape, (50, 1, 3))
            with self.assertRaises(ValueError):
                runner._schedule(path, dict(fold, train=["t2", "t1"]), (0, 684), digest)
            with self.assertRaises(ValueError):
                runner._schedule(path, fold, (0, 684), "0" * 64)

    def test_d4_cache_sidecar_and_cohort_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {key: root / (key + (".h5" if key.endswith("cache_path") else ".json"))
                     for key in ("d4_metadata_path", "d4_rejection_path", "audit_path", "catalog_path",
                                 "d4_cache_path", "cache_path", "cache_metadata_path")}
            paths["catalog_path"].write_text("catalog", encoding="utf-8")
            channels = [*(f"pitch_{i}" for i in range(12)), *(f"timbre_{i}" for i in range(12)),
                        "loudness", "duration"]
            paths["cache_metadata_path"].write_text(json.dumps({"channels": channels}), encoding="utf-8")
            d4_source = {"d4_audio": "b" * 64}
            audit = {"source_sha256": d4_source}
            paths["audit_path"].write_text(json.dumps(audit), encoding="utf-8")
            cohort = ["a", "b"]
            code_hashes = {key: hashlib.sha256((runner.ROOT / key).read_bytes()).hexdigest()
                           for key in runner.D4_CACHE_CODE_PATHS}
            audit_hash = hashlib.sha256(paths["audit_path"].read_bytes()).hexdigest()
            catalog_hash = hashlib.sha256(paths["catalog_path"].read_bytes()).hexdigest()
            meta = {"status": "complete", "window_seconds": 30,
                    "window_formula": "start=0; middle=max(0,(duration-30)/2); end=max(0,duration-30)",
                    "interval": "[start,start+30)", "channel_count": 26, "channels": channels,
                    "dtype": "float32", "normalization": "raw/unnormalized; deferred until train split",
                    "track_count": 2, "rejected_track_count": 0, "valid_track_ids": cohort, "rejected": [],
                    "audit_sha256": audit_hash, "catalog_sha256": catalog_hash,
                    "source_sha256": d4_source, "code_sha256": code_hashes}
            with h5py.File(paths["cache_path"], "w") as h5:
                h5.create_dataset("track_ids", data=np.asarray(cohort, dtype="S1"))
            with h5py.File(paths["d4_cache_path"], "w") as h5:
                h5.create_dataset("track_ids", data=np.asarray(cohort, dtype="S1"))
                h5.create_dataset("features", data=np.zeros((2, 26), dtype=np.float32))
                for key, value in {"window_seconds": 30, "window_formula": meta["window_formula"],
                                   "interval": meta["interval"], "channels": channels,
                                   "normalization": meta["normalization"], "audit_sha256": audit_hash,
                                   "catalog_sha256": catalog_hash, "source_sha256": d4_source,
                                   "code_sha256": code_hashes}.items():
                    h5.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
            paths["d4_rejection_path"].write_text("[]", encoding="utf-8")
            paths["d4_metadata_path"].write_text(json.dumps(meta), encoding="utf-8")
            with mock.patch.object(runner, "EXPECTED_TRACK_COUNT", 2):
                self.assertEqual(runner._validate_d4_cache(paths), cohort)
                for field, bad_value in (("window_formula", "wrong"), ("interval", "[0,30)"),
                                         ("dtype", "float64"), ("channel_count", 25)):
                    metadata = dict(meta); metadata[field] = bad_value
                    paths["d4_metadata_path"].write_text(json.dumps(metadata), encoding="utf-8")
                    with self.assertRaises(ValueError, msg=field):
                        runner._validate_d4_cache(paths)
                paths["d4_metadata_path"].write_text(json.dumps(meta), encoding="utf-8")
                stale = dict(meta); stale["code_sha256"] = dict(code_hashes)
                stale["code_sha256"][runner.D4_CACHE_CODE_PATHS[0]] = "0" * 64
                paths["d4_metadata_path"].write_text(json.dumps(stale), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner._validate_d4_cache(paths)
                paths["d4_metadata_path"].write_text(json.dumps(meta), encoding="utf-8")
                with h5py.File(paths["d4_cache_path"], "r+") as h5:
                    h5.attrs["window_formula"] = "wrong"
                with self.assertRaises(ValueError):
                    runner._validate_d4_cache(paths)
                with h5py.File(paths["d4_cache_path"], "r+") as h5:
                    h5.attrs["window_formula"] = json.dumps(meta["window_formula"])
                    h5.attrs["code_sha256"] = json.dumps({**code_hashes, runner.D4_CACHE_CODE_PATHS[0]: "stale"})
                with self.assertRaises(ValueError):
                    runner._validate_d4_cache(paths)
                with h5py.File(paths["d4_cache_path"], "r+") as h5:
                    h5.attrs["code_sha256"] = json.dumps({"malformed": "bad"})
                meta["code_sha256"] = {**code_hashes, "extra.py": "e" * 64}
                paths["d4_metadata_path"].write_text(json.dumps(meta), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner._validate_d4_cache(paths)

    def test_reference_rows_reject_duplicates_and_bad_selection(self):
        rows = [{"fold": f, "seed": s, "variant": "D", "selected_epoch": 5,
                 "validation": {"Recall@10": 0.2, "HitRate@10": 0.2, "MRR@10": 0.2,
                                "NDCG@10": 0.2, "query_count": 1, "eligible_positive_edges": 1},
                 "checkpoint_path": f"checkpoints/fold_{f}_D_seed_{s}.pt", "checkpoint_sha256": "a" * 64,
                 "schedule_sha256": "b" * 64, "transform_sha256": "c" * 64}
                for f in (0, 1, 2) for s in (684, 685, 686)]
        row = rows[0]
        self.assertEqual(runner._select_d0_rows(rows, ((0, 684),))[(0, 684)]["_checkpoint_path"],
                         "checkpoints/fold_0_D_seed_684.pt")
        with self.assertRaises(ValueError):
            runner._select_d0_rows(rows + [row], ((0, 684),))
        with self.assertRaises(ValueError):
            runner._select_d0_rows([dict(rows[1], selected_epoch=True), *rows[1:]], ((0, 684),))

    def test_pilot_and_full_successful_synthetic_runs(self):
        for pilot in (True, False):
            with self.subTest(pilot=pilot), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                data_root, results_root = base / "data", base / "results"
                input_paths = {key: data_root / value for key, value in runner.INPUTS.items()}
                for path in input_paths.values():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix == ".h5":
                        continue
                    if path.is_dir() or path.name == "mxm_path":
                        path.mkdir(exist_ok=True)
                    elif not path.exists():
                        path.write_text("{}\n", encoding="utf-8")

                ids_by_fold = {
                    fold: {"train": [f"tr{fold}a", f"tr{fold}b"], "validation": [f"va{fold}"],
                           "test": [f"te{fold}"]}
                    for fold in (0, 1, 2)
                }
                cohort = sorted({track_id for parts in ids_by_fold.values() for part in parts.values()
                                 for track_id in part})
                catalog_path = input_paths["catalog_path"]
                catalog_path.write_text("catalog\n", encoding="utf-8")
                audit = {"source_sha256": {"audio": "a" * 64}}
                input_paths["audit_path"].write_text(json.dumps(audit), encoding="utf-8")
                channels = [*(f"pitch_{i}" for i in range(12)), *(f"timbre_{i}" for i in range(12)),
                            "loudness", "duration"]
                input_paths["cache_metadata_path"].write_text(json.dumps({"channels": channels}), encoding="utf-8")
                input_paths["cache_path"].parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(input_paths["cache_path"], "w") as h5:
                    h5.create_dataset("track_ids", data=np.asarray(cohort, dtype="S8"))
                input_paths["d4_metadata_path"].parent.mkdir(parents=True, exist_ok=True)
                input_paths["d4_rejection_path"].write_text("[]\n", encoding="utf-8")
                source_hashes = {"audio": "a" * 64}
                code_hashes = {key: hashlib.sha256((runner.ROOT / key).read_bytes()).hexdigest()
                               for key in runner.D4_CACHE_CODE_PATHS}
                d4_metadata = {
                    "status": "complete", "window_seconds": 30,
                    "window_formula": "start=0; middle=max(0,(duration-30)/2); end=max(0,duration-30)",
                    "interval": "[start,start+30)", "channel_count": 26, "channels": channels,
                    "dtype": "float32", "normalization": "raw/unnormalized; deferred until train split",
                    "track_count": len(cohort), "rejected_track_count": 0,
                    "valid_track_ids": cohort, "rejected": [],
                    "audit_sha256": hashlib.sha256(input_paths["audit_path"].read_bytes()).hexdigest(),
                    "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                    "source_sha256": source_hashes, "code_sha256": code_hashes,
                }
                with h5py.File(input_paths["d4_cache_path"], "w") as h5:
                    h5.create_dataset("track_ids", data=np.asarray(cohort, dtype="S8"))
                    h5.create_dataset("features", data=np.zeros((len(cohort), 26), dtype=np.float32))
                    for key in ("window_seconds", "window_formula", "interval", "channels", "normalization",
                                "audit_sha256", "catalog_sha256", "source_sha256", "code_sha256"):
                        value = d4_metadata[key]
                        h5.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
                input_paths["d4_metadata_path"].write_text(json.dumps({
                    **d4_metadata}), encoding="utf-8")
                expected_count = len(cohort)

                result_root = results_root / "neurotune_v1_full_modal_ablation"
                (result_root / "checkpoints").mkdir(parents=True)
                schedule_root = results_root / "neurotune_v1_abcd_cv" / "triplet_schedules"
                schedule_root.mkdir(parents=True)
                references, cosine_rows, schedule_bytes = [], [], {}
                metrics = {"Recall@10": 0.1, "HitRate@10": 0.1, "MRR@10": 0.1,
                           "NDCG@10": 0.1, "query_count": 1, "eligible_positive_edges": 1}
                for fold in (0, 1, 2):
                    cosine_rows.append({"fold": fold, "variant": "D", "validation": metrics})
                    for seed in (684, 685, 686):
                        checkpoint_rel = f"checkpoints/fold_{fold}_D_seed_{seed}.pt"
                        checkpoint = result_root / checkpoint_rel
                        checkpoint.write_bytes(f"d0-{fold}-{seed}".encode())
                        schedule_path = schedule_root / f"fold_{fold}_seed_{seed}.npz"
                        np.savez(schedule_path,
                                 train_ids=np.asarray(ids_by_fold[fold]["train"]),
                                 validation_ids=np.asarray(ids_by_fold[fold]["validation"]),
                                 train_schedule=np.zeros((50, 1, 3), dtype=np.int32),
                                 validation_schedule=np.zeros((1, 1, 3), dtype=np.int32))
                        schedule_bytes[(fold, seed)] = schedule_path.read_bytes()
                        references.append({"fold": fold, "seed": seed, "variant": "D",
                                           "selected_epoch": 3, "validation": metrics,
                                           "checkpoint_path": checkpoint_rel,
                                           "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                           "schedule_sha256": hashlib.sha256(schedule_bytes[(fold, seed)]).hexdigest(),
                                           "transform_sha256": "f" * 64})
                result_root.joinpath("experiment.json").write_text(json.dumps({
                    "status": "complete", "model_validation": references,
                    "cosine_validation": cosine_rows,
                }), encoding="utf-8")

                calls, feature_ids, baseline_ids = [], [], []
                d0_audio_hash = hashlib.sha256(input_paths["cache_path"].read_bytes()).hexdigest()
                d4_audio_hash = hashlib.sha256(input_paths["d4_cache_path"].read_bytes()).hexdigest()
                non_audio = {"bpm_mean": 1.0, "bpm_std": 2.0, "idf": [], "genre_vocabulary": [],
                             "fit_ids_hash": "1" * 64, "split_hash": "2" * 64,
                             "source_sha256": {"audio_cache": d0_audio_hash, "d4_audio_cache": d4_audio_hash,
                                               "catalog_bpm_genre": "3" * 64, "lyrics": "4" * 64,
                                               "split_manifest": "2" * 64}}
                def loader(**_kwargs):
                    return {"manifest": {}, "catalog": {}, "cache_path": input_paths["cache_path"],
                            "folds": [{"fold": fold, **parts,
                                       "validation_edges": [(parts["validation"][0], parts["train"][0])],
                                       "train_edges": []}
                                      for fold, parts in ids_by_fold.items()]}

                def mxm_builder(*_args):
                    return {}, {"source_sha256": {}, "directory_sha256": "5" * 64}

                def d0_builder(_loaded, fold, _lyrics):
                    return {}, {key: value for key, value in non_audio.items() if key != "source_sha256"} | {
                        "source_sha256": {k: v for k, v in non_audio["source_sha256"].items() if k != "d4_audio_cache"}}

                def feature_builder(_loaded, fold, _lyrics, _cache):
                    self.assertNotIn("test", fold)
                    selected = list(fold["train"]) + list(fold["validation"])
                    test_ids = ids_by_fold[fold["fold"]]["test"]
                    self.assertTrue(set(selected).isdisjoint(test_ids),
                                    f"fold {fold['fold']} feature IDs overlap test IDs")
                    feature_ids.append(set(selected))
                    n = len(selected)
                    return {"audio_windows": {track_id: [np.ones((1, 26), dtype=np.float32) for _ in range(3)]
                                               for track_id in selected},
                            "window_valid": np.ones((n, 3), dtype=np.bool_),
                            "bpm": np.ones((n, 1), dtype=np.float32),
                            "lyrics": np.ones((n, 5000), dtype=np.float32),
                            "genre": np.ones((n, 1), dtype=np.float32)}, {
                                **non_audio, "source_sha256": {"d4_audio_cache": d4_audio_hash,
                                    "catalog_bpm_genre": "3" * 64, "lyrics": "4" * 64,
                                    "split_manifest": "2" * 64}, "audio_mean": [1.0], "audio_std": [1.0],
                                "audio_fit_rows": 2}

                def trainer(_train_ds, _val_ds, train_schedule, val_schedule, _edges, variant,
                            seed, candidate, _transform, provenance, **kwargs):
                    self.assertEqual(variant, "D")
                    self.assertEqual(kwargs, {"epochs": 50, "batch_size": 64})
                    self.assertFalse(provenance["d0_checkpoint_loaded"])
                    candidate.write_bytes(f"fresh-{seed}".encode())
                    calls.append((provenance["fold"], seed, train_schedule.copy(), val_schedule.copy()))
                    return {"history": [{"epoch": epoch, "train_triplet_loss": 1.0,
                                         "validation_triplet_loss": 1.0, "NDCG@10": 0.1}
                                        for epoch in range(1, 51)],
                            "best_epoch": 1, "best_metrics": metrics}

                dependencies = {"loader": loader, "mxm_builder": mxm_builder,
                                "feature_builder": feature_builder, "d0_transform_builder": d0_builder,
                                "trainer": trainer,
                                "baseline_builder": lambda *_args, **_kwargs: np.ones((1, 2)),
                                "evaluator": lambda *_args, **_kwargs: metrics}
                def baseline_builder(_features, song_ids, **_kwargs):
                    baseline_ids.extend(song_ids)
                    return np.ones((len(song_ids), 2))
                dependencies["baseline_builder"] = baseline_builder
                with mock.patch.object(runner, "EXPECTED_TRACK_COUNT", expected_count):
                    report = runner._run(data_root=data_root, results_root=results_root, pilot=pilot,
                                         _test_only=True, _dependencies=dependencies)
                requested = runner._scope(pilot)
                self.assertEqual([(f, s) for f, s, *_ in calls], list(requested))
                self.assertEqual(len(report["model_validation"]), len(requested))
                self.assertTrue(all(np.array_equal(train_schedule, np.zeros((50, 1, 3), dtype=np.int32))
                                    and np.array_equal(validation_schedule, np.zeros((1, 1, 3), dtype=np.int32))
                                    for _, _, train_schedule, validation_schedule in calls))
                self.assertEqual(len(feature_ids), 1 if pilot else 3)
                self.assertTrue(all(not track.startswith("te") for track in baseline_ids))
                for ids in feature_ids:
                    self.assertTrue(all(not any(track.startswith("te") for track in ids) for track in ids))
                output = results_root / (runner.EXPECTED_CONFIG["output_namespaces"]["pilot" if pilot else "full"])
                self.assertEqual(report["counts"]["new_checkpoints"], len(requested))
                self.assertEqual(report["counts"]["schedule_copies"], len(requested))
                self.assertEqual(report["counts"]["training_curve_rows"], len(requested) * 50)
                self.assertEqual(len(report["d0_cosine_validation_references"]), 3)
                self.assertEqual(len(report["cosine_validation"]), 1 if pilot else 3)
                self.assertFalse(report["independent_test_artifacts_created"])
                self.assertEqual(len((output / "training_curves.csv").read_text(encoding="utf-8").splitlines()),
                                 len(requested) * 50 + 1)
                self.assertEqual(len(list((output / "checkpoints").glob("*.pt"))), len(requested))
                self.assertEqual(len(list((output / "triplet_schedules").glob("*.npz"))), len(requested))
                for fold, seed in requested:
                    self.assertEqual((output / f"triplet_schedules/fold_{fold}_seed_{seed}.npz").read_bytes(),
                                     schedule_bytes[(fold, seed)])

    def test_existing_pilot_namespace_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "neurotune_v1_d4_multi_window_audio_pilot"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                runner._run(results_root=directory, pilot=True)


if __name__ == "__main__":
    unittest.main()

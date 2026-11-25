import unittest
import tempfile
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from scripts import run_neurotune_v1_official10k_cv as runner
from scripts.run_neurotune_v1_official10k_cv import _fold_split, _slice_feature_rows, _validate_cache, run_experiment
from src.triplets import sample_triplet_schedule


class Official10KCVRunnerTests(unittest.TestCase):
    def test_fold_split_is_canonical_and_isolates_test_ids(self):
        audit = {
            "track_ids": ["a", "b", "c", "d", "e"],
            "folds": [{"fold": 0, "track_ids": ["d", "e"],
                       "train_edges": [["a", "b"]], "validation_edges": [["a", "c"]],
                       "test_edges": [["d", "e"]], "non_test_pool_edges": [["a", "b"], ["a", "c"]]}],
        }
        split, digest = _fold_split(audit, audit["folds"][0])
        self.assertEqual(split["test_ids"], ["d", "e"])
        self.assertEqual(split["pool_ids"], ["a", "b", "c"])
        self.assertEqual(split["validation_edges"], [["a", "c"]])
        self.assertEqual(len(digest), 64)

    def test_test_feature_rows_are_sliced_by_id_not_source_order(self):
        source_ids = ["c", "a", "b"]
        features = {"audio": ["audio-c", "audio-a", "audio-b"],
                    "bpm": np.asarray([[30], [10], [20]]),
                    "genre": np.asarray([[3], [1], [2]])}
        sliced = _slice_feature_rows(features, source_ids, ["b", "c"])
        self.assertEqual(sliced["audio"], ["audio-b", "audio-c"])
        np.testing.assert_array_equal(sliced["bpm"], [[20], [30]])
        np.testing.assert_array_equal(sliced["genre"], [[2], [3]])

    def test_wrong_test_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_cache({"track_count": 3}, ["a", "b"])

    def test_cache_hash_keys_must_exactly_cover_catalog(self):
        meta = {"track_count": 2, "interval": "[0,30)", "channel_count": 26,
                "source_sha256": {"a": "a" * 64}}
        with self.assertRaises(ValueError):
            _validate_cache(meta, ["a", "b"])

    def test_valid_task2_metadata_contract_without_sequence_lengths(self):
        ids = ["a", "b"]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.h5"
            with h5py.File(cache, "w") as h5:
                h5.create_dataset("features", data=np.ones((3, 26), dtype=np.float32))
                h5.create_dataset("offsets", data=np.asarray([0, 2, 3], dtype=np.int64))
                h5.create_dataset("track_ids", data=np.asarray(ids, dtype="S1"))
            meta = {"track_count": 2, "segment_count": 3, "window_seconds": 30.0,
                    "interval": "[0,30)", "channel_count": 26,
                    "channel_schema": [*(f"pitch_{i}" for i in range(12)),
                                       *(f"timbre_{i}" for i in range(12)),
                                       "loudness_max", "clipped_segment_duration"],
                    "storage": {"features": "raw unnormalized float32 (T, 26)"},
                    "source_sha256": {track_id: "a" * 64 for track_id in ids}}
            self.assertIsNone(_validate_cache(meta, ids, cache))

    def test_cache_validation_reads_in_chunks_and_detects_late_nonfinite_values(self):
        ids = ["a", "b"]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.h5"
            values = np.ones((5, 26), dtype=np.float32)
            values[-1, 0] = np.nan
            with h5py.File(cache, "w") as h5:
                h5.create_dataset("features", data=values)
                h5.create_dataset("offsets", data=np.asarray([0, 3, 5], dtype=np.int64))
                h5.create_dataset("track_ids", data=np.asarray(ids, dtype="S1"))
            meta = {"track_count": 2, "segment_count": 5, "window_seconds": 30.0,
                    "interval": "[0,30)", "channel_count": 26,
                    "channel_schema": [*(f"pitch_{i}" for i in range(12)),
                                       *(f"timbre_{i}" for i in range(12)),
                                       "loudness_max", "clipped_segment_duration"],
                    "storage": {"features": "raw unnormalized float32 (T, 26)"},
                    "source_sha256": {track_id: "a" * 64 for track_id in ids}}
            with mock.patch.object(runner, "CACHE_VALIDATION_CHUNK_ROWS", 2):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    _validate_cache(meta, ids, cache)

    def test_output_reservation_excludes_competing_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "result"
            lock = runner._reserve_output_namespace(final)
            try:
                with self.assertRaises(FileExistsError):
                    runner._reserve_output_namespace(final)
            finally:
                lock.unlink(missing_ok=True)
            self.assertFalse(lock.exists())

    def test_schedules_exclude_withheld_known_positive_edges(self):
        ids = ["a", "b", "c", "d", "e"]
        train = [("a", "b")]
        known = [*train, ("a", "c"), ("a", "d")]
        schedule = sample_triplet_schedule(ids, train, 1, 2, known_positive_edges=known)
        forbidden = {tuple(sorted(edge)) for edge in known}
        self.assertTrue(all(tuple(sorted((ids[row[0]], ids[row[2]]))) not in forbidden
                            for row in schedule[0]))

    def test_output_namespace_collision_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "neurotune_v1_official10k_cv_test"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                run_experiment(directory, directory, _test_only=True)
            self.assertFalse((Path(directory) / ".neurotune_v1_official10k_cv_test.lock").exists())

    def test_barrier_and_rollback_on_midrun_failure(self):
        all_ids = [f"id{i}" for i in range(9)]
        folds = []
        for index in range(3):
            test_ids = all_ids[index * 3:(index + 1) * 3]
            pool = sorted(set(all_ids) - set(test_ids))
            folds.append({"fold": index, "track_ids": test_ids,
                          "train_edges": [[pool[0], pool[1]]],
                          "validation_edges": [[pool[2], pool[3]]],
                          "test_edges": [[test_ids[0], test_ids[1]]],
                          "non_test_pool_edges": [[pool[0], pool[1]], [pool[2], pool[3]]],
                          "counts": {"test_tracks": 3, "test_edges": 1}})
        audit = {"track_ids": all_ids, "folds": folds,
                 "counts": {"tracks": 9, "source_edges": 6, "cross_fold_edges": 0},
                 "source_sha256": {key: "f" * 64 for key in
                                   ("data_audit", "tracks", "mxm", "genres", "lastfm_train")}}
        paths = {name: Path("dummy") for name in runner.INPUTS}
        hashes = {"audit": "a" * 64, "cache": "b" * 64, "cache_meta": "c" * 64,
                  "sources": {}, "code": {"fake": "d" * 64}}
        row_by_id = {track_id: index for index, track_id in enumerate(all_ids)}
        transform = {"audio_mean": [0], "audio_std": [1], "audio_fit_rows": 1,
                     "bpm_mean": 0, "bpm_std": 1, "idf": {}, "genre_vocabulary": [],
                     "fit_ids_hash": "a" * 64, "split_hash": "b" * 64, "code_sha256": {}}

        class Dataset:
            def __init__(self, track_ids, values):
                self.track_ids, self.features = list(track_ids), values
            def __len__(self): return len(self.track_ids)

        feature_calls = []
        barrier_checks = []
        fold_by_test = {tuple(sorted(fold["track_ids"])): fold for fold in folds}
        def assert_barrier_for_test(test_ids):
            fold = fold_by_test[tuple(sorted(test_ids))]
            stage = next(Path(directory).glob(".neurotune_v1_official10k_cv_test.stage-*"))
            checkpoints = list((stage / "checkpoints").glob("*.pt"))
            schedules = list(stage.glob("*_schedule.npz"))
            self.assertEqual(len(checkpoints), 9)
            self.assertEqual(len(schedules), 9)
            self.assertTrue((stage / "training_curves.csv").is_file())
            for path in schedules:
                schedule_fold = int(path.name.split("_")[1])
                scheduled_fold = folds[schedule_fold]
                pool = sorted(set(all_ids) - set(scheduled_fold["track_ids"]))
                with np.load(path) as saved:
                    schedule_ids = saved["train_ids"].tolist()
                    self.assertEqual(schedule_ids, pool)
                    self.assertFalse(set(schedule_ids) & set(scheduled_fold["track_ids"]))
                    for key in ("train_schedule", "validation_schedule"):
                        schedule = saved[key]
                        self.assertTrue(np.all(schedule >= 0))
                        self.assertTrue(np.all(schedule < len(schedule_ids)))
            barrier_checks.append(tuple(test_ids))

        def feature_builder(*args, train_ids, song_ids, split_hash):
            feature_calls.append(tuple(song_ids))
            test_ids = tuple(sorted(set(all_ids) - set(train_ids)))
            expected_pool = sorted(set(all_ids) - set(test_ids))
            self.assertEqual(sorted(train_ids), expected_pool)
            self.assertLessEqual(set(train_ids), set(song_ids))
            if set(song_ids) == set(train_ids):
                self.assertEqual(sorted(song_ids), expected_pool)
                self.assertFalse(set(train_ids) & set(test_ids))
            else:
                self.assertEqual(list(song_ids), all_ids)
                self.assertEqual(sorted(train_ids), expected_pool)
                assert_barrier_for_test(test_ids)
            features = {"audio": [np.full((1, 26), row_by_id[track_id], dtype=np.float32)
                                   for track_id in song_ids],
                        "bpm": np.asarray([[row_by_id[track_id]] for track_id in song_ids], dtype=float),
                        "genre": np.asarray([[row_by_id[track_id]] for track_id in song_ids], dtype=float)}
            return features, transform

        import src.model_v1
        real_model = src.model_v1.NeuroTuneV1Encoder
        def fake_trainer(dataset, *args):
            checkpoint = args[6]
            if not checkpoint.parent.is_dir():
                raise AssertionError("runner must create checkpoint parent before trainer call")
            model = real_model("C", 1)
            torch.save({"model_state_dict": model.state_dict()}, checkpoint)
            return {"history": [{"epoch": 1, "NDCG@10": 0.5}],
                    "best_epoch": 1, "best_metrics": {"NDCG@10": 0.5}}

        metrics = {"Recall@10": 0.5, "HitRate@10": 0.5, "MRR@10": 0.4, "NDCG@10": 0.5}
        def assert_evaluation_barrier(embeddings, song_ids, edges):
            self.assertEqual(len(song_ids), 3)
            assert_barrier_for_test(song_ids)
            return metrics
        def assert_embed_barrier(model, dataset):
            assert_barrier_for_test(dataset.track_ids)
            return np.ones((len(dataset), 2))
        with tempfile.TemporaryDirectory() as directory:
            test_meta = {"source_sha256": {track_id: "e" * 64 for track_id in all_ids}}
            with mock.patch.object(runner, "_preflight", return_value=(paths, audit, test_meta, hashes)), \
                 mock.patch.object(runner, "parse_msd", return_value=[{"track_id": x} for x in all_ids]), \
                 mock.patch.object(runner, "parse_mxm", return_value={x: {} for x in all_ids}), \
                 mock.patch.object(runner, "parse_tagtraum", return_value={x: "g" for x in all_ids}), \
                 mock.patch.object(runner, "NeuroTuneV1Dataset", Dataset), \
                 mock.patch.object(runner, "build_neurotune_v1_features", feature_builder), \
                 mock.patch.object(runner, "train_neurotune_v1_cv", fake_trainer), \
                 mock.patch.object(runner, "build_neurotune_v1_cosine_baseline", return_value=np.ones((3, 2))), \
                  mock.patch.object(runner, "evaluate_neurotune_v1_retrieval", side_effect=assert_evaluation_barrier), \
                  mock.patch.object(runner, "_embed", side_effect=assert_embed_barrier), \
                 mock.patch.object(runner, "SEEDS", (1, 2, 3)), \
                 mock.patch.object(runner, "EPOCHS", 1):
                report = run_experiment(directory, directory, feature_builder=feature_builder,
                                         trainer=fake_trainer, _test_only=True)
                self.assertEqual(len(feature_calls), 6)
                self.assertEqual(report["aggregates"]["model_nine_fold_seed_observations"]["NDCG@10"], 0.5)
                self.assertEqual(report["aggregates"]["baseline_three_fold_observations"]["Recall@10"], 0.5)

            failure_dir = Path(directory) / "failure"
            calls = 0
            def failing_trainer(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected trainer failure")
                return fake_trainer(*args)
            with mock.patch.object(runner, "_preflight", return_value=(paths, audit, test_meta, hashes)), \
                 mock.patch.object(runner, "parse_msd", return_value=[{"track_id": x} for x in all_ids]), \
                 mock.patch.object(runner, "parse_mxm", return_value={x: {} for x in all_ids}), \
                 mock.patch.object(runner, "parse_tagtraum", return_value={x: "g" for x in all_ids}), \
                 mock.patch.object(runner, "NeuroTuneV1Dataset", Dataset), \
                 mock.patch.object(runner, "build_neurotune_v1_features", feature_builder), \
                 mock.patch.object(runner, "train_neurotune_v1_cv", failing_trainer), \
                 mock.patch.object(runner, "SEEDS", (1, 2, 3)), \
                 mock.patch.object(runner, "EPOCHS", 1):
                with self.assertRaisesRegex(RuntimeError, "injected trainer failure"):
                    run_experiment(directory, failure_dir, feature_builder=feature_builder,
                                   trainer=failing_trainer, _test_only=True)
            self.assertFalse((failure_dir / "neurotune_v1_official10k_cv_test").exists())
            self.assertEqual(list(failure_dir.glob(".neurotune_v1_official10k_cv_test.stage-*")), [])
            self.assertFalse((failure_dir / ".neurotune_v1_official10k_cv_test.lock").exists())


if __name__ == "__main__":
    unittest.main()

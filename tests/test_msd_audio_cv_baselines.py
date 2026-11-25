import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import hashlib

import numpy as np
import h5py

from scripts.evaluate_msd_audio_cv_baselines import (
    _mean_pool_audio, _random_rankings, _validate_inputs, run,
)


class TestRandomRankings(unittest.TestCase):
    def test_rankings_are_deterministic_unique_non_query_samples(self):
        gallery_ids = [f"TR{i:03d}" for i in range(20)]

        first = _random_rankings(gallery_ids, fold=1, seed=684)
        second = _random_rankings(gallery_ids, fold=1, seed=684)

        self.assertEqual(first, second)
        for query_id in gallery_ids:
            candidate_ids = [candidate["song_id"] for candidate in first[query_id]]
            self.assertNotIn(query_id, candidate_ids)
            self.assertEqual(len(candidate_ids), 10)
            self.assertEqual(len(set(candidate_ids)), 10)
            self.assertLessEqual(set(candidate_ids), set(gallery_ids) - {query_id})


class TestBaselineEndToEnd(unittest.TestCase):
    def test_three_fold_audio_and_random_baselines_with_model_reference(self):
        from scripts.audit_msd_audio_cv import _source_sha256
        from scripts.run_msd_audio_cv import _sha256
        from src.evaluate import build_test_relevance, evaluate_rankings
        from src.retrieve import rank_top_k

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audit_path = root / "msd_audio_only_cv_audit.json"
            lastfm = root / "lastfm"
            lastfm.mkdir()
            (lastfm / "edges.txt").write_text("synthetic Last.fm", encoding="utf-8")
            cache_dir = root / "cache"
            cache_dir.mkdir()
            ids_by_fold = {fold: [f"TR{fold}{index:02d}" for index in range(12)]
                           for fold in range(3)}
            all_ids = sorted(track_id for ids in ids_by_fold.values() for track_id in ids)
            edge_pairs = {fold: [[ids[0], ids[1]], [ids[2], ids[3]]]
                          for fold, ids in ids_by_fold.items()}
            audit = {"status": "ready", "catalog_counts": {"eligible_tracks": len(all_ids)},
                     "source_sha256": {"lastfm": _source_sha256(lastfm)},
                     "folds": {str(fold): {"track_ids": ids, "positive_edges": edge_pairs[fold]}
                               for fold, ids in ids_by_fold.items()}}
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            audit_hash = _sha256(audit_path)
            cache_paths, metadata_paths = {}, {}
            channel_names = ([f"pitch_{i + 1}" for i in range(12)] +
                             [f"timbre_{i + 1}" for i in range(12)] +
                             ["loudness_max", "beat_count", "bar_count", "section_count"])
            for fold in range(3):
                cache_path = cache_dir / f"msd_audio_only_cv_fold_{fold}.h5"
                metadata_path = cache_dir / f"msd_audio_only_cv_fold_{fold}_meta.json"
                cache_paths[fold], metadata_paths[fold] = cache_path, metadata_path
                audio = np.zeros((len(all_ids), 1024, 28), dtype=np.float32)
                for row, track_id in enumerate(all_ids):
                    # Distinct normalized audio vectors; event channels are deliberately large.
                    audio[row, :, :25] = (row + 1) * np.eye(1, 25, row % 25, dtype=np.float32)
                    audio[row, :, 25:] = 1000 + row
                with h5py.File(cache_path, "w") as handle:
                    handle.create_dataset("track_ids", data=np.asarray(all_ids, dtype="S18"))
                    handle.create_dataset("sequences", data=audio)
                    handle.create_dataset("channel_names", data=np.asarray(channel_names, dtype="S32"))
                train_ids = [item for other in range(3) if other != fold for item in ids_by_fold[other]]
                metadata = {"audit_sha256": audit_hash,
                            "split_ids": {"train": train_ids, "validation": [], "test": ids_by_fold[fold]},
                            "normalization": {"train_ids": train_ids}, "channel_names": channel_names,
                            "grid_size": 1024}
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            model_report_path = root / "msd_audio_only_cv_experiment.json"
            sources = {"audit": _sha256(audit_path), "lastfm_similar": _source_sha256(lastfm)}
            for fold in range(3):
                sources[f"fold_{fold}_cache"] = _sha256(cache_paths[fold])
                sources[f"fold_{fold}_metadata"] = _sha256(metadata_paths[fold])
            seeds = [11, 22, 33]
            model_folds = {}
            for fold in range(3):
                train_ids = [item for other in range(3) if other != fold for item in ids_by_fold[other]]
                model_folds[str(fold)] = {
                    "train_ids": train_ids, "test_ids": ids_by_fold[fold],
                    "test": {"gallery_count": 12, "query_count": 4, "unique_edge_count": 2},
                    "per_seed": {str(seed): {"metrics": {"MRR@10": seed / 100.0}}
                                 for seed in seeds}}
            model_report = {"fold_count": 3, "config": {"seeds": seeds},
                            "source_sha256": sources, "per_fold": model_folds}
            model_report_path.write_text(json.dumps(model_report), encoding="utf-8")
            output = root / "baselines.json"

            report = run(audit_path, cache_dir, lastfm, output, seeds=seeds)

            self.assertEqual(report["fold_count"], 3)
            self.assertIn("independently per query", report["random_sampling"])
            self.assertEqual(set(report["per_fold"]), {"0", "1", "2"})
            for fold in range(3):
                item = report["per_fold"][str(fold)]
                self.assertEqual(item["test_ids"], ids_by_fold[fold])
                self.assertEqual(item["test"], {"gallery_count": 12, "query_count": 4,
                                                 "unique_edge_count": 2})
                relevance = build_test_relevance(
                    [{"source": a, "target": b} for a, b in edge_pairs[fold]], ids_by_fold[fold])
                self.assertEqual(set(relevance), {track_id for edge in edge_pairs[fold]
                                                  for track_id in edge})
                self.assertEqual(sum(map(len, relevance.values())), 4)
                self.assertIn("audio_metrics", item)
                self.assertEqual(set(item["per_seed"]["11"]),
                                 {"random_metrics", "model_reference_metrics"})
                self.assertEqual(set(item["per_seed"]["22"]),
                                 {"random_metrics", "model_reference_metrics"})
                self.assertNotIn("audio_metrics", item["per_seed"]["11"])
                self.assertNotIn("audio_metrics", item["per_seed"]["22"])
                for seed in seeds:
                    self.assertEqual(
                        item["per_seed"][str(seed)]["model_reference_metrics"],
                        {"MRR@10": seed / 100.0})
                    random_rankings = _random_rankings(ids_by_fold[fold], fold, seed)
                    self.assertEqual(
                        item["per_seed"][str(seed)]["random_metrics"],
                        evaluate_rankings(random_rankings, relevance, k=10))
                    self.assertTrue(all(np.isfinite(value) for value in
                                        item["per_seed"][str(seed)]["random_metrics"].values()
                                        if isinstance(value, (int, float))))
                # The audio score is checked against rank_top_k on the fold's rows.
                with h5py.File(cache_paths[fold], "r") as handle:
                    cache_audio = handle["sequences"][...]
                    cache_ids = [value.decode() for value in handle["track_ids"][...]]
                indices = [cache_ids.index(track_id) for track_id in ids_by_fold[fold]]
                embeddings = _mean_pool_audio(cache_audio[indices])
                expected = {track_id: rank_top_k(embeddings, ids_by_fold[fold], track_id, k=10)
                            for track_id in ids_by_fold[fold]}
                self.assertEqual(item["audio_metrics"], evaluate_rankings(expected, relevance, k=10))
                self.assertTrue(all(np.isfinite(value) for value in item["audio_metrics"].values()
                                    if isinstance(value, (int, float))))
                for method, field in (("random", "random_metrics"),
                                      ("model_reference", "model_reference_metrics")):
                    self.assertEqual(item["summary"][method]["observation_count"], 3)
                    for metric in ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10"):
                        values = [item["per_seed"][str(seed)][field].get(metric, 0.0)
                                  for seed in seeds]
                        self.assertAlmostEqual(item["summary"][method]["metrics"][metric]["mean"],
                                               float(np.mean(values)))
                        self.assertAlmostEqual(item["summary"][method]["metrics"][metric]["sample_std"],
                                               float(np.std(values, ddof=1)))
            for method, count in (("audio", 3), ("random", 9), ("model_reference", 9)):
                self.assertEqual(report["summary"][method]["observation_count"], count)
                for metric in ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10"):
                    if method == "audio":
                        observations = [report["per_fold"][str(fold)]["audio_metrics"].get(metric, 0.0)
                                        for fold in range(3)]
                    else:
                        field = "random_metrics" if method == "random" else "model_reference_metrics"
                        observations = [seed_record[field].get(metric, 0.0)
                                        for fold in range(3)
                                        for seed_record in report["per_fold"][str(fold)]["per_seed"].values()]
                    expected_summary = report["summary"][method]["metrics"][metric]
                    self.assertAlmostEqual(expected_summary["mean"], float(np.mean(observations)))
                    self.assertAlmostEqual(expected_summary["sample_std"],
                                           float(np.std(observations, ddof=1)))
            implementation_files = {
                "scripts/evaluate_msd_audio_cv_baselines.py",
                "scripts/audit_msd_audio_cv.py",
                "scripts/run_msd_audio_cv.py",
                "src/data/preprocess_audio.py",
                "src/evaluate.py",
                "src/retrieve.py",
            }
            self.assertEqual(set(report["code_sha256"]), implementation_files)
            self.assertEqual(report["code_sha256"], {
                relative: _sha256(Path(__file__).resolve().parents[1] / relative)
                for relative in sorted(implementation_files)
            })
            self.assertEqual(report["source_sha256"], {**sources, "model_report": _sha256(model_report_path)})
            self.assertTrue(report["exploratory"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(hashlib.sha256(model_report_path.read_bytes()).hexdigest(),
                             report["source_sha256"]["model_report"])


class TestMeanPooledAudio(unittest.TestCase):
    def test_pools_continuous_channels_and_l2_normalizes_without_mutation(self):
        audio = np.arange(2 * 1024 * 28, dtype=np.float32).reshape(2, 1024, 28)
        original = audio.copy()

        pooled = _mean_pool_audio(audio)

        self.assertEqual(pooled.shape, (2, 25))
        np.testing.assert_allclose(pooled, audio[:, :, :25].mean(axis=1) /
                                   np.linalg.norm(audio[:, :, :25].mean(axis=1), axis=1, keepdims=True))
        np.testing.assert_allclose(np.linalg.norm(pooled, axis=1), np.ones(2))
        np.testing.assert_array_equal(audio, original)


class TestBaselineInputValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audit_path = self.root / "audit.json"
        self.lastfm = self.root / "lastfm"
        self.lastfm.mkdir()
        (self.lastfm / "edges.txt").write_text("source", encoding="utf-8")
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir()
        self.output = self.root / "baselines.json"
        self.report = self.root / "completed_report.json"
        self.checkpoint = self.root / "completed.pt"
        self.report.write_bytes(b"report sentinel")
        self.checkpoint.write_bytes(b"checkpoint sentinel")
        self.audit = {
            "status": "ready",
            "catalog_counts": {"eligible_tracks": 6},
            "source_sha256": {},
            "folds": {
                str(fold): {"track_ids": [f"TR{fold}{i}" for i in ("A", "B")],
                            "positive_edges": [[f"TR{fold}A", f"TR{fold}B"]]}
                for fold in range(3)
            },
        }
        self.ids_by_fold = {fold: self.audit["folds"][str(fold)]["track_ids"] for fold in range(3)}
        self._save_audit()
        self.metadata = {}
        for fold in range(3):
            self.metadata[fold] = {
                "audit_sha256": "set-by-test",
                "split_ids": {"train": [item for other in range(3) if other != fold
                                         for item in self.ids_by_fold[other]],
                              "validation": [], "test": self.ids_by_fold[fold]},
                "normalization": {"train_ids": [item for other in range(3) if other != fold
                                                  for item in self.ids_by_fold[other]]},
            }
            (self.cache_dir / f"msd_audio_only_cv_fold_{fold}.h5").write_bytes(b"cache")
            (self.cache_dir / f"msd_audio_only_cv_fold_{fold}_meta.json").write_text(
                json.dumps(self.metadata[fold]), encoding="utf-8")
        from scripts.audit_msd_audio_cv import _source_sha256
        from scripts.run_msd_audio_cv import _sha256
        sources = {"audit": _sha256(self.audit_path),
                   "lastfm_similar": _source_sha256(self.lastfm)}
        model_folds = {}
        for fold in range(3):
            sources[f"fold_{fold}_cache"] = _sha256(
                self.cache_dir / f"msd_audio_only_cv_fold_{fold}.h5")
            sources[f"fold_{fold}_metadata"] = _sha256(
                self.cache_dir / f"msd_audio_only_cv_fold_{fold}_meta.json")
            model_folds[str(fold)] = {
                "train_ids": [item for other in range(3) if other != fold
                              for item in self.ids_by_fold[other]],
                "test_ids": self.ids_by_fold[fold],
                "test": {"gallery_count": 2, "query_count": 2, "unique_edge_count": 1},
                "per_seed": {str(seed): {"metrics": {}} for seed in (684, 685, 686)},
            }
        (self.root / "msd_audio_only_cv_experiment.json").write_text(json.dumps({
            "fold_count": 3, "config": {"seeds": [684, 685, 686]},
            "source_sha256": sources, "per_fold": model_folds,
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _save_audit(self):
        from scripts.audit_msd_audio_cv import _source_sha256
        self.audit["source_sha256"]["lastfm"] = _source_sha256(self.lastfm)
        self.audit_path.write_text(json.dumps(self.audit), encoding="utf-8")

    def _validate(self):
        from scripts.audit_msd_audio_cv import _source_sha256
        from scripts.run_msd_audio_cv import _sha256
        for fold, metadata in self.metadata.items():
            (self.cache_dir / f"msd_audio_only_cv_fold_{fold}_meta.json").write_text(
                json.dumps(metadata), encoding="utf-8")
        model_path = self.root / "msd_audio_only_cv_experiment.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["source_sha256"]["audit"] = _sha256(self.audit_path)
        model["source_sha256"]["lastfm_similar"] = _source_sha256(self.lastfm)
        for fold in range(3):
            model["source_sha256"][f"fold_{fold}_cache"] = _sha256(
                self.cache_dir / f"msd_audio_only_cv_fold_{fold}.h5")
            model["source_sha256"][f"fold_{fold}_metadata"] = _sha256(
                self.cache_dir / f"msd_audio_only_cv_fold_{fold}_meta.json")
        model_path.write_text(json.dumps(model), encoding="utf-8")
        with patch("scripts.evaluate_msd_audio_cv_baselines.load_audio_sequence_cache",
                   return_value=([], np.empty((0, 1024, 28), dtype=np.float32), {})):
            return _validate_inputs(self.audit_path, self.cache_dir, self.lastfm, self.output)

    def test_rejects_stale_audit_and_changed_lastfm_before_output(self):
        self.audit["catalog_counts"]["eligible_tracks"] = 7
        self._save_audit()
        with self.assertRaisesRegex(ValueError, "cover the audited catalog"):
            self._validate()
        self.assertFalse(self.output.exists())

        self.audit["catalog_counts"]["eligible_tracks"] = 6
        self._save_audit()
        (self.lastfm / "edges.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Last.fm"):
            self._validate()
        self.assertFalse(self.output.exists())

    def test_rejects_cache_audit_hash_split_and_normalization_mismatches(self):
        with self.assertRaisesRegex(ValueError, "audit_sha256"):
            self._validate()
        audit_hash = __import__("hashlib").sha256(self.audit_path.read_bytes()).hexdigest()
        for metadata in self.metadata.values():
            metadata["audit_sha256"] = audit_hash
        self.metadata[1]["split_ids"]["test"] = self.ids_by_fold[0]
        with self.assertRaisesRegex(ValueError, "split_ids"):
            self._validate()
        self.metadata[1]["split_ids"]["test"] = self.ids_by_fold[1]
        self.metadata[2]["normalization"]["train_ids"] = self.ids_by_fold[2]
        with self.assertRaisesRegex(ValueError, "normalization"):
            self._validate()
        self.assertFalse(self.output.exists())

    def test_rejects_output_collisions_and_existing_report_before_writing(self):
        with self.assertRaisesRegex(ValueError, "collide"):
            with patch("scripts.evaluate_msd_audio_cv_baselines.load_audio_sequence_cache"):
                _validate_inputs(self.audit_path, self.cache_dir, self.lastfm, self.audit_path)
        self.output.write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._validate()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing")
        self.assertEqual(self.report.read_bytes(), b"report sentinel")
        self.assertEqual(self.checkpoint.read_bytes(), b"checkpoint sentinel")

    def test_completed_report_and_checkpoint_are_protected_outputs(self):
        protected_report = self.audit_path.parent / "msd_audio_only_cv_experiment.json"
        protected_report.write_bytes(b"report sentinel")
        checkpoint_dir = self.audit_path.parent / "checkpoints"
        checkpoint_dir.mkdir()
        protected_checkpoint = checkpoint_dir / "msd_audio_only_cv_fold_0_seed_684.pt"
        protected_checkpoint.write_bytes(b"checkpoint sentinel")
        for target in (protected_report, protected_checkpoint):
            with patch("scripts.evaluate_msd_audio_cv_baselines.load_audio_sequence_cache"):
                with self.assertRaisesRegex(ValueError, "collide"):
                    _validate_inputs(self.audit_path, self.cache_dir, self.lastfm, target)
        self.assertEqual(protected_report.read_bytes(), b"report sentinel")
        self.assertEqual(protected_checkpoint.read_bytes(), b"checkpoint sentinel")

    def test_rejects_output_nested_under_read_only_input_directories(self):
        targets = [self.lastfm / "baseline.json", self.cache_dir / "baseline.json"]
        for target in targets:
            with self.subTest(target=target), patch(
                    "scripts.evaluate_msd_audio_cv_baselines.load_audio_sequence_cache"):
                with self.assertRaisesRegex(ValueError, "inside a read-only input directory"):
                    _validate_inputs(self.audit_path, self.cache_dir, self.lastfm, target)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()

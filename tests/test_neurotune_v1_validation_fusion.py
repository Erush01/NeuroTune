import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock
import hashlib

import numpy as np

from scripts.evaluate_neurotune_v1_validation_fusion import (
    build_validation_relevance,
    evaluate_fused_rankings,
    fuse_score_matrices,
    rank_score_matrix,
    _stage_report,
    _validation_inputs,
    run_validation_fusion,
    METRIC_NAMES,
    SEEDS,
    WEIGHTS,
)


class ValidationFusionTests(unittest.TestCase):
    def setUp(self):
        self.model = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.4],
            [0.2, 0.4, 1.0],
        ])
        self.baseline = np.array([
            [1.0, 0.2, 0.8],
            [0.2, 1.0, 0.6],
            [0.8, 0.6, 1.0],
        ])
        self.song_ids = ["b", "a", "c"]

    def test_fusion_endpoints_and_exact_convex_blend(self):
        np.testing.assert_array_equal(fuse_score_matrices(self.model, self.baseline, 0), self.baseline)
        np.testing.assert_array_equal(fuse_score_matrices(self.model, self.baseline, 1), self.model)
        np.testing.assert_array_equal(
            fuse_score_matrices(self.model, self.baseline, 0.25),
            0.25 * self.model + 0.75 * self.baseline,
        )

    def test_fusion_rejects_invalid_matrices_and_weight(self):
        with self.assertRaises(ValueError):
            fuse_score_matrices(self.model, self.baseline[:2], 0.5)
        with self.assertRaises(ValueError):
            fuse_score_matrices(np.array([[1.0, 0.1], [0.2, 1.0]]), np.eye(2), 0.5)
        with self.assertRaises(ValueError):
            fuse_score_matrices(np.array([[1.0, np.nan], [np.nan, 1.0]]), np.eye(2), 0.5)
        for weight in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                fuse_score_matrices(self.model, self.baseline, weight)

    def test_fusion_rejects_complex_score_matrices(self):
        complex_scores = np.array([[1.0 + 1.0j, 0.5], [0.5, 1.0]])
        with self.assertRaises(ValueError):
            fuse_score_matrices(complex_scores, np.eye(2), 0.5)

    def test_rankings_exclude_self_and_include_full_gallery_with_id_tie_break(self):
        scores = np.ones((3, 3))
        rankings = rank_score_matrix(scores, ["z", "a", "m"])
        self.assertEqual([item["song_id"] for item in rankings["z"]], ["a", "m"])
        self.assertEqual([item["song_id"] for item in rankings["a"]], ["m", "z"])
        self.assertEqual(len(rankings["m"]), 2)
        self.assertTrue(all("score" in item for row in rankings.values() for item in row))

    def test_unique_undirected_validation_edges_feed_existing_metrics(self):
        relevance = build_validation_relevance(
            [("b", "a"), ("a", "b"), ("c", "a")], self.song_ids
        )
        self.assertEqual(relevance, {"b": {"a"}, "a": {"b", "c"}, "c": {"a"}})
        fused = fuse_score_matrices(self.model, self.baseline, 1)
        metrics = evaluate_fused_rankings(fused, self.song_ids, [("b", "a"), ("c", "a")], k=10)
        self.assertEqual(metrics["query_count"], 3)
        self.assertEqual(metrics["eligible_positive_edges"], 2)
        self.assertEqual(metrics["Recall@10"], 1.0)
        self.assertEqual(metrics["HitRate@10"], 1.0)
        self.assertEqual(metrics["MRR@10"], 1.0)
        self.assertEqual(metrics["NDCG@10"], 1.0)
        self.assertEqual(metrics["unique_undirected_edge_count"], 2)

    def test_validation_relevance_rejects_unhashable_endpoint(self):
        with self.assertRaises(ValueError):
            build_validation_relevance([(["b"], "a")], self.song_ids)

    def test_input_loader_uses_only_train_validation_ids_and_validation_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            data.mkdir()
            (data / "neurotune_v1_split.json").write_text(json.dumps({
                "track_ids": {"train": ["tr"], "validation": ["va"], "test": ["secret-test-id"]}
            }))
            (data / "ab_catalog_tracks.csv").write_text("track_id,tempo,genre\ntr,100,Rock\nva,110,Jazz\n")
            (data / "neurotune_v1_validation_edges.csv").write_text(
                "track_id_a,track_id_b\nva,va\n")
            with mock.patch("scripts.evaluate_neurotune_v1_validation_fusion.parse_mxm", return_value={}):
                _, train, validation, _, _, _, edges = _validation_inputs(data)
            self.assertEqual(train, ["tr"])
            self.assertEqual(validation, ["va"])
            self.assertEqual(edges, [("va", "va")])
            self.assertNotIn("secret-test-id", train + validation)

    def test_validation_feature_rows_follow_explicit_id_order(self):
        ordered_ids = ["train-a", "validation-b", "train-c", "validation-a"]
        by_id = {song_id: index for index, song_id in enumerate(ordered_ids)}
        validation_ids = ["validation-a", "validation-b"]
        rows = [by_id[song_id] for song_id in validation_ids]
        np.testing.assert_array_equal(np.array([10, 20, 30, 40])[rows], [40, 20])

    def test_staged_report_collision_refusal_and_rollback(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            _stage_report(output, {"ok": True})
            with self.assertRaises(FileExistsError):
                _stage_report(output, {"new": True})
            self.assertEqual(json.loads((output / "experiment.json").read_text()), {"ok": True})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            with mock.patch("scripts.evaluate_neurotune_v1_validation_fusion.os.link", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    _stage_report(output, {"ok": True})
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_staged_report_does_not_replace_concurrently_created_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            original_write_text = Path.write_text
            injected = False

            def write_then_create_destination(path, *args, **kwargs):
                nonlocal injected
                result = original_write_text(path, *args, **kwargs)
                if not injected and path.name == "experiment.json":
                    output.mkdir()
                    injected = True
                return result

            with mock.patch.object(Path, "write_text", new=write_then_create_destination):
                with self.assertRaises(FileExistsError):
                    _stage_report(output, {"ok": True})
            self.assertTrue(injected)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])

    def test_synthetic_runner_accepts_view_specific_hashes_and_writes_validation_only_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            checkpoint_dir = root / "checkpoints"
            data.mkdir()
            checkpoint_dir.mkdir()
            (data / "neurotune_v1_split.json").write_text("{}")
            (data / "ab_catalog_tracks.csv").write_text("track_id,tempo,genre\n")
            (data / "ab_audio_30s_26ch.h5").write_bytes(b"synthetic cache")
            (data / "neurotune_v1_validation_edges.csv").write_text("track_id_a,track_id_b\n")
            train_ids = ["train-only"]
            validation_ids = [f"validation-{index:03d}" for index in range(610)]
            edges = [(validation_ids[index], validation_ids[index + 1]) for index in range(133)]
            edges += [(validation_ids[index], validation_ids[index + 1]) for index in range(134, 238, 2)]
            self.assertEqual((len(edges), len({endpoint for edge in edges for endpoint in edge})), (185, 238))
            expected_fit_hash = hashlib.sha256(
                json.dumps(sorted(train_ids), separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            stable_fit = {
                "audio_mean": [0.0], "audio_std": [1.0], "audio_fit_rows": 1,
                "bpm_mean": 0.0, "bpm_std": 1.0, "idf": {"word": 1.0},
                "genre_vocabulary": ["Rock"], "fit_ids_hash": expected_fit_hash,
                "split_hash": hashlib.sha256((data / "neurotune_v1_split.json").read_bytes()).hexdigest(),
                "code_sha256": {"fit.py": "code-hash"},
            }
            current_manifest = {**stable_fit, "source_sha256": {
                "audio_cache": hashlib.sha256(b"synthetic cache").hexdigest(),
                "catalog_bpm_genre": "train-validation-view", "lyrics": "train-validation-lyrics",
            }}
            checkpoint_manifest = {**stable_fit, "source_sha256": {
                "audio_cache": current_manifest["source_sha256"]["audio_cache"],
                "catalog_bpm_genre": "train-validation-test-view", "lyrics": "all-partitions-lyrics",
            }}
            self.assertNotEqual(checkpoint_manifest, current_manifest)
            baseline_values = {name: 0.2 for name in METRIC_NAMES}
            stored_baseline_values = {name: 0.9 for name in METRIC_NAMES}
            model_values = {name: 0.4 for name in METRIC_NAMES}
            checkpoints = {}
            runs = []
            for seed in SEEDS:
                path = checkpoint_dir / f"C_seed_{seed}.pt"
                path.write_bytes(f"checkpoint-{seed}".encode())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checkpoints[str(path)] = {
                    "variant": "C", "seed": seed, "transform_manifest": checkpoint_manifest,
                    "provenance": {"split": current_manifest["split_hash"],
                                   "cache": current_manifest["source_sha256"]["audio_cache"]},
                }
                runs.append({"variant": "C", "seed": seed, "checkpoint_sha256": digest,
                             "validation": model_values})
            reference_path = root / "reference.json"
            reference_path.write_text(json.dumps({"baseline": {"validation": stored_baseline_values}, "runs": runs}))
            output = root / "output"
            feature_builder_ids = []
            eval_matrices = []
            embed_checkpoint_objects = []
            torch_load_calls = []
            deserialized_checkpoint_bytes = []

            def build_features(_cache, _tracks, _lyrics, _genres, train_ids, song_ids, split_hash):
                feature_builder_ids.append((list(train_ids), list(song_ids)))
                n = len(song_ids)
                return {"audio": [np.zeros((1, 26), dtype=np.float32) for _ in song_ids],
                        "bpm": np.zeros((n, 1), dtype=np.float32),
                        "lyrics": np.zeros((n, 5000), dtype=np.float32),
                        "genre": np.zeros((n, 1), dtype=np.float32)}, current_manifest

            def evaluate(scores, song_ids, positive_edges):
                self.assertEqual(scores.shape, (610, 610))
                eval_matrices.append(scores.copy())
                if np.allclose(scores, 0):
                    values = baseline_values
                elif np.allclose(scores, np.eye(len(song_ids))):
                    values = model_values
                else:
                    values = {name: 0.3 for name in METRIC_NAMES}
                return {**values, "query_count": 238, "eligible_positive_edges": 185,
                        "unique_undirected_edge_count": 185}

            def load_checkpoint(checkpoint_stream, **_kwargs):
                raw_bytes = checkpoint_stream.getvalue()
                deserialized_checkpoint_bytes.append(raw_bytes)
                path = next(path for path in checkpoints
                            if Path(path).read_bytes() == raw_bytes)
                torch_load_calls.append(path)
                return checkpoints[path]

            def embed_validation(_features, _ids, checkpoint, _seed):
                embed_checkpoint_objects.append(checkpoint)
                if not isinstance(checkpoint, dict):
                    raise AssertionError("embedding must receive the already-loaded checkpoint object")
                return np.eye(610)

            with mock.patch("scripts.evaluate_neurotune_v1_validation_fusion._validation_inputs",
                            return_value=({}, train_ids, validation_ids, [], {}, {}, edges)), \
                 mock.patch("scripts.evaluate_neurotune_v1_validation_fusion.build_neurotune_v1_features",
                            side_effect=build_features), \
                 mock.patch("scripts.evaluate_neurotune_v1_validation_fusion.build_neurotune_v1_cosine_baseline",
                            return_value=np.zeros((610, 2))) as baseline_builder, \
                 mock.patch("scripts.evaluate_neurotune_v1_validation_fusion._embed_validation",
                            side_effect=embed_validation), \
                 mock.patch("scripts.evaluate_neurotune_v1_validation_fusion.evaluate_fused_rankings",
                            side_effect=evaluate), \
                 mock.patch("scripts.evaluate_neurotune_v1_validation_fusion.evaluate_neurotune_v1_retrieval",
                            return_value=baseline_values, create=True) as baseline_evaluator, \
                 mock.patch("scripts.evaluate_neurotune_v1_validation_fusion.torch.load",
                            side_effect=load_checkpoint):
                report = run_validation_fusion(data, checkpoint_dir, reference_path, output)

            self.assertEqual(feature_builder_ids, [(train_ids, train_ids + validation_ids)])
            baseline_builder.assert_called_once()
            self.assertEqual(baseline_builder.call_args.kwargs["modalities"], ("audio", "bpm", "genre"))
            baseline_evaluator.assert_called_once()
            self.assertEqual(baseline_evaluator.call_args.args[0].shape, (610, 2))
            self.assertEqual(baseline_evaluator.call_args.args[1:], (validation_ids, edges))
            self.assertEqual(len(torch_load_calls), len(SEEDS))
            self.assertEqual(set(torch_load_calls), {str(checkpoint_dir / f"C_seed_{seed}.pt") for seed in SEEDS})
            self.assertEqual(deserialized_checkpoint_bytes,
                             [f"checkpoint-{seed}".encode() for seed in SEEDS])
            self.assertEqual(embed_checkpoint_objects, [checkpoints[str(checkpoint_dir / f"C_seed_{seed}.pt")]
                                                        for seed in SEEDS])
            self.assertEqual(report["hashes"]["checkpoint_sha256"], {
                str(seed): hashlib.sha256(f"checkpoint-{seed}".encode()).hexdigest() for seed in SEEDS
            })
            self.assertEqual(len(eval_matrices), 1 + len(SEEDS) * len(WEIGHTS))
            self.assertEqual(len(report["metrics_by_seed_and_weight"]), 3)
            self.assertTrue(all(len(rows) == 5 for rows in report["metrics_by_seed_and_weight"].values()))
            for rows in report["metrics_by_seed_and_weight"].values():
                self.assertEqual(set(rows), {str(weight) for weight in WEIGHTS})
                self.assertEqual(rows["0.0"], {**baseline_values, "query_count": 238,
                                               "eligible_positive_edges": 185})
                self.assertEqual(rows["1.0"], {**model_values, "query_count": 238,
                                               "eligible_positive_edges": 185})
            self.assertEqual(report["hashes"]["train_fit_ids"], expected_fit_hash)
            saved = json.loads((output / "experiment.json").read_text())
            serialized = json.dumps(saved)
            self.assertNotIn("secret-test-id", serialized)
            self.assertNotIn("test_metrics", saved)
            self.assertIn("No test IDs, edges, or metrics are used", saved["provenance_caveats"][-1])
            self.assertEqual(saved["counts"], {"train_ids": 1, "validation_tracks": 610,
                                                "validation_edges": 185, "positive_queries": 238})

    def test_synthetic_runner_rejects_missing_c_reference_run(self):
        # Reference-run completeness is checked before any checkpoint or model work.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "neurotune_v1_split.json").write_text("{}")
            reference = root / "missing.json"
            reference.write_text(json.dumps({"baseline": {"validation": {}}, "runs": []}))
            with mock.patch("scripts.evaluate_neurotune_v1_validation_fusion._validation_inputs",
                            return_value=({}, ["train"], ["validation"], [], {}, {}, [])), \
                 mock.patch("scripts.evaluate_neurotune_v1_validation_fusion.build_neurotune_v1_features") as build:
                with self.assertRaisesRegex(ValueError, "all three C validation runs"):
                    run_validation_fusion(data, root / "checkpoints", reference, root / "out")
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()

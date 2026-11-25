import unittest
import tempfile
import hashlib
import json
import csv
import contextlib
import io
from unittest import mock
from pathlib import Path

import numpy as np

from src.evaluate_neurotune_v1_p1a_intra_genre_retrieval import (
    build_validation_candidate_galleries,
    evaluate_intra_genre_retrieval,
    evaluate_query_metrics,
)
from src.evaluate import evaluate_rankings
import src.evaluate_neurotune_v1_p1a_intra_genre_retrieval as retrieval


class RetrievalCliTests(unittest.TestCase):
    def test_help_exposes_only_data_and_results_roots(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as error:
            retrieval.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("--data-root", help_text)
        self.assertIn("--results-root", help_text)
        self.assertNotIn("--fold", help_text)
        self.assertNotIn("--seed", help_text)
        self.assertNotIn("--model", help_text)


class ValidationCandidateGalleryTests(unittest.TestCase):
    def setUp(self):
        self.validation_ids = ["e", "d", "c", "b", "a", "j", "i", "h", "g", "f", "k"]
        self.genres = {
            "a": "rock", "b": "rock", "c": "rock", "d": "rock", "e": "rock",
            "f": "jazz", "g": "jazz", "h": "jazz", "i": "jazz", "j": "jazz",
        }
        self.edges = [("a", "b"), ("a", "f"), ("c", "d"), ("g", "h")]

    def test_galleries_filter_candidates_and_relevant_edges_deterministically(self):
        result = build_validation_candidate_galleries(self.validation_ids, self.genres, self.edges)

        self.assertEqual(result["queries"]["a"]["gallery_ids"], ["b", "c", "d", "e"])
        self.assertNotIn("a", result["queries"]["a"]["gallery_ids"])
        self.assertEqual(result["queries"]["a"]["relevant_ids"], ["b"])
        self.assertEqual(result["queries"]["f"]["gallery_ids"], ["g", "h", "i", "j"])
        self.assertEqual(result["queries"]["f"]["relevant_ids"], [])
        self.assertEqual(result["queries"]["c"]["relevant_ids"], ["d"])

    def test_reasons_counts_and_small_gallery_flag(self):
        result = build_validation_candidate_galleries(self.validation_ids, self.genres, self.edges)
        queries = result["queries"]

        self.assertEqual(queries["a"]["reason"], "eligible")
        self.assertTrue(queries["a"]["eligible"])
        self.assertTrue(queries["a"]["small_gallery"])
        self.assertEqual(queries["g"]["reason"], "eligible")
        self.assertEqual(queries["k"]["reason"], "missing_or_unusable_genre")
        self.assertEqual(queries["e"]["reason"], "no_in_gallery_positive")
        self.assertEqual(queries["j"]["reason"], "no_in_gallery_positive")
        # A labeled genre singleton is distinguishable from an unusable label.
        singleton = build_validation_candidate_galleries(["only"], {"only": "classical"}, [])
        self.assertEqual(singleton["queries"]["only"]["reason"], "no_candidates")
        blank_label = build_validation_candidate_galleries(["only"], {"only": "  "}, [])
        self.assertEqual(blank_label["queries"]["only"]["reason"], "missing_or_unusable_genre")
        self.assertEqual(result["counts"], {
            "missing_or_unusable_genre": 1,
            "no_candidates": 0,
            "no_in_gallery_positive": 4,
            "eligible": 6,
            "eligible_small_gallery": 6,
        })

    def test_duplicate_or_malformed_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            build_validation_candidate_galleries(["a", "a"], self.genres, self.edges)
        with self.assertRaises(ValueError):
            build_validation_candidate_galleries(["a", ""], self.genres, self.edges)
        with self.assertRaises(ValueError):
            build_validation_candidate_galleries(self.validation_ids, {**self.genres, "outside": "rock"}, self.edges)
        with self.assertRaises(ValueError):
            build_validation_candidate_galleries(self.validation_ids, {**self.genres, "a": 42}, self.edges)
        with self.assertRaises(ValueError):
            build_validation_candidate_galleries(self.validation_ids, self.genres, [("a", "outside")])
        with self.assertRaises(ValueError):
            build_validation_candidate_galleries(self.validation_ids, self.genres, [("a", "a")])
        with self.assertRaises(ValueError):
            build_validation_candidate_galleries(self.validation_ids, self.genres, [("a", "b"), ("b", "a")])
        with self.assertRaises(ValueError):
            build_validation_candidate_galleries(self.validation_ids, self.genres, [("a",)])


class IntraGenreRetrievalTests(unittest.TestCase):
    def test_per_query_metrics_match_shared_ranking_evaluator(self):
        ranking = [
            {"song_id": "x", "score": 1.0},
            {"song_id": "a", "score": 0.9},
            {"song_id": "y", "score": 0.8},
            {"song_id": "b", "score": 0.7},
        ]
        relevant = {"a", "b", "c"}

        query_metrics = evaluate_query_metrics(ranking, relevant)
        aggregate_metrics = evaluate_rankings({"q": ranking}, {"q": relevant})

        for metric in ("Recall@10", "MRR@10", "NDCG@10"):
            self.assertAlmostEqual(query_metrics[metric], aggregate_metrics[metric])

    def test_evaluator_rejects_malformed_embeddings_and_candidate_ids(self):
        ids = ["q", "a"]
        galleries = build_validation_candidate_galleries(ids, {"q": "rock", "a": "rock"}, [("q", "a")])

        invalid_inputs = [
            ([[1.0], [1.0]], ["q", "q"], galleries),
            ([[1.0]], ids, galleries),
            ([[float("nan")], [1.0]], ids, galleries),
        ]
        for embeddings, song_ids, candidates in invalid_inputs:
            with self.subTest(embeddings=embeddings, song_ids=song_ids):
                with self.assertRaises(ValueError):
                    evaluate_intra_genre_retrieval(embeddings, song_ids, candidates)

        duplicate_gallery = {"queries": dict(galleries["queries"]), "counts": galleries["counts"]}
        duplicate_gallery["queries"]["q"] = {
            **duplicate_gallery["queries"]["q"], "gallery_ids": ["a", "a"]
        }
        with self.assertRaises(ValueError):
            evaluate_intra_genre_retrieval([[1.0], [1.0]], ids, duplicate_gallery)

        unknown_gallery = {"queries": dict(galleries["queries"]), "counts": galleries["counts"]}
        unknown_gallery["queries"]["q"] = {
            **unknown_gallery["queries"]["q"], "gallery_ids": ["outside"]
        }
        with self.assertRaises(ValueError):
            evaluate_intra_genre_retrieval([[1.0], [1.0]], ids, unknown_gallery)

        misaligned = {"queries": {"q": galleries["queries"]["q"]}, "counts": galleries["counts"]}
        with self.assertRaises(ValueError):
            evaluate_intra_genre_retrieval([[1.0], [1.0]], ids, misaligned)

    def test_evaluator_rejects_inconsistent_galleries_for_same_genre(self):
        ids = ["q", "a", "b"]
        galleries = build_validation_candidate_galleries(
            ids, {"q": "rock", "a": "rock", "b": "rock"}, [("q", "a"), ("a", "b")]
        )
        inconsistent = {"queries": dict(galleries["queries"]), "counts": galleries["counts"]}
        inconsistent["queries"]["q"] = {
            **inconsistent["queries"]["q"], "gallery_ids": ["a"], "gallery_size": 1
        }

        with self.assertRaises(ValueError):
            evaluate_intra_genre_retrieval([[1.0], [0.0], [0.5]], ids, inconsistent)

    def test_evaluator_rejects_gallery_member_with_different_genre(self):
        ids = ["q", "a", "x"]
        galleries = build_validation_candidate_galleries(
            ids,
            {"q": "rock", "a": "rock", "x": "jazz"},
            [("q", "a")],
        )
        # Keep the same-genre union consistent so this specifically checks raw genre labels.
        galleries["queries"]["q"] = {
            **galleries["queries"]["q"], "gallery_ids": ["a", "x"], "gallery_size": 2
        }
        galleries["queries"]["a"] = {
            **galleries["queries"]["a"], "gallery_ids": ["q", "x"], "gallery_size": 2
        }

        with self.assertRaises(ValueError):
            evaluate_intra_genre_retrieval([[1.0, 0.0], [0.8, 0.6], [1.0, 0.0]], ids, galleries)

    def test_evaluator_rejects_inconsistent_gallery_size_and_status_counts(self):
        ids = ["q", "a", "b"]
        valid = build_validation_candidate_galleries(
            ids,
            {"q": "rock", "a": "rock", "b": "jazz"},
            [("q", "a")],
        )
        bad_size = {"queries": dict(valid["queries"]), "counts": dict(valid["counts"])}
        bad_size["queries"]["q"] = {**bad_size["queries"]["q"], "gallery_size": 2}
        with self.assertRaises(ValueError):
            evaluate_intra_genre_retrieval([[1.0], [1.0], [0.0]], ids, bad_size)

        bad_counts = {"queries": dict(valid["queries"]), "counts": dict(valid["counts"])}
        bad_counts["counts"]["eligible"] += 1
        with self.assertRaises(ValueError):
            evaluate_intra_genre_retrieval([[1.0], [1.0], [0.0]], ids, bad_counts)

    def test_cross_genre_distractor_is_excluded_and_off_gallery_positive_is_not_counted(self):
        ids = ["q", "a", "b", "x"]
        galleries = build_validation_candidate_galleries(
            ids,
            {"q": "rock", "a": "rock", "b": "rock", "x": "jazz"},
            [("q", "a"), ("q", "x")],
        )
        embeddings = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [1.0, 0.0]]

        result = evaluate_intra_genre_retrieval(embeddings, ids, galleries)

        self.assertEqual([item["song_id"] for item in result["rankings"]["q"]], ["a", "b"])
        self.assertEqual(result["per_query"]["q"]["positive_count"], 1)
        self.assertEqual(result["metrics"]["micro_recall@10"], 1.0)

    def test_no_eligible_queries_and_small_gallery_are_reported(self):
        small = build_validation_candidate_galleries(
            ["q", "a"], {"q": "rock", "a": "rock"}, [("q", "a")]
        )
        result = evaluate_intra_genre_retrieval([[1.0], [1.0]], ["q", "a"], small)
        self.assertEqual(result["counts"]["eligible_small_gallery"], 2)
        self.assertEqual(result["small_gallery_metrics"]["query_count"], 2)

        none = build_validation_candidate_galleries(["q"], {"q": "rock"}, [])
        empty = evaluate_intra_genre_retrieval([[1.0]], ["q"], none)
        self.assertEqual(empty["counts"]["eligible"], 0)
        self.assertIsNone(empty["metrics"]["macro_query"]["Recall@10"])
        self.assertIsNone(empty["metrics"]["micro_recall@10"])

    def test_genre_group_metrics_and_micro_recall_use_positive_edge_denominator(self):
        ids = ["q1", "a1", "q2", "a2", "b2", "c2", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
        genres = {track_id: "rock" for track_id in ids[:2]}
        genres.update({track_id: "jazz" for track_id in ids[2:]})
        galleries = build_validation_candidate_galleries(
            ids, genres, [("q1", "a1"), ("q2", "a2"), ("q2", "b2"), ("q2", "c2")]
        )
        distractor = [0.95, (1 - 0.95 ** 2) ** 0.5]
        vectors = {"q1": [1.0, 0.0], "a1": [1.0, 0.0], "q2": [1.0, 0.0],
                   "a2": [0.8, 0.6], "b2": [0.0, 1.0], "c2": [0.0, 1.0]}
        vectors.update({track_id: distractor for track_id in ids[6:]})
        result = evaluate_intra_genre_retrieval([vectors[i] for i in ids], ids, galleries)

        self.assertEqual(set(result["per_genre"]), {"jazz", "rock"})
        self.assertAlmostEqual(result["metrics"]["macro_query"]["Recall@10"], 5 / 9)
        self.assertAlmostEqual(result["metrics"]["genre_macro"]["Recall@10"], 2 / 3)
        self.assertEqual(result["metrics"]["micro_recall@10"], 0.5)


class IntegratedRunnerContractTests(unittest.TestCase):
    def test_protected_path_hash_matches_frozen_runner_for_files_and_directories(self):
        from scripts.run_neurotune_v1_full_modal_ablation import _hash_path as canonical_hash_path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file_path = root / "one.txt"
            file_path.write_text("file fixture", encoding="utf-8")
            nested_dir = root / "nested"
            (nested_dir / "subdir").mkdir(parents=True)
            (nested_dir / "first.txt").write_text("first", encoding="utf-8")
            (nested_dir / "subdir" / "second.txt").write_bytes(b"second\x00")

            self.assertEqual(retrieval._hash_path(file_path), canonical_hash_path(file_path))
            self.assertEqual(retrieval._hash_path(nested_dir), canonical_hash_path(nested_dir))

    def test_runner_integrates_validation_only_artifacts_and_publishes_atomically(self):
        from scripts.run_neurotune_v1_full_modal_ablation import FOLDS, SEEDS, INPUTS as D0_INPUTS
        from scripts.run_neurotune_v1_d4_multi_window_audio import INPUTS as D4_INPUTS
        from src.evaluate_neurotune_v1 import evaluate_neurotune_v1_retrieval

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data_root, results_root = base / "data", base / "results"
            data_root.mkdir(); results_root.mkdir()
            for relpath in set(D0_INPUTS.values()) | set(D4_INPUTS.values()):
                path = data_root / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("integrity fixture", encoding="utf-8")
            for name in ("neurotune_v1_full_modal_ablation", "neurotune_v1_d4_multi_window_audio",
                         "neurotune_v1_d4_modality_sensitivity", "neurotune_v1_d4_triplet_diagnostics",
                         "neurotune_v1_d_modality_sensitivity", "neurotune_v1_training_improvements",
                         "neurotune_v1_d2_modality_dropout", "neurotune_v1_d3_combined",
                         "neurotune_v1_d4_multi_window_audio_pilot", "neurotune_v1_abcd_cv"):
                (results_root / name).mkdir()

            test_id = "TEST_EDGE_CANARY_unsafe"
            validation = ["q", "a", "b", "j", "u", "m", "solo"]
            train = ["train_only"]
            catalog = {key: {"genre": genre} for key, genre in {
                "q": "rock", "a": "rock", "b": "rock", "j": "jazz", "u": "jazz",
                "m": None, "solo": "blues", "train_only": "rock", test_id: "rock"}.items()}
            folds = [{"fold": fold, "train": list(train), "validation": list(validation), "test": [test_id],
                      "validation_edges": [("q", "a")], "test_edges": [("q", test_id)]} for fold in FOLDS]
            loaded = {"folds": folds, "catalog": catalog, "manifest": {"synthetic": True},
                      "cache_path": base / "cache"}
            train_val = train + validation
            ids = list(validation)
            vectors = np.asarray([[1., 0.], [.9, .1], [.2, .8], [0., 1.], [.1, .9], [1., 1.], [.5, .5]])
            full = evaluate_neurotune_v1_retrieval(vectors, ids, [("q", "a")])
            expected = {key: full[key] for key in ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10", "query_count")}
            expected["eligible_positive_edges"] = full["unique_undirected_edge_count"]
            d0_transform = {"fit_ids_hash": "train", "code_sha256": {"src/features_neurotune_v1.py": "current"}}
            d0_manifest = {**d0_transform, "code_sha256": {"src/features_neurotune_v1.py": "frozen"}}
            d4_transform = {"fit_ids_hash": "train", "code_sha256": {"src/features_neurotune_v1.py": "d4"}}
            blob_map, checkpoint_objects, schedule_hashes, schedule_copy_hashes = {}, {}, {}, {}
            d0refs, d4refs = [], []
            schedules = results_root / "neurotune_v1_abcd_cv" / "triplet_schedules"
            d0root = results_root / "neurotune_v1_full_modal_ablation"
            d4root = results_root / "neurotune_v1_d4_multi_window_audio"
            for fold in FOLDS:
                for seed in SEEDS:
                    train_schedule = np.asarray(["train_only"])
                    val_schedule = np.asarray(validation)
                    schedules.mkdir(parents=True, exist_ok=True)
                    source = schedules / f"fold_{fold}_seed_{seed}.npz"
                    np.savez(source, train_ids=train_schedule, validation_ids=val_schedule,
                             train_schedule=np.asarray([["train_only"]]), validation_schedule=np.asarray([["q", "a"]]))
                    schedule_bytes = source.read_bytes()
                    schedule_hashes[(fold, seed)] = hashlib.sha256(schedule_bytes).hexdigest()
                    d4_copy = d4root / "triplet_schedules" / source.name
                    d4_copy.parent.mkdir(parents=True, exist_ok=True); d4_copy.write_bytes(schedule_bytes)
                    schedule_copy_hashes[f"{fold}/{seed}"] = schedule_hashes[(fold, seed)]
                    for variant, root, transform, manifest in (("D0", d0root, d0_transform, d0_manifest),
                                                               ("D4", d4root, d4_transform, d4_transform)):
                        checkpoint_path = root / "checkpoints" / f"fold_{fold}_D_seed_{seed}.pt"
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        blob = f"checkpoint-{fold}-{seed}-{variant}".encode()
                        blob_map[(fold, seed, variant)] = blob
                        checkpoint_path.write_bytes(blob)
                        cp_hash = hashlib.sha256(blob).hexdigest()
                        transform_hash = retrieval._digest(manifest)
                        obj = {"variant": "D", "fold": fold, "seed": seed, "epoch": 2,
                               "transform_manifest": manifest, "selection_metrics": expected,
                               "provenance": {"fold": fold, "seed": seed, "variant": "D",
                                              "schedule_sha256": schedule_hashes[(fold, seed)],
                                              "transform_sha256": transform_hash}}
                        checkpoint_objects[blob] = obj
                        ref = {"variant": "D", "fold": fold, "seed": seed,
                               "checkpoint_path": f"checkpoints/fold_{fold}_D_seed_{seed}.pt",
                               "checkpoint_sha256": cp_hash, "schedule_sha256": schedule_hashes[(fold, seed)],
                               "transform_sha256": transform_hash, "selected_epoch": 2, "validation": expected}
                        (d0refs if variant == "D0" else d4refs).append(ref)
            d0report = {"status": "complete", "model_validation": d0refs,
                        "hashes": {"data_sources": {}}}
            d4report = {"status": "complete", "model_validation": d4refs,
                        "evaluation_scope": "validation_only_development", "independent_test_artifacts_created": False,
                        "hashes": {"schedule_copies": schedule_copy_hashes}}
            hash_calls = {}
            def hash_path(path):
                path = Path(path)
                if path.exists() and path.is_file():
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                else:
                    digest = hashlib.sha256(str(path).encode()).hexdigest()
                hash_calls[str(path)] = digest
                return digest
            d0report["hashes"]["data_sources"] = {
                key: hash_path(data_root / relpath) for key, relpath in D0_INPUTS.items()
            }
            d4report["hashes"]["protected_before"] = {
                **{f"input:{key}": hash_path(data_root / relpath)
                   for key, relpath in {**D0_INPUTS, **D4_INPUTS}.items()},
                "d0_report": hash_path(results_root / "neurotune_v1_full_modal_ablation" / "experiment.json"),
                **{f"checkpoint:{fold}/{seed}": hashlib.sha256(blob_map[(fold, seed, "D0")]).hexdigest()
                   for fold in FOLDS for seed in SEEDS},
                **{f"schedule:{fold}/{seed}": schedule_hashes[(fold, seed)]
                   for fold in FOLDS for seed in SEEDS},
            }
            def loader(**kwargs):
                return loaded
            def mxm_builder(view, path):
                self.assertNotIn(test_id, view["catalog"], "test canary reached lyric/MXM feature parsing")
                self.assertEqual(set(view["catalog"]), set(train_val))
                return {track_id: np.zeros(5000, dtype=np.float32) for track_id in train_val}, {"synthetic": True}
            def features_builder(_loaded, fold, lyrics):
                self.assertNotIn(test_id, fold["train"] + fold["validation"])
                n = len(train_val)
                return {"audio": [np.ones((1, 26), dtype=np.float32) for _ in train_val],
                        "bpm": np.ones((n, 1), dtype=np.float32), "genre": np.ones((n, 2), dtype=np.float32),
                        "lyrics": np.ones((n, 5000), dtype=np.float32)}, d0_transform
            def d4_builder(_loaded, fold, lyrics, cache):
                self.assertNotIn(test_id, fold["train"] + fold["validation"])
                n = len(train_val)
                return {"audio_windows": {track_id: [np.ones((1, 26), dtype=np.float32), np.zeros((0, 26), dtype=np.float32), np.zeros((0, 26), dtype=np.float32)] for track_id in train_val},
                        "window_valid": np.tile(np.asarray([[True, False, False]]), (n, 1)),
                        "bpm": np.ones((n, 1), dtype=np.float32), "genre": np.ones((n, 2), dtype=np.float32),
                        "lyrics": np.ones((n, 5000), dtype=np.float32)}, d4_transform
            feature_calls = []
            original_features_builder, original_d4_builder = features_builder, d4_builder
            def features_builder(_loaded, fold, lyrics):
                feature_calls.append((fold["fold"], tuple(fold["train"]), tuple(fold["validation"])))
                return original_features_builder(_loaded, fold, lyrics)
            def d4_builder(_loaded, fold, lyrics, cache):
                feature_calls.append((fold["fold"], tuple(fold["train"]), tuple(fold["validation"])))
                return original_d4_builder(_loaded, fold, lyrics, cache)
            baseline_calls = []
            def passthrough_baseline(features, ids):
                baseline_calls.append(tuple(ids))
                return vectors
            deps = {"loader": loader, "mxm_builder": mxm_builder, "d0_builder": features_builder,
                    "d4_builder": d4_builder, "reports": {"D0": d0report, "D4": d4report},
                    "checkpoint_bytes": lambda key: blob_map[key],
                    "checkpoint_loader": lambda blob: checkpoint_objects[blob],
                    "model_loader": lambda checkpoint, variant, features: object(),
                    "embedding_builder": lambda model, batch, variant, fold, seed: vectors,
                    "d0_baseline_builder": passthrough_baseline, "d4_baseline_builder": passthrough_baseline,
                    "hash_path": hash_path}
            protected_before = {str(path): hash_path(path) for path in [
                *(data_root / relpath for relpath in set(D0_INPUTS.values()) | set(D4_INPUTS.values())),
                *(results_root / name for name in ("neurotune_v1_full_modal_ablation", "neurotune_v1_d4_multi_window_audio",
                    "neurotune_v1_d4_modality_sensitivity", "neurotune_v1_d4_triplet_diagnostics", "neurotune_v1_abcd_cv"))]}
            report = retrieval._run(data_root, results_root, _test_only=True, _dependencies=deps)
            self.assertEqual(report["status"], "complete")
            published = results_root / retrieval.OUTPUT_NAME
            self.assertTrue(published.is_dir())
            expected_files = {"experiment.json", "per_query_metrics.csv", "per_fold_seed_genre_metrics.csv", "gallery_coverage.csv", "analysis.md"}
            self.assertEqual({path.name for path in published.iterdir()}, expected_files)
            experiment = json.loads((published / "experiment.json").read_text(encoding="utf-8"))
            self.assertGreater(experiment["runtime"]["seconds"], 0.0)
            p1_code_key = "code:src/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py"
            expected_p1_hash = hashlib.sha256(Path(retrieval.__file__).read_bytes()).hexdigest()
            self.assertEqual(experiment["hashes"]["protected_before"][p1_code_key], expected_p1_hash)
            self.assertEqual(experiment["hashes"]["protected_after"][p1_code_key], expected_p1_hash)
            cli_code_key = "code:scripts/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py"
            cli_path = Path(retrieval.__file__).resolve().parents[1] / "scripts" / "evaluate_neurotune_v1_p1a_intra_genre_retrieval.py"
            expected_cli_hash = hashlib.sha256(cli_path.read_bytes()).hexdigest()
            self.assertEqual(experiment["hashes"]["protected_before"][cli_code_key], expected_cli_hash)
            self.assertEqual(experiment["hashes"]["protected_after"][cli_code_key], expected_cli_hash)
            self.assertEqual(experiment["protocol"]["publication_strategy"], "linux_renameat2_rename_noreplace")
            self.assertEqual(set(experiment["protocol"]["systems"]), {"D0-neural", "D4-neural", "D0-cosine", "D4-cosine"})
            self.assertEqual(len(feature_calls), 6)
            self.assertEqual(len(baseline_calls), 6)
            self.assertTrue(all(call[1] == tuple(train) and call[2] == tuple(validation) for call in feature_calls))
            self.assertTrue(all(ids == tuple(validation) for ids in baseline_calls))
            with (published / "per_query_metrics.csv").open(newline="", encoding="utf-8") as stream:
                query_rows = list(csv.DictReader(stream))
            for name in ("per_fold_seed_genre_metrics.csv", "gallery_coverage.csv"):
                with (published / name).open(newline="", encoding="utf-8") as stream:
                    self.assertTrue(list(csv.DictReader(stream)))
            self.assertFalse(any(test_id in json.dumps(row) for row in query_rows))
            by_query = {}
            for row in query_rows:
                key = (row["fold"], row["query_id"])
                by_query.setdefault(key, set()).add((row["candidate_hash"], row["label_hash"]))
                self.assertEqual(int(row["in_gallery_positive_count"]), 0 if row["query_id"] not in ("q", "a") else 1)
            self.assertTrue(all(len(hashes) == 1 for hashes in by_query.values()))
            self.assertEqual(len(by_query), len(FOLDS) * len(validation))
            self.assertEqual(len({row["system"] for row in query_rows}), 4)
            self.assertTrue(all(row["reason"] in {"eligible", "no_in_gallery_positive", "missing_or_unusable_genre", "no_candidates"}
                                for row in query_rows))
            self.assertEqual({row["reason"] for row in query_rows},
                             {"eligible", "no_in_gallery_positive", "missing_or_unusable_genre", "no_candidates"})
            self.assertTrue(all(row["small_gallery"] == "True" for row in query_rows if row["query_id"] in ("q", "a")))
            for row in experiment["records"]:
                self.assertEqual(row["directed_query_relevance_count"], 2)
                self.assertEqual(row["unique_undirected_in_genre_positive_edge_count"], 1)
            genre_csv = (published / "per_fold_seed_genre_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("Recall@10", genre_csv); self.assertIn("MRR@10", genre_csv); self.assertIn("NDCG@10", genre_csv)
            with (published / "per_fold_seed_genre_metrics.csv").open(newline="", encoding="utf-8") as stream:
                macro_rows = [row for row in csv.DictReader(stream) if row["genre"] == "__macro_query__"]
            self.assertTrue(macro_rows)
            for row in macro_rows:
                self.assertEqual(float(row["genre_macro_recall@10"]), 1.0)
                self.assertEqual(float(row["genre_macro_mrr@10"]), 1.0)
                self.assertEqual(float(row["genre_macro_ndcg@10"]), 1.0)
                self.assertEqual(int(row["directed_query_relevance_count"]), 2)
                self.assertEqual(int(row["unique_undirected_in_genre_positive_edge_count"]), 1)
            analysis = (published / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("no_in_gallery_positive", analysis)
            self.assertIn("incomplete", analysis.lower())
            self.assertIn("Genre-macro R@10 / MRR@10 / NDCG@10", analysis)
            self.assertIn("Directed relevance / unique undirected edges", analysis)
            self.assertIn("1.0 / 1.0 / 1.0", analysis)
            self.assertIn("2 / 1", analysis)
            self.assertNotIn(test_id, "".join(path.read_text(encoding="utf-8") for path in published.iterdir()))
            protected_after = {str(path): hash_path(path) for path in map(Path, protected_before)}
            self.assertEqual(protected_before, protected_after)
            with self.assertRaises(FileExistsError):
                retrieval._run(data_root, results_root, _test_only=True, _dependencies=deps)
            staged = results_root / ".publication-stage"
            staged.mkdir(); (staged / "partial.txt").write_text("staged only", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                retrieval._atomic_publish(staged, published)
            self.assertTrue((staged / "partial.txt").is_file())
            self.assertFalse((published / "partial.txt").exists())

    def test_renameat2_unavailable_fails_fast_without_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage, final = Path(temporary) / "stage-fixture", Path(temporary) / "final-fixture"
            stage.mkdir()
            (stage / "payload").write_text("stage", encoding="utf-8")
            with mock.patch.object(retrieval.ctypes, "CDLL", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "renameat2"):
                    retrieval._atomic_publish(stage, final)
                calls = []
                with self.assertRaisesRegex(RuntimeError, "renameat2"):
                    retrieval._run(_test_only=True, _dependencies={"loader": lambda **_: calls.append(True)})
            self.assertEqual(calls, [], "unsupported platform must fail before loading production data")
            with mock.patch.object(retrieval.platform, "system", return_value="Darwin"):
                with self.assertRaisesRegex(RuntimeError, "Linux renameat2"):
                    retrieval._run(_test_only=True, _dependencies={"loader": lambda **_: calls.append(True)})
                self.assertEqual(calls, [])
            self.assertFalse(final.exists())
            self.assertEqual((stage / "payload").read_text(encoding="utf-8"), "stage")

    def test_runner_rejects_injected_seams_in_production_mode(self):
        with self.assertRaisesRegex(ValueError, "test-only"):
            retrieval._run(_dependencies={"loader": lambda **_: {}})


if __name__ == "__main__":
    unittest.main()

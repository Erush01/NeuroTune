import tempfile
import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from src.evaluate_neurotune_v1_p1a_intra_genre_retrieval import build_validation_candidate_galleries
from src.data.neurotune_v1_dataset import neurotune_v1_collate
from src.evaluate_neurotune_v1_p1b_audio_branch_diagnostic import (
    OUTPUT_NAME, _require_fresh_output, _run, evaluate_paired_representations,
    extract_pre_fusion_audio,
)


class ProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_projection = nn.Linear(2, 64, bias=False)
        with torch.no_grad():
            self.audio_projection.weight.zero_()
            self.audio_projection.weight[0, 0] = 1
            self.audio_projection.weight[1, 1] = 1
        self.grad_enabled = None
        self.fused = None

    def forward(self, **batch):
        self.grad_enabled = torch.is_grad_enabled()
        if "audio_windows" in batch:
            values = batch["audio_windows"][batch["window_mask"]].mean(dim=1)
        else:
            values = batch["audio"]
        raw = self.audio_projection(values)
        self.fused = torch.full((raw.shape[0], 64), 999.0)
        return self.fused


class AudioExtractionTests(unittest.TestCase):
    def test_d0_is_raw_projection_and_not_fused_embedding(self):
        model = ProjectionModel().eval()
        result = extract_pre_fusion_audio(model, {"audio": torch.tensor([[3., 4.]])}, "D0")
        np.testing.assert_allclose(result[0, :2], [0.6, 0.8])
        self.assertNotEqual(float(result[0, 0]), 999.0)
        self.assertFalse(model.grad_enabled)
        self.assertEqual(model.audio_projection._forward_hooks, {})

    def test_d4_averages_only_masked_valid_window_projections(self):
        model = ProjectionModel().eval()
        batch = {
            "audio_windows": torch.tensor([[[[1., 0.]], [[0., 2.]], [[100., 100.]]]]),
            "window_mask": torch.tensor([[True, True, False]]),
        }
        result = extract_pre_fusion_audio(model, batch, "D4")
        np.testing.assert_allclose(result[0, :2], [1 / np.sqrt(5), 2 / np.sqrt(5)])
        self.assertEqual(model.audio_projection._forward_hooks, {})

    def test_projection_call_count_and_eval_mode_are_required(self):
        model = ProjectionModel()
        with self.assertRaisesRegex(ValueError, "eval-mode"):
            extract_pre_fusion_audio(model, {"audio": torch.ones(1, 2)}, "D0")
        model.eval()

        class Twice(ProjectionModel):
            def forward(self, **batch):
                self.audio_projection(batch["audio"])
                return self.audio_projection(batch["audio"])

        model = Twice().eval()
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            extract_pre_fusion_audio(model, {"audio": torch.ones(1, 2)}, "D0")
        self.assertEqual(model.audio_projection._forward_hooks, {})

    def test_real_collate_metadata_is_not_forwarded_to_strict_model(self):
        class StrictModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_projection = nn.Linear(2, 64, bias=False)

            def forward(self, audio, mask, bpm, genre, lyrics):
                return self.audio_projection(audio[:, 0, :2])

        row = {"track_id": "track-1", "audio": np.ones((1, 26), np.float32),
               "bpm": np.ones(1, np.float32), "genre": np.ones(2, np.float32),
               "lyrics": np.ones(5000, np.float32)}
        batch = neurotune_v1_collate([row])
        self.assertIn("lengths", batch)
        self.assertIn("track_ids", batch)
        result = extract_pre_fusion_audio(StrictModel().eval(), batch, "D0")
        self.assertEqual(result.shape, (1, 64))


class PairedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["a", "b", "c", "d"]
        galleries = build_validation_candidate_galleries(
            self.ids, {i: "rock" for i in self.ids}, [("a", "b"), ("c", "d")])
        self.records = {(0, 684): {
            variant: {"validation_ids": list(self.ids), "candidate_galleries": galleries,
                      "embeddings": np.eye(4, 64)} for variant in ("D0", "D4")}}

    def test_same_gallery_parity_and_two_branch_diagnostic_without_training(self):
        with mock.patch("src.train_neurotune_v1.train_neurotune_v1") as trainer:
            result = evaluate_paired_representations(self.records)
        trainer.assert_not_called()
        self.assertEqual(set(result[(0, 684)]), {"D0", "D4"})
        self.assertEqual(result[(0, 684)]["D0"]["candidate_hashes"],
                         result[(0, 684)]["D4"]["candidate_hashes"])
        self.assertEqual(result[(0, 684)]["D0"]["label_hashes"],
                         result[(0, 684)]["D4"]["label_hashes"])

    def test_test_id_leakage_and_gallery_provenance_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "test IDs"):
            evaluate_paired_representations(self.records, {"a"})
        broken = {(0, 684): dict(self.records[(0, 684)])}
        changed = dict(broken[(0, 684)]["D4"])
        changed["candidate_galleries"] = build_validation_candidate_galleries(
            self.ids, {i: "jazz" for i in self.ids}, [("a", "b"), ("c", "d")])
        broken[(0, 684)]["D4"] = changed
        with self.assertRaisesRegex(ValueError, "identical ordered validation IDs and galleries"):
            evaluate_paired_representations(broken)

    def test_output_namespace_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                _require_fresh_output(target)
            _require_fresh_output(Path(directory) / "new")


class SyntheticRunnerTests(unittest.TestCase):
    class BranchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_projection = nn.Linear(2, 64, bias=False)
            with torch.no_grad():
                self.audio_projection.weight.zero_()
                self.audio_projection.weight[0, 0] = 1
                self.audio_projection.weight[1, 1] = 1

        def forward(self, audio=None, audio_windows=None, mask=None, window_mask=None,
                    bpm=None, genre=None, lyrics=None):
            if audio_windows is None:
                values = audio[:, :2]
            else:
                values = audio_windows[window_mask].mean(dim=1)
            return self.audio_projection(values)

    def _run_synthetic(self, output_root, *, bad_provenance=False, bad_selection=False):
        ids = ["a", "b", "c", "d"]
        blobs = {"D0": b"frozen-d0", "D4": b"frozen-d4"}
        cp_records, refs = {}, {"D0": {}, "D4": {}}
        for variant in ("D0", "D4"):
            cp_records[variant] = {
                "variant": "D", "fold": 0, "seed": 684, "epoch": 1,
                "provenance": {"variant": "D", "fold": 0 if not bad_provenance else 9,
                               "seed": 684, "schedule_sha256": "schedule"},
                "selection_metrics": {"Recall@10": 0.0, "HitRate@10": 0.0, "MRR@10": 0.0,
                                      "NDCG@10": 0.0, "query_count": 1, "eligible_positive_edges": 1},
                "transform_manifest": {},
            }
        if bad_selection:
            cp_records["D0"]["selection_metrics"]["Recall@10"] = 0.25
        expected_checkpoint_hash = {v: hashlib.sha256(blobs[v]).hexdigest() for v in blobs}
        input_names = {"train_edges_path": "train.csv", "cache_path": "cache.h5",
                       "cache_metadata_path": "cache.json", "audit_path": "audit.json",
                       "catalog_path": "catalog.csv", "test_edges_path": "test.csv",
                       "rejection_path": "reject.json", "ab_catalog_path": "ab.csv",
                       "ab_split_path": "split.json", "ab_test_edges_path": "ab_test.csv",
                       "cv_dir": "cv", "mxm_path": "mxm", "d4_cache_path": "d4.h5"}
        def hash_path(path):
            path = str(path)
            if "triplet_schedules" in path:
                return "schedule"
            for v, blob in blobs.items():
                if ("full_modal_ablation" in path and v == "D0") or ("d4_multi_window_audio" in path and v == "D4"):
                    if "fold_0_D_seed_684.pt" in path:
                        return expected_checkpoint_hash[v]
            return hashlib.sha256(path.encode()).hexdigest()
        def loader(**_kwargs):
            return {"manifest": {}, "cache_path": "fake-cache",
                    "catalog": {track: {"genre": "rock"} for track in [*ids, "train", "test"]},
                    "folds": [{"fold": 0, "train": ["train"], "validation": ids,
                               "test": ["test"], "validation_edges": [("a", "b"), ("c", "d")]}]}
        def builder(fi, development, lyrics):
            self.assertNotIn("test", fi["catalog"])
            return {"genre": np.ones((5, 2), np.float32)}, {"identity": "synthetic"}
        def slice_features(features, _ids, chosen):
            return {"genre": np.ones((len(chosen), 2), np.float32)}
        def batch_builder(variant, chosen, _features):
            base = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])
            batch = {"mask": torch.ones((len(chosen), 1), dtype=torch.bool),
                     "bpm": torch.ones((len(chosen), 1)), "genre": torch.ones((len(chosen), 2)),
                     "lyrics": torch.ones((len(chosen), 2)), "track_ids": list(chosen),
                     "lengths": torch.ones(len(chosen), dtype=torch.long)}
            if variant == "D0":
                batch["audio"] = base
            else:
                windows = torch.zeros((len(chosen), 3, 1, 2))
                windows[:, 0, 0] = base
                windows[:, 1, 0] = base
                batch.update(audio_windows=windows, mask=torch.ones((len(chosen), 3, 1), dtype=torch.bool),
                             window_mask=torch.tensor([[True, True, False]] * len(chosen)))
            return batch
        references = {v: {(0, 684): {
            "checkpoint_path": f"checkpoints/fold_0_D_seed_684.pt",
            "checkpoint_sha256": expected_checkpoint_hash[v], "schedule_sha256": "schedule",
            "selected_epoch": 1, "validation": {"Recall@10": 0.0, "HitRate@10": 0.0,
                                                  "MRR@10": 0.0, "NDCG@10": 0.0,
                                                  "query_count": 1, "eligible_positive_edges": 1}}}
            for v in ("D0", "D4")}
        report = {"status": "complete", "evaluation_scope": "validation_only_development",
                  "independent_test_artifacts_created": False,
                  "hashes": {"schedule_copies": {"0/684": "schedule"}}}
        deps = {
            "folds": (0,), "seeds": (684,), "d0_inputs": input_names,
            "d4_inputs": {"d4_cache_path": "d4.h5"}, "hash_path": hash_path,
            "reports": {"D0": report, "D4": report}, "references": references,
            "report_validator": lambda *_: None, "loader": loader,
            "mxm_builder": lambda *_: ({}, {"mapping_sha256": "synthetic"}),
            "d0_builder": builder, "d4_builder": lambda fi, dev, lyrics, _cache: builder(fi, dev, lyrics),
            "d0_slice": slice_features, "d4_slice": slice_features,
            "checkpoint_bytes": lambda key: blobs[key[2]],
            "checkpoint_loader": lambda blob: cp_records["D0" if blob == blobs["D0"] else "D4"],
            "schedule_validator": lambda *args: "schedule",
            "checkpoint_validator": lambda *_: None,
            "d0_transform_validator": lambda *_: None,
            "model_loader": lambda *_: self.BranchModel().eval(),
            "batch_builder": batch_builder,
        }
        result = _run(Path("/fake-data"), output_root, _test_only=True, _dependencies=deps)
        return result

    def test_synthetic_runner_publishes_paired_results_without_training_or_test_use(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("src.train_neurotune_v1.train_neurotune_v1") as trainer:
            result = self._run_synthetic(Path(directory))
            trainer.assert_not_called()
            output = Path(directory) / OUTPUT_NAME
            self.assertTrue((output / "experiment.json").is_file())
            self.assertTrue((output / "per_query_metrics.csv").is_file())
            self.assertTrue((output / "fold_query_coverage.csv").is_file())
            self.assertTrue((output / "analysis.md").is_file())
            per_query_csv = (output / "per_query_metrics.csv").read_text()
            header = per_query_csv.splitlines()[0]
            self.assertNotIn("gallery_ids", header)
            self.assertNotIn("relevant_ids", header)
            self.assertNotIn("gallery_ids", per_query_csv)
            self.assertNotIn("relevant_ids", per_query_csv)
            report = json.loads((output / "experiment.json").read_text())
            self.assertEqual(len(report["records"]), 2)
            self.assertIn("mxm_feature_source_provenance", report)
            self.assertNotIn("negative_exclusion_provenance", report)
            self.assertIn("MXM text source files are read", report["test_boundary"])
            self.assertIn("train+validation catalog IDs", report["test_boundary"])
            self.assertIn("Held-out ABCD test IDs/labels are not materialized", report["test_boundary"])
            self.assertIn("only as frozen schedule/checkpoint provenance",
                          report["schedule_positive_exclusion_policy"])
            self.assertIn("does not resample triplets or train",
                          report["schedule_positive_exclusion_policy"])
            self.assertEqual(result["records"][0]["candidate_hashes"], result["records"][1]["candidate_hashes"])
            self.assertEqual(result["records"][0]["label_hashes"], result["records"][1]["label_hashes"])
            analysis = (output / "analysis.md").read_text()
            self.assertIn("not as independently trained audio-only models", analysis)
            self.assertIn("D4−D0", analysis)
            self.assertIn("Fold/seed eligibility coverage", analysis)
            self.assertIn("MXM text source files are read", analysis)
            self.assertIn("Held-out ABCD test IDs/labels are not materialized", analysis)
            self.assertIn("official/legacy test-positive exclusion policy", analysis)
            lines = analysis.splitlines()
            aggregate_heading = lines.index("## Fold/seed aggregate metrics")
            coverage_heading = lines.index("## Fold/seed eligibility coverage")
            aggregate_header = next(i for i in range(aggregate_heading, coverage_heading)
                                    if lines[i].startswith("| Fold | Seed | D0 R@10"))
            aggregate_rows = [line for line in lines[aggregate_header + 2:coverage_heading]
                              if line.startswith("|")]
            self.assertTrue(aggregate_rows)
            self.assertLess(aggregate_header, lines.index(aggregate_rows[0]))
            self.assertLess(lines.index(aggregate_rows[-1]), coverage_heading)
            aggregate_columns = lines[aggregate_header].count("|")
            self.assertTrue(all(row.count("|") == aggregate_columns for row in aggregate_rows))
            coverage_header = next(i for i in range(coverage_heading, len(lines))
                                    if lines[i].startswith("| Fold | Seed | System |"))
            coverage_rows = [line for line in lines[coverage_header + 2:] if line.startswith("|")]
            self.assertTrue(coverage_rows)
            self.assertTrue(all(row.count("|") == lines[coverage_header].count("|")
                                for row in coverage_rows))

    def test_synthetic_checkpoint_provenance_failure_and_existing_output_do_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / OUTPUT_NAME
            with self.assertRaisesRegex(ValueError, "checkpoint fold/seed/epoch/schedule"):
                self._run_synthetic(Path(directory), bad_provenance=True)
            self.assertFalse(target.exists())
            target.mkdir()
            with self.assertRaises(FileExistsError):
                self._run_synthetic(Path(directory))

    def test_synthetic_selection_metrics_must_match_reference_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "parity mismatch"):
                self._run_synthetic(Path(directory), bad_selection=True)
            self.assertFalse((Path(directory) / OUTPUT_NAME).exists())


if __name__ == "__main__":
    unittest.main()

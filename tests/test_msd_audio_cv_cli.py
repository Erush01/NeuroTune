import csv
import hashlib
import json
import tempfile
import unittest
import gc
import weakref
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from scripts.audit_msd_audio_cv import _source_sha256
import scripts.run_msd_audio_cv as cli


def _channel_names():
    return ([f"pitch_{index + 1}" for index in range(12)]
            + [f"timbre_{index + 1}" for index in range(12)]
            + ["loudness_max", "beat_count", "bar_count", "section_count"])


class MsdAudioCvCliTests(unittest.TestCase):
    def _fixture(self, root, bad_normalization_fold=None):
        ids_by_fold = {fold: [f"TR{fold}A", f"TR{fold}B"] for fold in range(3)}
        all_ids = sorted(track_id for ids in ids_by_fold.values() for track_id in ids)
        folds = {}
        for fold, ids in ids_by_fold.items():
            folds[str(fold)] = {"track_ids": ids, "positive_edges": [[*ids]], "track_count": 2,
                                "artist_count": 2, "positive_edge_count": 1}
        lastfm = root / "lastfm"
        lastfm.mkdir()
        (lastfm / "source.json").write_text("{}")
        audit = {"status": "ready", "fold_count": 3, "catalog_counts": {"eligible_tracks": 6},
                 "folds": folds, "source_sha256": {"lastfm": _source_sha256(lastfm)}}
        audit_path = root / "audit.json"
        audit_path.write_text(json.dumps(audit))
        audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        cache_dir, metadata_dir = root / "cache", root / "metadata"
        cache_dir.mkdir()
        metadata_dir.mkdir()
        sequences = {track_id: np.full((1024, 28), index + 1, dtype=np.float32)
                     for index, track_id in enumerate(all_ids)}
        for fold in range(3):
            train_ids = [track_id for other in range(3) if other != fold for track_id in ids_by_fold[other]]
            normalization_ids = list(train_ids)
            if fold == bad_normalization_fold:
                normalization_ids = ids_by_fold[fold]
            metadata = {"split_ids": {"train": train_ids, "validation": [], "test": ids_by_fold[fold]},
                        "normalization": {"train_ids": normalization_ids}, "grid_size": 1024,
                        "channel_names": _channel_names(), "audit_sha256": audit_sha256}
            (metadata_dir / f"msd_audio_only_cv_fold_{fold}_meta.json").write_text(json.dumps(metadata))
            with h5py.File(cache_dir / f"msd_audio_only_cv_fold_{fold}.h5", "w") as handle:
                # The cache's sorted order intentionally differs from the split's concatenated order.
                handle.create_dataset("track_ids", data=np.asarray(all_ids, dtype="S"))
                handle.create_dataset("sequences", data=np.asarray([sequences[track_id] for track_id in all_ids]))
                handle.create_dataset("channel_names", data=np.asarray(_channel_names(), dtype="S"))
        return audit_path, cache_dir, metadata_dir, lastfm, ids_by_fold

    def test_three_fold_run_checkpoints_trace_and_provenance_before_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, ids_by_fold = self._fixture(root)
            output = root / "results/msd_audio_only_cv_experiment.json"
            checkpoint_dir = root / "results/checkpoints"
            expected_checkpoints = [checkpoint_dir / f"msd_audio_only_cv_fold_{fold}_seed_{seed}.pt"
                                    for fold in range(3) for seed in (684, 685)]
            original_train = cli.train_audio_only_seed
            original_relevance = cli.build_test_relevance
            original_encode = cli._encode
            observed_fit_ids, observed_traces = [], []
            original_loader = cli.load_audio_sequence_cache
            loaded_sequence_refs = []
            loader_phases = []

            def checked_loader(*args, **kwargs):
                gc.collect()
                self.assertTrue(all(reference() is None for reference in loaded_sequence_refs),
                                "a prior full-fold cache array was retained")
                all_ready = all(path.is_file() for path in expected_checkpoints)
                if len(loader_phases) >= 6:
                    self.assertTrue(all_ready, "held-out cache loaded before all checkpoints")
                result = original_loader(*args, **kwargs)
                loaded_sequence_refs.append(weakref.ref(result[1]))
                loader_phases.append("evaluation" if all_ready else "pre-test")
                return result

            def checked_train(audio, song_ids, pools, validation_pools, seed, **kwargs):
                gc.collect()
                self.assertTrue(all(reference() is None for reference in loaded_sequence_refs),
                                "full-fold cache retained while training")
                self.assertIsNone(validation_pools)
                self.assertEqual(set(song_ids), set(pools["partition_ids"]))
                model, history, trace = original_train(audio, song_ids, pools, validation_pools, seed, **kwargs)
                known = set(pools["positives"])
                for rows in trace["train_by_epoch"]:
                    for left, right, label in rows:
                        self.assertIn(left, song_ids)
                        self.assertIn(right, song_ids)
                        if label == 0:
                            self.assertNotIn(tuple(sorted((left, right))), known)
                observed_fit_ids.append(set(song_ids))
                observed_traces.extend(trace["train_by_epoch"])
                return model, history, trace

            def assert_checkpoints(*args, **kwargs):
                self.assertTrue(all(path.is_file() for path in expected_checkpoints))
                return original_relevance(*args, **kwargs)

            def checked_encode(model, audio):
                self.assertTrue(all(path.is_file() for path in expected_checkpoints))
                return original_encode(model, audio)

            with patch.object(cli, "load_audio_sequence_cache", side_effect=checked_loader), \
                    patch.object(cli, "train_audio_only_seed", side_effect=checked_train), \
                    patch.object(cli, "build_test_relevance", side_effect=assert_checkpoints), \
                    patch.object(cli, "_encode", side_effect=checked_encode):
                report = cli.run(audit, cache_dir, metadata_dir, lastfm, output, checkpoint_dir,
                                 epochs=1, seeds=[684, 685])

            self.assertEqual(report["source_sha256"]["lastfm_similar"], _source_sha256(lastfm))
            self.assertEqual(len(observed_fit_ids), 6)
            self.assertEqual(len(observed_traces), 6)
            self.assertEqual(len(loader_phases), 9)
            self.assertEqual(loader_phases[6:], ["evaluation"] * 3)
            for index, fit_ids in enumerate(observed_fit_ids):
                fold = index // 2
                expected_train = set(track_id for other in range(3) if other != fold for track_id in ids_by_fold[other])
                self.assertEqual(fit_ids, expected_train)
                self.assertFalse(set(ids_by_fold[fold]) & fit_ids)
            self.assertTrue(all(np.isfinite(value) for fold in report["per_fold"].values()
                                for seed in fold["per_seed"].values()
                                for value in seed["metrics"].values()))
            with Path(report["sampled_pairs_csv"]).open(newline="") as stream:
                traces = list(csv.DictReader(stream))
            self.assertEqual(len(traces), 6 * 4)
            trace_hash = hashlib.sha256(Path(report["sampled_pairs_csv"]).read_bytes()).hexdigest()
            self.assertEqual(report["sampled_pairs_csv_sha256"], trace_hash)
            self.assertTrue({
                "src/data/audio_only_cv.py",
                "src/data/audio_only_audit.py",
                "src/data/parse_msd.py",
                "src/data/parse_msd_duplicates.py",
                "src/data/parse_lastfm.py",
                "src/data/table.py",
                "scripts/preprocess_audio_cv.py",
            }.issubset(report["code_sha256"]))
            for path in expected_checkpoints:
                checkpoint = cli.torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["sampled_pairs_csv_sha256"], trace_hash)
                self.assertEqual(checkpoint["code_sha256"], report["code_sha256"])
                self.assertIn("source_sha256", checkpoint)
                self.assertEqual(checkpoint["fold"], int(path.name.split("_fold_")[1].split("_")[0]))
            self.assertNotEqual(Path(report["training_curves_csv"]), Path(report["sampled_pairs_csv"]))
            self.assertNotEqual(Path(report["paths"]["output"]), Path(report["sampled_pairs_csv"]))

    def test_existing_outputs_are_preserved_and_no_new_artifacts_are_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, _ = self._fixture(root)
            output = root / "results/msd_audio_only_cv_experiment.json"
            pairs = root / "results/msd_audio_only_cv_sampled_pairs.csv"
            curves = root / "results/msd_audio_only_cv_training_curves.csv"
            checkpoint_dir = root / "results/checkpoints"
            checkpoint_dir.mkdir(parents=True)
            existing = {
                output: b"keep report",
                pairs: b"keep pairs",
                checkpoint_dir / "msd_audio_only_cv_fold_1_seed_684.pt": b"keep checkpoint",
            }
            for path, content in existing.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            with self.assertRaisesRegex(FileExistsError, "output target already exists"):
                cli.run(audit, cache_dir, metadata_dir, lastfm, output, checkpoint_dir,
                        epochs=1, seeds=[684, 685])

            for path, content in existing.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertFalse(curves.exists())
            self.assertFalse((checkpoint_dir / "msd_audio_only_cv_fold_0_seed_684.pt").exists())

    def test_training_failure_removes_every_output_created_by_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, _ = self._fixture(root)
            output = root / "results/msd_audio_only_cv_experiment.json"
            checkpoint_dir = root / "results/checkpoints"
            original_train = cli.train_audio_only_seed
            calls = 0

            def fail_second_train(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected training failure")
                return original_train(*args, **kwargs)

            with patch.object(cli, "train_audio_only_seed", side_effect=fail_second_train):
                with self.assertRaisesRegex(RuntimeError, "injected training failure"):
                    cli.run(audit, cache_dir, metadata_dir, lastfm, output, checkpoint_dir,
                            epochs=1, seeds=[684, 685])

            targets = [output, root / "results/msd_audio_only_cv_training_curves.csv",
                       root / "results/msd_audio_only_cv_sampled_pairs.csv"]
            targets.extend(checkpoint_dir / f"msd_audio_only_cv_fold_{fold}_seed_{seed}.pt"
                           for fold in range(3) for seed in (684, 685))
            self.assertFalse([path for path in targets if path.exists()])

    def test_rejects_changed_lastfm_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, _ = self._fixture(root)
            (lastfm / "changed.json").write_text("changed after audit")
            output = root / "results/report.json"
            checkpoint_dir = root / "results/checkpoints"
            with self.assertRaisesRegex(ValueError, "Last.fm source provenance"):
                cli.run(audit, cache_dir, metadata_dir, lastfm, output, checkpoint_dir,
                        epochs=1, seeds=[684])
            self.assertFalse(output.parent.exists())
            self.assertFalse(checkpoint_dir.exists())

    def test_rejects_stale_fold_metadata_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, _ = self._fixture(root)
            metadata_path = metadata_dir / "msd_audio_only_cv_fold_1_meta.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["audit_sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata))
            output = root / "results/report.json"
            checkpoint_dir = root / "results/checkpoints"
            with self.assertRaisesRegex(ValueError, "fold 1 metadata audit_sha256"):
                cli.run(audit, cache_dir, metadata_dir, lastfm, output, checkpoint_dir,
                        epochs=1, seeds=[684])
            self.assertFalse(output.parent.exists())
            self.assertFalse(checkpoint_dir.exists())

    def test_empty_seeds_are_rejected_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, _ = self._fixture(root)
            output = root / "results/report.json"
            checkpoint_dir = root / "results/checkpoints"
            with self.assertRaisesRegex(ValueError, "seeds must not be empty"):
                cli.run(audit, cache_dir, metadata_dir, lastfm, output, checkpoint_dir,
                        epochs=1, seeds=[])
            self.assertFalse(output.parent.exists())
            self.assertFalse(checkpoint_dir.exists())

    def test_seed_iterable_is_materialized_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, _ = self._fixture(root)
            report = cli.run(
                audit, cache_dir, metadata_dir, lastfm,
                root / "results/report.json", root / "results/checkpoints",
                epochs=1, seeds=iter([684]),
            )

            self.assertEqual(report["config"]["seeds"], [684])
            self.assertTrue(all(set(fold["per_seed"]) == {"684"}
                                for fold in report["per_fold"].values()))
            self.assertTrue(all(np.isfinite(values["mean"])
                                for values in report["summary"].values()))

    def test_cleanup_errors_do_not_mask_training_failure_or_stop_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, _ = self._fixture(root)
            output = root / "results/report.json"
            checkpoint_dir = root / "results/checkpoints"
            pair_path = root / "results/msd_audio_only_cv_sampled_pairs.csv"
            later_checkpoint = checkpoint_dir / "msd_audio_only_cv_fold_0_seed_684.pt"
            original_train = cli.train_audio_only_seed
            original_unlink = Path.unlink
            calls = 0
            attempted = []

            def fail_second_train(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("original training failure")
                return original_train(*args, **kwargs)

            def fail_one_unlink(path, *args, **kwargs):
                attempted.append(path.resolve())
                if path.resolve() == pair_path.resolve():
                    raise OSError("injected cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with patch.object(cli, "train_audio_only_seed", side_effect=fail_second_train), \
                    patch.object(Path, "unlink", fail_one_unlink):
                with self.assertRaisesRegex(RuntimeError, "original training failure"):
                    cli.run(audit, cache_dir, metadata_dir, lastfm, output, checkpoint_dir,
                            epochs=1, seeds=[684, 685])

            self.assertIn(pair_path.resolve(), attempted)
            self.assertIn(later_checkpoint.resolve(), attempted)
            self.assertFalse(output.exists())
            self.assertFalse(later_checkpoint.exists())
            self.assertTrue(pair_path.exists())
            self.assertNotIn(audit.resolve(), attempted)

    def test_rejects_wrong_fold_normalization_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, _ = self._fixture(root, bad_normalization_fold=1)
            with self.assertRaisesRegex(ValueError, "normalization train_ids"):
                cli.run(audit, cache_dir, metadata_dir, lastfm, root / "out/report.json",
                        root / "out/checkpoints", epochs=1, seeds=[1])

    def test_rejects_duplicate_seeds_and_output_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, cache_dir, metadata_dir, lastfm, _ = self._fixture(root)
            args = (audit, cache_dir, metadata_dir, lastfm, root / "out/report.json", root / "out/checkpoints")
            with self.assertRaisesRegex(ValueError, "seeds must be unique"):
                cli.run(*args, epochs=1, seeds=[2, 2])
            with self.assertRaisesRegex(ValueError, "output paths must be distinct"):
                cli.run(audit, cache_dir, metadata_dir, lastfm, root / "out/msd_audio_only_cv_sampled_pairs.csv",
                        root / "out/checkpoints", epochs=1, seeds=[2])


if __name__ == "__main__":
    unittest.main()

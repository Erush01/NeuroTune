import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.preprocess_audio_cv import main
from src.data.preprocess_audio import preprocess_audio


class PreprocessAudioCVTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.input_root = self.root / "raw"
        self.output_dir = self.root / "processed"
        self.audit_path = self.root / "audit.json"
        self.folds = {
            "0": ["TRACK_A1", "TRACK_A2"],
            "1": ["TRACK_B1", "TRACK_B2"],
            "2": ["TRACK_C1", "TRACK_C2"],
        }
        self._write_sources()
        self._write_audit()

    def _write_sources(self):
        for index, track_id in enumerate(sum(self.folds.values(), [])):
            path = self.input_root / "A" / "B" / f"{track_id}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as handle:
                analysis = handle.create_group("analysis")
                songs = np.zeros(1, dtype=[("track_id", "S18"), ("duration", "f8")])
                songs[0] = (track_id.encode("ascii"), 2.0)
                analysis.create_dataset("songs", data=songs)
                analysis.create_dataset(
                    "segments_pitches", data=np.full((3, 12), index * 10.0)
                )
                analysis.create_dataset(
                    "segments_timbre", data=np.full((3, 12), index + 1.0)
                )
                analysis.create_dataset("segments_loudness_max", data=[-20.0, -10.0, 0.0])
                analysis.create_dataset("segments_start", data=[0.0, 1.0, 2.0])
                analysis.create_dataset("beats_start", data=[0.0, 0.5, 2.0])
                analysis.create_dataset("bars_start", data=[1.0])
                analysis.create_dataset("sections_start", data=[])

    def _write_audit(self, **updates):
        audit = {"status": "ready", "fold_count": 3, "folds": {
            key: {"track_ids": ids} for key, ids in self.folds.items()
        }}
        audit.update(updates)
        self.audit_path.write_text(json.dumps(audit), encoding="utf-8")

    def _run(self):
        return main([
            "--input-root", str(self.input_root),
            "--audit", str(self.audit_path),
            "--output-dir", str(self.output_dir),
            "--grid-size", "5",
        ])

    def test_builds_independent_full_catalog_caches_with_train_only_normalization(self):
        self._run()
        catalog = set(sum(self.folds.values(), []))
        self.assertEqual(len(list(self.output_dir.glob("*.h5"))), 3)
        self.assertEqual(len(list(self.output_dir.glob("*_meta.json"))), 3)

        for fold_index, held_out_ids in self.folds.items():
            cache_path = self.output_dir / f"msd_audio_only_cv_fold_{fold_index}.h5"
            metadata_path = self.output_dir / f"msd_audio_only_cv_fold_{fold_index}_meta.json"
            self.assertTrue(cache_path.is_file())
            self.assertTrue(metadata_path.is_file())
            with h5py.File(cache_path, "r") as handle:
                cache_ids = {value.decode("utf-8") for value in handle["track_ids"][...]}
                self.assertEqual(cache_ids, catalog)
                track_ids = [value.decode("utf-8") for value in handle["track_ids"][...]]
                sequences = handle["sequences"][...]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            train_ids = [track_id for key, ids in self.folds.items()
                         if key != fold_index for track_id in ids]
            self.assertEqual(metadata["split_ids"]["validation"], [])
            self.assertEqual(set(metadata["split_ids"]["test"]), set(held_out_ids))
            self.assertEqual(set(metadata["split_ids"]["train"]), set(train_ids))
            normalization = metadata["normalization"]
            self.assertEqual(set(normalization["train_ids"]), set(train_ids))
            source_indices = [sum(self.folds.values(), []).index(track_id) for track_id in train_ids]
            self.assertAlmostEqual(normalization["mean"][0], np.mean(source_indices) * 10.0)
            held_out_index = track_ids.index(held_out_ids[0])
            train_raw = np.asarray(source_indices, dtype=np.float64) * 10.0
            held_out_raw = sum(self.folds.values(), []).index(held_out_ids[0]) * 10.0
            expected_value = (held_out_raw - train_raw.mean()) / train_raw.std()
            self.assertAlmostEqual(
                float(sequences[held_out_index, 0, 0]),
                float(np.float32(expected_value)),
                places=6,
            )
            raw_path = self.input_root / "A" / "B" / f"{held_out_ids[0]}.h5"
            raw_sequence = preprocess_audio(raw_path, held_out_ids[0], grid_size=5)
            np.testing.assert_array_equal(
                sequences[held_out_index, :, 25:], raw_sequence[:, 25:]
            )
            self.assertEqual(metadata["audit_sha256"], __import__("hashlib").sha256(
                self.audit_path.read_bytes()).hexdigest())

    def test_invalid_audit_fails_before_any_outputs_are_created(self):
        bad_audits = [
            {"status": "blocked"},
            {"status": "ready", "fold_count": 0},
            {"status": "ready", "fold_count": 3,
             "folds": {"0": {"track_ids": ["TRACK_A1"]}}},
            {"status": "ready", "fold_count": 3, "folds": {
                "0": {"track_ids": ["TRACK_A1"]},
                "1": {"track_ids": ["TRACK_A1"]},
                "2": {"track_ids": ["TRACK_C1"]},
            }},
        ]
        for audit in bad_audits:
            with self.subTest(audit=audit):
                self._write_audit(**audit)
                with self.assertRaises(ValueError):
                    self._run()
                self.assertEqual(list(self.output_dir.glob("*")) if self.output_dir.exists() else [], [])

    def test_path_collisions_with_inputs_or_audit_fail_before_outputs(self):
        with self.assertRaises(ValueError):
            main([
                "--input-root", str(self.input_root),
                "--audit", str(self.audit_path),
                "--output-dir", str(self.input_root),
                "--grid-size", "5",
            ])

        colliding_audit = self.output_dir / "msd_audio_only_cv_fold_0.h5"
        colliding_audit.parent.mkdir(parents=True)
        colliding_audit.write_text(self.audit_path.read_text(encoding="utf-8"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "collides with CV audit"):
            main([
                "--input-root", str(self.input_root),
                "--audit", str(colliding_audit),
                "--output-dir", str(self.output_dir),
                "--grid-size", "5",
            ])

    def test_missing_raw_catalog_track_fails_before_any_fold_is_written(self):
        missing_track_path = self.input_root / "A" / "B" / "TRACK_C2.h5"
        missing_track_path.unlink()
        with self.assertRaisesRegex(ValueError, "missing source HDF5 files"):
            self._run()
        self.assertFalse(self.output_dir.exists())

    def test_existing_generated_target_is_unchanged_and_blocks_all_fold_outputs(self):
        self.output_dir.mkdir(parents=True)
        existing = self.output_dir / "msd_audio_only_cv_fold_1.h5"
        existing_bytes = b"preserve existing target"
        existing.write_bytes(existing_bytes)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self._run()
        self.assertEqual(existing.read_bytes(), existing_bytes)
        self.assertEqual(list(self.output_dir.iterdir()), [existing])


if __name__ == "__main__":
    unittest.main()

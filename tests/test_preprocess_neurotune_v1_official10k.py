import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from src.data import preprocess_neurotune_v1_official10k as cache


class Official10KAudioCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.subset = self.root / "subset"
        self.subset.mkdir()
        self.ids = ["TRA000000000000002", "TRA000000000000001"]
        for track_id in self.ids:
            path = cache.official_h5_path(self.subset, track_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(track_id.encode())
        self.audit = self.root / "audit.json"
        self.audit.write_text(json.dumps({"track_ids": self.ids}))
        self.output = self.root / "cache.h5"
        self.meta = self.root / "cache_meta.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_ragged_round_trip_sorted_raw_float32_and_source_hashes(self):
        values = {
            self.ids[0]: np.arange(52, dtype=np.float32).reshape(2, 26),
            self.ids[1]: np.arange(78, dtype=np.float32).reshape(3, 26) + 100,
        }
        with patch.object(cache, "preprocess_audio_window", side_effect=lambda path, tid, window_seconds: (values[tid], {"track_id": tid})):
            result = cache.build_official10k_audio_cache(self.audit, self.subset, self.output, self.meta)
        sorted_ids = sorted(self.ids)
        with h5py.File(self.output, "r") as h5:
            self.assertEqual(list(h5["track_ids"].asstr()[:]), sorted_ids)
            np.testing.assert_array_equal(h5["offsets"][:], [0, 3, 5])
            np.testing.assert_array_equal(h5["features"][:], np.concatenate([values[i] for i in sorted_ids]))
            self.assertEqual(h5["features"].dtype, np.dtype(np.float32))
        metadata = json.loads(self.meta.read_text())
        self.assertEqual(metadata["window_seconds"], 30.0)
        self.assertEqual(metadata["interval"], "[0,30)")
        self.assertEqual(metadata["track_count"], 2)
        self.assertEqual(metadata["segment_count"], 5)
        self.assertEqual(metadata["audit_sha256"], hashlib.sha256(self.audit.read_bytes()).hexdigest())
        self.assertEqual(set(metadata["source_sha256"]), set(self.ids))
        self.assertEqual(result["track_count"], 2)

    def test_refuses_either_existing_target(self):
        self.output.touch()
        with self.assertRaises(FileExistsError):
            cache.build_official10k_audio_cache(self.audit, self.subset, self.output, self.meta)
        self.assertEqual(self.output.read_bytes(), b"")

    def test_identical_resolved_output_paths_rejected_before_extraction(self):
        shared = self.root / "same-target"
        shared.write_bytes(b"keep existing file")
        equivalent = self.root / "unused" / ".." / "same-target"
        with patch.object(cache, "preprocess_audio_window") as extractor:
            with self.assertRaisesRegex(ValueError, "output paths must be different"):
                cache.build_official10k_audio_cache(self.audit, self.subset, shared, equivalent)
        extractor.assert_not_called()
        self.assertEqual(shared.read_bytes(), b"keep existing file")

    def test_official_path_rejects_noncanonical_track_ids(self):
        for malformed in ("../TRAAAAAAAAAAAAAAA", "TRAaaaaaaaaaaaaaaa", "TRA123"):
            with self.subTest(track_id=malformed), self.assertRaisesRegex(ValueError, "canonical 18-character"):
                cache.official_h5_path(self.subset, malformed)

    def test_official_path_rejects_symlink_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        shutil.rmtree(self.subset / "A")
        (self.subset / "A").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes the official subset root"):
            cache.official_h5_path(self.subset, self.ids[0])

    def test_extraction_failure_rolls_back_outputs(self):
        with patch.object(cache, "preprocess_audio_window", side_effect=RuntimeError("extract failed")):
            with self.assertRaisesRegex(RuntimeError, "extract failed"):
                cache.build_official10k_audio_cache(self.audit, self.subset, self.output, self.meta)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.meta.exists())
        self.assertEqual(list(self.root.glob("*.stage*")), [])

    def test_write_failure_rolls_back_published_pair(self):
        values = np.ones((1, 26), dtype=np.float32)
        def fail_after_hdf5(*args, **kwargs):
            raise OSError("metadata write failed")
        with patch.object(cache, "preprocess_audio_window", side_effect=lambda path, tid, window_seconds: (values, {"track_id": tid})), patch.object(cache, "_write_metadata", side_effect=fail_after_hdf5):
            with self.assertRaisesRegex(OSError, "metadata write failed"):
                cache.build_official10k_audio_cache(self.audit, self.subset, self.output, self.meta)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.meta.exists())

    def test_second_publish_failure_removes_first_published_file(self):
        values = np.ones((1, 26), dtype=np.float32)
        real_link = os.link
        calls = 0
        def fail_metadata_publish(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("metadata publish failed")
            return real_link(source, destination)
        with patch.object(cache, "preprocess_audio_window", side_effect=lambda path, tid, window_seconds: (values, {"track_id": tid})), patch.object(cache.os, "link", side_effect=fail_metadata_publish):
            with self.assertRaisesRegex(OSError, "metadata publish failed"):
                cache.build_official10k_audio_cache(self.audit, self.subset, self.output, self.meta)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.meta.exists())

    def test_metadata_target_race_preserves_competing_file_and_rolls_back_cache(self):
        values = np.ones((1, 26), dtype=np.float32)
        real_link = os.link
        calls = 0
        competing_bytes = b"created by competing writer"
        def create_competing_metadata(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                Path(destination).write_bytes(competing_bytes)
            return real_link(source, destination)
        with patch.object(cache, "preprocess_audio_window", side_effect=lambda path, tid, window_seconds: (values, {"track_id": tid})), patch.object(cache.os, "link", side_effect=create_competing_metadata):
            with self.assertRaises(FileExistsError):
                cache.build_official10k_audio_cache(self.audit, self.subset, self.output, self.meta)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.meta.read_bytes(), competing_bytes)


if __name__ == "__main__":
    unittest.main()

import csv
import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scripts import preprocess_msd_hf_ab as cache


def _track_bytes(track_id, artist_id, song_id, pitch_value, starts):
    payload = io.BytesIO()
    with h5py.File(payload, "w") as handle:
        analysis = handle.create_group("analysis")
        songs_dtype = np.dtype([("track_id", "S18"), ("tempo", "f8"), ("duration", "f8")])
        analysis.create_dataset("songs", data=np.array([(track_id.encode(), 120.0, 60.0)], dtype=songs_dtype))
        analysis.create_dataset("segments_start", data=np.asarray(starts, dtype=np.float64))
        analysis.create_dataset("segments_pitches", data=np.full((len(starts), 12), pitch_value))
        analysis.create_dataset("segments_timbre", data=np.full((len(starts), 12), pitch_value + 1))
        analysis.create_dataset("segments_loudness_max", data=np.full(len(starts), pitch_value + 2))
        metadata = handle.create_group("metadata")
        metadata_dtype = np.dtype([("artist_id", "S16"), ("song_id", "S16")])
        metadata.create_dataset("songs", data=np.array([(artist_id.encode(), song_id.encode())], dtype=metadata_dtype))
    return payload.getvalue()


class PreprocessMsdHfAbTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = ["TRAAAAAAAAAAAAAAAA", "TRBBBBBBBBBBBBBBBB"]
        self.members = {
            self.ids[0]: _track_bytes(self.ids[0], "AR1", "SO1", 1, [0.0, 2.0, 31.0]),
            self.ids[1]: _track_bytes(self.ids[1], "AR2", "SO2", 5, [0.0, 4.0, 8.0, 35.0]),
        }
        self.a_tar = self.root / "A.tar.gz"
        self.b_tar = self.root / "B.tar.gz"
        self._write_tar(self.a_tar, [self.ids[0]])
        self._write_tar(self.b_tar, [self.ids[1]])
        self.audit = self.root / "ab_catalog_audit.json"
        self.catalog = self.root / "ab_catalog_tracks.csv"
        self.output = self.root / "ab_audio_30s_26ch.h5"
        self.meta = self.root / "ab_audio_30s_26ch_meta.json"
        self._write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def _write_tar(self, path, track_ids):
        with tarfile.open(path, "w:gz") as archive:
            for track_id in track_ids:
                payload = self.members[track_id]
                info = tarfile.TarInfo(f"{track_id[2]}/{track_id[3]}/{track_id[4]}/{track_id}.h5")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    def _write_inputs(self, ids=None):
        ids = sorted(ids if ids is not None else self.ids)
        hashes = {"A": cache._sha256(self.a_tar), "B": cache._sha256(self.b_tar)}
        self.audit.write_text(json.dumps({
            "status": "complete", "source_sha256": hashes,
            "counts": {"eligible_tracks": len(ids)},
            "window": {"seconds": 30, "interval": "[0,30)"},
            "channels": cache.CHANNELS,
        }), encoding="utf-8")
        with self.catalog.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["track_id", "tempo", "artist_id", "song_id"])
            writer.writeheader()
            for track_id in ids:
                index = self.ids.index(track_id) if track_id in self.ids else 2
                writer.writerow({"track_id": track_id, "tempo": 120,
                                 "artist_id": f"AR{index + 1}", "song_id": f"SO{index + 1}"})

    def _run(self):
        return cache.run(self.a_tar, self.b_tar, self.audit, self.catalog,
                         self.output, self.meta)

    def test_ragged_round_trip_sorted_ids_and_first_window_channels(self):
        self._run()
        expected_lengths = {self.ids[0]: 2, self.ids[1]: 3}
        expected_audit_hash = hashlib.sha256(self.audit.read_bytes()).hexdigest()
        expected_code_hashes = {
            "scripts/preprocess_msd_hf_ab.py": cache._sha256(Path(cache.__file__)),
            "src/data/preprocess_audio_window.py": cache._sha256(
                cache.REPO_ROOT / "src/data/preprocess_audio_window.py"
            ),
        }
        source_hashes = {"A": cache._sha256(self.a_tar), "B": cache._sha256(self.b_tar)}
        with h5py.File(self.output, "r") as handle:
            ids = [value.decode() if isinstance(value, bytes) else value for value in handle["track_ids"][:]]
            np.testing.assert_array_equal(ids, sorted(self.ids))
            offsets = handle["offsets"][:]
            np.testing.assert_array_equal(offsets, [0, 2, 5])
            self.assertTrue(np.all(np.diff(offsets) > 0))
            features = handle["features"][:]
            self.assertEqual(features.shape, (5, 26))
            self.assertEqual(features.dtype, np.float32)
            np.testing.assert_array_equal(features[:2, :12], np.ones((2, 12)))
            np.testing.assert_array_equal(features[:2, 12:24], np.full((2, 12), 2))
            np.testing.assert_array_equal(features[:2, 24], [3, 3])
            np.testing.assert_array_equal(features[:2, 25], [2, 28])
            self.assertEqual(handle.attrs["audit_sha256"], expected_audit_hash)
            self.assertEqual(json.loads(handle.attrs["source_sha256"]), source_hashes)
            self.assertEqual(json.loads(handle.attrs["code_sha256"]), expected_code_hashes)
            self.assertEqual(json.loads(handle.attrs["sequence_lengths"]), expected_lengths)
            self.assertEqual(handle.attrs["track_count"], len(expected_lengths))
        metadata = json.loads(self.meta.read_text())
        self.assertEqual(metadata["window_seconds"], 30)
        self.assertEqual(metadata["interval"], "[0,30)")
        self.assertEqual(metadata["channels"], cache.CHANNELS)
        self.assertEqual(metadata["dtype"], "float32")
        self.assertEqual(metadata["normalization"], "raw/unnormalized; deferred until train split")
        self.assertEqual(metadata["track_count"], 2)
        self.assertEqual(metadata["sequence_lengths"], expected_lengths)
        self.assertEqual(metadata["track_count"], len(expected_lengths))
        self.assertEqual(metadata["audit_sha256"], expected_audit_hash)
        self.assertEqual(metadata["source_sha256"], source_hashes)
        self.assertEqual(metadata["code_sha256"], expected_code_hashes)
        for value in [expected_audit_hash, *source_hashes.values(), *expected_code_hashes.values()]:
            self.assertRegex(value, r"^[0-9a-f]{64}$")

    def test_missing_expected_track_fails(self):
        self._write_inputs([self.ids[0], "TRCCCCCCCCCCCCCCCC"])
        with self.assertRaisesRegex(ValueError, "coverage|missing"):
            self._run()
        self.assertFalse(self.output.exists())

    def test_duplicate_source_track_fails(self):
        self._write_tar(self.b_tar, [self.ids[0], self.ids[1]])
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._run()

    def test_catalog_id_order_or_uniqueness_is_validated(self):
        self._write_inputs()
        with self.catalog.open() as stream:
            rows = list(csv.DictReader(stream))
        with self.catalog.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows[::-1])
        with self.assertRaisesRegex(ValueError, "sorted"):
            self._run()

    def test_archive_track_id_mismatch_fails_and_cleans_outputs(self):
        wrong = self.ids[0]
        self.members[wrong] = _track_bytes("TRCCCCCCCCCCCCCCCC", "AR1", "SO1", 1, [0.0])
        self._write_tar(self.a_tar, [wrong])
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "track ID"):
            self._run()
        self.assertFalse(self.output.exists())

    def test_stale_audit_source_hash_fails(self):
        audit = json.loads(self.audit.read_text())
        audit["source_sha256"]["A"] = "0" * 64
        self.audit.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self._run()

    def test_archive_hash_is_rechecked_after_scan(self):
        original_scan = cache._scan_archives

        def scan_then_mutate(paths, records):
            found = original_scan(paths, records)
            with self.a_tar.open("ab") as stream:
                stream.write(b"changed")
            return found

        with mock.patch.object(cache, "_scan_archives", side_effect=scan_then_mutate):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self._run()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.meta.exists())

    def test_existing_target_is_rejected_before_work(self):
        self.output.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self._run()
        self.assertEqual(self.output.read_bytes(), b"preserve")

    def test_existing_metadata_target_is_rejected_before_work(self):
        self.meta.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self._run()
        self.assertEqual(self.meta.read_bytes(), b"preserve")

    def test_metadata_mismatch_fails(self):
        with self.catalog.open() as source:
            rows = list(csv.DictReader(source))
        rows[0]["artist_id"] = "WRONG"
        with self.catalog.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "metadata"):
            self._run()

    def test_publish_failure_cleans_targets_and_preserves_error(self):
        error = OSError("publish injected failure")
        original_link = cache.os.link
        calls = 0

        def link_then_fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise error
            return original_link(source, target)

        with mock.patch.object(cache.os, "link", side_effect=link_then_fail_second):
            with self.assertRaisesRegex(OSError, "publish injected failure"):
                try:
                    self._run()
                except OSError as caught:
                    self.assertIs(caught, error)
                    raise
        self.assertEqual(calls, 2)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.meta.exists())

    def test_stream_member_closes_fd_and_source_when_fdopen_fails(self):
        track_id = self.ids[0]
        member = mock.Mock(size=len(self.members[track_id]), name=f"{track_id}.h5")
        source = io.BytesIO(self.members[track_id])
        archive = mock.Mock()
        archive.extractfile.return_value = source
        descriptors = []

        def fail_fdopen(fd, mode):
            descriptors.append(fd)
            raise OSError("fdopen injected failure")

        try:
            with mock.patch.object(cache, "TEMP_DIR", self.root), mock.patch.object(
                cache.os, "fdopen", side_effect=fail_fdopen
            ):
                with self.assertRaisesRegex(OSError, "fdopen injected failure"):
                    cache._stream_member(archive, member, track_id)
            self.assertEqual(len(descriptors), 1)
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])
            self.assertTrue(source.closed)
        finally:
            if descriptors:
                try:
                    os.close(descriptors[0])
                except OSError:
                    pass

    def test_stream_member_does_not_close_transferred_descriptor_again(self):
        track_id = self.ids[0]
        payload = self.members[track_id]
        member = mock.Mock(size=len(payload), name=f"{track_id}.h5")
        source = io.BytesIO(payload)
        archive = mock.Mock()
        archive.extractfile.return_value = source
        original_close = os.close
        with mock.patch.object(cache, "TEMP_DIR", self.root), mock.patch.object(
            cache.os, "close", wraps=original_close
        ) as close:
            features, metadata = cache._stream_member(archive, member, track_id)
        self.assertEqual(features.shape, (2, 26))
        self.assertEqual(metadata["track_id"], track_id)
        close.assert_not_called()
        self.assertTrue(source.closed)


if __name__ == "__main__":
    unittest.main()

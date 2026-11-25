import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from unittest.mock import patch

import src.data.preprocess_audio_window as preprocessing
from src.data.preprocess_audio_window import preprocess_audio_three_windows, preprocess_audio_window


class AudioWindowPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "track.h5"
        self._write_record()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_record(self, *, track_id=b"TRTEST000001", duration=40.0, tempo=120.0,
                      starts=None, pitches=None, timbre=None, loudness=None,
                      omit=(), artist_id=b"ARTEST000001", song_id=b"SO TEST"):
        starts = np.asarray([0.0, 10.0, 29.0, 30.0, 35.0] if starts is None else starts)
        pitches = np.asarray(np.arange(len(starts) * 12).reshape(len(starts), 12)
                             if pitches is None else pitches)
        timbre = np.asarray(100 + np.arange(len(starts) * 12).reshape(len(starts), 12)
                            if timbre is None else timbre)
        loudness = np.asarray(np.arange(len(starts), dtype=float) - 20
                              if loudness is None else loudness)
        with h5py.File(self.path, "w") as handle:
            analysis = handle.create_group("analysis")
            songs = np.zeros(1, dtype=[("track_id", "S12"), ("duration", "f8"), ("tempo", "f8")])
            songs[0] = (track_id, duration, tempo)
            analysis.create_dataset("songs", data=songs)
            for name, values in (("segments_pitches", pitches), ("segments_timbre", timbre),
                                 ("segments_loudness_max", loudness), ("segments_start", starts)):
                if name not in omit:
                    analysis.create_dataset(name, data=values)
            metadata = handle.create_group("metadata")
            rows = np.zeros(1, dtype=[("artist_id", "S12"), ("song_id", "S12")])
            rows[0] = (artist_id, song_id)
            metadata.create_dataset("songs", data=rows)

    def test_returns_ordered_26_channel_rows_clipped_to_window_without_mutation(self):
        with h5py.File(self.path, "r") as handle:
            before = {name: handle[f"analysis/{name}"][...].copy() for name in (
                "segments_start", "segments_pitches", "segments_timbre", "segments_loudness_max"
            )}

        features, metadata = preprocess_audio_window(self.path, " trtest000001 ")

        self.assertEqual(features.shape, (3, 26))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))
        np.testing.assert_array_equal(features[:, :12], np.arange(60).reshape(5, 12)[:3])
        np.testing.assert_array_equal(features[:, 12:24], (100 + np.arange(60).reshape(5, 12))[:3])
        np.testing.assert_array_equal(features[:, 24], [-20, -19, -18])
        np.testing.assert_array_equal(features[:, 25], [10, 19, 1])
        self.assertEqual(metadata, {
            "track_id": "TRTEST000001", "tempo": 120.0, "duration": 40.0,
            "artist_id": "ARTEST000001", "song_id": "SO TEST",
        })
        with h5py.File(self.path, "r") as handle:
            for name, original in before.items():
                np.testing.assert_array_equal(handle[f"analysis/{name}"][...], original)

    def test_three_windows_use_start_middle_end_placement_and_half_open_boundaries(self):
        starts = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        self._write_record(duration=70.0, starts=starts)

        windows, metadata = preprocess_audio_three_windows(self.path, "TRTEST000001")

        self.assertEqual([window["window_start"] for window in windows], [0.0, 20.0, 40.0])
        self.assertEqual(metadata["duration"], 70.0)
        self.assertEqual([window["source_indices"].tolist() for window in windows],
                         [[0, 1, 2], [2, 3, 4], [4, 5, 6]])
        self.assertEqual([window["timestamps"].tolist() for window in windows],
                         [[0.0, 10.0, 20.0], [20.0, 30.0, 40.0], [40.0, 50.0, 60.0]])

    def test_three_window_durations_clip_at_next_segment_song_and_window_end(self):
        self._write_record(duration=65.0, starts=[0.0, 29.0, 35.0, 64.0])
        windows, _ = preprocess_audio_three_windows(self.path, "TRTEST000001")

        np.testing.assert_array_equal(windows[0]["features"][:, 25], [29.0, 1.0])
        np.testing.assert_array_equal(windows[1]["features"][:, 25], [6.0, 12.5])
        np.testing.assert_array_equal(windows[2]["features"][:, 25], [29.0, 1.0])

    def test_three_windows_overlap_naturally_and_short_tracks_repeat_opening(self):
        self._write_record(duration=40.0, starts=[0.0, 10.0, 20.0, 30.0, 35.0])
        windows, _ = preprocess_audio_three_windows(self.path, "TRTEST000001")
        self.assertEqual(windows[0]["source_indices"].tolist(), [0, 1, 2])
        self.assertEqual(windows[1]["source_indices"].tolist(), [1, 2, 3])
        self.assertEqual(windows[2]["source_indices"].tolist(), [1, 2, 3, 4])

        self._write_record(duration=12.0, starts=[0.0, 4.0, 8.0])
        short_windows, _ = preprocess_audio_three_windows(self.path, "TRTEST000001")
        self.assertEqual([window["window_start"] for window in short_windows], [0.0, 0.0, 0.0])
        for window in short_windows[1:]:
            np.testing.assert_array_equal(window["features"], short_windows[0]["features"])
            np.testing.assert_array_equal(window["source_indices"], [0, 1, 2])

    def test_empty_individual_window_is_invalid_and_has_aligned_empty_provenance(self):
        self._write_record(duration=60.0, starts=[0.0, 55.0])
        windows, _ = preprocess_audio_three_windows(self.path, "TRTEST000001")

        self.assertEqual([window["valid"] for window in windows], [True, False, True])
        self.assertEqual(windows[1]["features"].shape, (0, 26))
        self.assertEqual(windows[1]["features"].dtype, np.float32)
        self.assertEqual(windows[1]["source_indices"].shape, (0,))
        self.assertEqual(windows[1]["timestamps"].shape, (0,))

    def test_existing_opening_window_function_is_unchanged(self):
        features, metadata = preprocess_audio_window(self.path, "TRTEST000001")
        self.assertEqual(features.shape, (3, 26))
        np.testing.assert_array_equal(features[:, 25], [10.0, 19.0, 1.0])
        self.assertEqual(metadata["track_id"], "TRTEST000001")

    def test_short_track_clips_last_segment_to_song_duration(self):
        self._write_record(duration=2.5, starts=[0.0, 1.0], pitches=np.zeros((2, 12)),
                           timbre=np.zeros((2, 12)), loudness=[-1.0, -2.0])
        features, _ = preprocess_audio_window(self.path, "TRTEST000001")
        self.assertEqual(features.shape, (2, 26))
        np.testing.assert_array_equal(features[:, 25], [1.0, 1.5])

    def test_rejects_track_with_no_segments_in_window(self):
        self._write_record(duration=40.0, starts=[30.0, 35.0],
                           pitches=np.zeros((2, 12)), timbre=np.zeros((2, 12)),
                           loudness=[-1.0, -2.0])
        with self.assertRaisesRegex(preprocessing.AudioWindowTrackError, "at least one segment"):
            preprocess_audio_window(self.path, "TRTEST000001")

    def test_rejects_missing_arrays_and_malformed_or_misaligned_shapes(self):
        self._write_record(omit=("segments_timbre",))
        with self.assertRaisesRegex(preprocessing.AudioWindowTrackError, "segments_timbre"):
            preprocess_audio_window(self.path, "TRTEST000001")
        self._write_record(pitches=np.zeros((5, 11)))
        with self.assertRaisesRegex(ValueError, "segments_pitches"):
            preprocess_audio_window(self.path, "TRTEST000001")
        self._write_record(loudness=np.zeros(4))
        with self.assertRaisesRegex(preprocessing.AudioWindowTrackError, "aligned"):
            preprocess_audio_window(self.path, "TRTEST000001")
        self._write_record(starts=np.zeros((5, 1)))
        with self.assertRaisesRegex(ValueError, "segments_start"):
            preprocess_audio_window(self.path, "TRTEST000001")

    def test_rejects_non_finite_arrays_and_unordered_or_duplicate_starts(self):
        pitches = np.zeros((5, 12)); pitches[0, 0] = np.nan
        self._write_record(pitches=pitches)
        with self.assertRaisesRegex(preprocessing.AudioWindowTrackError, "finite"):
            preprocess_audio_window(self.path, "TRTEST000001")
        for starts in ([0.0, 2.0, 1.0], [0.0, 1.0, 1.0]):
            self._write_record(starts=starts, pitches=np.zeros((3, 12)),
                               timbre=np.zeros((3, 12)), loudness=np.zeros(3))
            with self.assertRaisesRegex(ValueError, "ordered"):
                preprocess_audio_window(self.path, "TRTEST000001")

        unsigned_starts = np.array([2, 1], dtype=np.uint64)
        self._write_record(starts=unsigned_starts, pitches=np.zeros((2, 12)),
                           timbre=np.zeros((2, 12)), loudness=np.zeros(2))
        with self.assertRaisesRegex(ValueError, "ordered"):
            preprocess_audio_window(self.path, "TRTEST000001")

    def test_rejects_invalid_track_metadata_and_window(self):
        with self.assertRaisesRegex(ValueError, "track ID"):
            preprocess_audio_window(self.path, "TROTHER000001")
        for duration in (0.0, -1.0, np.nan, np.inf):
            self._write_record(duration=duration)
            with self.assertRaisesRegex(ValueError, "duration"):
                preprocess_audio_window(self.path, "TRTEST000001")
        for tempo in (0.0, -1.0, np.nan, np.inf):
            self._write_record(tempo=tempo)
            with self.assertRaisesRegex(ValueError, "tempo"):
                preprocess_audio_window(self.path, "TRTEST000001")
        self._write_record()
        for window in (0, -1, np.nan, np.inf, "30"):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "window_seconds"):
                preprocess_audio_window(self.path, "TRTEST000001", window_seconds=window)

    def test_rejects_zero_clipped_duration(self):
        self._write_record(duration=2.0, starts=[0.0, 2.0], pitches=np.zeros((2, 12)),
                           timbre=np.zeros((2, 12)), loudness=np.zeros(2))
        with self.assertRaisesRegex(preprocessing.AudioWindowTrackError, "positive"):
            preprocess_audio_window(self.path, "TRTEST000001")

    def test_rejects_segment_count_and_logical_bytes_before_reading_arrays(self):
        for name, shape, cap_name, cap in (
            ("segments_start", (4,), "MAX_SEGMENTS_PER_TRACK", 3),
            ("segments_pitches", (3, 12), "MAX_SEGMENT_ARRAY_BYTES", 100),
        ):
            self._write_record()
            with h5py.File(self.path, "r+") as handle:
                del handle[f"analysis/{name}"]
                dtype = "f8"
                handle[f"analysis"].create_dataset(name, shape=shape, dtype=dtype)
            with patch.object(preprocessing, cap_name, cap):
                with self.assertRaisesRegex(preprocessing.AudioWindowUnsafeError, "segment"):
                    preprocess_audio_window(self.path, "TRTEST000001")

    def test_rejects_zero_length_segment_arrays_as_track_data_errors(self):
        self._write_record()
        with h5py.File(self.path, "r+") as handle:
            del handle["analysis/segments_start"]
            handle["analysis"].create_dataset("segments_start", data=np.array([], dtype="f8"))
        with self.assertRaises(preprocessing.AudioWindowTrackError):
            preprocess_audio_window(self.path, "TRTEST000001")

    def test_rejects_non_internal_links_and_malformed_song_rows(self):
        self._write_record()
        with h5py.File(self.path, "r+") as handle:
            del handle["analysis/segments_start"]
            handle["analysis"]["segments_start"] = h5py.SoftLink("/analysis/segments_pitches")
        with self.assertRaisesRegex(preprocessing.AudioWindowUnsafeError, "hard link"):
            preprocess_audio_window(self.path, "TRTEST000001")

        self._write_record()
        with h5py.File(self.path, "r+") as handle:
            del handle["analysis/songs"]
            handle["analysis"]["songs"] = h5py.ExternalLink("elsewhere.h5", "/songs")
        with self.assertRaisesRegex(preprocessing.AudioWindowUnsafeError, "hard link"):
            preprocess_audio_window(self.path, "TRTEST000001")

        self._write_record()
        with h5py.File(self.path, "r+") as handle:
            del handle["metadata/songs"]
            handle["metadata"].create_dataset("songs", shape=(2,), dtype=[("artist_id", "S12"), ("song_id", "S12")])
        with self.assertRaisesRegex(preprocessing.AudioWindowUnsafeError, "single-row"):
            preprocess_audio_window(self.path, "TRTEST000001")

    def test_rejects_vlen_song_fields_and_accepts_large_fixed_metadata(self):
        self._write_record()
        with h5py.File(self.path, "r+") as handle:
            metadata = handle["metadata"]
            del metadata["songs"]
            songs = metadata.create_dataset("songs", (1,), dtype=np.dtype([
                ("artist_id", h5py.string_dtype("utf-8")), ("song_id", "S12")
            ]))
            songs[0] = ("ARTEST000001", b"SO TEST")
        with self.assertRaisesRegex(preprocessing.AudioWindowUnsafeError, "variable-length"):
            preprocess_audio_window(self.path, "TRTEST000001")

        self._write_record()
        with h5py.File(self.path, "r+") as handle:
            metadata = handle["metadata"]
            del metadata["songs"]
            dtype = np.dtype([
                ("artist_id", "S12"), ("song_id", "S12"),
                ("artist_name", "S1024"), ("title", "S1024"), ("reserved", "S3248"),
            ])
            self.assertEqual(dtype.itemsize, 5320)
            rows = metadata.create_dataset("songs", (1,), dtype=dtype)
            rows[0] = (b"ARTEST000001", b"SO TEST", b"Artist", b"Title", b"")
        _, metadata = preprocess_audio_window(self.path, "TRTEST000001")
        self.assertEqual(metadata["artist_id"], "ARTEST000001")

    def test_error_categories_remain_value_errors_and_identity_is_fatal(self):
        self.assertTrue(issubclass(preprocessing.AudioWindowTrackError, ValueError))
        self.assertTrue(issubclass(preprocessing.AudioWindowUnsafeError, ValueError))
        with self.assertRaises(ValueError) as error:
            preprocess_audio_window(self.path, "TROTHER000001")
        self.assertNotIsInstance(error.exception, preprocessing.AudioWindowTrackError)
        self.assertNotIsInstance(error.exception, preprocessing.AudioWindowUnsafeError)

    def test_virtual_segment_dataset_is_unsafe(self):
        self._write_record()
        source_path = Path(self.temp_dir.name) / "source.h5"
        with h5py.File(source_path, "w") as source:
            source.create_dataset("values", data=np.zeros((5, 12)))
        with h5py.File(self.path, "r+") as handle:
            del handle["analysis/segments_pitches"]
            handle["analysis"].create_virtual_dataset(
                "segments_pitches", h5py.VirtualLayout((5, 12), dtype="f8"), fillvalue=0
            )
        with self.assertRaises(preprocessing.AudioWindowUnsafeError):
            preprocess_audio_window(self.path, "TRTEST000001")


if __name__ == "__main__":
    unittest.main()

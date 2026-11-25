import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.preprocess_audio import (
    apply_audio_normalization,
    build_audio_sequence_cache,
    fit_audio_normalization,
    load_audio_sequence_cache,
    preprocess_audio,
)


class AudioPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "track.h5"
        self._write_record()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_record(self, *, track_id=b"TRTEST000001", duration=2.0,
                      starts=None, pitches=None, timbre=None, loudness=None,
                      beats=None, bars=None, sections=None, omit=()):
        starts = np.array([0.0, 1.0, 2.0]) if starts is None else np.asarray(starts)
        pitches = np.tile(np.array([[0.0] * 12, [10.0] * 12, [20.0] * 12]), (1, 1)) if pitches is None else np.asarray(pitches)
        timbre = np.tile(np.array([[100.0] * 12, [110.0] * 12, [120.0] * 12]), (1, 1)) if timbre is None else np.asarray(timbre)
        loudness = np.array([-20.0, -10.0, 0.0]) if loudness is None else np.asarray(loudness)
        beats = np.array([0.0, 0.5, 0.51, 2.0]) if beats is None else np.asarray(beats)
        bars = np.array([1.0]) if bars is None else np.asarray(bars)
        sections = np.array([], dtype=np.float64) if sections is None else np.asarray(sections)
        with h5py.File(self.path, "w") as handle:
            analysis = handle.create_group("analysis")
            songs = np.zeros(1, dtype=[("track_id", "S12"), ("duration", "f8")])
            songs[0] = (track_id, duration)
            analysis.create_dataset("songs", data=songs)
            values = {
                "segments_pitches": pitches,
                "segments_timbre": timbre,
                "segments_loudness_max": loudness,
                "segments_start": starts,
                "beats_start": beats,
                "bars_start": bars,
                "sections_start": sections,
            }
            for name, value in values.items():
                if name not in omit:
                    analysis.create_dataset(name, data=value)

    def _write_sequence_cache(self, *, ids=None, sequences=None, channel_names=None,
                              metadata=None):
        ids = ["TRCACHE00001", "TRCACHE00002"] if ids is None else ids
        sequences = np.arange(len(ids) * 1024 * 28, dtype=np.float32).reshape(
            len(ids), 1024, 28
        ) if sequences is None else sequences
        channel_names = (
            [f"pitch_{index}" for index in range(1, 13)]
            + [f"timbre_{index}" for index in range(1, 13)]
            + ["loudness_max", "beat_count", "bar_count", "section_count"]
            if channel_names is None else channel_names
        )
        metadata = {"grid_size": 1024, "channel_names": channel_names} if metadata is None else metadata
        hdf5_path = Path(self.temp_dir.name) / "sequence-cache.h5"
        metadata_path = Path(self.temp_dir.name) / "sequence-cache.json"
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset("track_ids", data=np.asarray(ids, dtype=string_dtype))
            handle.create_dataset("sequences", data=sequences)
            handle.create_dataset("channel_names", data=np.asarray(channel_names, dtype=string_dtype))
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return hdf5_path, metadata_path

    def test_load_sequence_cache_round_trips_values_and_normalizes_ids(self):
        sequences = np.arange(2 * 1024 * 28, dtype=np.float32).reshape(2, 1024, 28)
        hdf5_path, metadata_path = self._write_sequence_cache(
            ids=[b" trcache00001 ", "TRCACHE00002"], sequences=sequences
        )

        track_ids, actual, metadata = load_audio_sequence_cache(
            hdf5_path, metadata_path, expected_ids=[" trcache00002", "TRCACHE00001 "]
        )

        self.assertEqual(track_ids, ["TRCACHE00001", "TRCACHE00002"])
        np.testing.assert_array_equal(actual, sequences)
        self.assertEqual(metadata["grid_size"], 1024)

    def test_load_sequence_cache_rejects_non_object_metadata_json(self):
        hdf5_path, metadata_path = self._write_sequence_cache()
        metadata_path.write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "metadata JSON must be an object"):
            load_audio_sequence_cache(hdf5_path, metadata_path)

    def test_load_sequence_cache_rejects_bad_dtype_and_shape(self):
        hdf5_path, metadata_path = self._write_sequence_cache(
            sequences=np.zeros((2, 1024, 28), dtype=np.float64)
        )
        with self.assertRaisesRegex(ValueError, "float32"):
            load_audio_sequence_cache(hdf5_path, metadata_path)

        hdf5_path, metadata_path = self._write_sequence_cache(
            sequences=np.zeros((2, 1023, 28), dtype=np.float32)
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            load_audio_sequence_cache(hdf5_path, metadata_path)

        sequences = np.zeros((2, 1024, 28), dtype=np.float32)
        sequences[0, 0, 0] = np.nan
        hdf5_path, metadata_path = self._write_sequence_cache(sequences=sequences)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_audio_sequence_cache(hdf5_path, metadata_path)

    def test_load_sequence_cache_rejects_unsorted_or_duplicate_ids(self):
        for ids, message in ((["TRCACHE00002", "TRCACHE00001"], "sorted"),
                             (["TRCACHE00001", "TRCACHE00001"], "unique")):
            with self.subTest(ids=ids):
                hdf5_path, metadata_path = self._write_sequence_cache(ids=ids)
                with self.assertRaisesRegex(ValueError, message):
                    load_audio_sequence_cache(hdf5_path, metadata_path)

    def test_load_sequence_cache_rejects_channel_and_grid_metadata_mismatches(self):
        names = [f"pitch_{index}" for index in range(1, 13)] + [
            f"timbre_{index}" for index in range(1, 13)
        ] + ["loudness_max", "beat_count", "bar_count", "wrong"]
        hdf5_path, metadata_path = self._write_sequence_cache(channel_names=names)
        with self.assertRaisesRegex(ValueError, "channel"):
            load_audio_sequence_cache(hdf5_path, metadata_path)

        hdf5_path, metadata_path = self._write_sequence_cache()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["channel_names"][0] = "wrong"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "channel"):
            load_audio_sequence_cache(hdf5_path, metadata_path)

        hdf5_path, metadata_path = self._write_sequence_cache(
            metadata={"grid_size": 512, "channel_names": [
                *[f"pitch_{index}" for index in range(1, 13)],
                *[f"timbre_{index}" for index in range(1, 13)],
                "loudness_max", "beat_count", "bar_count", "section_count",
            ]}
        )
        with self.assertRaisesRegex(ValueError, "grid_size"):
            load_audio_sequence_cache(hdf5_path, metadata_path)

    def test_load_sequence_cache_requires_expected_id_set_without_duplicates(self):
        hdf5_path, metadata_path = self._write_sequence_cache()
        for expected_ids, message in ((["TRCACHE00001", "TRCACHE00003"], "match"),
                                      (["TRCACHE00001", "TRCACHE00001"], "unique")):
            with self.subTest(expected_ids=expected_ids):
                with self.assertRaisesRegex(ValueError, message):
                    load_audio_sequence_cache(hdf5_path, metadata_path, expected_ids)

    def test_interpolates_channels_and_counts_event_collisions(self):
        actual = preprocess_audio(self.path, "TRTEST000001", grid_size=5)
        self.assertEqual(actual.shape, (5, 28))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual[:, 0], [0, 5, 10, 15, 20])
        np.testing.assert_allclose(actual[:, 12], [100, 105, 110, 115, 120])
        np.testing.assert_allclose(actual[:, 24], [-20, -15, -10, -5, 0])
        # 0.5 and 0.51 both round to bin 1; endpoints land in bins 0 and 4.
        np.testing.assert_array_equal(actual[:, 25], [1, 2, 0, 0, 1])
        np.testing.assert_array_equal(actual[:, 26], [0, 0, 1, 0, 0])
        np.testing.assert_array_equal(actual[:, 27], [0, 0, 0, 0, 0])

    def test_large_unsigned_event_times_do_not_overflow_when_binned(self):
        self._write_record(
            duration=1.5e19,
            beats=np.array([8_000_000_000_000_000_000], dtype=np.uint64),
        )

        actual = preprocess_audio(self.path, "TRTEST000001", grid_size=4)

        np.testing.assert_array_equal(actual[:, 25], [0, 0, 1, 0])

    def test_output_is_deterministic(self):
        first = preprocess_audio(self.path, "TRTEST000001", grid_size=7)
        second = preprocess_audio(self.path, "TRTEST000001", grid_size=7)
        np.testing.assert_array_equal(first, second)

    def test_default_grid_size(self):
        result = preprocess_audio(self.path, "TRTEST000001")
        self.assertEqual(result.shape, (1024, 28))

    def test_rejects_track_id_mismatch(self):
        with self.assertRaisesRegex(ValueError, "track ID"):
            preprocess_audio(self.path, "TROTHER000001", grid_size=3)

    def test_rejects_missing_required_arrays(self):
        self._write_record(omit=("segments_timbre",))
        with self.assertRaisesRegex(ValueError, "segments_timbre"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_rejects_malformed_segment_shape(self):
        self._write_record(pitches=np.zeros((3, 11)))
        with self.assertRaisesRegex(ValueError, "segments_pitches"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_rejects_misaligned_segment_arrays(self):
        self._write_record(loudness=[-20.0, -10.0])
        with self.assertRaisesRegex(ValueError, "aligned"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_rejects_non_finite_continuous_or_event_values(self):
        self._write_record(timbre=np.array([[0.] * 12, [np.nan] * 12, [0.] * 12]))
        with self.assertRaisesRegex(ValueError, "finite"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)
        self._write_record(beats=[0.0, np.inf])
        with self.assertRaisesRegex(ValueError, "finite"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_rejects_nonnumeric_segment_and_event_arrays(self):
        self._write_record(pitches=np.full((3, 12), b"bad", dtype="S3"))
        with self.assertRaisesRegex(ValueError, "segments_pitches.*real numeric"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)
        self._write_record(beats=np.array([b"bad"], dtype="S3"))
        with self.assertRaisesRegex(ValueError, "beats_start.*real numeric"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_rejects_invalid_duration(self):
        for duration in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(duration=duration):
                self._write_record(duration=duration)
                with self.assertRaisesRegex(ValueError, "duration"):
                    preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_rejects_unordered_or_out_of_range_segment_times(self):
        self._write_record(starts=[0.0, 1.5, 1.0])
        with self.assertRaisesRegex(ValueError, "ordered"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)
        self._write_record(starts=[0.0, 1.0, 2.1])
        with self.assertRaisesRegex(ValueError, "duration"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_rejects_duplicate_segment_times(self):
        self._write_record(starts=[0.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_rejects_event_times_outside_duration(self):
        self._write_record(beats=[-0.1])
        with self.assertRaisesRegex(ValueError, "beats_start.*duration"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_rejects_empty_segment_arrays(self):
        self._write_record(starts=[], pitches=np.empty((0, 12)), timbre=np.empty((0, 12)), loudness=[])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            preprocess_audio(self.path, "TRTEST000001", grid_size=3)

    def test_normalization_fits_only_training_tracks_and_preserves_event_channels(self):
        sequences = np.zeros((3, 2, 28), dtype=np.float32)
        sequences[0, :, :25] = 1.0
        sequences[1, :, :25] = 3.0
        sequences[2, :, :25] = 1000.0
        sequences[:, :, 25:] = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
        raw = sequences.copy()

        metadata = fit_audio_normalization(sequences, ["train-a", "train-b", "test"], ["train-a", "train-b"])
        normalized = apply_audio_normalization(sequences, metadata)

        self.assertEqual(metadata["continuous_channel_count"], 25)
        self.assertEqual(metadata["train_ids"], ["train-a", "train-b"])
        np.testing.assert_allclose(metadata["mean"], np.full(25, 2.0))
        np.testing.assert_allclose(metadata["std"], np.full(25, 1.0))
        np.testing.assert_allclose(normalized[0, :, :25], -1.0)
        np.testing.assert_allclose(normalized[1, :, :25], 1.0)
        np.testing.assert_allclose(normalized[2, :, :25], 998.0)
        np.testing.assert_array_equal(normalized[:, :, 25:], sequences[:, :, 25:])
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(sequences, raw)

    def test_normalization_uses_scale_one_for_zero_variance(self):
        sequences = np.ones((2, 3, 28), dtype=np.float32)
        metadata = fit_audio_normalization(sequences, ["a", "b"], ["a"])
        normalized = apply_audio_normalization(sequences, metadata)
        np.testing.assert_array_equal(metadata["std"], np.zeros(25))
        np.testing.assert_array_equal(normalized[:, :, :25], np.zeros((2, 3, 25)))
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_normalization_rejects_event_values_not_exact_in_float32(self):
        sequences = np.zeros((1, 1, 28), dtype=np.float64)
        sequences[0, 0, 25] = 16_777_217
        metadata = {"mean": [0.0] * 25, "std": [1.0] * 25, "continuous_channel_count": 25}
        with self.assertRaisesRegex(ValueError, "exactly representable as float32"):
            apply_audio_normalization(sequences, metadata)

    def test_normalization_accepts_float64_with_exact_event_values(self):
        sequences = np.zeros((1, 2, 28), dtype=np.float64)
        sequences[0, :, 0] = [1.0, 3.0]
        sequences[0, :, 25:] = [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
        metadata = fit_audio_normalization(sequences, ["train"], ["train"])

        normalized = apply_audio_normalization(sequences, metadata)

        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(normalized[:, :, 25:], sequences[:, :, 25:].astype(np.float32))

    def test_normalization_accepts_integer_sequences_with_exact_event_values(self):
        sequences = np.zeros((1, 2, 28), dtype=np.int32)
        sequences[0, :, 0] = [1, 3]
        sequences[0, :, 25:] = [[0, 1, 2], [2, 1, 0]]
        metadata = fit_audio_normalization(sequences, ["train"], ["train"])

        normalized = apply_audio_normalization(sequences, metadata)

        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(normalized[:, :, 25:], sequences[:, :, 25:].astype(np.float32))

    def test_normalization_rejects_invalid_inputs(self):
        valid = np.zeros((2, 3, 28), dtype=np.float32)
        cases = (
            (valid, ["a"], ["a"], "matching"),
            (valid, ["a", "a"], ["a"], "unique"),
            (valid, ["a", "b"], [], "non-empty"),
            (valid, ["a", "b"], ["missing"], "present"),
            (np.zeros((2, 3, 27)), ["a", "b"], ["a"], "shape"),
            (np.full((2, 3, 28), np.nan), ["a", "b"], ["a"], "finite"),
            (np.full((2, 3, 28), "bad"), ["a", "b"], ["a"], "numeric"),
        )
        for sequences, track_ids, train_ids, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    fit_audio_normalization(sequences, track_ids, train_ids)

    def test_cache_round_trip_is_sorted_normalized_and_auditable(self):
        input_root = Path(self.temp_dir.name) / "sources"
        input_root.mkdir()
        values = {
            "TRCACH00003": (30.0, [0.0, 1.0]),
            "TRCACH00001": (1.0, [1.0, 0.0]),
            "TRCACH00002": (3.0, [2.0, 1.0]),
        }
        source_paths = {}
        for track_id, (offset, beats) in values.items():
            path = input_root / f"{track_id}.h5"
            self.path = path
            self._write_record(
                track_id=track_id.encode(),
                pitches=np.tile(np.array([[offset] * 12, [offset + 2] * 12, [offset + 4] * 12]), (1, 1)),
                timbre=np.tile(np.array([[offset] * 12, [offset + 2] * 12, [offset + 4] * 12]), (1, 1)),
                loudness=[offset, offset + 2, offset + 4],
                beats=beats,
                bars=[],
                sections=[],
            )
            source_paths[track_id] = path

        audit_path = Path(self.temp_dir.name) / "audit.json"
        audit_path.write_text('{"split": "audit"}', encoding="utf-8")
        output_path = Path(self.temp_dir.name) / "cache.h5"
        metadata_path = Path(self.temp_dir.name) / "cache_meta.json"
        splits = {
            "train": ["TRCACH00002", "TRCACH00001"],
            "validation": ["TRCACH00003"],
            "test": [],
        }

        build_audio_sequence_cache(
            input_root, splits, audit_path, output_path, metadata_path, grid_size=3
        )

        expected_ids = ["TRCACH00001", "TRCACH00002", "TRCACH00003"]
        raw = np.stack([
            preprocess_audio(source_paths[track_id], track_id, grid_size=3)
            for track_id in expected_ids
        ])
        normalization = fit_audio_normalization(raw, expected_ids, splits["train"])
        expected = apply_audio_normalization(raw, normalization)
        with h5py.File(output_path, "r") as handle:
            ids = [value.decode() for value in handle["track_ids"][...]]
            self.assertEqual(ids, expected_ids)
            self.assertEqual(handle["sequences"].shape, (3, 3, 28))
            self.assertEqual(handle["sequences"].dtype, np.dtype(np.float32))
            sequences = handle["sequences"][...]
            names = [value.decode() for value in handle["channel_names"][...]]
        np.testing.assert_array_equal(sequences, expected)
        np.testing.assert_array_equal(sequences[:, :, 25:], raw[:, :, 25:])
        self.assertEqual(len(names), 28)
        self.assertEqual(
            names,
            [f"pitch_{index}" for index in range(1, 13)]
            + [f"timbre_{index}" for index in range(1, 13)]
            + ["loudness_max", "beat_count", "bar_count", "section_count"],
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["split_counts"], {"train": 2, "validation": 1, "test": 0})
        self.assertEqual(metadata["grid_size"], 3)
        self.assertEqual(metadata["split_ids"], splits)
        self.assertEqual(metadata["grid_definition"]["points"], 3)
        self.assertEqual(metadata["normalization"]["train_ids"], splits["train"])
        self.assertEqual(metadata["normalization"], normalization)
        self.assertEqual(metadata["channel_names"], names)
        self.assertEqual(metadata["audit_sha256"], hashlib.sha256(audit_path.read_bytes()).hexdigest())
        self.assertEqual(
            metadata["source_sha256"],
            {track_id: hashlib.sha256(source_paths[track_id].read_bytes()).hexdigest()
             for track_id in expected_ids},
        )

        repeated_path = Path(self.temp_dir.name) / "repeated.h5"
        repeated_metadata_path = Path(self.temp_dir.name) / "repeated_meta.json"
        build_audio_sequence_cache(
            input_root, splits, audit_path, repeated_path, repeated_metadata_path, grid_size=3
        )
        with h5py.File(repeated_path, "r") as handle:
            self.assertEqual([value.decode() for value in handle["track_ids"][...]], expected_ids)
            np.testing.assert_array_equal(handle["sequences"][...], sequences)
        self.assertEqual(
            json.loads(repeated_metadata_path.read_text(encoding="utf-8")), metadata
        )

    def test_cache_rejects_missing_and_duplicate_selected_sources(self):
        input_root = Path(self.temp_dir.name) / "sources"
        input_root.mkdir()
        audit_path = Path(self.temp_dir.name) / "audit.json"
        audit_path.write_text("{}", encoding="utf-8")
        splits = {"train": ["TRCACH00001"], "validation": [], "test": []}
        output_path = Path(self.temp_dir.name) / "cache.h5"
        metadata_path = Path(self.temp_dir.name) / "cache_meta.json"

        with self.assertRaisesRegex(ValueError, "missing"):
            build_audio_sequence_cache(
                input_root, splits, audit_path, output_path, metadata_path, grid_size=3
            )

        for filename in ("first.h5", "second.h5"):
            self.path = input_root / filename
            self._write_record(track_id=b"TRCACH00001")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_audio_sequence_cache(
                input_root, splits, audit_path, output_path, metadata_path, grid_size=3
            )

    def test_cache_preserves_existing_outputs_when_audit_hashing_fails(self):
        input_root = Path(self.temp_dir.name) / "sources"
        input_root.mkdir()
        self.path = input_root / "track.h5"
        self._write_record(track_id=b"TRCACH00001")
        missing_audit = Path(self.temp_dir.name) / "missing-audit.json"
        output_path = Path(self.temp_dir.name) / "cache.h5"
        metadata_path = Path(self.temp_dir.name) / "cache_meta.json"
        original_hdf5 = b"existing HDF5 output"
        original_metadata = b"existing metadata"
        output_path.write_bytes(original_hdf5)
        metadata_path.write_bytes(original_metadata)

        with self.assertRaises(FileNotFoundError):
            build_audio_sequence_cache(
                input_root,
                {"train": ["TRCACH00001"], "validation": [], "test": []},
                missing_audit,
                output_path,
                metadata_path,
                grid_size=3,
            )

        self.assertEqual(output_path.read_bytes(), original_hdf5)
        self.assertEqual(metadata_path.read_bytes(), original_metadata)


if __name__ == "__main__":
    unittest.main()

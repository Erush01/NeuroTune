import tempfile
import unittest
import json
import hashlib
from pathlib import Path

import h5py
import numpy as np

from src.features_neurotune_v1 import (
    apply_audio_stats,
    build_neurotune_v1_features,
    build_train_fitted_modalities,
    fit_audio_stats,
    fit_d4_audio_stats,
    read_d4_audio_cache,
    build_d4_neurotune_v1_features,
)
from src.features import build_train_fitted_modalities as build_existing_modalities


class NeuroTuneV1FeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name) / "cache.h5"
        ids = np.asarray([b"A", b"B", b"C"])
        features = np.zeros((5, 26), dtype=np.float32)
        features[:, 0] = [1, 3, 2, 200, 300]
        offsets = [0, 2, 3, 5]
        with h5py.File(self.cache, "w") as h5:
            h5["track_ids"] = ids
            h5["features"] = features
            h5["offsets"] = offsets

    def _auxiliary(self, validation_bpm=1000, validation_lyrics=None, validation_genre="ZZ"):
        return (
            [{"track_id": "A", "tempo": "60"}, {"track_id": "B", "tempo": "120"},
             {"track_id": "C", "tempo": str(validation_bpm)}],
            {"A": {"1": 2}, "B": {"1": 1, "2": 1}, "C": validation_lyrics or {"5000": 9}},
            {"A": "Rock", "B": "Jazz", "C": validation_genre},
        )

    def _d4_cache(self, path=None):
        path = path or (Path(self.tmp.name) / "d4.h5")
        # A has one overlapping source row with duration-dependent values; B is
        # validation-only and deliberately far from the training distribution.
        values = np.zeros((6, 26), dtype=np.float32)
        values[:, 0] = [0, 10, 20, 30, 1000, 2000]
        with h5py.File(path, "w") as h5:
            h5.create_dataset("track_ids", data=np.asarray([b"A", b"B"]))
            h5.create_dataset("features", data=values)
            h5.create_dataset("window_offsets", data=np.asarray([[0, 2, 4, 4], [4, 5, 6, 6]], dtype=np.int64))
            h5.create_dataset("segment_indices", data=np.asarray([7, 8, 7, 8, 10, 11], dtype=np.int64))
            h5.create_dataset("segment_starts", data=np.arange(6, dtype=np.float64))
            h5.create_dataset("window_starts", data=np.zeros((2, 3), dtype=np.float64))
            h5.create_dataset("window_valid", data=np.asarray([[True, True, False], [True, True, False]], dtype=np.bool_))
            h5.create_dataset("song_durations", data=np.asarray([60, 70], dtype=np.float64))
        return path

    def _partitioned_d4_cache(self, path):
        ids = [b"A", b"B", b"C"]
        values = np.zeros((9, 26), dtype=np.float32)
        values[:, 0] = np.arange(9, dtype=np.float32)
        values[6, 0] = np.nan  # test-only corruption must not be read by train/validation.
        with h5py.File(path, "w") as h5:
            h5.create_dataset("track_ids", data=np.asarray(ids))
            h5.create_dataset("features", data=values)
            h5.create_dataset("window_offsets", data=np.asarray(
                [[i * 3, i * 3 + 1, i * 3 + 2, i * 3 + 3] for i in range(3)], dtype=np.int64))
            h5.create_dataset("segment_indices", data=np.arange(9, dtype=np.int64))
            h5.create_dataset("segment_starts", data=np.arange(9, dtype=np.float64))
            h5.create_dataset("window_starts", data=np.zeros((3, 3), dtype=np.float64))
            h5.create_dataset("window_valid", data=np.ones((3, 3), dtype=np.bool_))
            h5.create_dataset("song_durations", data=np.full(3, 20, dtype=np.float64))
        return path

    def test_d4_cache_validates_schema_and_reads_empty_windows(self):
        cache = self._d4_cache()
        d4 = read_d4_audio_cache(cache, ["A"])
        self.assertEqual(d4["track_ids"], ["A"])
        self.assertEqual([len(window) for window in d4["windows"]["A"]], [2, 2, 0])
        self.assertEqual(d4["window_valid"][0].tolist(), [True, True, False])
        with h5py.File(cache, "a") as h5:
            del h5["window_offsets"]
            h5.create_dataset("window_offsets", data=np.asarray([[0, 2, 5, 4], [4, 5, 6, 6]], dtype=np.int64))
        with self.assertRaises(ValueError):
            read_d4_audio_cache(cache, ["A"])

    def test_d4_cache_rejects_invalid_source_segment_metadata(self):
        mutations = (
            ("segment_indices", 0, -1),
            ("segment_starts", 0, -1.0),
            ("segment_starts", 0, 61.0),
            ("song_durations", 0, 0.0),
        )
        for dataset, row, value in mutations:
            with self.subTest(dataset=dataset, value=value):
                cache = self._d4_cache(Path(self.tmp.name) / f"invalid-{dataset}-{value}.h5")
                with h5py.File(cache, "a") as h5:
                    h5[dataset][row] = value
                with self.assertRaises(ValueError):
                    read_d4_audio_cache(cache, ["A"])

    def test_d4_cache_rejects_nonincreasing_segment_order_within_window(self):
        for dataset, values in (("segment_indices", [8, 7]),
                                ("segment_starts", [1.0, 0.0])):
            with self.subTest(dataset=dataset):
                cache = self._d4_cache(Path(self.tmp.name) / f"unordered-{dataset}.h5")
                with h5py.File(cache, "a") as h5:
                    h5[dataset][0:2] = values
                with self.assertRaises(ValueError):
                    read_d4_audio_cache(cache, ["A"])

    def test_d4_selected_reads_and_fit_ignore_unselected_test_features(self):
        cache = self._partitioned_d4_cache(Path(self.tmp.name) / "partitioned-d4.h5")
        selected = read_d4_audio_cache(cache, ["A", "B"])
        self.assertEqual(set(selected["windows"]), {"A", "B"})
        self.assertEqual(fit_d4_audio_stats(cache, ["A"])["row_count"], 3)
        tracks = [{"track_id": x, "tempo": bpm} for x, bpm in (("A", 60), ("B", 90))]
        features, _ = build_d4_neurotune_v1_features(
            cache, tracks, {"A": {}, "B": {}}, {"A": "Rock", "B": "Jazz"},
            ["A"], ["A", "B"], split_hash="a" * 64,
        )
        self.assertEqual(set(features["audio_windows"]), {"A", "B"})

    def test_d4_selected_corrupt_rows_are_rejected(self):
        for selected_id, row in (("A", 0), ("B", 3)):
            with self.subTest(track=selected_id):
                cache = self._partitioned_d4_cache(Path(self.tmp.name) / f"corrupt-{selected_id}.h5")
                with h5py.File(cache, "a") as h5:
                    h5["features"][row, 0] = np.nan
                with self.assertRaises(ValueError):
                    read_d4_audio_cache(cache, [selected_id])
                if selected_id == "A":
                    with self.assertRaises(ValueError):
                        fit_d4_audio_stats(cache, ["A"])
                else:
                    with self.assertRaises(ValueError):
                        build_d4_neurotune_v1_features(
                            cache, [{"track_id": "A", "tempo": 60}, {"track_id": "B", "tempo": 90}],
                            {"A": {}, "B": {}}, {"A": "Rock", "B": "Jazz"},
                            ["A"], ["A", "B"], split_hash="a" * 64,
                        )

    def test_d4_stats_average_overlapping_segment_rows_once_and_fit_train_only(self):
        cache = self._d4_cache()
        stats = fit_d4_audio_stats(cache, ["A"])
        self.assertEqual(stats["row_count"], 2)
        self.assertEqual(stats["mean"][0], 15)
        self.assertEqual(stats["std"][0], 5)

    def test_d4_builder_keeps_raw_rows_aligned_and_empty_window_invalid(self):
        cache = self._d4_cache()
        tracks = [{"track_id": "A", "tempo": 60}, {"track_id": "B", "tempo": 120}]
        raw = read_d4_audio_cache(cache, ["A", "B"])
        original = [window.copy() for window in raw["windows"]["A"]]
        features, manifest = build_d4_neurotune_v1_features(
            cache, tracks, {"A": {}, "B": {}}, {"A": "Rock", "B": "Jazz"},
            ["A"], ["A", "B"], split_hash="a" * 64,
        )
        self.assertEqual([len(window) for window in features["audio_windows"]["A"]], [2, 2, 0])
        np.testing.assert_array_equal(features["window_valid"], [[True, True, False], [True, True, False]])
        for got, before in zip(raw["windows"]["A"], original):
            np.testing.assert_array_equal(got, before)
        self.assertEqual(features["audio_windows"]["A"][0][0, 0], -3)
        self.assertEqual(manifest["audio_fit_rows"], 2)
        self.assertEqual(set(features) - {"audio_windows", "window_valid"}, {"bpm", "genre", "lyrics"})

    def test_audio_statistics_use_train_rows_only_and_constant_channels(self):
        stats = fit_audio_stats(self.cache, ["A", "B"])
        self.assertAlmostEqual(stats["mean"][0], 2)
        self.assertAlmostEqual(stats["std"][0], np.sqrt(2 / 3))
        np.testing.assert_array_equal(stats["std"][1:], np.ones(25))

    def test_audio_apply_is_float32_and_does_not_mutate_input(self):
        stats = fit_audio_stats(self.cache, ["A", "B"])
        original = np.asarray([[1, 4] + [0] * 24], dtype=np.float64)
        saved = original.copy()
        normalized = apply_audio_stats(original, stats)
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(original, saved)
        self.assertAlmostEqual(float(normalized[0, 1]), 4.0)

    def test_audio_stats_are_stable_for_large_float32_values(self):
        cache = Path(self.tmp.name) / "large_values.h5"
        values = np.zeros((2, 26), dtype=np.float32)
        values[:, 0] = np.asarray([1e9, 1e9 + 64], dtype=np.float32)
        with h5py.File(cache, "w") as h5:
            h5["track_ids"] = np.asarray([b"A"])
            h5["features"] = values
            h5["offsets"] = np.asarray([0, 2], dtype=np.int64)
        stats = fit_audio_stats(cache, ["A"])
        self.assertEqual(stats["std"][0], 32)
        normalized = apply_audio_stats(values, stats)
        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_array_equal(normalized[:, 0], [-1, 1])

    def test_audio_apply_rejects_float32_overflow(self):
        values = np.zeros((1, 26), dtype=np.float64)
        values[0, 0] = 3e38
        stats = {"mean": np.zeros(26, dtype=np.float64), "std": np.ones(26, dtype=np.float64)}
        stats["mean"][0] = -3e38
        with self.assertRaises(ValueError):
            apply_audio_stats(values, stats)

    def test_auxiliary_outputs_train_fit_and_validation_independent(self):
        args = self._auxiliary()
        first, metadata = build_train_fitted_modalities(*args, ["A", "B"], ["C", "A", "B"])
        changed, metadata_changed = build_train_fitted_modalities(
            *self._auxiliary(1e8, {"2": 10000}, "Novel"), ["A", "B"], ["C", "A", "B"]
        )
        self.assertEqual(metadata["bpm_mean"], 90)
        self.assertEqual(metadata["bpm_std"], 30)
        self.assertEqual(metadata["genre_vocabulary"], ["Jazz", "Rock"])
        self.assertEqual(len(metadata["idf"]), 5000)
        self.assertEqual(metadata["idf"][0], 1.0)
        self.assertAlmostEqual(metadata["idf"][1], np.log(3 / 2) + 1)
        np.testing.assert_allclose(metadata["idf"], metadata_changed["idf"])
        np.testing.assert_allclose(first["bpm"][1:], changed["bpm"][1:])
        np.testing.assert_allclose(first["lyrics"][1:], changed["lyrics"][1:])
        np.testing.assert_allclose(first["genre"][1:], changed["genre"][1:])
        self.assertEqual(first["bpm"].dtype, np.float32)
        self.assertEqual(first["lyrics"].shape, (3, 5000))
        np.testing.assert_array_equal(first["genre"][0], [0, 0])
        self.assertAlmostEqual(float(np.linalg.norm(first["lyrics"][1])), 1.0)
        self.assertEqual(first["genre"][1].tolist(), [0, 1])

    def test_constant_bpm_and_duplicate_partition_ids(self):
        tracks = [{"track_id": x, "tempo": "90"} for x in "ABC"]
        lyrics = {x: {} for x in "ABC"}
        genres = {x: "Rock" for x in "ABC"}
        modalities, metadata = build_train_fitted_modalities(
            tracks, lyrics, genres, ["A", "B"], ["A", "B", "C"]
        )
        self.assertEqual(metadata["bpm_std"], 0)
        np.testing.assert_array_equal(modalities["bpm"], [[0], [0], [0]])
        with self.assertRaises(ValueError):
            fit_audio_stats(self.cache, ["A", "A"])

    def test_full_builder_preserves_requested_order_and_adds_provenance(self):
        tracks, lyrics, genres = self._auxiliary()
        features, manifest = build_neurotune_v1_features(
            self.cache, tracks, lyrics, genres, ["A", "B"], ["C", "A", "B"],
            split_hash="a" * 64,
        )
        self.assertEqual([len(sequence) for sequence in features["audio"]], [2, 2, 1])
        stats = fit_audio_stats(self.cache, ["A", "B"])
        expected_c = apply_audio_stats(np.asarray([[200] + [0] * 25, [300] + [0] * 25]), stats)
        expected_a = apply_audio_stats(np.asarray([[1] + [0] * 25, [3] + [0] * 25]), stats)
        expected_b = apply_audio_stats(np.asarray([[2] + [0] * 25]), stats)
        np.testing.assert_array_equal(features["audio"][0], expected_c)
        np.testing.assert_array_equal(features["audio"][1], expected_a)
        np.testing.assert_array_equal(features["audio"][2], expected_b)
        self.assertEqual(features["bpm"].shape, (3, 1))
        self.assertEqual(features["lyrics"].shape, (3, 5000))
        self.assertEqual(features["genre"].shape, (3, 2))
        for name in ("bpm", "lyrics", "genre"):
            self.assertEqual(features[name].dtype, np.float32)
        np.testing.assert_allclose(features["bpm"][0], [(1000 - 90) / 30])
        np.testing.assert_array_equal(features["bpm"][1:], [[-1], [1]])
        np.testing.assert_array_equal(features["genre"], [[0, 0], [0, 1], [1, 0]])
        np.testing.assert_array_equal(features["lyrics"][0, :2], [0, 0])
        self.assertGreater(features["lyrics"][0, 4999], 0)
        np.testing.assert_array_equal(features["lyrics"][1, :2] != 0, [True, False])
        np.testing.assert_array_equal(features["lyrics"][2, :2] != 0, [True, True])
        expected_fit_hash = hashlib.sha256(json.dumps(["A", "B"], separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(manifest["fit_ids_hash"], expected_fit_hash)
        self.assertEqual(manifest["audio_fit_rows"], 3)
        self.assertEqual(manifest["bpm_mean"], 90)
        self.assertEqual(manifest["genre_vocabulary"], ["Jazz", "Rock"])
        self.assertEqual(manifest["split_hash"], "a" * 64)
        self.assertEqual(manifest["source_sha256"]["split_manifest"], "a" * 64)
        self.assertEqual(set(manifest["source_sha256"]), {
            "audio_cache", "catalog_bpm_genre", "lyrics", "split_manifest"
        })
        self.assertEqual(set(manifest["code_sha256"]), {
            "src/features_neurotune_v1.py", "src/features.py", "src/data/parse_mxm.py"
        })
        for collection in (manifest["source_sha256"], manifest["code_sha256"]):
            for digest in collection.values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(json.loads(json.dumps(manifest, allow_nan=False)), manifest)

    def test_full_builder_rejects_invalid_split_hash(self):
        tracks, lyrics, genres = self._auxiliary()
        with self.assertRaises(ValueError):
            build_neurotune_v1_features(
                self.cache, tracks, lyrics, genres, ["A", "B"], ["C", "A", "B"], split_hash="BAD"
            )

    def test_cache_rejects_non_int64_and_non_increasing_offsets(self):
        for offsets in ([0.0, 2.0, 3.0, 5.0], [0, 2, 2, 5]):
            with h5py.File(self.cache, "a") as h5:
                del h5["offsets"]
                h5.create_dataset("offsets", data=offsets)
            with self.assertRaises(ValueError):
                fit_audio_stats(self.cache, ["A", "B"])
            # Restore a valid cache for the next malformed-offset case.
            with h5py.File(self.cache, "a") as h5:
                del h5["offsets"]
                h5.create_dataset("offsets", data=np.asarray([0, 2, 3, 5], dtype=np.int64))

    def test_auxiliary_builder_generator_input_matches_existing_transform(self):
        tracks, lyrics, genres = self._auxiliary()
        expected, expected_meta = build_existing_modalities(
            tracks, lyrics, genres, ["A", "B"], ["C", "A", "B"]
        )
        actual, metadata = build_train_fitted_modalities(
            (track for track in tracks), lyrics, genres, ["A", "B"], ["C", "A", "B"]
        )
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(metadata["idf"], expected_meta["idf"])

    def test_full_builder_generator_input_matches_list_input(self):
        tracks, lyrics, genres = self._auxiliary()
        expected, expected_manifest = build_neurotune_v1_features(
            self.cache, tracks, lyrics, genres, ["A", "B"], ["C", "A", "B"], "a" * 64
        )
        actual, manifest = build_neurotune_v1_features(
            self.cache, (track for track in tracks), lyrics, genres,
            ["A", "B"], ["C", "A", "B"], "a" * 64
        )
        for name in ("bpm", "lyrics", "genre"):
            np.testing.assert_array_equal(actual[name], expected[name])
        self.assertEqual(manifest["source_sha256"], expected_manifest["source_sha256"])

    def test_duplicate_track_records_and_nonfinite_audio_stats_are_rejected(self):
        tracks, lyrics, genres = self._auxiliary()
        with self.assertRaises(ValueError):
            build_train_fitted_modalities(tracks + [tracks[0]], lyrics, genres, ["A", "B"], ["A", "B"])
        stats = fit_audio_stats(self.cache, ["A", "B"])
        stats["mean"][0] = np.nan
        with self.assertRaises(ValueError):
            apply_audio_stats(np.zeros((1, 26)), stats)

    def test_fit_id_hash_uses_canonical_json(self):
        tracks, lyrics, genres = self._auxiliary()
        _, manifest = build_neurotune_v1_features(
            self.cache, tracks, lyrics, genres, ["B", "A"], ["C", "A", "B"], "a" * 64
        )
        expected = hashlib.sha256(json.dumps(["A", "B"], separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(manifest["fit_ids_hash"], expected)


if __name__ == "__main__":
    unittest.main()

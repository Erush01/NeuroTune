import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import h5py

from src.data.neurotune_v1_abcd_cv import build_abcd_cv
from src.data.neurotune_v1_abcd_cv_training import load_abcd_cv_training
from src.data.neurotune_v1_abcd_cv_training import build_abcd_cv_fold_mxm
from src.data.neurotune_v1_abcd_cv_training import build_abcd_cv_fold_features
from src.data.neurotune_v1_abcd_cv_training import build_abcd_cv_development_features
from src.data.neurotune_v1_abcd_cv_training import build_abcd_cv_d4_multi_window_features
from src.data.neurotune_v1_abcd_cv_training import build_abcd_cv_fold_schedules


def tid(i):
    return f"TR{i:016d}"


class AbcdCvTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / "cv"
        self.tracks = {tid(i): {"track_id": tid(i), "artist_id": f"artist-{i // 2}", "artist_name": f"Artist {i // 2}", "tempo": 60 + i, "genre": f"genre-{i % 3}"} for i in range(12)}
        self.audit = self.root / "audit.json"
        self.audit.write_text(json.dumps({"status": "complete", "window": {"seconds": 30, "interval": "[0,30)"}, "channels": [f"c{i}" for i in range(26)], "source_sha256": {k: k.lower() * 64 for k in "ABCD"}}))
        self.catalog = self.root / "catalog.csv"
        self._catalog(self.catalog, self.tracks)
        self.cache = self.root / "cache.h5"
        with h5py.File(self.cache, "w") as h:
            h.create_dataset("track_ids", data=[x.encode() for x in sorted(self.tracks)])
            h.create_dataset("features", shape=(24, 26), dtype="f4")
            h.create_dataset("offsets", data=list(range(0, 25, 2)))
            for key, value in {"window_seconds": 30, "interval": "[0,30)", "channels": json.dumps([f"c{i}" for i in range(26)]), "normalization": "raw/unnormalized; deferred until train split", "audit_sha256": self._sha(self.audit), "catalog_sha256": self._sha(self.catalog), "source_sha256": json.dumps(json.loads(self.audit.read_text())["source_sha256"], sort_keys=True), "code_sha256": json.dumps({}, sort_keys=True)}.items():
                h.attrs[key] = value
        self.metadata = self.root / "metadata.json"
        self.metadata.write_text(json.dumps({"status": "complete", "window_seconds": 30, "interval": "[0,30)", "channel_count": 26, "channels": [f"c{i}" for i in range(26)], "dtype": "float32", "normalization": "raw/unnormalized; deferred until train split", "segment_count": 24, "track_count": 12, "rejected_track_count": 0, "valid_track_ids": sorted(self.tracks), "sequence_lengths": {key: 2 for key in self.tracks}, "rejected": [], "audit_sha256": self._sha(self.audit), "catalog_sha256": self._sha(self.catalog), "source_sha256": json.loads(self.audit.read_text())["source_sha256"], "code_sha256": {}}))
        self.rejections = self.root / "rejections.json"
        self.rejections.write_text("[]")
        self.train = self.root / "train.csv"
        self.official = self.root / "official.csv"
        self._edges(self.train, [(tid(0), tid(1)), (tid(2), tid(3)), (tid(4), tid(5)), (tid(6), tid(7)), (tid(8), tid(9)), (tid(10), tid(11))])
        self._edges(self.official, [(tid(0), tid(2)), (tid(4), tid(6)), (tid(8), tid(10))])
        self.abdir = self.root / "legacy"
        self.abdir.mkdir()
        self.ab_catalog = self.abdir / "catalog.csv"
        self._catalog(self.ab_catalog, {tid(i): self.tracks[tid(i)] for i in range(4)})
        self.ab_split = self.root / "ab-split.json"
        self.ab_split.write_text(json.dumps({"artist_keys": {"train": ["artist-1"], "validation": [], "test": ["artist-0"]}, "track_ids": {"train": [tid(2), tid(3)], "validation": [], "test": [tid(0), tid(1)]}}))
        self.ab_test = self.root / "ab-test.csv"
        self.legacy_edge = (tid(2), tid(3))
        self._edges(self.ab_test, [self.legacy_edge])
        build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.official,
                      cache_path=self.cache, cache_metadata_path=self.metadata, rejection_path=self.rejections,
                      ab_catalog_path=self.ab_catalog, ab_split_path=self.ab_split, ab_test_edges_path=self.ab_test, output_dir=self.out)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _sha(path):
        import hashlib
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _catalog(path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["track_id", "artist_id", "artist_name", "tempo", "genre"])
            writer.writeheader()
            writer.writerows(rows.values())

    @staticmethod
    def _edges(path, edges):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["track_id_a", "track_id_b"])
            writer.writerows(edges)

    def load(self):
        return load_abcd_cv_training(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train,
                                     test_edges_path=self.official, cache_path=self.cache, cache_metadata_path=self.metadata,
                                     rejection_path=self.rejections, ab_catalog_path=self.ab_catalog, ab_split_path=self.ab_split,
                                     ab_test_edges_path=self.ab_test, cv_dir=self.out)

    def test_valid_load_keeps_sources_separate_and_known_positives_exclusion_only(self):
        loaded = self.load()
        self.assertEqual(loaded["cache_ids"], sorted(self.tracks))
        self.assertEqual(len(loaded["folds"]), 3)
        self.assertIn(self.legacy_edge, loaded["ab_test_source_edges"])
        self.assertIn(self.legacy_edge, loaded["known_positive_edges"])
        for fold in loaded["folds"]:
            self.assertFalse(set(fold["train_edges"]) & set(fold["test_train_source_edges"]))
            self.assertFalse(set(fold["test_train_source_edges"]) & set(fold["test_official_source_edges"]))
            self.assertNotIn(self.legacy_edge, sum((fold[k] for k in ("train_edges", "validation_edges", "test_train_source_edges", "test_official_source_edges")), []))
            self.assertNotIn("known_positive_edges", fold)

    def test_fold_features_fit_only_training_ids_without_lyrics(self):
        from unittest.mock import patch
        from src.features_neurotune_v1 import build_neurotune_v1_features

        loaded = self.load()
        fold = loaded["folds"][0]
        with patch("src.data.parse_mxm.parse_mxm", side_effect=AssertionError("MXM must not be parsed")), \
                patch("src.data.neurotune_v1_abcd_cv_training.build_neurotune_v1_features",
                      wraps=build_neurotune_v1_features) as builder:
            features, manifest = build_abcd_cv_fold_features(loaded, fold)
        self.assertEqual(builder.call_args.args[2], {})
        self.assertEqual(builder.call_args.args[4], fold["train"])
        self.assertEqual(builder.call_args.args[5], loaded["cache_ids"])
        self.assertEqual(manifest["fit_ids_hash"], __import__("src.features_neurotune_v1", fromlist=["_canonical_hash"])._canonical_hash(sorted(fold["train"])))
        self.assertEqual(manifest["audio_fit_rows"], len(fold["train"]) * 2)
        self.assertEqual(set(features), {"audio", "bpm", "genre"})
        self.assertEqual(len(features["audio"]), len(loaded["cache_ids"]))
        self.assertEqual(features["bpm"].shape[0], len(loaded["cache_ids"]))
        self.assertEqual(features["genre"].shape[0], len(loaded["cache_ids"]))
        train_bpms = [float(self.tracks[track_id]["tempo"]) for track_id in fold["train"]]
        bpm_mean = sum(train_bpms) / len(train_bpms)
        bpm_std = (sum((value - bpm_mean) ** 2 for value in train_bpms) / len(train_bpms)) ** 0.5
        self.assertAlmostEqual(manifest["bpm_mean"], bpm_mean)
        self.assertAlmostEqual(manifest["bpm_std"], bpm_std)
        self.assertEqual(manifest["genre_vocabulary"], sorted({self.tracks[track_id]["genre"] for track_id in fold["train"]}))

    def test_development_features_are_aligned_and_only_request_train_validation_ids(self):
        from unittest.mock import patch
        from src.features_neurotune_v1 import build_neurotune_v1_features

        loaded = self.load()
        fold = dict(loaded["folds"][0])
        held_out = sorted(set(loaded["cache_ids"]) - set(fold["train"]))
        fold["validation"], fold["test"] = held_out[:1], held_out[1:]
        requested_ids = sorted(fold["train"] + fold["validation"])
        lyrics = {track_id: {"1": index + 1} for index, track_id in enumerate(loaded["cache_ids"])}
        with patch("src.data.neurotune_v1_abcd_cv_training.build_neurotune_v1_features",
                   wraps=build_neurotune_v1_features) as builder:
            features, manifest = build_abcd_cv_development_features(loaded, fold, lyrics)

        self.assertEqual(set(features), {"audio", "bpm", "genre", "lyrics"})
        self.assertEqual(builder.call_args.args[4], fold["train"])
        self.assertEqual(builder.call_args.args[5], requested_ids)
        self.assertEqual(set(builder.call_args.args[2]), set(requested_ids))
        self.assertEqual(len(features["audio"]), len(requested_ids))
        for key in ("bpm", "genre", "lyrics"):
            self.assertEqual(features[key].shape[0], len(requested_ids))
        self.assertEqual(features["lyrics"].shape[1], 5000)
        self.assertEqual(manifest["fit_ids_hash"], __import__(
            "src.features_neurotune_v1", fromlist=["_canonical_hash"]
        )._canonical_hash(sorted(fold["train"])))
        self.assertFalse(set(fold["test"]) & set(builder.call_args.args[5]))

    def test_development_features_reject_missing_train_or_validation_lyrics(self):
        loaded = self.load()
        fold = dict(loaded["folds"][0])
        missing_id = next(track_id for track_id in loaded["cache_ids"] if track_id not in fold["train"])
        fold["validation"], fold["test"] = [missing_id], []
        lyrics = {track_id: {} for track_id in loaded["cache_ids"] if track_id != missing_id}
        with self.assertRaisesRegex(ValueError, "lyrics are missing"):
            build_abcd_cv_development_features(loaded, fold, lyrics)

    def test_development_feature_idf_and_transform_parameters_ignore_validation_test_lyrics(self):
        import numpy as np

        loaded = self.load()
        fold = dict(loaded["folds"][0])
        held_out = sorted(set(loaded["cache_ids"]) - set(fold["train"]))
        fold["validation"], fold["test"] = held_out[:1], held_out[1:]
        lyrics = {track_id: {"1": 2} for track_id in loaded["cache_ids"]}
        changed = {track_id: dict(document) for track_id, document in lyrics.items()}
        for track_id in fold["validation"] + fold["test"]:
            changed[track_id] = {"2": 999, "5000": 3}

        first_features, first_manifest = build_abcd_cv_development_features(loaded, fold, lyrics)
        second_features, second_manifest = build_abcd_cv_development_features(loaded, fold, changed)

        self.assertEqual(first_manifest["idf"], second_manifest["idf"])
        for key in ("audio_mean", "audio_std", "audio_fit_rows", "bpm_mean", "bpm_std", "genre_vocabulary"):
            self.assertEqual(first_manifest[key], second_manifest[key])
        np.testing.assert_array_equal(first_features["audio"], second_features["audio"])
        for key in ("bpm", "genre"):
            np.testing.assert_array_equal(first_features[key], second_features[key])

    def test_d4_builder_uses_train_validation_only_and_preserves_auxiliaries(self):
        import numpy as np

        loaded = self.load()
        fold = {"train": [tid(0), tid(1)], "validation": [tid(2)], "test": [tid(3)]}
        song_ids = sorted(fold["train"] + fold["validation"])
        cache_ids = sorted(loaded["cache_ids"])
        cache = self.root / "d4.h5"
        values = np.zeros((len(cache_ids) * 3, 26), dtype=np.float32)
        values[:, 0] = np.arange(len(values), dtype=np.float32)
        with h5py.File(cache, "w") as h5:
            h5.create_dataset("track_ids", data=np.asarray([x.encode() for x in cache_ids]))
            h5.create_dataset("features", data=values)
            h5.create_dataset("window_offsets", data=np.asarray([[i * 3, i * 3 + 1, i * 3 + 2, i * 3 + 3] for i in range(len(cache_ids))], dtype=np.int64))
            h5.create_dataset("segment_indices", data=np.arange(len(values), dtype=np.int64))
            h5.create_dataset("segment_starts", data=np.arange(len(values), dtype=np.float64))
            h5.create_dataset("window_starts", data=np.zeros((len(cache_ids), 3), dtype=np.float64))
            h5.create_dataset("window_valid", data=np.ones((len(cache_ids), 3), dtype=np.bool_))
            h5.create_dataset("song_durations", data=np.full(len(cache_ids), 60, dtype=np.float64))
        loaded["d4_cache_path"] = self.cache
        lyrics = {track_id: {"1": i + 1} for i, track_id in enumerate(song_ids)}

        features, manifest = build_abcd_cv_d4_multi_window_features(loaded, fold, lyrics, cache)

        self.assertEqual(list(features["audio_windows"]), song_ids)
        self.assertFalse(set(features["audio_windows"]) & set(fold["test"]))
        self.assertEqual(manifest["fit_ids_hash"], hashlib.sha256(json.dumps(sorted(fold["train"]), separators=(",", ":")).encode()).hexdigest())
        self.assertEqual(features["bpm"].shape[0], len(song_ids))
        self.assertEqual(features["genre"].shape[0], len(song_ids))
        self.assertEqual(features["lyrics"].shape, (len(song_ids), 5000))
        self.assertEqual(manifest["audio_fit_rows"], len(fold["train"]) * 3)

    def test_fold_mxm_filters_catalog_and_records_empty_and_hash_provenance(self):
        mxm = self.root / "mxm"
        mxm.mkdir()
        (mxm / "part_a.txt").write_text("%word\n" + f"{tid(0)},100,1:2\n{tid(1)},101\n{tid(2)},102,1:0\n" + "".join(f"{tid(i)},{100 + i}\n" for i in range(3, 6)))
        (mxm / "part_b.txt").write_text("%word\n" + "".join(f"{tid(i)},{100 + i}\n" for i in range(6, 12)) + "TR9999999999999999,999,1:9\n")
        audit = json.loads(self.audit.read_text())
        audit["source_sha256"]["mxm"] = self._mxm_directory_sha(mxm)
        self.audit.write_text(json.dumps(audit))
        # Refresh fixture manifests/cache provenance after changing the audit.
        self._refresh_audit_provenance()
        loaded = self.load()
        lyrics, provenance = build_abcd_cv_fold_mxm(loaded, mxm)
        self.assertEqual(set(lyrics), set(self.tracks))
        self.assertEqual(lyrics[tid(0)], {"1": 2})
        self.assertEqual(lyrics[tid(1)], {})
        self.assertEqual(lyrics[tid(2)], {"1": 0})
        self.assertEqual(provenance["empty_document_count"], 11)
        self.assertEqual(provenance["source_sha256"], {"part_a.txt": self._sha(mxm / "part_a.txt"), "part_b.txt": self._sha(mxm / "part_b.txt")})
        self.assertEqual(provenance["directory_sha256"], self._mxm_directory_sha(mxm))
        self.assertEqual(len(provenance["mapping_sha256"]), 64)

    def test_fold_mxm_rejects_missing_catalog_ids_and_audit_hash_mismatch(self):
        mxm = self.root / "mxm"
        mxm.mkdir()
        (mxm / "part_a.txt").write_text("%word\n" + f"{tid(0)},100,1:2\n")
        loaded = self.load()
        loaded["manifest"]["audit_source_sha256"]["mxm"] = self._mxm_directory_sha(mxm)
        with self.assertRaisesRegex(ValueError, "missing catalog"):
            build_abcd_cv_fold_mxm(loaded, mxm)
        (mxm / "part_b.txt").write_text("%word\n" + "".join(f"{tid(i)},{100 + i}\n" for i in range(1, 12)))
        loaded = self.load()
        loaded["manifest"]["audit_source_sha256"]["mxm"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "MXM.*hash"):
            build_abcd_cv_fold_mxm(loaded, mxm)

    def test_fold_mxm_rejects_file_path_when_audit_records_directory_hash(self):
        mxm_file = self.root / "mxm.txt"
        mxm_file.write_text("%word\n" + "".join(f"{tid(i)},{100 + i}\n" for i in range(12)))
        loaded = self.load()
        loaded["manifest"]["audit_source_sha256"]["mxm"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "directory"):
            build_abcd_cv_fold_mxm(loaded, mxm_file)

    def test_fold_mxm_requires_recorded_audit_hash(self):
        mxm = self.root / "mxm"
        mxm.mkdir()
        (mxm / "part.txt").write_text("%word\n" + "".join(f"{tid(i)},{100 + i}\n" for i in range(12)))
        loaded = self.load()
        loaded["manifest"]["audit_source_sha256"].pop("mxm", None)
        with self.assertRaisesRegex(ValueError, "audit.*MXM.*hash"):
            build_abcd_cv_fold_mxm(loaded, mxm)

    def test_fold_mxm_rejects_source_mutation_during_parse(self):
        from unittest.mock import patch
        from src.data.parse_mxm import parse_mxm

        mxm = self.root / "mxm-race"
        mxm.mkdir()
        source = mxm / "part.txt"
        source.write_text("%word\n" + "".join(f"{tid(i)},{100 + i}\n" for i in range(12)))
        loaded = self.load()
        loaded["manifest"]["audit_source_sha256"]["mxm"] = self._mxm_directory_sha(mxm)

        def parse_then_mutate(path, *, track_ids):
            lyrics = parse_mxm(path, track_ids=track_ids)
            source.write_text(source.read_text() + f"{tid(0)},900\n")
            return lyrics

        with patch("src.data.neurotune_v1_abcd_cv_training.parse_mxm", side_effect=parse_then_mutate):
            with self.assertRaisesRegex(ValueError, "changed during parsing"):
                build_abcd_cv_fold_mxm(loaded, mxm)

    def _refresh_audit_provenance(self):
        audit_sha = self._sha(self.audit)
        metadata = json.loads(self.metadata.read_text())
        metadata["audit_sha256"] = audit_sha
        metadata["source_sha256"] = json.loads(self.audit.read_text())["source_sha256"]
        self.metadata.write_text(json.dumps(metadata))
        with h5py.File(self.cache, "r+") as h:
            h.attrs["audit_sha256"] = audit_sha
            h.attrs["source_sha256"] = json.dumps(metadata["source_sha256"], sort_keys=True)
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["input_sha256"]["audit_path"] = audit_sha
        manifest["input_sha256"]["cache_metadata_path"] = self._sha(self.metadata)
        manifest["input_sha256"]["cache_path"] = self._sha(self.cache)
        manifest["audit_source_sha256"] = json.loads(self.audit.read_text())["source_sha256"]
        manifest["cache_metadata_provenance"]["audit_sha256"] = audit_sha
        (self.out / "manifest.json").write_text(json.dumps(manifest))

    @staticmethod
    def _mxm_directory_sha(path):
        digest = hashlib.sha256()
        for source in sorted(Path(path).rglob("*")):
            if source.is_file():
                relative = source.relative_to(path).as_posix().encode("utf-8")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                with source.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()

    def test_fold_schedules_are_partition_local_deterministic_and_exclude_known_positives(self):
        from unittest.mock import patch
        from src.triplets import sample_triplet_schedule

        fold = {
            "train": [tid(i) for i in range(4)],
            "validation": [tid(i) for i in range(4, 8)],
            "train_edges": [(tid(0), tid(1))],
            "validation_edges": [(tid(4), tid(5))],
        }
        loaded = {"known_positive_edges": [
            (tid(0), tid(1)), (tid(0), tid(2)), (tid(1), tid(3)),
            (tid(4), tid(5)), (tid(4), tid(6)), (tid(5), tid(7)),
            (tid(8), tid(9)),
        ]}
        with patch("src.data.neurotune_v1_abcd_cv_training.sample_triplet_schedule",
                   wraps=sample_triplet_schedule) as sampler:
            train_schedule, validation_schedule = build_abcd_cv_fold_schedules(loaded, fold, seed=17)
            repeated_train, repeated_validation = build_abcd_cv_fold_schedules(loaded, fold, seed=17)

        import numpy as np
        np.testing.assert_array_equal(train_schedule, repeated_train)
        np.testing.assert_array_equal(validation_schedule, repeated_validation)
        self.assertEqual(sampler.call_count, 4)
        for call, partition, positive_edges, epochs in (
            (sampler.call_args_list[0], "train", fold["train_edges"], 50),
            (sampler.call_args_list[1], "validation", fold["validation_edges"], 1),
        ):
            ids, edges = call.args[:2]
            kwargs = call.kwargs
            self.assertEqual(set(ids), set(fold[partition]))
            self.assertEqual(set(edges), set(positive_edges))
            expected_known = {edge for edge in loaded["known_positive_edges"]
                              if set(edge) <= set(fold[partition])}
            self.assertEqual(set(kwargs["known_positive_edges"]), expected_known)
            self.assertEqual(kwargs["epochs"], epochs)
            self.assertEqual(kwargs["seed"], 17)

        for schedule, partition, positives in (
            (train_schedule, "train", fold["train_edges"]),
            (validation_schedule, "validation", fold["validation_edges"]),
        ):
            ids = fold[partition]
            id_to_index = {track_id: index for index, track_id in enumerate(ids)}
            expected_directed = {(id_to_index[left], id_to_index[right])
                                 for edge in positives for left, right in (edge, edge[::-1])}
            known_neighbors = {track_id: set() for track_id in ids}
            for left, right in loaded["known_positive_edges"]:
                if left in known_neighbors and right in known_neighbors:
                    known_neighbors[left].add(right)
                    known_neighbors[right].add(left)
            for epoch_rows in schedule:
                self.assertEqual({tuple(row[:2]) for row in epoch_rows}, expected_directed)
                for anchor, _, negative in epoch_rows:
                    self.assertNotIn(ids[int(negative)], known_neighbors[ids[int(anchor)]])

    def test_manifest_input_hash_mismatch_is_rejected(self):
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["input_sha256"]["catalog_path"] = "0" * 64
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "hash"):
            self.load()

    def test_manifest_source_edge_counts_must_match_input_sources(self):
        manifest_path = self.out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_edge_counts"]["train_source"] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "source edge counts"):
            self.load()

    def test_hdf5_sidecar_provenance_mismatch_is_rejected(self):
        with h5py.File(self.cache, "r+") as h:
            h.attrs["catalog_sha256"] = "0" * 64
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["input_sha256"]["cache_path"] = self._sha(self.cache)
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.load()

    def test_missing_or_duplicate_manifest_cache_ids_are_rejected(self):
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["track_ids"].pop()
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "track IDs"):
            self.load()

    def test_overlapping_fold_artist_partitions_are_rejected(self):
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["folds"][0]["train_artist_keys"].append(manifest["folds"][0]["test_artist_keys"][0])
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "artist"):
            self.load()

    def test_test_artist_from_entire_legacy_catalog_is_rejected(self):
        manifest = json.loads((self.out / "manifest.json").read_text())
        test_artist = manifest["folds"][0]["test_artist_keys"][0].removeprefix("artist_id:")
        ab_rows = {tid(i): dict(self.tracks[tid(i)]) for i in range(4)}
        ab_rows[tid(0)]["artist_id"] = test_artist
        self._catalog(self.ab_catalog, ab_rows)
        manifest["input_sha256"]["ab_catalog_path"] = self._sha(self.ab_catalog)
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "legacy A/B"):
            self.load()

    def test_legacy_test_edge_in_any_fold_target_is_rejected(self):
        self._edges(self.out / "fold_0_validation_edges.csv", [self.legacy_edge])
        with self.assertRaisesRegex(ValueError, "A/B test-source"):
            self.load()

    def test_known_positive_must_match_source_union_even_with_updated_digest_and_count(self):
        path = self.out / "known_positive_edges.csv"
        edges = sorted(set(self._read_edges(path)))
        edges.pop()
        self._edges(path, edges)
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["known_positive_edge_count"] = len(edges)
        manifest["known_positive_edges_sha256"] = hashlib.sha256(
            json.dumps([list(edge) for edge in edges], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "source edge union"):
            self.load()

    def test_legacy_ab_test_edges_must_remain_in_known_positive_exclusions(self):
        source_edges = self._read_edges(self.train)
        self._edges(self.train, [edge for edge in source_edges if edge != self.legacy_edge])
        self._update_manifest_input_hash("train_edges_path", self.train)
        known_path = self.out / "known_positive_edges.csv"
        known_edges = [edge for edge in self._read_edges(known_path) if edge != self.legacy_edge]
        self._edges(known_path, known_edges)
        manifest_path = self.out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["known_positive_edge_count"] = len(known_edges)
        manifest["known_positive_edges_sha256"] = hashlib.sha256(
            json.dumps([list(edge) for edge in known_edges], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest["source_edge_counts"]["train_source"] = len(source_edges) - 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "legacy A/B test-source edge missing from known-positive set"):
            self.load()

    def test_legacy_split_is_semantically_validated_after_input_hash_update(self):
        split = json.loads(self.ab_split.read_text())
        split["track_ids"]["train"] = [tid(2)]
        self.ab_split.write_text(json.dumps(split))
        self._update_manifest_input_hash("ab_split_path", self.ab_split)
        with self.assertRaisesRegex(ValueError, "legacy A/B split"):
            self.load()

    def test_manifest_old_test_exclusion_set_must_equal_full_input(self):
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["ab_test_source_policy_exclusions"] = []
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "old A/B test-source exclusion"):
            self.load()

    def test_silently_truncated_target_fails_even_if_manifest_count_is_updated(self):
        target = self.out / "fold_0_train_edges.csv"
        edges = self._read_edges(target)
        self.assertTrue(edges)
        self._edges(target, edges[:-1])
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["folds"][0]["counts"]["train"] = len(edges) - 1
        manifest["folds"][0]["train"] = [list(edge) for edge in edges[:-1]]
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "recomputed targets"):
            self.load()

    def test_non_mapping_fold_entry_is_rejected_clearly(self):
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["folds"][0] = "not a fold mapping"
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fold 0.*mapping"):
            self.load()

    def test_non_mapping_fold_counts_is_rejected_clearly(self):
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest["folds"][0]["counts"] = [1, 2, 3]
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fold 0.*counts.*mapping"):
            self.load()

    def _update_manifest_input_hash(self, key, path):
        manifest_path = self.out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["input_sha256"][key] = self._sha(path)
        manifest_path.write_text(json.dumps(manifest))

    @staticmethod
    def _read_edges(path):
        with Path(path).open(newline="", encoding="utf-8") as stream:
            return [tuple(row) for row in list(csv.reader(stream))[1:]]


if __name__ == "__main__":
    unittest.main()

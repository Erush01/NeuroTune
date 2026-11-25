import csv
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from scripts import preprocess_msd_hf_abcd as cli
from src.data import preprocess_msd_hf_abcd as builder
from src.data.preprocess_audio_window import AudioWindowUnsafeError


def make_h5(track_id, artist="artist", song="song", invalid=False, unsafe=False, zero_segments=False, duration=40.0):
    stream = io.BytesIO()
    with h5py.File(stream, "w") as h5:
        analysis = h5.create_group("analysis")
        metadata = h5.create_group("metadata")
        songs = np.zeros(1, dtype=[("track_id", "S18"), ("duration", "f8"), ("tempo", "f8")])
        songs[0] = (track_id.encode(), duration, 120.0)
        analysis.create_dataset("songs", data=songs)
        metas = np.zeros(1, dtype=[("artist_id", "S20"), ("song_id", "S20")])
        metas[0] = (artist.encode(), song.encode())
        metadata.create_dataset("songs", data=metas)
        if unsafe:
            analysis["segments_start"] = h5py.ExternalLink("elsewhere.h5", "/segments_start")
        else:
            count = 0 if zero_segments else (1 if invalid else 2)
            analysis.create_dataset("segments_start", data=np.array([]) if zero_segments else np.array([31.0] if invalid else [0.0, 1.0]))
        count = 0 if zero_segments else (1 if invalid else 2)
        analysis.create_dataset("segments_pitches", data=np.ones((count, 12)))
        analysis.create_dataset("segments_timbre", data=np.ones((count, 12)))
        analysis.create_dataset("segments_loudness_max", data=np.ones(count))
    return stream.getvalue()


class PreprocessMsdHfAbcdTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = ["TRAAAAAAAAAAAAAAAA", "TRBBBBBBBBBBBBBBBB", "TRCCCCCCCCCCCCCCCC", "TRDDDDDDDDDDDDDDDD"]
        self.catalog = self.root / "catalog.csv"
        with self.catalog.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["track_id", "tempo", "artist_id", "song_id"])
            writer.writeheader()
            for tid in self.ids:
                writer.writerow({"track_id": tid, "tempo": 120, "artist_id": "artist", "song_id": "song"})
        self.archives = {}
        for shard, tid in zip("ABCD", self.ids):
            path = self.root / f"{shard}.tar.gz"
            with tarfile.open(path, "w:gz") as tar:
                payload = make_h5(tid, invalid=(shard == "D"))
                info = tarfile.TarInfo(f"{tid[2]}/{tid[3]}/{tid[4]}/{tid}.h5")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            self.archives[shard] = path
        hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in self.archives.items()}
        self.audit = self.root / "audit.json"
        self.audit.write_text(json.dumps({"status": "complete", "window": {"seconds": 30, "interval": "[0,30)"},
                                          "channels": builder.CHANNELS, "source_sha256": hashes,
                                          "counts": {"eligible_tracks": 4}}), encoding="utf-8")
        self.outputs = [self.root / name for name in ("out.h5", "meta.json", "reject.json")]

    def tearDown(self):
        self.temp.cleanup()

    def run_build(self, **changes):
        args = {"a_tar": self.archives["A"], "b_tar": self.archives["B"], "c_tar": self.archives["C"],
                "d_tar": self.archives["D"], "audit_path": self.audit, "catalog_path": self.catalog,
                "output_h5": self.outputs[0], "output_meta": self.outputs[1], "output_rejections": self.outputs[2]}
        args.update(changes)
        return builder.run(**args)

    def test_sorted_raw_cache_and_track_rejection(self):
        result = self.run_build()
        self.assertEqual((result["track_count"], result["rejected_track_count"]), (3, 1))
        with h5py.File(self.outputs[0], "r") as h5:
            ids = [v.decode() for v in h5["track_ids"][:]]
            self.assertEqual(ids, sorted(self.ids[:3]))
            features = h5["features"][:]
            self.assertEqual(features.dtype, np.float32)
            self.assertEqual(features.shape, (6, 26))
            self.assertTrue(np.isfinite(features).all())
            self.assertIn("raw/unnormalized", h5.attrs["normalization"])
            expected_catalog_hash = hashlib.sha256(self.catalog.read_bytes()).hexdigest()
            self.assertEqual(h5.attrs["catalog_sha256"], expected_catalog_hash)
        meta = json.loads(self.outputs[1].read_text())
        self.assertEqual(meta["catalog_sha256"], hashlib.sha256(self.catalog.read_bytes()).hexdigest())
        self.assertEqual(meta["valid_track_ids"], sorted(self.ids[:3]))
        self.assertEqual(meta["rejected"][0]["track_id"], self.ids[3])
        self.assertEqual(set(meta["code_sha256"]), {"src/data/preprocess_msd_hf_abcd.py", "scripts/preprocess_msd_hf_abcd.py", "src/data/preprocess_audio_window.py"})

    def test_missing_candidate_member_fails(self):
        with tarfile.open(self.archives["D"], "w:gz"):
            pass
        audit = json.loads(self.audit.read_text())
        audit["source_sha256"]["D"] = hashlib.sha256(self.archives["D"].read_bytes()).hexdigest()
        self.audit.write_text(json.dumps(audit), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing catalog tracks"):
            self.run_build()

    def test_source_hash_mismatch_fails(self):
        self.archives["A"].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.run_build()

    def test_catalog_artist_mismatch_is_fatal(self):
        text = self.catalog.read_text().replace(",artist,song", ",wrong-artist,song", 1)
        self.catalog.write_text(text)
        with self.assertRaisesRegex(ValueError, "artist/song mismatch"):
            self.run_build()

    def test_catalog_mutation_during_scan_aborts_without_publishing(self):
        scan = builder._scan_archives

        def mutate_after_scan(paths, records):
            result = scan(paths, records)
            self.catalog.write_text(self.catalog.read_text() + "\n", encoding="utf-8")
            return result

        with patch.object(builder, "_scan_archives", side_effect=mutate_after_scan):
            with self.assertRaisesRegex(ValueError, "catalog.*changed"):
                self.run_build()
        self.assertFalse(any(path.exists() for path in self.outputs))

    def test_unsafe_hdf5_aborts(self):
        path = self.archives["A"]
        with tarfile.open(path, "w:gz") as tar:
            payload = make_h5(self.ids[0], unsafe=True)
            info = tarfile.TarInfo(f"{self.ids[0][2]}/{self.ids[0][3]}/{self.ids[0][4]}/{self.ids[0]}.h5")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in self.archives.items()}
        audit = json.loads(self.audit.read_text()); audit["source_sha256"] = hashes
        self.audit.write_text(json.dumps(audit))
        with self.assertRaises(AudioWindowUnsafeError):
            self.run_build()

    def test_unsafe_metadata_link_raises_audio_window_unsafe_error(self):
        path = self.root / "unsafe-metadata.h5"
        with h5py.File(path, "w") as h5:
            h5["analysis"] = h5py.ExternalLink("external.h5", "/analysis")
        with self.assertRaises(AudioWindowUnsafeError):
            builder._validate_catalog_metadata(path, self.ids[0], {"tempo": 120, "artist_id": "artist", "song_id": "song"})

    def test_output_collision_refused(self):
        self.outputs[0].touch()
        with self.assertRaises(FileExistsError):
            self.run_build()

    def test_duplicate_candidate_member_fails(self):
        path = self.archives["A"]
        payload = make_h5(self.ids[0])
        with tarfile.open(path, "w:gz") as tar:
            for _ in range(2):
                info = tarfile.TarInfo(f"{self.ids[0][2]}/{self.ids[0][3]}/{self.ids[0][4]}/{self.ids[0]}.h5")
                info.size = len(payload); tar.addfile(info, io.BytesIO(payload))
        hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in self.archives.items()}
        audit = json.loads(self.audit.read_text()); audit["source_sha256"] = hashes
        self.audit.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            self.run_build()

    def test_publication_rollback(self):
        original = builder.os.link
        calls = 0
        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("publication failure")
            return original(source, target)
        with patch.object(builder.os, "link", side_effect=fail_second), self.assertRaisesRegex(OSError, "publication"):
            self.run_build()
        self.assertFalse(any(path.exists() for path in self.outputs))

    def test_cli_defaults_and_strict_argument_routing(self):
        with patch.object(cli, "run", return_value={"track_count": 1, "rejected_track_count": 0, "segment_count": 1}) as mocked:
            cli.main([])
        self.assertEqual(set(mocked.call_args.kwargs), {"a_tar", "b_tar", "c_tar", "d_tar", "audit_path", "catalog_path", "output_h5", "output_meta", "output_rejections"})
        self.assertEqual(mocked.call_args.kwargs["output_h5"].name, "abcd_audio_30s_26ch.h5")
        overrides = [self.root / f"override-{name}" for name in range(9)]
        argv = []
        for option, path in zip(("--a-tar", "--b-tar", "--c-tar", "--d-tar", "--audit", "--catalog", "--output-h5", "--output-meta", "--output-rejections"), overrides):
            argv.extend((option, str(path)))
        with patch.object(cli, "run", return_value={"track_count": 1, "rejected_track_count": 0, "segment_count": 1}) as mocked:
            cli.main(argv)
        self.assertEqual(list(mocked.call_args.args), [])
        self.assertEqual(list(mocked.call_args.kwargs.values()), overrides)

    def test_three_window_cache_schema_and_row_provenance(self):
        path = self.archives["D"]
        with tarfile.open(path, "w:gz") as tar:
            payload = make_h5(self.ids[3])
            info = tarfile.TarInfo(f"{self.ids[3][2]}/{self.ids[3][3]}/{self.ids[3][4]}/{self.ids[3]}.h5")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        audit = json.loads(self.audit.read_text())
        audit["source_sha256"]["D"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.audit.write_text(json.dumps(audit), encoding="utf-8")
        with patch.object(builder, "EXPECTED_TRACK_COUNT", len(self.ids)):
            result = builder.run_three_windows(
                *(self.archives[s] for s in "ABCD"), self.audit, self.catalog,
                self.root / "windows.h5", self.root / "windows-meta.json", self.root / "windows-reject.json",
            )
        self.assertEqual(result["track_count"], 4)
        with h5py.File(self.root / "windows.h5", "r") as h5:
            self.assertEqual(set(h5.keys()), {"features", "track_ids", "window_offsets", "segment_indices", "segment_starts", "window_starts", "window_valid", "song_durations"})
            self.assertEqual(h5["features"].shape, (8, 26))
            self.assertEqual(h5["features"].dtype, np.dtype("float32"))
            self.assertEqual([v.decode() for v in h5["track_ids"][:]], self.ids)
            np.testing.assert_array_equal(h5["window_offsets"][:], [[0, 2, 2, 2], [2, 4, 4, 4], [4, 6, 6, 6], [6, 8, 8, 8]])
            np.testing.assert_array_equal(h5["window_valid"][:], [[True, False, False]] * 4)
            np.testing.assert_array_equal(h5["segment_indices"][:], [0, 1] * 4)
            np.testing.assert_array_equal(h5["segment_starts"][:], [0., 1.] * 4)
            np.testing.assert_array_equal(h5["window_starts"][:], [[0., 5., 10.]] * 4)
            np.testing.assert_array_equal(h5["song_durations"][:], [40.] * 4)
        meta = json.loads((self.root / "windows-meta.json").read_text())
        self.assertEqual(meta["source_sha256"], json.loads(self.audit.read_text())["source_sha256"])
        self.assertEqual(meta["catalog_sha256"], hashlib.sha256(self.catalog.read_bytes()).hexdigest())

    def test_three_window_zero_valid_track_aborts_without_publication(self):
        path = self.archives["D"]
        with tarfile.open(path, "w:gz") as tar:
            payload = make_h5(self.ids[3], zero_segments=True)
            info = tarfile.TarInfo(f"{self.ids[3][2]}/{self.ids[3][3]}/{self.ids[3][4]}/{self.ids[3]}.h5")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        audit = json.loads(self.audit.read_text())
        audit["source_sha256"]["D"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.audit.write_text(json.dumps(audit), encoding="utf-8")
        outputs = [self.root / name for name in ("zero.h5", "zero-meta.json", "zero-reject.json")]
        with patch.object(builder, "EXPECTED_TRACK_COUNT", len(self.ids)):
            with self.assertRaisesRegex(ValueError, "zero valid windows"):
                builder.run_three_windows(*(self.archives[s] for s in "ABCD"), self.audit, self.catalog, *outputs)
        self.assertFalse(any(path.exists() for path in outputs))

    def test_three_window_output_collision_refused(self):
        outputs = [self.root / name for name in ("collision.h5", "collision-meta.json", "collision-reject.json")]
        outputs[0].touch()
        with patch.object(builder, "EXPECTED_TRACK_COUNT", len(self.ids)), self.assertRaises(FileExistsError):
            builder.run_three_windows(*(self.archives[s] for s in "ABCD"), self.audit, self.catalog, *outputs)

    def test_three_window_creates_missing_output_parent(self):
        parent = self.root / "nested" / "d4-output"
        outputs = [parent / name for name in ("cache.h5", "meta.json", "rejections.json")]
        with patch.object(builder, "EXPECTED_TRACK_COUNT", len(self.ids)):
            result = builder.run_three_windows(*(self.archives[s] for s in "ABCD"), self.audit, self.catalog, *outputs)
        self.assertEqual(result["track_count"], len(self.ids))
        self.assertTrue(all(path.is_file() for path in outputs))

    def test_three_window_short_tracks_duplicate_all_views(self):
        for shard, tid in zip("ABCD", self.ids):
            with tarfile.open(self.archives[shard], "w:gz") as tar:
                payload = make_h5(tid, duration=15.0)
                info = tarfile.TarInfo(f"{tid[2]}/{tid[3]}/{tid[4]}/{tid}.h5")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        audit = json.loads(self.audit.read_text())
        audit["source_sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in self.archives.items()}
        self.audit.write_text(json.dumps(audit), encoding="utf-8")
        with patch.object(builder, "EXPECTED_TRACK_COUNT", len(self.ids)):
            builder.run_three_windows(*(self.archives[s] for s in "ABCD"), self.audit, self.catalog,
                                      self.root / "short.h5", self.root / "short-meta.json", self.root / "short-reject.json")
        with h5py.File(self.root / "short.h5", "r") as h5:
            np.testing.assert_array_equal(h5["window_starts"][:], np.zeros((4, 3)))
            np.testing.assert_array_equal(h5["window_valid"][:], np.ones((4, 3), dtype=bool))
            np.testing.assert_array_equal(h5["window_offsets"][:], [[0, 2, 4, 6], [6, 8, 10, 12], [12, 14, 16, 18], [18, 20, 22, 24]])

    def test_three_window_archive_and_catalog_hash_checks(self):
        with patch.object(builder, "EXPECTED_TRACK_COUNT", len(self.ids)):
            self.archives["A"].write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                builder.run_three_windows(*(self.archives[s] for s in "ABCD"), self.audit, self.catalog,
                                          self.root / "hash.h5", self.root / "hash-meta.json", self.root / "hash-reject.json")
        with tarfile.open(self.archives["A"], "w:gz") as tar:
            payload = make_h5(self.ids[0])
            info = tarfile.TarInfo(f"{self.ids[0][2]}/{self.ids[0][3]}/{self.ids[0][4]}/{self.ids[0]}.h5")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        audit = json.loads(self.audit.read_text())
        audit["source_sha256"]["A"] = hashlib.sha256(self.archives["A"].read_bytes()).hexdigest()
        self.audit.write_text(json.dumps(audit), encoding="utf-8")
        # A catalog mutation during scanning must abort before the staged cache is published.
        with patch.object(builder, "EXPECTED_TRACK_COUNT", len(self.ids)):
            scan = builder._scan_three_window_archives
            def mutate(paths, records):
                result = scan(paths, records)
                self.catalog.write_text(self.catalog.read_text() + "\n", encoding="utf-8")
                return result
            with patch.object(builder, "_scan_three_window_archives", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "catalog.*changed"):
                    builder.run_three_windows(*(self.archives[s] for s in "ABCD"), self.audit, self.catalog,
                                              self.root / "catalog-hash.h5", self.root / "catalog-hash-meta.json", self.root / "catalog-hash-reject.json")
        self.assertFalse((self.root / "catalog-hash.h5").exists())

    def test_three_window_requires_full_expected_catalog(self):
        with self.assertRaisesRegex(ValueError, "exactly 13822 expected catalog IDs"):
            builder.run_three_windows(*(self.archives[s] for s in "ABCD"), self.audit, self.catalog,
                                      self.root / "cohort.h5", self.root / "cohort-meta.json", self.root / "cohort-reject.json")
        self.assertFalse((self.root / "cohort.h5").exists())

    def test_three_window_refuses_opening_cache_namespace_even_if_absent(self):
        with self.assertRaisesRegex(ValueError, "must not target the existing opening-window cache namespace"):
            builder.run_three_windows(*(self.archives[s] for s in "ABCD"), self.audit, self.catalog,
                                      builder.PILOT_DIR / "abcd_audio_30s_26ch.h5",
                                      self.root / "protected-meta.json", self.root / "protected-reject.json")

    def test_three_window_cli_defaults(self):
        from scripts import preprocess_neurotune_v1_d4_audio_windows as d4_cli
        with patch.object(d4_cli, "run_three_windows", return_value={"track_count": 1, "rejected_track_count": 0, "segment_count": 1}) as mocked:
            d4_cli.main([])
        self.assertEqual(mocked.call_args.kwargs["output_h5"].name, "abcd_audio_start_mid_end_30s_26ch.h5")
        self.assertEqual(mocked.call_args.kwargs["output_meta"].name, "abcd_audio_start_mid_end_30s_26ch_meta.json")
        self.assertEqual(mocked.call_args.kwargs["output_rejections"].name, "abcd_audio_start_mid_end_30s_26ch_rejections.json")
        self.assertEqual(mocked.call_args.kwargs["a_tar"].name, "A.tar.gz")
        self.assertEqual(mocked.call_args.kwargs["audit_path"].name, "abcd_catalog_audit.json")
        self.assertEqual(mocked.call_args.kwargs["catalog_path"].name, "abcd_catalog_tracks.csv")


if __name__ == "__main__":
    unittest.main()

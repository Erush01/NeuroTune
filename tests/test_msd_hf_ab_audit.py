import csv
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import scripts.audit_msd_hf_ab as audit_module
from scripts.audit_msd_hf_ab import run


def track_id(number):
    return f"TR{number:016d}"


def add_track(tar, identifier, artist, song, tempo=120.0, archive_id=None):
    import io
    import tarfile

    archive_id = archive_id or identifier
    data = io.BytesIO()
    with h5py.File(data, "w") as h5:
        analysis = h5.create_group("analysis")
        songs = analysis.create_dataset("songs", (1,), dtype=np.dtype([
            ("track_id", "S18"), ("tempo", "f8")
        ]))
        songs[0] = (identifier.encode(), tempo)
        metadata = h5.create_group("metadata")
        songs_meta = metadata.create_dataset("songs", (1,), dtype=np.dtype([
            ("artist_id", "S18"), ("song_id", "S18"),
            ("artist_name", "S40"), ("title", "S40")
        ]))
        songs_meta[0] = (artist.encode(), song.encode(), b"Artist", b"Title")
    payload = data.getvalue()
    member = f"{archive_id[2]}/{archive_id[3]}/{archive_id[4]}/{archive_id}.h5"
    info = __import__("tarfile").TarInfo(member)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a = self.root / "A.tar.gz"
        self.b = self.root / "B.tar.gz"
        self.lyrics = self.root / "mxm"
        self.lyrics.mkdir()
        self.genres = self.root / "genres.csv"
        self.duplicates = self.root / "duplicates.txt"
        self.train = self.root / "lastfm_train.zip"
        self.test = self.root / "lastfm_test.zip"
        self.output = self.root / "out"
        self.ids = [track_id(i) for i in range(1, 9)]
        self._make_sources()

    def tearDown(self):
        self.temp.cleanup()

    def _make_sources(self):
        import tarfile

        specs = [
            (self.ids[0], "AR1", "SO1", 120),
            (self.ids[1], "AR1", "SO1", 121),  # artist/song duplicate; first wins
            (self.ids[2], "AR2", "SO2", 122),
            (self.ids[3], "AR3", "SO3", 123),
            (self.ids[4], "AR4", "SO4", 124),
            (self.ids[5], "AR5", "SO5", 125),
            (self.ids[6], "AR6", "SO6", 0),
        ]
        with tarfile.open(self.a, "w:gz") as archive:
            for spec in specs[:4]:
                add_track(archive, *spec)
        with tarfile.open(self.b, "w:gz") as archive:
            for spec in specs[4:]:
                add_track(archive, *spec)
        self.lyrics.joinpath("mxm.txt").write_text(
            "# header\n" + "\n".join(f"{identifier},1,1:1" for identifier in self.ids),
            encoding="utf-8",
        )
        self.genres.write_text("track_id,genre\n" + "\n".join(
            f"{identifier},rock" for identifier in self.ids
        ) + "\n", encoding="utf-8")
        self.duplicates.write_text(f"% group\n{self.ids[2]}\n{self.ids[3]}\n", encoding="utf-8")
        train_rows = {
            self.ids[0]: [self.ids[0], self.ids[2], self.ids[2], self.ids[4]],
            self.ids[2]: [self.ids[0]],
        }
        test_rows = {self.ids[0]: [self.ids[2], self.ids[5]]}
        self._write_zip(self.train, "train", train_rows)
        self._write_zip(self.test, "test", test_rows)

    def _write_zip(self, path, split, rows):
        with zipfile.ZipFile(path, "w") as archive:
            for identifier, similars in rows.items():
                value = {"track_id": identifier, "similars": similars}
                member = f"lastfm_{split}/{identifier[2]}/{identifier[3]}/{identifier[4]}/{identifier}.json"
                archive.writestr(member, json.dumps(value))

    def invoke(self):
        return run(self.a, self.b, self.lyrics, self.genres, self.duplicates,
                   self.train, self.test, self.output)

    def test_duplicate_intersections_edges_and_order_are_deterministic(self):
        audit = self.invoke()
        with (self.output / "ab_catalog_tracks.csv").open() as stream:
            tracks = list(csv.DictReader(stream))
        ids = [row["track_id"] for row in tracks]
        self.assertEqual(ids, sorted([self.ids[0], self.ids[2], self.ids[4], self.ids[5]]))
        self.assertEqual(audit["counts"]["eligible_tracks"], 4)
        with (self.output / "ab_lastfm_train_edges.csv").open() as stream:
            train = list(csv.DictReader(stream))
        with (self.output / "ab_lastfm_test_edges.csv").open() as stream:
            test = list(csv.DictReader(stream))
        self.assertEqual([(r["track_id_a"], r["track_id_b"]) for r in train], [(self.ids[0], self.ids[4])])
        self.assertEqual([(r["track_id_a"], r["track_id_b"]) for r in test],
                         [(self.ids[0], self.ids[2]), (self.ids[0], self.ids[5])])
        self.assertEqual(audit["counts"]["train_test_overlap_removed"], 1)

    def test_rejects_unsafe_path_mismatch_missing_sources_and_existing_outputs(self):
        self.a.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_rejects_tar_traversal_before_creating_outputs(self):
        import io
        import tarfile

        with tarfile.open(self.a, "w:gz") as archive:
            info = tarfile.TarInfo("../../escape.h5")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_rejects_hdf5_track_id_path_mismatch(self):
        import tarfile

        with tarfile.open(self.a, "w:gz") as archive:
            add_track(archive, self.ids[0], "AR1", "SO1", archive_id=self.ids[1])
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_input_member_order_does_not_change_catalog_or_edges(self):
        self.invoke()
        catalog = (self.output / "ab_catalog_tracks.csv").read_bytes()
        train = (self.output / "ab_lastfm_train_edges.csv").read_bytes()
        test = (self.output / "ab_lastfm_test_edges.csv").read_bytes()
        import tarfile
        specs = [
            (self.ids[0], "AR1", "SO1", 120),
            (self.ids[1], "AR1", "SO1", 121),
            (self.ids[2], "AR2", "SO2", 122),
            (self.ids[3], "AR3", "SO3", 123),
        ]
        with tarfile.open(self.a, "w:gz") as archive:
            for spec in reversed(specs):
                add_track(archive, *spec)
        second_output = self.root / "other-out"
        run(self.a, self.b, self.lyrics, self.genres, self.duplicates,
            self.train, self.test, second_output)
        self.assertEqual(catalog, (second_output / "ab_catalog_tracks.csv").read_bytes())
        self.assertEqual(train, (second_output / "ab_lastfm_train_edges.csv").read_bytes())
        self.assertEqual(test, (second_output / "ab_lastfm_test_edges.csv").read_bytes())

    def test_lastfm_json_without_track_id_uses_archive_path_source_id(self):
        with zipfile.ZipFile(self.train) as source:
            members = {name: source.read(name) for name in source.namelist()}
        source_path = f"lastfm_train/{self.ids[0][2]}/{self.ids[0][3]}/{self.ids[0][4]}/{self.ids[0]}.json"
        record = json.loads(members[source_path])
        del record["track_id"]
        members[source_path] = json.dumps(record).encode("utf-8")
        with zipfile.ZipFile(self.train, "w") as target:
            for name, payload in members.items():
                target.writestr(name, payload)

        self.invoke()
        with (self.output / "ab_lastfm_train_edges.csv").open() as stream:
            train = list(csv.DictReader(stream))
        self.assertEqual([(row["track_id_a"], row["track_id_b"]) for row in train],
                         [(self.ids[0], self.ids[4])])

    def test_rejects_oversized_hdf5_member_without_outputs(self):
        with patch.object(audit_module, "MAX_HDF5_MEMBER_BYTES", 1, create=True):
            with self.assertRaisesRegex(ValueError, "HDF5.*size"):
                self.invoke()
        self.assertFalse(self.output.exists())

    def test_rejects_oversized_lastfm_json_without_outputs(self):
        with patch.object(audit_module, "MAX_LASTFM_JSON_BYTES", 1, create=True):
            with self.assertRaisesRegex(ValueError, "Last.fm JSON.*size"):
                self.invoke()
        self.assertFalse(self.output.exists())

    def test_hdf5_metadata_preflights_links_and_song_shape(self):
        import io

        def payload(link=None, shape=(1,)):
            data = io.BytesIO()
            with h5py.File(data, "w") as h5:
                analysis = h5.create_group("analysis")
                rows = analysis.create_dataset("songs", shape=shape, dtype=[("track_id", "S18"), ("tempo", "f8")])
                if shape == (1,):
                    rows[0] = (self.ids[0].encode(), 120.0)
                metadata = h5.create_group("metadata")
                meta = metadata.create_dataset("songs", (1,), dtype=[
                    ("artist_id", "S18"), ("song_id", "S18"), ("artist_name", "S40"), ("title", "S40")
                ])
                meta[0] = (b"AR1", b"SO1", b"Artist", b"Title")
                if link == "soft":
                    del metadata["songs"]
                    metadata["songs"] = h5py.SoftLink("/analysis/songs")
                elif link == "external":
                    del metadata["songs"]
                    metadata["songs"] = h5py.ExternalLink("elsewhere.h5", "/songs")
            return data.getvalue()

        self.assertEqual(audit_module._hdf5_metadata(payload(), self.ids[0])["track_id"], self.ids[0])
        for link in ("soft", "external"):
            with self.subTest(link=link), self.assertRaisesRegex(ValueError, "hard link"):
                audit_module._hdf5_metadata(payload(link=link), self.ids[0])
        with self.assertRaisesRegex(ValueError, "single-row"):
            audit_module._hdf5_metadata(payload(shape=(2,)), self.ids[0])

    def test_hdf5_metadata_rejects_vlen_fields_and_accepts_5320_byte_fixed_record(self):
        import io

        def payload(vlen=False):
            data = io.BytesIO()
            with h5py.File(data, "w") as h5:
                analysis = h5.create_group("analysis")
                songs = analysis.create_dataset("songs", (1,), dtype=[("track_id", "S18"), ("tempo", "f8")])
                songs[0] = (self.ids[0].encode(), 120.0)
                metadata = h5.create_group("metadata")
                if vlen:
                    dtype = np.dtype([
                        ("artist_id", h5py.string_dtype("utf-8")), ("song_id", "S18"),
                        ("artist_name", "S40"), ("title", "S40"),
                    ])
                else:
                    dtype = np.dtype([
                        ("artist_id", "S18"), ("song_id", "S18"),
                        ("artist_name", "S1024"), ("title", "S1024"), ("reserved", "S3236"),
                    ])
                if not vlen:
                    self.assertEqual(dtype.itemsize, 5320)
                meta = metadata.create_dataset("songs", (1,), dtype=dtype)
                meta[0] = (("AR1", "SO1", "Artist", "Title") if vlen else
                           ("AR1", "SO1", "Artist", "Title", ""))
            return data.getvalue()

        with self.assertRaisesRegex(ValueError, "variable-length"):
            audit_module._hdf5_metadata(payload(vlen=True), self.ids[0])
        metadata = audit_module._hdf5_metadata(payload(), self.ids[0])
        self.assertEqual(metadata["artist_name"], "Artist")

    def test_keyboard_interrupt_during_publication_rolls_back(self):
        real_link = audit_module.os.link
        calls = 0

        def link_then_interrupt(source, target):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_link(source, target)
            raise KeyboardInterrupt("publication interrupted")

        with patch.object(audit_module.os, "link", side_effect=link_then_interrupt):
            with self.assertRaisesRegex(KeyboardInterrupt, "publication interrupted"):
                self.invoke()
        self.assertFalse((self.output / "ab_catalog_audit.json").exists())
        self.assertFalse(self.output.exists())

    def test_cli_defaults_use_root_lastfm_archives(self):
        with patch.object(sys, "argv", ["audit_msd_hf_ab"]), patch.object(
            audit_module, "run", return_value={"counts": {}}
        ) as mock_run:
            audit_module.main()
        self.assertEqual(mock_run.call_args.args[5], audit_module.REPO_ROOT / "lastfm_train.zip")
        self.assertEqual(mock_run.call_args.args[6], audit_module.REPO_ROOT / "lastfm_test.zip")

    def test_existing_target_is_preserved(self):
        self.output.mkdir()
        target = self.output / "ab_catalog_audit.json"
        target.write_text("keep", encoding="utf-8")
        with self.assertRaises((FileExistsError, ValueError)):
            self.invoke()
        self.assertEqual(target.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()

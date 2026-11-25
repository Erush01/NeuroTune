import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.extract_msd import extract_dataset, extract_track, write_tracks_csv

H5PY_AVAILABLE = True
try:
    import h5py
except ImportError:  # pragma: no cover
    H5PY_AVAILABLE = False


@unittest.skipUnless(H5PY_AVAILABLE, "h5py is required for MSD HDF5 extraction")
class TestExtractMsd(unittest.TestCase):
    def _write_h5(self, path, track_id, tempo, artist_id, song_id, artist_name, title):
        import numpy
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            meta = f.create_dataset("metadata/songs", (1,), dtype=numpy.dtype([
                ("artist_id", h5py.string_dtype("utf-8")),
                ("song_id", h5py.string_dtype("utf-8")),
                ("artist_name", h5py.string_dtype("utf-8")),
                ("title", h5py.string_dtype("utf-8")),
            ]))
            meta[0] = (artist_id, song_id, artist_name, title)
            analysis = f.create_dataset("analysis/songs", (1,), dtype=numpy.dtype([
                ("track_id", h5py.string_dtype("utf-8")),
                ("tempo", "f8"),
            ]))
            analysis[0] = (track_id, tempo)

    def test_extract_track_reads_h5_fields(self):
        with TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "A" / "A" / "A" / "TRAAAAW128F429D538.h5"
            self._write_h5(
                h5_path, "TRAAAAW128F429D538", 92.198,
                "AR000118703C9F81F3", "SOBAYQD12A8C131F37",
                "Massive Attack", "Dissolved Girl",
            )
            track = extract_track(h5_path)
            self.assertEqual(track["track_id"], "TRAAAAW128F429D538")
            self.assertEqual(track["artist_id"], "AR000118703C9F81F3")
            self.assertEqual(track["song_id"], "SOBAYQD12A8C131F37")
            self.assertEqual(track["artist_name"], "Massive Attack")
            self.assertEqual(track["title"], "Dissolved Girl")
            self.assertAlmostEqual(track["tempo"], 92.198)

    def test_extract_track_missing_tempo_is_none(self):
        with TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "A" / "A" / "A" / "TRBBBBB128F429D538.h5"
            self._write_h5(
                h5_path, "TRBBBBB128F429D538", 0.0,
                "AR1", "SO1", "X", "Y",
            )
            track = extract_track(h5_path)
            self.assertIsNone(track["tempo"])

    def test_extract_dataset_walks_and_deduplicates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "subset"
            first = root / "A" / "A" / "A" / "TRAAAAW128F429D538.h5"
            self._write_h5(first, "TRAAAAW128F429D538", 92.0, "AR1", "SO1", "X", "Y")
            second = root / "B" / "B" / "B" / "TRBBBBB128F429D539.h5"
            self._write_h5(second, "TRBBBBB128F429D539", 120.0, "AR2", "SO2", "P", "Q")
            tracks = extract_dataset(root)
            self.assertEqual(len(tracks), 2)
            self.assertEqual({t["track_id"] for t in tracks},
                             {"TRAAAAW128F429D538", "TRBBBBB128F429D539"})

    def test_write_tracks_csv_header_and_order(self):
        with TemporaryDirectory() as tmp:
            tracks = [
                {"track_id": "TR2", "tempo": None, "artist_id": "A2", "song_id": "S2",
                 "artist_name": "N2", "title": "T2"},
                {"track_id": "TR1", "tempo": 90.5, "artist_id": "A1", "song_id": "S1",
                 "artist_name": "N1", "title": "T1"},
            ]
            out = Path(tmp) / "msd_tracks.csv"
            write_tracks_csv(tracks, out)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "track_id,tempo,artist_id,song_id,artist_name,title")
            self.assertIn("TR1,90.5,A1,S1,N1,T1", lines)
            self.assertIn("TR2,,A2,S2,N2,T2", lines)


if __name__ == "__main__":
    unittest.main()

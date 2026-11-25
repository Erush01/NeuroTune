import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.extract_tagtraum import convert_cls


class TestExtractTagtraum(unittest.TestCase):
    def _write_cls(self, path, lines):
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_skips_header_and_takes_majority_genre(self):
        with TemporaryDirectory() as tmp:
            cls_path = Path(tmp) / "msd_tagtraum_cd2c.cls"
            out_csv = Path(tmp) / "tagtraum_genres.csv"
            self._write_cls(cls_path, [
                "# comments ...",
                "TRAAAAK128F9318786\tRock\tPop",
                "TRBBBBB128F931879X\tJazz",
                "#""",
            ])
            convert_cls(cls_path, out_csv)
            rows = out_csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "track_id,genre")
            self.assertIn("TRAAAAK128F9318786,Rock", rows)
            self.assertIn("TRBBBBB128F931879X,Jazz", rows)

    def test_conflicting_repeat_track_ids_write_dedup_first(self):
        with TemporaryDirectory() as tmp:
            cls_path = Path(tmp) / "f.cls"
            out_csv = Path(tmp) / "g.csv"
            self._write_cls(cls_path, [
                "TRAAAAK128F9318786\tRock",
                "TRAAAAK128F9318786\tJazz",
                "TRBBBBB128F931879X\tJazz",
            ])
            with self.assertRaises(ValueError):
                convert_cls(cls_path, out_csv)


if __name__ == "__main__":
    unittest.main()

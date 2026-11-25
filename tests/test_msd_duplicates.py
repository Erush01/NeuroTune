import tempfile
import unittest
from pathlib import Path

from src.data.parse_msd_duplicates import parse_msd_duplicates


class TestParseMsdDuplicates(unittest.TestCase):
    def test_parses_track_id_groups_and_ignores_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "msd_duplicates.txt"
            path.write_text(
                "# duplicate list\n%1 artist - title\nTRABC\nTRDEF\n"
                "%2 other - title\nTRGHI\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_msd_duplicates(path),
                [["TRABC", "TRDEF"], ["TRGHI"]],
            )


if __name__ == "__main__":
    unittest.main()

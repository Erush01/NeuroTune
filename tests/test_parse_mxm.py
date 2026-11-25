import tempfile
import unittest
from pathlib import Path

from src.data.parse_mxm import parse_mxm


class ParseMxmTests(unittest.TestCase):
    def test_exact_filter_normalizes_ids_merges_duplicates_and_keeps_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mxm_train.txt").write_text("%word\ntr0000000000000001,101,1:2\nTR0000000000000002,102\n")
            (root / "mxm_test.txt").write_text("%word\nTR0000000000000001,201,1:3\nTR0000000000000003,202,1:9\n")
            nested = root / "nested"
            nested.mkdir()
            (nested / "ignored.txt").write_text("%word\nTR0000000000000004,204,1:4\n")
            result = parse_mxm(root, track_ids={"tr0000000000000001", "TR0000000000000002"})
            unfiltered = parse_mxm(root)
        self.assertEqual(result, {"TR0000000000000001": {"1": 5}, "TR0000000000000002": {}})
        self.assertNotIn("TR0000000000000003", result)
        self.assertNotIn("TR0000000000000004", unfiltered)

    def test_individual_text_and_long_form_are_filtered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = root / "one.txt"
            text.write_text("%word\nTR0000000000000001,101,1:2\nTR0000000000000002,102\n")
            extract = root / "extract.csv"
            extract.write_text("track_id,word_id,count\ntr0000000000000001,1,2\nTR0000000000000002,1,0\n")
            requested = {"TR0000000000000001"}
            self.assertEqual(parse_mxm(text, track_ids=requested), {"TR0000000000000001": {"1": 2}})
            self.assertEqual(parse_mxm(extract, track_ids=requested), {"TR0000000000000001": {"1": 2}})

    def test_malformed_non_candidate_official_row_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            text = Path(temp) / "bad.txt"
            text.write_text("%word\nTR0000000000000001,101,1:2\nTR0000000000000002,not-an-id,1:5\n")
            with self.assertRaisesRegex(ValueError, "MXM|musiXmatch"):
                parse_mxm(text, track_ids={"TR0000000000000001"})


if __name__ == "__main__":
    unittest.main()

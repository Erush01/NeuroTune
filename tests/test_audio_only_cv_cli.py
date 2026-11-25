import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import scripts.audit_msd_audio_cv as cli


class AudioOnlyCVCliTests(unittest.TestCase):
    def _sources(self, root):
        tracks = root / "tracks.csv"
        with tracks.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "track_id", "tempo", "artist_id", "song_id", "artist_name", "title",
            ])
            writer.writeheader()
            for index in range(9):
                writer.writerow({
                    "track_id": f"TR{index:06d}", "tempo": 120 + index,
                    "artist_id": f"AR{index:06d}", "song_id": f"SO{index:06d}",
                    "artist_name": f"Artist {index}", "title": f"Song {index}",
                })
        lastfm = root / "lastfm"
        lastfm.mkdir()
        for index in range(9):
            (lastfm / f"TR{index:06d}.json").write_text(json.dumps({
                "track_id": f"TR{index:06d}",
                "similars": [[f"TR{(index + 1) % 9:06d}", 1.0]],
            }))
        duplicates = root / "duplicates.txt"
        duplicates.write_text("")
        return tracks, lastfm, duplicates

    def test_overrides_hashes_and_deterministic_fold_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks, lastfm, duplicates = self._sources(root)
            output = root / "audit.json"
            args = ["--tracks", str(tracks), "--lastfm", str(lastfm), "--duplicates", str(duplicates),
                    "--output", str(output), "--seed", "17", "--fold-count", "3"]
            self.assertEqual(cli.main(args), 0)
            first = json.loads(output.read_text())
            self.assertEqual(first["seed"], 17)
            self.assertEqual(first["fold_count"], 3)
            self.assertEqual(first["status"], "ready")
            self.assertEqual(first["catalog_counts"]["eligible_tracks"], 9)
            self.assertEqual(sum(fold["track_count"] for fold in first["folds"].values()), 9)
            self.assertEqual(sum(fold["positive_edge_count"] for fold in first["folds"].values())
                             + first["cross_fold_edges"], 9)
            self.assertEqual(first["source_sha256"]["tracks"], hashlib.sha256(tracks.read_bytes()).hexdigest())
            self.assertEqual(len(first["source_sha256"]["lastfm"]), 64)
            self.assertEqual(cli.main(args), 0)
            self.assertEqual(json.loads(output.read_text()), first)

    def test_missing_source_writes_blocked_audit_with_unknown_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks, lastfm, duplicates = self._sources(root)
            missing = root / "missing.txt"
            output = root / "blocked.json"
            self.assertEqual(cli.main(["--tracks", str(tracks), "--lastfm", str(lastfm),
                                       "--duplicates", str(missing), "--output", str(output)]), 0)
            audit = json.loads(output.read_text())
            self.assertEqual(audit["status"], "blocked")
            self.assertIsNone(audit["source_counts"])
            self.assertIsNone(audit["catalog_counts"])
            self.assertIsNone(audit["folds"])
            self.assertIn("duplicates", audit["missing_sources"])


if __name__ == "__main__":
    unittest.main()

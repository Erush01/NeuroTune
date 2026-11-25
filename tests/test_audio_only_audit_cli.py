import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_msd_audio.py"


class AudioOnlyAuditCliTests(unittest.TestCase):
    def _write_sources(self, root):
        tracks = root / "custom_tracks.csv"
        with tracks.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["track_id", "artist_id", "song_id"])
            writer.writeheader()
            for i in range(10):
                writer.writerow({"track_id": f"TR{i:018d}", "artist_id": f"AR{i}", "song_id": f"SO{i}"})

        lastfm = root / "custom_lastfm"
        lastfm.mkdir()
        # Complete undirected coverage ensures the test split has usable edges.
        for i in range(10):
            record = {"track_id": f"TR{i:018d}", "similars": [[f"TR{j:018d}", 1.0] for j in range(10) if j != i]}
            (lastfm / f"TR{i:018d}.json").write_text(json.dumps(record), encoding="utf-8")

        duplicates = root / "custom_duplicates.txt"
        duplicates.write_text("% group\nTR000000000000000000\nTR000000000000000001\n", encoding="utf-8")
        return tracks, lastfm, duplicates

    def test_overrides_generate_audit_with_hashes_and_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tracks, lastfm, duplicates = self._write_sources(root)
            output = root / "custom-output.json"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--tracks", str(tracks), "--lastfm", str(lastfm),
                 "--duplicates", str(duplicates), "--output", str(output), "--seed", "17"],
                cwd=REPO, capture_output=True, text=True, check=True,
            )

            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["seed"], 17)
            self.assertEqual(audit["source_counts"]["msd"], 10)
            self.assertEqual(audit["eligible_track_count"], 9)
            splits = audit["splits"]["artist_disjoint"]
            self.assertEqual(sum(splits["track_counts"].values()), 9)
            self.assertGreater(splits["usable_positive_edges"]["test"], 0)
            self.assertGreater(audit["positive_edges"]["unique_edges_within_catalog"], 0)
            self.assertEqual(set(audit["source_sha256"]), {"msd", "lastfm", "msd_duplicates"})
            self.assertEqual(audit["source_sha256"]["msd"], hashlib.sha256(tracks.read_bytes()).hexdigest())
            self.assertEqual(audit["source_sha256"]["msd_duplicates"], hashlib.sha256(duplicates.read_bytes()).hexdigest())
            self.assertEqual(len(audit["source_sha256"]["lastfm"]), 64)
            self.assertIn(str(output), result.stdout)
            self.assertIn(audit["status"], result.stdout)

    def test_missing_source_writes_blocked_audit_with_unknown_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "missing.json"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--tracks", str(root / "no-tracks.csv"),
                 "--lastfm", str(root / "no-lastfm"), "--duplicates", str(root / "no-dupes.txt"),
                 "--output", str(output)],
                cwd=REPO, capture_output=True, text=True, check=True,
            )
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "blocked_missing_sources")
            self.assertIsNone(audit["source_counts"]["msd"])
            self.assertIsNone(audit["source_counts"]["lastfm_edges_raw"])
            self.assertEqual(len(audit["missing_sources"]), 3)


if __name__ == "__main__":
    unittest.main()

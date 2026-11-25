import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BaselineCliTests(unittest.TestCase):
    def test_cli_writes_metrics_split_fit_metadata_and_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "lastfm_similar").mkdir()
            audit_path = root / "data_audit.json"
            output_path = root / "baseline.json"

            train_ids = ["TRTRAINA", "TRTRAINB"]
            test_ids = ["TRTESTA", "TRTESTB", "TRTESTC"]
            with (data_dir / "msd_tracks.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["track_id", "tempo", "artist_id", "song_id", "artist_name", "title"])
                writer.writeheader()
                for track_id, tempo in [("TRTRAINA", 100), ("TRTRAINB", 120), ("TRTESTA", 110), ("TRTESTB", 130), ("TRTESTC", 90)]:
                    writer.writerow({"track_id": track_id, "tempo": tempo})
            with (data_dir / "mxm_counts.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["track_id", "word_id", "count"])
                writer.writeheader()
                writer.writerows([
                    {"track_id": "TRTRAINA", "word_id": 1, "count": 2},
                    {"track_id": "TRTRAINB", "word_id": 2, "count": 1},
                    {"track_id": "TRTESTA", "word_id": 1, "count": 1},
                    {"track_id": "TRTESTB", "word_id": 2, "count": 1},
                ])
            with (data_dir / "tagtraum_genres.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["track_id", "genre"])
                writer.writeheader()
                writer.writerows([
                    {"track_id": "TRTRAINA", "genre": "Rock"},
                    {"track_id": "TRTRAINB", "genre": "Pop"},
                    {"track_id": "TRTESTA", "genre": "Rock"},
                    {"track_id": "TRTESTB", "genre": "Pop"},
                    {"track_id": "TRTESTC", "genre": "Unseen"},
                ])
            (data_dir / "lastfm_similar" / "TEST_A.json").write_text(
                json.dumps({"track_id": "TRTESTA", "similars": [["TRTESTB", 1.0], ["TRTRAINA", 0.5]]})
            )
            audit_path.write_text(json.dumps({
                "seed": 684,
                "splits": {"artist_disjoint": {"track_ids": {"train": train_ids, "validation": [], "test": test_ids}}},
            }))

            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_baseline", "--data-dir", str(data_dir), "--audit", str(audit_path), "--output", str(output_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output_path.read_text())
            self.assertEqual(report["split_counts"], {"train": 2, "validation": 0, "test": 3})
            self.assertEqual(report["metrics"]["query_count"], 2)
            self.assertEqual(report["metrics"]["eligible_positive_edges"], 1)
            self.assertIn("Recall@10", report["metrics"])
            for metric_name in ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10"):
                self.assertEqual(report["metrics"][metric_name], 1.0)
            self.assertEqual(report["preprocessing"]["fit_train_ids"], train_ids)
            self.assertEqual(report["preprocessing"]["genre_vocabulary"], ["Pop", "Rock"])
            self.assertEqual(report["k"], 10)
            self.assertEqual(
                set(report["source_sha256"]),
                {"data_audit", "msd_tracks", "mxm_counts", "tagtraum_genres", "lastfm_similar"},
            )
            for source_hash in report["source_sha256"].values():
                self.assertRegex(source_hash, re.compile(r"^[0-9a-f]{64}$"))
            self.assertIn("exploratory", report["warning"].lower())
            self.assertIn("1 positive edge", report["warning"])
            self.assertNotIn("11 positive edges", report["warning"])

    def test_cli_rejects_overlapping_partitions_before_reading_feature_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "missing-data"
            data_dir.mkdir()
            audit_path = root / "overlapping_audit.json"
            output_path = root / "baseline.json"
            audit_path.write_text(json.dumps({
                "seed": 684,
                "splits": {"artist_disjoint": {"track_ids": {
                    "train": ["TRSHARED"],
                    "validation": [],
                    "test": ["TRSHARED"],
                }}},
            }))

            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_baseline", "--data-dir", str(data_dir), "--audit", str(audit_path), "--output", str(output_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("train", result.stderr.lower())
            self.assertIn("test", result.stderr.lower())
            self.assertIn("overlap", result.stderr.lower())
            self.assertNotIn("msd_tracks.csv", result.stderr)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()

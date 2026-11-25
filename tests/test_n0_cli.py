import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


class N0CliTests(unittest.TestCase):
    def test_duplicate_seeds_fail_before_feature_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "empty-data"
            data_dir.mkdir()
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({"seed": 684, "splits": {"artist_disjoint": {
                "track_ids": {"train": ["TRTRAIN"], "validation": ["TRVALID"], "test": ["TRTEST"]}
            }}}))
            result = subprocess.run([
                sys.executable, "-m", "scripts.run_n0",
                "--seeds", "684", "684", "--data-dir", str(data_dir),
                "--audit", str(audit_path), "--output", str(root / "report.json"),
                "--checkpoint-dir", str(root / "checkpoints"),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("seeds must be unique", result.stderr)
            self.assertNotIn("msd_tracks.csv", result.stderr)

    def test_synthetic_one_epoch_cli_report_and_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            lastfm_dir = data_dir / "lastfm_similar"
            lastfm_dir.mkdir(parents=True)
            train = [f"TRTRAIN{i}" for i in range(6)]
            validation = [f"TRVALID{i}" for i in range(4)]
            test = [f"TRTEST{i}" for i in range(3)]
            all_ids = train + validation + test

            with (data_dir / "msd_tracks.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["track_id", "tempo"])
                writer.writeheader()
                for index, track_id in enumerate(all_ids):
                    writer.writerow({"track_id": track_id, "tempo": 80 + index * 3})
            with (data_dir / "mxm_counts.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["track_id", "word_id", "count"])
                writer.writeheader()
                for index, track_id in enumerate(all_ids):
                    writer.writerow({"track_id": track_id, "word_id": index + 1, "count": 1})
            with (data_dir / "tagtraum_genres.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["track_id", "genre"])
                writer.writeheader()
                for index, track_id in enumerate(all_ids):
                    writer.writerow({"track_id": track_id, "genre": "A" if index % 2 == 0 else "B"})

            # Two internal positives per fitting partition, with eligible random
            # and same-genre non-edges for the exact 50/50 negative sample.
            edges = [(train[0], train[1]), (train[2], train[3]),
                     (validation[0], validation[1]), (validation[2], validation[3]),
                     (test[0], test[1])]
            for source, target in edges:
                path = lastfm_dir / f"{source}.json"
                record = json.loads(path.read_text()) if path.exists() else {"track_id": source, "similars": []}
                record["similars"].append([target, 1.0])
                path.write_text(json.dumps(record))
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({"seed": 684, "splits": {"artist_disjoint": {
                "track_ids": {"train": train, "validation": validation, "test": test}
            }}}))
            output = root / "n0.json"
            checkpoint_dir = root / "checkpoints"
            subprocess.run([
                sys.executable, "-m", "scripts.run_n0",
                "--epochs", "1", "--seeds", "684", "--data-dir", str(data_dir),
                "--audit", str(audit_path), "--output", str(output),
                "--checkpoint-dir", str(checkpoint_dir),
            ], check=True, cwd=ROOT)

            report = json.loads(output.read_text())
            self.assertEqual(report["config"]["epochs"], 1)
            self.assertEqual(report["config"]["seeds"], [684])
            self.assertEqual(report["config"]["embedding_dim"], 64)
            self.assertEqual(report["config"]["negative_policy"]["allocation"], {
                "same_genre": "floor(total_negatives / 2)",
                "random": "remainder",
            })
            sampled_pairs_path = root / "n0_sampled_pairs.csv"
            self.assertEqual(report["sampled_pairs_csv"], str(sampled_pairs_path))
            with sampled_pairs_path.open(newline="") as stream:
                sampled_rows = list(csv.DictReader(stream))
            self.assertEqual(set(sampled_rows[0]), {
                "seed", "partition", "epoch", "left_track_id", "right_track_id", "label"
            })
            train_rows = [row for row in sampled_rows if row["partition"] == "train"]
            validation_rows = [row for row in sampled_rows if row["partition"] == "validation"]
            self.assertEqual({row["epoch"] for row in train_rows}, {"1"})
            self.assertEqual({row["epoch"] for row in validation_rows}, {"0"})
            self.assertTrue(all(row["seed"] == "684" for row in sampled_rows))
            self.assertTrue(all(
                row["left_track_id"] in train and row["right_track_id"] in train
                for row in train_rows
            ))
            self.assertTrue(all(
                row["left_track_id"] in validation and row["right_track_id"] in validation
                for row in validation_rows
            ))
            self.assertTrue(report["runtime_versions"]["python"])
            self.assertTrue(report["runtime_versions"]["numpy"])
            self.assertTrue(report["runtime_versions"]["pytorch"])
            self.assertEqual(set(report["code_sha256"]), {
                "scripts/run_n0.py", "src/features.py", "src/pairs.py", "src/model.py",
                "src/train.py", "src/retrieve.py", "src/evaluate.py", "requirements.txt",
                "src/data/parse_lastfm.py", "src/data/parse_msd.py",
                "src/data/parse_mxm.py", "src/data/parse_tagtraum.py", "src/data/table.py",
            })
            self.assertTrue(all(
                len(value) == 64 and all(char in "0123456789abcdef" for char in value)
                for value in report["code_sha256"].values()
            ))
            self.assertEqual(report["split_counts"], {"train": 6, "validation": 4, "test": 3})
            self.assertEqual(report["test"]["unique_edge_count"], 1)
            self.assertEqual(report["test"]["query_count"], 2)
            self.assertEqual(set(report["source_sha256"]), {
                "data_audit", "msd_tracks", "mxm_counts", "tagtraum_genres", "lastfm_similar"
            })
            self.assertTrue(all(len(value) == 64 for value in report["source_sha256"].values()))
            self.assertTrue(all(
                len(value) == 64 and all(char in "0123456789abcdef" for char in value)
                for value in report["source_sha256"].values()
            ))
            self.assertEqual(
                report["source_sha256"]["data_audit"],
                hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(report["training_input_ids"], train + validation)
            self.assertFalse(set(report["training_input_ids"]) & set(test))
            metrics = report["per_seed"]["684"]["metrics"]
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.assertTrue(math.isfinite(value), metric)
            curves = report["per_seed"]["684"]["history"]
            self.assertEqual(len(curves), 1)
            self.assertTrue(math.isfinite(curves[0]["train_loss"]))
            self.assertTrue(math.isfinite(curves[0]["validation_loss"]))
            self.assertTrue((checkpoint_dir / "n0_seed_684.pt").is_file())
            checkpoint = torch.load(
                checkpoint_dir / "n0_seed_684.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["runtime_versions"], report["runtime_versions"])
            self.assertEqual(checkpoint["code_sha256"], report["code_sha256"])
            self.assertEqual(checkpoint["sampled_pairs_csv"], report["sampled_pairs_csv"])
            with (root / "n0_training_curves.csv").open(newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["seed"], "684")
            self.assertIn("1 positive edge", report["warning"])
            self.assertNotIn("11 positive edges", report["warning"])


if __name__ == "__main__":
    unittest.main()

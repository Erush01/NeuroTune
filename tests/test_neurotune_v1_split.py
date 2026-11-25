import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.data.neurotune_v1_split import build_split, publish_split


def fixture(root):
    root = Path(root)
    zip_path = root / "lastfm_test.zip"
    # Two test-source tracks share an artist; IDs with missing artist metadata exercise fallback.
    tracks = [
        ("TRAAAAA00000000001", "artist-a", "Alpha"),
        ("TRAAAAA00000000002", "artist-a", "Alpha"),
        ("TRBBBBB00000000001", "artist-b", "Beta"),
        ("TRCCCCC00000000001", "", "Gamma"),
        ("TRDDDDD00000000001", "", ""),
        ("TREEEEE00000000001", "artist-e", "Epsilon"),
        ("TRFFFFF00000000001", "artist-f", "Phi"),
        ("TRGGGGG00000000001", "artist-g", "Gamma"),
        ("TRHHHHH00000000001", "artist-h", "Eta"),
        ("TRIIIII00000000001", "artist-i", "Iota"),
        ("TRJJJJJ00000000001", "artist-j", "Jota"),
        ("TRKKKKK00000000001", "artist-k", "Kappa"),
        ("TRLLLLL00000000001", "artist-l", "Lambda"),
    ]
    with zipfile.ZipFile(zip_path, "w") as archive:
        for track_id in (tracks[0][0], tracks[1][0]):
            archive.writestr(f"lastfm_test/{track_id[2]}/{track_id[3]}/{track_id[4]}/{track_id}.json", "{}")
    audit_path = root / "audit.json"
    audit_path.write_text(json.dumps({"status": "complete", "channels": list(range(26)),
                                      "window": {"seconds": 30},
                                      "source_sha256": {"lastfm_test": hashlib.sha256(zip_path.read_bytes()).hexdigest()}}))
    catalog_path = root / "catalog.csv"
    with catalog_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("track_id", "artist_id", "artist_name"))
        writer.writerows(tracks)
    train_path, test_path = root / "train.csv", root / "test.csv"
    def write_edges(path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("track_id_a", "track_id_b"))
            writer.writerows(rows)
    write_edges(train_path, [(tracks[2][0], tracks[3][0]), (tracks[2][0], tracks[0][0])])
    write_edges(test_path, [(tracks[0][0], tracks[1][0]), (tracks[0][0], tracks[2][0])])
    return audit_path, catalog_path, train_path, test_path, zip_path


class NeuroTuneSplitTests(unittest.TestCase):
    def test_artist_source_assignment_grouping_fallback_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit, catalog, train, test, archive = fixture(tmp)
            manifest, edges = build_split(audit, catalog, train, test, archive, tmp)
            self.assertEqual(set(manifest["track_ids"]["test"]), {"TRAAAAA00000000001", "TRAAAAA00000000002"})
            self.assertEqual(edges["test"], [("TRAAAAA00000000001", "TRAAAAA00000000002")])
            self.assertEqual(sum(manifest["counts"]["edges"].values()), 2)
            self.assertEqual(manifest["counts"]["excluded_cross_partition_edges"], {"train_source": 1, "test_source": 1})
            self.assertEqual(manifest["counts"]["test_queries"], 2)
            artist_sets = manifest["artist_keys"]
            self.assertFalse(set(artist_sets["train"]) & set(artist_sets["validation"]))
            self.assertFalse(set(artist_sets["train"]) & set(artist_sets["test"]))
            self.assertFalse(set(artist_sets["validation"]) & set(artist_sets["test"]))
            self.assertIn("TRDDDDD00000000001", manifest["track_ids"]["train"] + manifest["track_ids"]["validation"])
            self.assertIn("catalog", manifest["source_sha256"])
            self.assertIn("lastfm_test", manifest["source_sha256"])

    def test_validation_artist_selection_is_seeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(tmp)
            first, _ = build_split(*paths[:4], paths[4], tmp)
            second, _ = build_split(*paths[:4], paths[4], tmp)
            self.assertEqual(first["artist_keys"]["validation"], ["TRDDDDD00000000001"])
            self.assertEqual(len(first["artist_keys"]["validation"]), 1)
            self.assertEqual(first["artist_keys"]["validation"], second["artist_keys"]["validation"])

    def test_publish_creates_all_outputs_and_refuses_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(tmp)
            manifest, edges = build_split(*paths[:4], paths[4], tmp)
            outputs = publish_split(manifest, edges, tmp)
            self.assertTrue(all(path.is_file() for path in outputs))
            with self.assertRaises(FileExistsError):
                publish_split(manifest, edges, tmp)


if __name__ == "__main__":
    unittest.main()

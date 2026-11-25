import csv
import json
import sys
import tarfile
import unittest
import zipfile
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

from scripts.audit_msd_hf_ab import _run_named_shards, run
import tests.test_msd_hf_ab_audit as ab_audit
import scripts.audit_msd_hf_abcd as abcd_cli


class NamedShardAuditTest(unittest.TestCase):
    def setUp(self):
        self.fixture = ab_audit.AuditTest("test_duplicate_intersections_edges_and_order_are_deterministic")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.c = self.fixture.root / "C.tar.gz"
        self.d = self.fixture.root / "D.tar.gz"
        self._write_shard(self.c, (ab_audit.track_id(8), "AR8", "SO8", 128),
                          (ab_audit.track_id(10), "AR10", "SO10", 130))
        self._write_shard(self.d, (ab_audit.track_id(9), "AR8", "SO8", 129))
        with self.fixture.lyrics.joinpath("mxm.txt").open("a", encoding="utf-8") as stream:
            stream.write(f"\n{ab_audit.track_id(8)},1,1:1\n{ab_audit.track_id(9)},1,1:1\n"
                         f"{ab_audit.track_id(10)},1,1:1")
        with self.fixture.genres.open("a", encoding="utf-8") as stream:
            stream.write(f"{ab_audit.track_id(8)},rock\n{ab_audit.track_id(9)},rock\n"
                         f"{ab_audit.track_id(10)},rock\n")
        self._add_lastfm_rows(self.fixture.train, "train", {
            ab_audit.track_id(8): [ab_audit.track_id(1), ab_audit.track_id(10)],
            ab_audit.track_id(10): [ab_audit.track_id(8)],
        })
        self._add_lastfm_rows(self.fixture.test, "test", {
            ab_audit.track_id(8): [ab_audit.track_id(6)],
        })
        self.reference_catalog = self.fixture.root / "ab_catalog_tracks.csv"
        self.reference_train = self.fixture.root / "ab_lastfm_train_edges.csv"
        self.reference_test = self.fixture.root / "ab_lastfm_test_edges.csv"
        self._write_reference_catalog([1, 3, 5, 6, 7])
        self._write_reference_edges(self.reference_train, [(1, 5), (1, 3)])
        self._write_reference_edges(self.reference_test, [(1, 3), (1, 6)])

    @staticmethod
    def _write_shard(path, *specs):
        with tarfile.open(path, "w:gz") as archive:
            for spec in specs:
                ab_audit.add_track(archive, *spec)

    def _add_lastfm_rows(self, path, split, rows):
        with zipfile.ZipFile(path) as source:
            members = {name: source.read(name) for name in source.namelist()}
        for identifier, similars in rows.items():
            member = f"lastfm_{split}/{identifier[2]}/{identifier[3]}/{identifier[4]}/{identifier}.json"
            members[member] = json.dumps({"track_id": identifier, "similars": similars}).encode()
        with zipfile.ZipFile(path, "w") as target:
            for name, payload in members.items():
                target.writestr(name, payload)

    def _write_reference_catalog(self, identifiers):
        with self.reference_catalog.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["track_id", "tempo"])
            writer.writeheader()
            for number in identifiers:
                writer.writerow({"track_id": ab_audit.track_id(number), "tempo": 120})

    @staticmethod
    def _write_reference_edges(path, edges):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["track_id_a", "track_id_b"])
            writer.writeheader()
            for left, right in edges:
                writer.writerow({"track_id_a": ab_audit.track_id(left),
                                 "track_id_b": ab_audit.track_id(right)})

    def invoke_comparison(self, output=None, comparison_paths=None):
        return _run_named_shards(
            OrderedDict((label, path) for label, path in (
                ("A", self.fixture.a), ("B", self.fixture.b), ("C", self.c), ("D", self.d),
            )),
            "abcd", self.fixture.lyrics, self.fixture.genres, self.fixture.duplicates,
            self.fixture.train, self.fixture.test, output or self.fixture.root / "abcd-out",
            comparison_paths=comparison_paths or {
                "catalog": self.reference_catalog,
                "train_edges": self.reference_train,
                "test_edges": self.reference_test,
            },
            third_party_caveat=None,
            project_use_authorization="The user confirmed project-use permission.",
        )

    def test_named_shards_preserve_provenance_and_resolve_cross_shard_duplicate(self):
        output = self.fixture.root / "named-out"
        audit = _run_named_shards(
            OrderedDict((label, path) for label, path in (
                ("A", self.fixture.a), ("B", self.fixture.b), ("C", self.c), ("D", self.d),
            )),
            "abcd", self.fixture.lyrics, self.fixture.genres, self.fixture.duplicates,
            self.fixture.train, self.fixture.test, output,
        )
        for label, path in (("A", self.fixture.a), ("B", self.fixture.b), ("C", self.c), ("D", self.d)):
            self.assertEqual(audit["source_sha256"][label], ab_audit.audit_module._sha256(path))
        with (output / "abcd_catalog_tracks.csv").open(encoding="utf-8", newline="") as stream:
            ids = [row["track_id"] for row in csv.DictReader(stream)]
        self.assertIn(ab_audit.track_id(8), ids)
        self.assertNotIn(ab_audit.track_id(9), ids)
        self.assertEqual(audit["counts"]["duplicate_tracks_removed"], 3)
        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            sorted(("abcd_catalog_audit.json", "abcd_catalog_tracks.csv",
                    "abcd_lastfm_train_edges.csv", "abcd_lastfm_test_edges.csv")),
        )

    def test_legacy_run_keeps_ab_outputs_and_source_keys(self):
        audit = run(
            self.fixture.a, self.fixture.b, self.fixture.lyrics, self.fixture.genres,
            self.fixture.duplicates, self.fixture.train, self.fixture.test, self.fixture.output,
        )
        self.assertEqual(
            sorted(path.name for path in self.fixture.output.iterdir()),
            sorted(("ab_catalog_audit.json", "ab_catalog_tracks.csv",
                    "ab_lastfm_train_edges.csv", "ab_lastfm_test_edges.csv")),
        )
        saved = json.loads((self.fixture.output / "ab_catalog_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["source_sha256"]["A"], audit["source_sha256"]["A"])
        self.assertEqual(saved["source_sha256"]["B"], audit["source_sha256"]["B"])
        self.assertNotIn("C", saved["source_sha256"])

    def test_rejects_unsafe_output_prefix_before_creating_paths(self):
        output = self.fixture.root / "out"
        with self.assertRaises(ValueError):
            _run_named_shards(
                OrderedDict((("A", self.fixture.a),)), "../escape",
                self.fixture.lyrics, self.fixture.genres, self.fixture.duplicates,
                self.fixture.train, self.fixture.test, output,
            )
        self.assertFalse(output.exists())
        self.assertFalse((self.fixture.root / "escape_catalog_audit.json").exists())

    def test_ab_comparison_reports_exact_catalog_edge_and_edge_class_counts(self):
        audit = self.invoke_comparison()
        comparison = audit["ab_comparison"]
        self.assertEqual(comparison["eligible_tracks"], {
            "added_count": 2, "removed_count": 1, "net_count": 1,
            "added_ids": [ab_audit.track_id(8), ab_audit.track_id(10)],
            "removed_ids": [ab_audit.track_id(7)],
        })
        self.assertEqual(comparison["train_edges"], {
            "added_count": 2, "removed_count": 1, "net_count": 1,
        })
        self.assertEqual(comparison["test_edges"], {
            "added_count": 1, "removed_count": 0, "net_count": 1,
        })
        self.assertEqual(comparison["edge_classes"], {
            "train": {"new_new": 1, "new_existing": 1, "existing_existing": 1},
            "test": {"new_new": 0, "new_existing": 1, "existing_existing": 2},
        })
        self.assertEqual(set(comparison["reference_sha256"]), {"catalog", "train_edges", "test_edges"})
        for name, path in (("catalog", self.reference_catalog),
                           ("train_edges", self.reference_train),
                           ("test_edges", self.reference_test)):
            self.assertEqual(comparison["reference_sha256"][name], ab_audit.audit_module._sha256(path))
        self.assertEqual(audit["project_use_authorization"], "The user confirmed project-use permission.")
        self.assertNotIn("third_party_caveat", audit)

    def test_rejects_reference_output_collision_before_publication(self):
        output = self.fixture.root / "collision-out"
        output.mkdir()
        catalog_inside_output = output / "ab_catalog_tracks.csv"
        catalog_inside_output.write_bytes(self.reference_catalog.read_bytes())
        bad_paths = {"catalog": catalog_inside_output,
                     "train_edges": self.reference_train, "test_edges": self.reference_test}
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.invoke_comparison(output, bad_paths)
        self.assertFalse((output / "abcd_catalog_audit.json").exists())

    def test_rejects_shard_output_collision_before_publication(self):
        source_bytes = self.fixture.a.read_bytes()
        with self.assertRaises(ValueError):
            self.invoke_comparison(self.fixture.a)
        self.assertEqual(self.fixture.a.read_bytes(), source_bytes)


    def test_rejects_invalid_reference_before_publishing(self):
        invalid_catalog = self.fixture.root / "bad_catalog.csv"
        invalid_catalog.write_text("wrong\nvalue\n", encoding="utf-8")
        output = self.fixture.root / "invalid-ref-out"
        paths = {"catalog": invalid_catalog, "train_edges": self.reference_train,
                 "test_edges": self.reference_test}
        with self.assertRaisesRegex(ValueError, "reference CSV header"):
            self.invoke_comparison(output, paths)
        self.assertFalse(output.exists())

    def test_abcd_cli_uses_rooted_defaults_and_abcd_publication_contract(self):
        self.assertEqual(abcd_cli.DEFAULTS["a_tar"], abcd_cli.REPO_ROOT / "data/msd_hf_pilot/A.tar.gz")
        self.assertEqual(abcd_cli.DEFAULTS["d_tar"], abcd_cli.REPO_ROOT / "data/msd_hf_pilot/D.tar.gz")
        self.assertEqual(abcd_cli.DEFAULTS["output_dir"], abcd_cli.REPO_ROOT / "data/msd_hf_abcd_pilot")
        output = self.fixture.root / "cli-out"
        audit = abcd_cli.run_abcd(
            self.fixture.a, self.fixture.b, self.c, self.d, self.fixture.lyrics,
            self.fixture.genres, self.fixture.duplicates, self.fixture.train, self.fixture.test,
            self.reference_catalog, self.reference_train, self.reference_test, output,
        )
        self.assertEqual(set(audit["source_sha256"]) & {"A", "B", "C", "D"},
                         {"A", "B", "C", "D"})
        wrapper_path = abcd_cli.REPO_ROOT / "scripts/audit_msd_hf_abcd.py"
        wrapper_key = "scripts/audit_msd_hf_abcd.py"
        self.assertIn(wrapper_key, audit["code_sha256"])
        self.assertEqual(audit["code_sha256"][wrapper_key],
                         ab_audit.audit_module._sha256(wrapper_path))
        self.assertEqual(audit["project_use_authorization"], abcd_cli.AUTHORIZATION)
        self.assertTrue((output / "abcd_catalog_audit.json").is_file())
        with self.assertRaises(ValueError):
            abcd_cli.run_abcd(output_dir=abcd_cli.REPO_ROOT / "data/msd_hf_pilot")

    def test_cli_routes_options_to_run_abcd_keyword_parameters(self):
        calls = []

        def strict_capture(a_tar, b_tar, c_tar, d_tar, lyrics_dir, genres_csv,
                           duplicates_path, lastfm_train_zip, lastfm_test_zip,
                           reference_catalog, reference_train_edges, reference_test_edges,
                           output_dir):
            calls.append({
                "a_tar": a_tar, "b_tar": b_tar, "c_tar": c_tar, "d_tar": d_tar,
                "lyrics_dir": lyrics_dir, "genres_csv": genres_csv,
                "duplicates_path": duplicates_path, "lastfm_train_zip": lastfm_train_zip,
                "lastfm_test_zip": lastfm_test_zip, "reference_catalog": reference_catalog,
                "reference_train_edges": reference_train_edges,
                "reference_test_edges": reference_test_edges, "output_dir": output_dir,
            })
            return {"counts": {}}

        argv = [
            "audit_msd_hf_abcd", "--a-tar", "a.tar.gz", "--b-tar", "b.tar.gz",
            "--c-tar", "c.tar.gz", "--d-tar", "d.tar.gz", "--lyrics-dir", "lyrics",
            "--genres", "genres.csv", "--duplicates", "duplicates.txt",
            "--lastfm-train", "train.zip", "--lastfm-test", "test.zip",
            "--ab-catalog", "catalog.csv", "--ab-train-edges", "train-edges.csv",
            "--ab-test-edges", "test-edges.csv", "--output-dir", "output",
        ]
        with patch.object(sys, "argv", argv), patch.object(abcd_cli, "run_abcd", strict_capture):
            abcd_cli.main()
        self.assertEqual(calls, [{
            "a_tar": Path("a.tar.gz"), "b_tar": Path("b.tar.gz"),
            "c_tar": Path("c.tar.gz"), "d_tar": Path("d.tar.gz"),
            "lyrics_dir": Path("lyrics"), "genres_csv": Path("genres.csv"),
            "duplicates_path": Path("duplicates.txt"),
            "lastfm_train_zip": Path("train.zip"), "lastfm_test_zip": Path("test.zip"),
            "reference_catalog": Path("catalog.csv"),
            "reference_train_edges": Path("train-edges.csv"),
            "reference_test_edges": Path("test-edges.csv"), "output_dir": Path("output"),
        }])


if __name__ == "__main__":
    unittest.main()

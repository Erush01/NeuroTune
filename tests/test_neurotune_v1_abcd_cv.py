import csv
import contextlib
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py

from src.data.neurotune_v1_abcd_cv import build_abcd_cv
from scripts import audit_neurotune_v1_abcd_cv as abcd_cv_cli


def tid(i):
    return f"TR{i:016d}"


class AbcdCvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / "out"
        self.tracks = {tid(i): {"track_id": tid(i), "artist_id": f"artist-{i // 2}", "artist_name": f"Name {i // 2}"} for i in range(12)}
        self.tracks[tid(10)]["artist_id"] = ""
        self.tracks[tid(10)]["artist_name"] = "fallback"
        self.tracks[tid(11)]["artist_id"] = ""
        self.tracks[tid(11)]["artist_name"] = "fallback"
        self.catalog = self.root / "catalog.csv"
        self._write_catalog(self.catalog, self.tracks)
        self.audit = self.root / "audit.json"
        self.audit.write_text(json.dumps({"status": "complete", "window": {"seconds": 30, "interval": "[0,30)"}, "channels": [f"c{i}" for i in range(26)], "source_sha256": {k: k.lower() * 64 for k in "ABCD"}}))
        self.cache = self.root / "cache.h5"
        with h5py.File(self.cache, "w") as h:
            h.create_dataset("track_ids", data=[x.encode() for x in sorted(self.tracks)])
            h.create_dataset("features", shape=(24, 26), dtype="f4")
            h.create_dataset("offsets", data=list(range(0, 25, 2)))
            self._cache_attrs(h)
        self.meta = self.root / "metadata.json"
        self._write_metadata(sorted(self.tracks), 0)
        self.reject = self.root / "rejections.json"
        self.reject.write_text("[]")
        self.train = self.root / "train.csv"
        self.test = self.root / "test.csv"
        self.legacy_pilot = self.root / "legacy-pilot"
        self.legacy_pilot.mkdir()
        self.abcat = self.legacy_pilot / "ab.csv"
        self._write_catalog(self.abcat, {tid(i): self.tracks[tid(i)] for i in range(4)})
        self.absplit = self.root / "split.json"
        self.absplit.write_text(json.dumps({"artist_keys": {"train": ["artist-1"], "validation": [], "test": ["artist-0"]}, "track_ids": {"train": [tid(2), tid(3)], "validation": [], "test": [tid(0), tid(1)]}}))
        self.abtest = self.root / "abtest.csv"
        self.ab_test_edge = (tid(2), tid(3))
        self.write_edges(self.abtest, [self.ab_test_edge])
        self.train_edges = [(tid(0), tid(1)), (tid(0), tid(2)), (tid(2), tid(3)), (tid(4), tid(6)), (tid(6), tid(8)), (tid(8), tid(10)), (tid(0), tid(8)), (tid(1), tid(9))]
        self.test_edges = [(tid(4), tid(5)), (tid(6), tid(7)), (tid(8), tid(9)), (tid(10), tid(11))]
        self.write_edges(self.train, self.train_edges)
        self.write_edges(self.test, self.test_edges)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def _cache_attrs(self, handle):
        handle.attrs["dtype"] = "float32"
        handle.attrs["window_seconds"] = 30
        handle.attrs["interval"] = "[0,30)"
        handle.attrs["channels"] = json.dumps([f"c{i}" for i in range(26)])
        handle.attrs["normalization"] = "raw/unnormalized; deferred until train split"
        handle.attrs["audit_sha256"] = self.sha(self.audit)
        handle.attrs["catalog_sha256"] = self.sha(self.catalog)
        handle.attrs["source_sha256"] = json.dumps(json.loads(self.audit.read_text())["source_sha256"], sort_keys=True)
        handle.attrs["code_sha256"] = json.dumps({}, sort_keys=True)

    def _write_metadata(self, track_ids, rejected_count):
        count = len(track_ids)
        self.meta.write_text(json.dumps({"status": "complete", "window_seconds": 30, "interval": "[0,30)", "channel_count": 26, "channels": [f"c{i}" for i in range(26)], "dtype": "float32", "normalization": "raw/unnormalized; deferred until train split", "segment_count": 2 * count, "track_count": count, "rejected_track_count": rejected_count, "valid_track_ids": track_ids, "sequence_lengths": {key: 2 for key in track_ids}, "rejected": [], "audit_sha256": self.sha(self.audit), "catalog_sha256": self.sha(self.catalog), "source_sha256": json.loads(self.audit.read_text())["source_sha256"], "code_sha256": {}}))

    @staticmethod
    def _write_catalog(path, rows):
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "artist_id", "artist_name"])
            writer.writeheader()
            writer.writerows(rows.values())

    @staticmethod
    def write_edges(path, edges):
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["track_id_a", "track_id_b"])
            w.writerows(edges)

    def run_build(self):
        return build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train,
                             test_edges_path=self.test, cache_path=self.cache, cache_metadata_path=self.meta,
                             rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit,
                              ab_test_edges_path=self.abtest, output_dir=self.out)

    def cli_args(self, output=None):
        output = output or self.out
        values = {
            "--audit": self.audit, "--catalog": self.catalog,
            "--train-edges": self.train, "--test-edges": self.test,
            "--cache": self.cache, "--cache-metadata": self.meta,
            "--rejections": self.reject, "--ab-catalog": self.abcat,
            "--ab-split": self.absplit, "--ab-test-edges": self.abtest,
            "--output-dir": output,
        }
        return [arg for option, path in values.items() for arg in (option, str(path))]

    def test_cli_defaults_and_strict_keyword_routing(self):
        manifest_path = self.root / "mock-manifest.json"
        manifest_path.write_text(json.dumps({"folds": [], "exclusions": {"invalid_endpoint": {}, "by_fold": []}, "ab_test_source_policy_exclusions": []}))
        with patch.object(abcd_cv_cli, "build_abcd_cv", return_value={"manifest": str(manifest_path)}) as build, contextlib.redirect_stdout(io.StringIO()):
            abcd_cv_cli.main([])
        self.assertEqual(build.call_args.kwargs, {**abcd_cv_cli.DEFAULTS, "seed": 684})

        custom_out = self.root / "custom-output"
        args = self.cli_args(custom_out) + ["--seed", "23"]
        with patch.object(abcd_cv_cli, "build_abcd_cv", return_value={"manifest": str(manifest_path)}) as build, contextlib.redirect_stdout(io.StringIO()):
            abcd_cv_cli.main(args)
        expected = {key: Path(value) for key, value in {
            "audit_path": self.audit, "catalog_path": self.catalog, "train_edges_path": self.train,
            "test_edges_path": self.test, "cache_path": self.cache, "cache_metadata_path": self.meta,
            "rejection_path": self.reject, "ab_catalog_path": self.abcat, "ab_split_path": self.absplit,
            "ab_test_edges_path": self.abtest, "output_dir": custom_out,
        }.items()}
        self.assertEqual(build.call_args.kwargs, {**expected, "seed": 23})

    def test_cli_integrates_core_and_prints_fold_and_exclusion_counts(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            abcd_cv_cli.main(self.cli_args())
        summary = json.loads(output.getvalue())
        self.assertEqual(len(summary["folds"]), 3)
        self.assertIn("valid_tracks", summary["folds"][0])
        self.assertIn("valid_artists", summary["folds"][0])
        self.assertIn("train_source", summary["folds"][0]["edges"])
        self.assertIn("official_test_source", summary["folds"][0]["edges"])
        self.assertEqual(summary["exclusions"]["ab_test_source_policy"], 1)
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["input_sha256"]["ab_split_path"], self.sha(self.absplit))

    def test_cli_rejects_audit_cache_and_provenance_hash_mismatches(self):
        cases = ("audit", "cache", "provenance", "hash")
        for case in cases:
            with self.subTest(case=case):
                self._write_metadata(sorted(self.tracks), 0)
                self.audit.write_text(json.dumps({"status": "complete", "window": {"seconds": 30, "interval": "[0,30)"}, "channels": [f"c{i}" for i in range(26)], "source_sha256": {k: k.lower() * 64 for k in "ABCD"}}))
                self._write_metadata(sorted(self.tracks), 0)
                if case == "audit":
                    audit = json.loads(self.audit.read_text())
                    audit["status"] = "incomplete"
                    self.audit.write_text(json.dumps(audit))
                elif case == "cache":
                    with h5py.File(self.cache, "w") as h:
                        h.create_dataset("track_ids", data=[x.encode() for x in sorted(self.tracks)])
                        h.create_dataset("features", shape=(24, 26), dtype="f8")
                        h.create_dataset("offsets", data=list(range(0, 25, 2)))
                        self._cache_attrs(h)
                elif case == "provenance":
                    meta = json.loads(self.meta.read_text())
                    meta["source_sha256"] = {k: "0" * 64 for k in "ABCD"}
                    self.meta.write_text(json.dumps(meta))
                else:
                    meta = json.loads(self.meta.read_text())
                    meta["audit_sha256"] = "0" * 64
                    self.meta.write_text(json.dumps(meta))
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    abcd_cv_cli.main(self.cli_args(self.root / f"cli-{case}"))
                self.assertFalse((self.root / f"cli-{case}").exists())

    def test_cli_collision_refusal_and_rollback_no_clobber(self):
        self.cli_args()
        with contextlib.redirect_stdout(io.StringIO()):
            abcd_cv_cli.main(self.cli_args())
        original = (self.out / "manifest.json").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            abcd_cv_cli.main(self.cli_args())
        self.assertEqual((self.out / "manifest.json").read_bytes(), original)

        rollback_dir = self.root / "rollback"
        from src.data import neurotune_v1_abcd_cv
        real_link = neurotune_v1_abcd_cv.os.link
        calls = 0

        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated publication failure")
            return real_link(source, target)

        with patch.object(neurotune_v1_abcd_cv.os, "link", side_effect=fail_second):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(OSError, "simulated"):
                abcd_cv_cli.main(self.cli_args(rollback_dir))
        self.assertEqual(list(rollback_dir.iterdir()), [])

    def test_three_folds_assignment_routing_and_known_positive_superset(self):
        result = self.run_build()
        manifest = json.loads(Path(result["manifest"]).read_text())
        cli_source = Path(__file__).resolve().parents[1] / "scripts/audit_neurotune_v1_abcd_cv.py"
        self.assertEqual(manifest["code_sha256"]["scripts/audit_neurotune_v1_abcd_cv.py"], self.sha(cli_source))
        groups = manifest["artist_groups"]
        self.assertEqual(set(groups.values()), {"artist_id:artist-0", "artist_id:artist-1", "artist_id:artist-2", "artist_id:artist-3", "artist_id:artist-4", "artist_name:fallback"})
        self.assertEqual(len(manifest["folds"]), 3)
        self.assertTrue(all("artist_id:artist-0" not in f["test_artist_keys"] for f in manifest["folds"]))
        expected = sorted({"artist_id:artist-2", "artist_id:artist-3", "artist_id:artist-4", "artist_name:fallback"})
        random.Random(684).shuffle(expected)
        self.assertEqual([f["test_artist_keys"] for f in manifest["folds"]], [sorted(expected[i::3]) for i in range(3)])
        self.assertEqual(manifest["known_positive_edge_count"], len(set(map(lambda e: tuple(sorted(e)), self.train_edges + self.test_edges))))
        self.assertEqual(manifest["ab_test_source_policy_exclusions"], [list(self.ab_test_edge)])
        expected_digest = hashlib.sha256(json.dumps([list(edge) for edge in sorted(set(tuple(sorted(e)) for e in self.train_edges + self.test_edges))], separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(manifest["known_positive_edges_sha256"], expected_digest)
        for i, fold in enumerate(manifest["folds"]):
            self.assertEqual(fold["test_artist_keys"], sorted(set(fold["test_artist_keys"])))
            for name in ("train", "validation", "test_train_source", "test_official_source"):
                with (self.out / f"fold_{i}_{name}_edges.csv").open() as f:
                    rows = list(csv.reader(f))[1:]
                edges = {tuple(row) for row in rows}
                self.assertFalse(edges & {self.ab_test_edge})
            tr = set(tuple(e) for e in self.read_edges(self.out / f"fold_{i}_test_train_source_edges.csv"))
            te = set(tuple(e) for e in self.read_edges(self.out / f"fold_{i}_test_official_source_edges.csv"))
            self.assertFalse(tr & te)
            self.assertEqual(fold["counts"]["test_train_source"], len(tr))
            self.assertEqual(fold["counts"]["test_official_source"], len(te))
        known = set(map(tuple, self.read_edges(self.out / "known_positive_edges.csv")))
        self.assertGreaterEqual(known, set(tuple(sorted(e)) for e in self.train_edges + self.test_edges))

    @staticmethod
    def read_edges(path):
        with Path(path).open(newline="", encoding="utf-8") as f:
            return [tuple(row) for row in list(csv.reader(f))[1:]]

    def test_cache_valid_filter_and_provenance_validation(self):
        # One rejected candidate is excluded from cache, and one edge touching it becomes invalid-endpoint exclusion.
        rejected_id = tid(11)
        with h5py.File(self.cache, "w") as h:
            h.create_dataset("track_ids", data=[x.encode() for x in sorted(set(self.tracks) - {rejected_id})])
            h.create_dataset("features", shape=(22, 26), dtype="f4")
            h.create_dataset("offsets", data=list(range(0, 23, 2)))
            self._cache_attrs(h)
        self.reject.write_text(json.dumps([{"track_id": rejected_id, "reason": "bad"}]))
        valid_ids = sorted(set(self.tracks) - {rejected_id})
        self._write_metadata(valid_ids, 1)
        meta = json.loads(self.meta.read_text())
        meta["rejected"] = [{"track_id": rejected_id, "reason": "bad"}]
        self.meta.write_text(json.dumps(meta))
        self.run_build()
        with self.assertRaisesRegex(ValueError, "hashes"):
            meta = json.loads(self.meta.read_text())
            meta["audit_sha256"] = "0" * 64
            self.meta.write_text(json.dumps(meta))
            self.out2 = self.root / "out2"
            build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.test, cache_path=self.cache, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=self.out2)

    def test_overlap_duplicates_and_collisions_are_rejected_deterministically(self):
        first = self.run_build()
        original = Path(first["manifest"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_build()
        self.assertEqual(Path(first["manifest"]).read_bytes(), original)
        clone = self.root / "clone"
        build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.test, cache_path=self.cache, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=clone)
        self.assertEqual((clone / "manifest.json").read_bytes(), original)
        duplicate = self.root / "dup.csv"
        self.write_edges(duplicate, self.train_edges + [self.train_edges[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=duplicate, test_edges_path=self.test, cache_path=self.cache, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=self.root / "dupout")
        overlap = self.root / "overlap.csv"
        self.write_edges(overlap, self.train_edges)
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=overlap, cache_path=self.cache, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=self.root / "overlapout")

    def test_publication_rolls_back_if_a_link_fails(self):
        from src.data import neurotune_v1_abcd_cv
        real_link = neurotune_v1_abcd_cv.os.link
        calls = 0

        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated publication failure")
            return real_link(source, target)

        with patch.object(neurotune_v1_abcd_cv.os, "link", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.run_build()
        self.assertEqual(list(self.out.iterdir()), [])

    def test_legacy_artist_name_fallback_and_stale_artist_manifest(self):
        old_rows = {tid(i): dict(self.tracks[tid(i)]) for i in range(4)}
        old_rows[tid(0)]["artist_id"] = ""
        old_rows[tid(1)]["artist_id"] = ""
        old_rows[tid(0)]["artist_name"] = "legacy name"
        old_rows[tid(1)]["artist_name"] = "legacy name"
        self._write_catalog(self.abcat, old_rows)
        self.absplit.write_text(json.dumps({"artist_keys": {"train": ["artist-1"], "validation": [], "test": ["legacy name"]}, "track_ids": {"train": [tid(2), tid(3)], "validation": [], "test": [tid(0), tid(1)]}}))
        result = self.run_build()
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertNotIn("artist_name:legacy name", {key for fold in manifest["folds"] for key in fold["test_artist_keys"]})
        self.absplit.write_text(json.dumps({"artist_keys": {"train": ["artist-1"], "validation": [], "test": ["wrong legacy name"]}, "track_ids": {"train": [tid(2), tid(3)], "validation": [], "test": [tid(0), tid(1)]}}))
        with self.assertRaisesRegex(ValueError, "artist keys do not match"):
            build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.test, cache_path=self.cache, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=self.root / "bad-split")

    def test_legacy_train_and_validation_artist_key_declarations_are_validated(self):
        valid = {"artist_keys": {"train": ["artist-1"], "validation": [], "test": ["artist-0"]},
                 "track_ids": {"train": [tid(2), tid(3)], "validation": [], "test": [tid(0), tid(1)]}}
        for partition, bad_keys in (("train", ["stale-train-key"]), ("validation", ["stale-validation-key"])):
            split = json.loads(json.dumps(valid))
            split["artist_keys"][partition] = bad_keys
            self.absplit.write_text(json.dumps(split))
            with self.subTest(partition=partition), self.assertRaisesRegex(ValueError, "artist keys do not match"):
                build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.test, cache_path=self.cache, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=self.root / f"bad-{partition}")

    def test_output_cannot_overwrite_legacy_pilot_directory(self):
        pilot = self.legacy_pilot
        ab_catalog = pilot / "catalog.csv"
        self._write_catalog(ab_catalog, {tid(i): self.tracks[tid(i)] for i in range(4)})
        before = sorted(path.name for path in pilot.iterdir())
        kwargs = dict(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.test, cache_path=self.cache, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=ab_catalog, ab_split_path=self.absplit, ab_test_edges_path=self.abtest)
        for output in (pilot, pilot / "nested"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "legacy A/B pilot"):
                build_abcd_cv(**kwargs, output_dir=output)
        self.assertEqual(sorted(path.name for path in pilot.iterdir()), before)

    def test_invalid_feature_dtype_values_and_empty_sequences_fail_before_output(self):
        for case in ("float64", "nan", "empty_sequence"):
            with self.subTest(case=case):
                cache_path = self.root / f"{case}.h5"
                with h5py.File(cache_path, "w") as h:
                    h.create_dataset("track_ids", data=[x.encode() for x in sorted(self.tracks)])
                    dtype = "f8" if case == "float64" else "f4"
                    features = h.create_dataset("features", shape=(24, 26), dtype=dtype)
                    if case == "nan":
                        features[0, 0] = float("nan")
                    offsets = list(range(0, 25, 2))
                    if case == "empty_sequence":
                        offsets[1] = 0
                    h.create_dataset("offsets", data=offsets)
                    self._cache_attrs(h)
                out = self.root / f"out-{case}"
                with self.assertRaises(ValueError):
                    build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.test, cache_path=cache_path, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=out)
                self.assertFalse(out.exists())

    def test_permuted_hdf5_track_ids_are_rejected_before_output(self):
        cache_path = self.root / "permuted-cache.h5"
        ids = sorted(self.tracks)
        ids[0], ids[1] = ids[1], ids[0]
        with h5py.File(cache_path, "w") as h:
            h.create_dataset("track_ids", data=[track_id.encode() for track_id in ids])
            h.create_dataset("features", shape=(24, 26), dtype="f4")
            h.create_dataset("offsets", data=list(range(0, 25, 2)))
            self._cache_attrs(h)
        out = self.root / "out-permuted-ids"
        with self.assertRaisesRegex(ValueError, "sorted"):
            build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.test, cache_path=cache_path, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=out)
        self.assertFalse(out.exists())

    def test_ab_test_source_edges_must_reference_catalog_tracks(self):
        self.write_edges(self.abtest, [(tid(0), tid(99))])
        out = self.root / "out-bad-ab-edge"
        with self.assertRaisesRegex(ValueError, "outside the legacy A/B catalog"):
            build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.test, cache_path=self.cache, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=out)
        self.assertFalse(out.exists())

    def test_hdf5_provenance_must_match_sidecar_metadata(self):
        cache_path = self.root / "bad-provenance-cache.h5"
        with h5py.File(self.cache, "r") as source, h5py.File(cache_path, "w") as target:
            for name in source:
                source.copy(name, target)
            for key, value in source.attrs.items():
                target.attrs[key] = value
            target.attrs["catalog_sha256"] = "0" * 64
        out = self.root / "out-bad-provenance"
        with self.assertRaisesRegex(ValueError, "HDF5 provenance"):
            build_abcd_cv(audit_path=self.audit, catalog_path=self.catalog, train_edges_path=self.train, test_edges_path=self.test, cache_path=cache_path, cache_metadata_path=self.meta, rejection_path=self.reject, ab_catalog_path=self.abcat, ab_split_path=self.absplit, ab_test_edges_path=self.abtest, output_dir=out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()

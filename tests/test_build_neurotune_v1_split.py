import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_neurotune_v1_split import main
from src.data.neurotune_v1_split import OUTPUT_NAMES, build_split, publish_split
from tests.test_neurotune_v1_split import fixture


class BuildNeuroTuneSplitTests(unittest.TestCase):
    def test_missing_source_and_stale_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(tmp)
            missing = Path(tmp) / "missing.zip"
            with self.assertRaises(FileNotFoundError):
                build_split(*paths[:4], missing, tmp)
            audit = json.loads(paths[0].read_text())
            audit["source_sha256"]["lastfm_test"] = "0" * 64
            paths[0].write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_split(*paths[:4], paths[4], tmp)

    def test_bad_audit_contract_and_edge_endpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(tmp)
            audit = json.loads(paths[0].read_text())
            audit["window"]["seconds"] = 60
            paths[0].write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "30s/26-channel"):
                build_split(*paths[:4], paths[4], tmp)

    def test_collision_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(tmp)
            manifest, edges = build_split(*paths[:4], paths[4], tmp)
            collision = Path(tmp) / OUTPUT_NAMES[0]
            collision.write_text("keep")
            with self.assertRaises(FileExistsError):
                publish_split(manifest, edges, tmp)
            self.assertEqual(collision.read_text(), "keep")
            collision.unlink()
            original = Path.hardlink_to
            calls = 0
            def fail_second(path, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated publication failure")
                return original(path, target)
            with patch.object(Path, "hardlink_to", fail_second):
                with self.assertRaisesRegex(OSError, "simulated"):
                    publish_split(manifest, edges, tmp)
            self.assertFalse(any((Path(tmp) / name).exists() for name in OUTPUT_NAMES))
            self.assertFalse(list(Path(tmp).glob("*.stage")))

    def test_keyboard_interrupt_rolls_back_published_targets_and_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(tmp)
            manifest, edges = build_split(*paths[:4], paths[4], tmp)
            original = Path.hardlink_to
            calls = 0
            def interrupt_second(path, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt("simulated interruption")
                return original(path, target)
            with patch.object(Path, "hardlink_to", interrupt_second):
                with self.assertRaisesRegex(KeyboardInterrupt, "simulated interruption"):
                    publish_split(manifest, edges, tmp)
            self.assertFalse(any((Path(tmp) / name).exists() for name in OUTPUT_NAMES))
            self.assertFalse(list(Path(tmp).glob("*.stage")))

    def test_partial_stage_write_failure_cleans_stage_and_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(tmp)
            manifest, edges = build_split(*paths[:4], paths[4], tmp)
            original = Path.write_text
            def partial_write(path, data, *args, **kwargs):
                original(path, data[:10], *args, **kwargs)
                raise OSError("simulated partial staging failure")
            with patch.object(Path, "write_text", partial_write):
                with self.assertRaisesRegex(OSError, "simulated partial staging failure"):
                    publish_split(manifest, edges, tmp)
            self.assertFalse(any((Path(tmp) / name).exists() for name in OUTPUT_NAMES))
            self.assertFalse(list(Path(tmp).glob("*.stage")))

    def test_cli_builds_outputs_from_mockable_local_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            paths = fixture(data)
            for source, expected in zip(paths[:4], ("ab_catalog_audit.json", "ab_catalog_tracks.csv",
                                                      "ab_lastfm_train_edges.csv", "ab_lastfm_test_edges.csv")):
                source.rename(data / expected)
            self.assertEqual(main(["--data-dir", str(data), "--test-zip", str(paths[4])]), 0)
            manifest = json.loads((data / OUTPUT_NAMES[0]).read_text())
            self.assertEqual(manifest["counts"]["tracks"]["test"], 2)
            self.assertTrue(all((data / name).is_file() for name in OUTPUT_NAMES))


if __name__ == "__main__":
    unittest.main()

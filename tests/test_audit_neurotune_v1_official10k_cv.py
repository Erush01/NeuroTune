import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_neurotune_v1_official10k_cv import _atomic_json


class Official10kCvAuditCliTests(unittest.TestCase):
    def test_atomic_json_publishes_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "audit.json"
            _atomic_json(target, {"status": "complete", "counts": {"tracks": 939}})
            self.assertEqual(json.loads(target.read_text()), {
                "status": "complete", "counts": {"tracks": 939}
            })

    def test_existing_target_is_refused_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "audit.json"
            target.write_text("preserve me")
            with self.assertRaises(FileExistsError):
                _atomic_json(target, {"status": "complete"})
            self.assertEqual(target.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()

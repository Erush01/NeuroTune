import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.preprocess_audio import main


class PreprocessAudioCliTests(unittest.TestCase):
    def test_main_rejects_colliding_output_paths_without_changing_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audit_path = root / "audit.json"
            output_path = root / "output.dat"
            original_bytes = b"keep this file unchanged"
            output_path.write_bytes(original_bytes)
            audit_path.write_text(
                json.dumps({
                    "splits": {
                        "artist_disjoint": {
                            "track_ids": {
                                "train": ["TRTRAIN000001"],
                                "validation": ["TRVALID000001"],
                                "test": ["TRTEST000001"],
                            }
                        }
                    }
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "output paths.*must be different"):
                main([
                    "--input-root", str(root / "missing-source"),
                    "--audit", str(audit_path),
                    "--output", str(output_path),
                    "--metadata-output", str(output_path),
                    "--grid-size", "5",
                ])

            self.assertEqual(output_path.read_bytes(), original_bytes)

    def test_main_builds_cache_from_audit_and_input_overrides(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "raw"
            audit_path = root / "audit.json"
            output_path = root / "processed" / "audio.h5"
            metadata_path = root / "processed" / "audio_meta.json"
            split_ids = {
                "train": ["TRTRAIN000001"],
                "validation": ["TRVALID000001"],
                "test": ["TRTEST000001"],
            }
            audit_path.write_text(
                json.dumps({"splits": {"artist_disjoint": {"track_ids": split_ids}}}),
                encoding="utf-8",
            )

            for index, track_id in enumerate(
                split_ids["train"] + split_ids["validation"] + split_ids["test"]
            ):
                track_path = input_root / "A" / "B" / f"{track_id}.h5"
                track_path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(track_path, "w") as handle:
                    analysis = handle.create_group("analysis")
                    songs = np.zeros(1, dtype=[("track_id", "S18"), ("duration", "f8")])
                    songs[0] = (track_id.encode("ascii"), 2.0)
                    analysis.create_dataset("songs", data=songs)
                    analysis.create_dataset(
                        "segments_pitches", data=np.full((3, 12), index, dtype=np.float64)
                    )
                    analysis.create_dataset(
                        "segments_timbre", data=np.full((3, 12), index + 1, dtype=np.float64)
                    )
                    analysis.create_dataset("segments_loudness_max", data=[-20.0, -10.0, 0.0])
                    analysis.create_dataset("segments_start", data=[0.0, 1.0, 2.0])
                    analysis.create_dataset("beats_start", data=[0.0, 0.5, 2.0])
                    analysis.create_dataset("bars_start", data=[1.0])
                    analysis.create_dataset("sections_start", data=[])

            main([
                "--input-root", str(input_root),
                "--audit", str(audit_path),
                "--output", str(output_path),
                "--metadata-output", str(metadata_path),
                "--grid-size", "5",
            ])

            self.assertTrue(output_path.is_file())
            self.assertTrue(metadata_path.is_file())
            with h5py.File(output_path, "r") as handle:
                track_ids = [value.decode("utf-8") for value in handle["track_ids"][...]]
                self.assertEqual(track_ids, sorted(sum(split_ids.values(), [])))
                self.assertEqual(handle["sequences"].shape, (3, 5, 28))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["split_counts"], {
                "train": 1,
                "validation": 1,
                "test": 1,
            })


if __name__ == "__main__":
    unittest.main()

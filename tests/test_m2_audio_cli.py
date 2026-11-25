import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

import scripts.run_m2_audio as runner
from scripts.run_m2_audio import main


class M2AudioCliTests(unittest.TestCase):
    def test_audio_normalization_provenance_required_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            train, validation, test = ["123"], ["VALID"], ["TEST"]
            all_ids = train + validation + test
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({"splits": {"artist_disjoint": {
                "track_ids": {"train": train, "validation": validation, "test": test}
            }}}))
            channel_names = ([f"pitch_{i}" for i in range(1, 13)]
                             + [f"timbre_{i}" for i in range(1, 13)]
                             + ["loudness_max", "beat_count", "bar_count", "section_count"])
            cache_path = data_dir / "audio_sequences.h5"
            with h5py.File(cache_path, "w") as handle:
                handle.create_dataset("track_ids", data=np.asarray(sorted(all_ids), dtype="S"))
                handle.create_dataset("sequences", data=np.zeros((3, 1024, 28), dtype=np.float32))
                handle.create_dataset("channel_names", data=np.asarray(channel_names, dtype="S"))
            metadata_path = data_dir / "audio_sequences_meta.json"
            base_metadata = {
                "grid_size": 1024, "channel_names": channel_names,
                "split_ids": {"train": train, "validation": validation, "test": test},
            }
            args = [
                "--data-dir", str(data_dir), "--audit", str(audit_path),
                "--audio-cache", str(cache_path), "--audio-metadata", str(metadata_path),
                "--output", str(root / "report.json"), "--checkpoint-dir", str(root / "checkpoints"),
            ]
            for normalization in ({"train_ids": ["OTHER"]}, {"train_ids": [123]}, None):
                metadata = dict(base_metadata)
                if normalization is not None:
                    metadata["normalization"] = normalization
                metadata_path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "audio normalization.*train_ids"):
                    main(args)

    def test_synthetic_one_epoch_audio_report_and_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            lastfm_dir = data_dir / "lastfm_similar"
            lastfm_dir.mkdir(parents=True)
            train = [f"TRTRAIN{i}" for i in reversed(range(6))]
            validation = [f"TRVALID{i}" for i in reversed(range(4))]
            test = [f"TRTEST{i}" for i in reversed(range(3))]
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

            for source, target in [
                (train[0], train[1]), (train[2], train[3]),
                (validation[0], validation[1]), (validation[2], validation[3]),
                (test[0], test[1]),
            ]:
                path = lastfm_dir / f"{source}.json"
                record = json.loads(path.read_text()) if path.exists() else {"track_id": source, "similars": []}
                record["similars"].append([target, 1.0])
                path.write_text(json.dumps(record))

            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({"seed": 684, "splits": {"artist_disjoint": {
                "track_ids": {"train": train, "validation": validation, "test": test}
            }}}))
            channel_names = ([f"pitch_{i}" for i in range(1, 13)]
                             + [f"timbre_{i}" for i in range(1, 13)]
                             + ["loudness_max", "beat_count", "bar_count", "section_count"])
            cache_path = data_dir / "audio_sequences.h5"
            cache_ids = sorted(all_ids)
            audio_value_by_id = {track_id: float(index + 1) for index, track_id in enumerate(cache_ids)}
            sequences = np.empty((len(cache_ids), 1024, 28), dtype=np.float32)
            for index, track_id in enumerate(cache_ids):
                sequences[index].fill(audio_value_by_id[track_id])
            with h5py.File(cache_path, "w") as handle:
                handle.create_dataset("track_ids", data=np.asarray(cache_ids, dtype="S"))
                handle.create_dataset("sequences", data=sequences)
                handle.create_dataset("channel_names", data=np.asarray(channel_names, dtype="S"))
            metadata_path = data_dir / "audio_sequences_meta.json"
            metadata_path.write_text(json.dumps({
                "grid_size": 1024, "channel_names": channel_names,
                "split_ids": {"train": train, "validation": validation, "test": test},
                "normalization": {"train_ids": train},
            }))
            output = root / "m2_audio_experiment.json"
            checkpoint_dir = root / "checkpoints"
            build_modalities = runner.build_train_fitted_modalities

            def checked_build_modalities(*args, **kwargs):
                if list(args[4]) == test:
                    self.assertTrue((checkpoint_dir / "m2_audio_seed_684.pt").is_file())
                    self.assertTrue((checkpoint_dir / "m2_audio_seed_685.pt").is_file())
                return build_modalities(*args, **kwargs)

            train_audio_seed = runner.train_audio_seed

            def checked_train_audio_seed(*args, **kwargs):
                expected = [audio_value_by_id[track_id] for track_id in args[2]]
                actual = np.asarray(args[0])[:, 0, 0].tolist()
                self.assertEqual(actual, expected)
                return train_audio_seed(*args, **kwargs)

            encode = runner._encode

            def checked_encode(model, audio, modalities):
                expected = [audio_value_by_id[track_id] for track_id in test]
                actual = np.asarray(audio)[:, 0, 0].tolist()
                self.assertEqual(actual, expected)
                return encode(model, audio, modalities)

            with (
                patch.object(runner, "build_train_fitted_modalities", side_effect=checked_build_modalities),
                patch.object(runner, "train_audio_seed", side_effect=checked_train_audio_seed),
                patch.object(runner, "_encode", side_effect=checked_encode),
            ):
                main([
                    "--epochs", "1", "--seeds", "684", "685", "--data-dir", str(data_dir),
                    "--audit", str(audit_path), "--audio-cache", str(cache_path),
                    "--audio-metadata", str(metadata_path), "--output", str(output),
                    "--checkpoint-dir", str(checkpoint_dir),
                ])

            report = json.loads(output.read_text())
            self.assertEqual(report["config"]["epochs"], 1)
            self.assertEqual(report["config"]["seeds"], [684, 685])
            self.assertEqual(report["split_counts"], {"train": 6, "validation": 4, "test": 3})
            self.assertEqual(report["training_input_ids"], train + validation)
            self.assertFalse(set(report["training_input_ids"]) & set(test))
            self.assertEqual(report["test"]["gallery_count"], 3)
            self.assertEqual(report["test"]["unique_edge_count"], 1)
            self.assertEqual(report["fit_train_ids"], train)
            sampled_path = root / "m2_audio_sampled_pairs.csv"
            self.assertEqual(report["sampled_pairs_csv"], str(sampled_path))
            with sampled_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            train_rows = [row for row in rows if row["partition"] == "train"]
            validation_rows = [row for row in rows if row["partition"] == "validation"]
            self.assertTrue(all(row["left_track_id"] in train and row["right_track_id"] in train for row in train_rows))
            self.assertTrue(all(row["left_track_id"] in validation and row["right_track_id"] in validation for row in validation_rows))
            self.assertFalse(any(row["left_track_id"] in test or row["right_track_id"] in test for row in rows))
            for value in report["source_sha256"].values():
                self.assertEqual(len(value), 64)
            self.assertEqual(report["source_sha256"]["data_audit"], hashlib.sha256(audit_path.read_bytes()).hexdigest())
            self.assertTrue(report["runtime_versions"]["python"] and report["runtime_versions"]["numpy"] and report["runtime_versions"]["pytorch"])
            self.assertIn("scripts/run_m2_audio.py", report["code_sha256"])
            self.assertEqual(set(report["per_seed"]), {"684", "685"})
            for result in report["per_seed"].values():
                self.assertTrue(math.isfinite(result["history"][0]["train_loss"]))
                self.assertTrue(math.isfinite(result["history"][0]["validation_loss"]))
                self.assertTrue(all(math.isfinite(value) for value in result["metrics"].values()))
            self.assertTrue(output.is_file())
            self.assertTrue((root / "m2_audio_training_curves.csv").is_file())
            checkpoint_path = checkpoint_dir / "m2_audio_seed_684.pt"
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["fit_train_ids"], train)
            self.assertEqual(checkpoint["code_sha256"], report["code_sha256"])

            args = [
                "--epochs", "1", "--seeds", "684", "--data-dir", str(data_dir),
                "--audit", str(audit_path), "--audio-cache", str(cache_path),
                "--audio-metadata", str(metadata_path), "--output", str(root / "rejected.json"),
                "--checkpoint-dir", str(root / "rejected-checkpoints"),
            ]
            for bad_normalization in ({"train_ids": ["NOT-TRAIN"]}, None):
                metadata = json.loads(metadata_path.read_text())
                if bad_normalization is None:
                    del metadata["normalization"]
                else:
                    metadata["normalization"] = bad_normalization
                metadata_path.write_text(json.dumps(metadata))
                with patch.object(runner, "parse_msd", side_effect=AssertionError("parsed features before provenance validation")):
                    with self.assertRaisesRegex(ValueError, "audio normalization.*train_ids"):
                        main(args)


if __name__ == "__main__":
    unittest.main()

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

import scripts.run_msd_audio_only as runner


class MSDAudioOnlyCliTests(unittest.TestCase):
    def _fixture(self, root):
        train = [f"TRTRAIN{i}" for i in reversed(range(6))]
        validation = [f"TRVALID{i}" for i in reversed(range(4))]
        test = [f"TRTEST{i}" for i in reversed(range(3))]
        all_ids = train + validation + test
        audit = root / "audit.json"
        audit.write_text(json.dumps({"splits": {"artist_disjoint": {"track_ids": {
            "train": train, "validation": validation, "test": test,
        }}}}))
        channels = ([f"pitch_{i}" for i in range(1, 13)]
                    + [f"timbre_{i}" for i in range(1, 13)]
                    + ["loudness_max", "beat_count", "bar_count", "section_count"])
        cache = root / "audio.h5"
        cache_ids = sorted(all_ids)
        values = {track_id: float(index + 1) for index, track_id in enumerate(cache_ids)}
        sequences = np.empty((len(cache_ids), 1024, 28), dtype=np.float32)
        for i, track_id in enumerate(cache_ids):
            sequences[i].fill(values[track_id])
        with h5py.File(cache, "w") as handle:
            handle.create_dataset("track_ids", data=np.asarray(cache_ids, dtype="S"))
            handle.create_dataset("sequences", data=sequences)
            handle.create_dataset("channel_names", data=np.asarray(channels, dtype="S"))
        metadata = root / "audio.json"
        metadata.write_text(json.dumps({"grid_size": 1024, "channel_names": channels,
            "split_ids": {"train": train, "validation": validation, "test": test},
            "normalization": {"train_ids": train}}))
        lastfm = root / "lastfm"
        lastfm.mkdir()
        for ids in (train, validation, test):
            for source, target in ((ids[0], ids[1]), (ids[2], ids[3]) if len(ids) > 3 else (ids[0], ids[1])):
                path = lastfm / f"{source}.json"
                record = json.loads(path.read_text()) if path.exists() else {"track_id": source, "similars": []}
                record["similars"].append([target, 1.0])
                path.write_text(json.dumps(record))
        return train, validation, test, values, audit, cache, metadata, lastfm

    def test_rejects_bad_normalization_before_cache_or_lastfm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, validation, test, _, audit, cache, metadata, lastfm = self._fixture(root)
            args = ["--audit", str(audit), "--audio-cache", str(cache), "--audio-metadata", str(metadata),
                    "--lastfm", str(lastfm), "--output", str(root / "out.json"),
                    "--checkpoint-dir", str(root / "ckpt")]
            for normalization in (None, {"train_ids": ["WRONG"]}, {"train_ids": [123]}):
                record = json.loads(metadata.read_text())
                if normalization is None:
                    record.pop("normalization")
                else:
                    record["normalization"] = normalization
                metadata.write_text(json.dumps(record))
                with patch.object(runner, "load_audio_sequence_cache", side_effect=AssertionError("cache read")), \
                     patch.object(runner, "parse_lastfm", side_effect=AssertionError("Last.fm read")):
                    with self.assertRaisesRegex(ValueError, "audio normalization.*train_ids"):
                        runner.main(args)

    def test_synthetic_audio_only_multi_seed_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, validation, test, values, audit, cache, metadata, lastfm = self._fixture(root)
            output, checkpoints = root / "report.json", root / "checkpoints"
            original_train = runner.train_audio_only_seed
            original_pair_builder = runner.build_pair_pools
            original_relevance = runner.build_test_relevance

            def checked_pairs(edges, ids, genres):
                self.assertEqual(genres, {})
                return original_pair_builder(edges, ids, genres)

            def checked_train(audio, ids, *args, **kwargs):
                self.assertEqual(np.asarray(audio)[:, 0, 0].tolist(), [values[i] for i in ids])
                return original_train(audio, ids, *args, **kwargs)

            def checked_relevance(edges, ids):
                for seed in (684, 685):
                    self.assertTrue((checkpoints / f"msd_audio_only_seed_{seed}.pt").is_file())
                return original_relevance(edges, ids)

            encode = runner._encode
            def checked_encode(model, audio):
                self.assertEqual(np.asarray(audio)[:, 0, 0].tolist(), [values[i] for i in test])
                return encode(model, audio)

            with (patch.object(runner, "build_pair_pools", side_effect=checked_pairs),
                  patch.object(runner, "train_audio_only_seed", side_effect=checked_train),
                  patch.object(runner, "build_test_relevance", side_effect=checked_relevance),
                  patch.object(runner, "_encode", side_effect=checked_encode)):
                runner.main(["--epochs", "1", "--seeds", "684", "685", "--audit", str(audit),
                    "--audio-cache", str(cache), "--audio-metadata", str(metadata), "--lastfm", str(lastfm),
                    "--output", str(output), "--checkpoint-dir", str(checkpoints)])

            report = json.loads(output.read_text())
            self.assertEqual(report["split_counts"], {"train": 6, "validation": 4, "test": 3})
            self.assertEqual(report["training_input_ids"], train + validation)
            self.assertEqual(report["architecture"]["embedding_dim"], 32)
            self.assertEqual(set(report["per_seed"]), {"684", "685"})
            self.assertIn("sparse", report["warning"].lower())
            for values_by_seed in report["per_seed"].values():
                self.assertTrue(math.isfinite(values_by_seed["history"][0]["train_loss"]))
                self.assertTrue(math.isfinite(values_by_seed["history"][0]["validation_loss"]))
                self.assertTrue(all(math.isfinite(v) for v in values_by_seed["metrics"].values()))
            for digest in report["source_sha256"].values():
                self.assertEqual(len(digest), 64)
            self.assertEqual(report["source_sha256"]["audit"], hashlib.sha256(audit.read_bytes()).hexdigest())
            self.assertIn("scripts/run_msd_audio_only.py", report["code_sha256"])
            self.assertEqual(report["normalization"]["train_ids"], train)
            trace = root / "msd_audio_only_sampled_pairs.csv"
            with trace.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
            self.assertEqual(report["sampled_pairs_csv_sha256"], trace_hash)
            self.assertFalse(any(row["left_track_id"] in test or row["right_track_id"] in test for row in rows))
            self.assertTrue((root / "msd_audio_only_training_curves.csv").is_file())
            for seed in (684, 685):
                path = checkpoints / f"msd_audio_only_seed_{seed}.pt"
                self.assertTrue(path.is_file())
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["fit_train_ids"], train)
                self.assertEqual(checkpoint["source_sha256"], report["source_sha256"])
                self.assertEqual(checkpoint["sampled_pairs_csv_sha256"], trace_hash)
            self.assertTrue(report["normalization"]["method"])


if __name__ == "__main__":
    unittest.main()

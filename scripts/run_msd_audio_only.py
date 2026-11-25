"""Run the fixed MSD audio-only Siamese retrieval experiment."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from src.data.parse_lastfm import parse_lastfm
from src.data.preprocess_audio import load_audio_sequence_cache
from src.evaluate import build_test_relevance, evaluate_rankings
from src.model import AudioOnlySiameseEncoder
from src.pairs import build_pair_pools
from src.retrieve import rank_top_k
from src.train_audio_only import train_audio_only_seed


ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 50
SEEDS = [684, 685, 686]
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MARGIN = 1.0
EMBEDDING_DIM = 32
K = 10


def _sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_partitions(split):
    partitions = {name: list(split[name]) for name in ("train", "validation", "test")}
    for name, ids in partitions.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate track IDs in {name} partition")
    names = list(partitions)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = set(partitions[first]) & set(partitions[second])
            if overlap:
                raise ValueError(f"overlap between {first} and {second} partitions: {sorted(overlap)}")
    return partitions


def _runtime_versions():
    return {"python": platform.python_version(), "numpy": str(np.__version__), "pytorch": str(torch.__version__)}


def _encode(model, audio):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(audio, dtype=torch.float32)).cpu().numpy()


def run(audit_path, audio_cache_path, audio_metadata_path, lastfm_path, output_path,
        checkpoint_dir, epochs=EPOCHS, seeds=None):
    audit_path, audio_cache_path, audio_metadata_path, lastfm_path, output_path, checkpoint_dir = map(
        Path, (audit_path, audio_cache_path, audio_metadata_path, lastfm_path, output_path, checkpoint_dir)
    )
    seeds = list(SEEDS if seeds is None else seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    with audit_path.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    partitions = _validate_partitions(audit["splits"]["artist_disjoint"]["track_ids"])
    train_ids, validation_ids, test_ids = (partitions["train"], partitions["validation"], partitions["test"])
    all_ids = train_ids + validation_ids + test_ids

    with audio_metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    metadata_split = _validate_partitions(metadata["split_ids"])
    if any(metadata_split[name] != partitions[name] for name in partitions):
        raise ValueError("audio metadata split_ids do not match audit split")
    normalization = metadata.get("normalization")
    if not isinstance(normalization, dict) or not isinstance(normalization.get("train_ids"), list):
        raise ValueError("audio normalization metadata must be an object with train_ids")
    if any(not isinstance(track_id, str) for track_id in normalization["train_ids"]):
        raise ValueError("audio normalization train_ids must contain only strings")
    normalized_train_ids = [track_id.strip().upper() for track_id in normalization["train_ids"]]
    expected_train_ids = [str(track_id).strip().upper() for track_id in train_ids]
    if normalized_train_ids != expected_train_ids:
        raise ValueError("audio normalization train_ids do not match audit train IDs")

    audio_ids, sequences, _ = load_audio_sequence_cache(
        audio_cache_path, audio_metadata_path, expected_ids=all_ids
    )
    audio_by_id = {track_id: sequences[index] for index, track_id in enumerate(audio_ids)}
    edges = parse_lastfm(lastfm_path)
    train_pools = build_pair_pools(edges, train_ids, {})
    validation_pools = build_pair_pools(edges, validation_ids, {})
    training_ids = train_ids + validation_ids
    training_audio = np.asarray([audio_by_id[track_id] for track_id in training_ids])

    config = {"epochs": int(epochs), "seeds": seeds, "batch_size": BATCH_SIZE,
              "learning_rate": LEARNING_RATE, "optimizer": "Adam", "margin": MARGIN,
              "embedding_dim": EMBEDDING_DIM, "k": K,
              "negative_policy": {"negatives_per_positive": 1, "same_genre_fraction": 0.0,
                                  "allocation": "uniform random negatives only",
                                  "resample_train_each_epoch": True, "validation_sample": "fixed per seed"}}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path = output_path.parent / "msd_audio_only_training_curves.csv"
    pairs_path = output_path.parent / "msd_audio_only_sampled_pairs.csv"
    source_paths = {"audit": audit_path, "audio_cache": audio_cache_path,
                    "audio_metadata": audio_metadata_path, "lastfm_similar": lastfm_path}
    source_sha256 = {name: _sha256(path) for name, path in source_paths.items()}
    code_files = ("scripts/run_msd_audio_only.py", "src/data/preprocess_audio.py", "src/data/parse_lastfm.py",
                  "src/pairs.py", "src/model.py", "src/train_audio_only.py", "src/retrieve.py",
                  "src/evaluate.py", "requirements.txt")
    code_sha256 = {name: _sha256(ROOT / name) for name in code_files}
    runtime_versions = _runtime_versions()
    per_seed = {}
    with pairs_path.open("w", encoding="utf-8", newline="") as stream:
        pair_writer = csv.DictWriter(stream, fieldnames=["seed", "partition", "epoch", "left_track_id", "right_track_id", "label"])
        pair_writer.writeheader()
        for seed in seeds:
            model, history, trace = train_audio_only_seed(
                training_audio, training_ids, train_pools, validation_pools, seed,
                epochs=epochs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, margin=MARGIN,
            )
            per_seed[str(seed)] = {"history": history}
            for epoch, rows in enumerate(trace["train_by_epoch"], start=1):
                for left, right, label in rows:
                    pair_writer.writerow({"seed": seed, "partition": "train", "epoch": epoch,
                                          "left_track_id": left, "right_track_id": right, "label": label})
            for left, right, label in trace["validation"]:
                pair_writer.writerow({"seed": seed, "partition": "validation", "epoch": 0,
                                      "left_track_id": left, "right_track_id": right, "label": label})
            stream.flush()
            torch.save({"state_dict": model.state_dict(), "seed": seed, "config": config,
                        "fit_train_ids": train_ids, "normalization_train_ids": normalized_train_ids,
                        "runtime_versions": runtime_versions, "code_sha256": code_sha256,
                        "source_sha256": source_sha256, "sampled_pairs_csv": str(pairs_path)},
                       checkpoint_dir / f"msd_audio_only_seed_{seed}.pt")
            del trace, model

    sampled_pairs_csv_sha256 = _sha256(pairs_path)
    for seed in seeds:
        path = checkpoint_dir / f"msd_audio_only_seed_{seed}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["sampled_pairs_csv_sha256"] = sampled_pairs_csv_sha256
        torch.save(checkpoint, path)

    # Test features, relevance labels, and embeddings are created only after every fixed-epoch checkpoint.
    test_audio = np.asarray([audio_by_id[track_id] for track_id in test_ids])
    relevance = build_test_relevance(edges, test_ids)
    for seed in seeds:
        path = checkpoint_dir / f"msd_audio_only_seed_{seed}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = AudioOnlySiameseEncoder().cpu()
        model.load_state_dict(checkpoint["state_dict"])
        embeddings = _encode(model, test_audio)
        rankings = {track_id: rank_top_k(embeddings, test_ids, track_id, k=K) for track_id in test_ids}
        per_seed[str(seed)]["metrics"] = evaluate_rankings(rankings, relevance, k=K)

    metrics = [f"Recall@{K}", f"HitRate@{K}", f"MRR@{K}", f"NDCG@{K}"]
    summary = {name: {"mean": float(np.mean([per_seed[str(seed)]["metrics"][name] for seed in seeds])),
                     "sample_std": float(np.std([per_seed[str(seed)]["metrics"][name] for seed in seeds], ddof=1))
                     if len(seeds) > 1 else 0.0} for name in metrics}
    edge_count = len({frozenset((left, right)) for left, targets in relevance.items() for right in targets})
    warning = (f"Sparse-edge exploratory warning: measured test graph contains only {edge_count} positive edges; "
               "metrics do not support a statistical-significance claim.")
    with curves_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        for seed in seeds:
            for row in per_seed[str(seed)]["history"]:
                writer.writerow({"seed": seed, **row})
    report = {"experiment": "MSD_audio_only_siamese_encoder", "exploratory": True,
              "config": config, "split_counts": {name: len(ids) for name, ids in partitions.items()},
              "edge_counts": {"all_lastfm_edges": len(edges), "train_positive_pairs": len(train_pools["positives"]),
                              "validation_positive_pairs": len(validation_pools["positives"]),
                              "test_positive_pairs": edge_count},
              "training_input_ids": training_ids, "fit_train_ids": train_ids,
              "normalization": {"method": "train-only z-score on channels 0–24; channels 25–27 unchanged",
                                "train_ids": normalized_train_ids},
              "test": {"gallery_count": len(test_ids), "query_count": len(relevance), "unique_edge_count": edge_count},
              "architecture": {"audio": "Conv1d(28,32,7,stride=2) + Conv1d(32,64,5,stride=2) + Conv1d(64,64,3,stride=2) + AdaptiveAvgPool1d(16) + Linear(1024,32)",
                              "projection": "L2 normalization", "embedding_dim": EMBEDDING_DIM},
              "per_seed": per_seed, "summary": summary,
              "source_sha256": source_sha256,
              "sampled_pairs_csv_sha256": sampled_pairs_csv_sha256,
              "runtime_versions": runtime_versions, "code_sha256": code_sha256,
              "paths": {"audit": str(audit_path), "audio_cache": str(audio_cache_path),
                        "audio_metadata": str(audio_metadata_path), "lastfm_similar": str(lastfm_path),
                        "output": str(output_path), "checkpoint_dir": str(checkpoint_dir)},
              "training_curves_csv": str(curves_path), "sampled_pairs_csv": str(pairs_path),
              "checkpoint_dir": str(checkpoint_dir), "warning": warning}
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--audit", type=Path, default=ROOT / "results" / "msd_audio_only_audit.json")
    parser.add_argument("--audio-cache", type=Path, default=ROOT / "data" / "processed" / "msd_audio_only_sequences.h5")
    parser.add_argument("--audio-metadata", type=Path, default=ROOT / "data" / "processed" / "msd_audio_only_sequences_meta.json")
    parser.add_argument("--lastfm", type=Path, default=ROOT / "data" / "processed" / "lastfm_similar")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "msd_audio_only_experiment.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "results" / "checkpoints")
    args = parser.parse_args(argv)
    run(args.audit, args.audio_cache, args.audio_metadata, args.lastfm, args.output,
        args.checkpoint_dir, args.epochs, args.seeds)


if __name__ == "__main__":
    main()

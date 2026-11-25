"""Run the fixed, exploratory N0 Siamese retrieval experiment."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from src.data.parse_lastfm import parse_lastfm
from src.data.parse_msd import parse_msd
from src.data.parse_mxm import parse_mxm
from src.data.parse_tagtraum import parse_tagtraum
from src.evaluate import build_test_relevance, evaluate_rankings
from src.features import build_train_fitted_modalities
from src.model import SiameseEncoder
from src.pairs import build_pair_pools
from src.retrieve import rank_top_k
from src.train import train_seed


ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 50
SEEDS = [684, 685, 686]
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MARGIN = 1.0
EMBEDDING_DIM = 64
K = 10
NEGATIVES_PER_POSITIVE = 1
SAME_GENRE_FRACTION = 0.5


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
        for second in names[index + 1 :]:
            overlap = set(partitions[first]) & set(partitions[second])
            if overlap:
                raise ValueError(f"overlap between {first} and {second} partitions: {sorted(overlap)}")
    return partitions


def _json_fit_metadata(metadata):
    return {
        "bpm_mean": float(metadata["bpm_mean"]),
        "bpm_std": float(metadata["bpm_std"]),
        "idf_vocabulary_size": int(metadata["idf"].size),
        "genre_vocabulary": metadata["genre_vocabulary"],
    }


def _runtime_versions():
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "pytorch": str(torch.__version__),
    }


def _encode(model, modalities):
    model.eval()
    with torch.no_grad():
        return model(
            torch.as_tensor(modalities["bpm"], dtype=torch.float32),
            torch.as_tensor(modalities["lyrics"], dtype=torch.float32),
            torch.as_tensor(modalities["genre"], dtype=torch.float32),
        ).cpu().numpy()


def run(data_dir, audit_path, output_path, checkpoint_dir, epochs=EPOCHS, seeds=None):
    data_dir, audit_path, output_path, checkpoint_dir = map(
        Path, (data_dir, audit_path, output_path, checkpoint_dir)
    )
    seeds = list(SEEDS if seeds is None else seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    with audit_path.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    split = audit["splits"]["artist_disjoint"]["track_ids"]
    partitions = _validate_partitions(split)
    train_ids = partitions["train"]
    validation_ids = partitions["validation"]
    test_ids = partitions["test"]

    tracks_path = data_dir / "msd_tracks.csv"
    mxm_path = data_dir / "mxm_counts.csv"
    if not mxm_path.exists():
        mxm_path = data_dir / "mxm_counts"
    genres_path = data_dir / "tagtraum_genres.csv"
    lastfm_path = data_dir / "lastfm_similar"
    tracks = parse_msd(tracks_path)
    lyrics = parse_mxm(mxm_path)
    genres = parse_tagtraum(genres_path)
    edges = parse_lastfm(lastfm_path)

    train_validation_ids = train_ids + validation_ids
    modalities, fit = build_train_fitted_modalities(
        tracks, lyrics, genres, train_ids, train_validation_ids
    )
    train_pools = build_pair_pools(edges, train_ids, genres)
    validation_pools = build_pair_pools(edges, validation_ids, genres)
    fit_record = _json_fit_metadata(fit)
    config = {
        "epochs": int(epochs), "seeds": seeds, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "optimizer": "Adam", "margin": MARGIN,
        "embedding_dim": EMBEDDING_DIM, "k": K,
        "negative_policy": {
            "negatives_per_positive": NEGATIVES_PER_POSITIVE,
            "same_genre_fraction": SAME_GENRE_FRACTION,
            "allocation": {
                "same_genre": "floor(total_negatives / 2)",
                "random": "remainder",
            },
            "resample_train_each_epoch": True,
            "validation_sample": "fixed per seed",
        },
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sampled_pairs_path = output_path.parent / "n0_sampled_pairs.csv"
    runtime_versions = _runtime_versions()
    code_files = (
        "scripts/run_n0.py", "src/features.py", "src/pairs.py", "src/model.py",
        "src/train.py", "src/retrieve.py", "src/evaluate.py", "requirements.txt",
        "src/data/parse_lastfm.py", "src/data/parse_msd.py",
        "src/data/parse_mxm.py", "src/data/parse_tagtraum.py", "src/data/table.py",
    )
    code_sha256 = {name: _sha256(ROOT / name) for name in code_files}
    per_seed = {}
    sampled_pair_rows = []
    for seed in seeds:
        model, history, pair_trace = train_seed(
            modalities, train_validation_ids, train_pools, validation_pools,
            seed, epochs=epochs, batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE, margin=MARGIN,
        )
        per_seed[str(seed)] = {"history": history}
        for epoch, rows in enumerate(pair_trace["train_by_epoch"], start=1):
            sampled_pair_rows.extend(
                {
                    "seed": seed, "partition": "train", "epoch": epoch,
                    "left_track_id": left_id, "right_track_id": right_id, "label": label,
                }
                for left_id, right_id, label in rows
            )
        sampled_pair_rows.extend(
            {
                "seed": seed, "partition": "validation", "epoch": 0,
                "left_track_id": left_id, "right_track_id": right_id, "label": label,
            }
            for left_id, right_id, label in pair_trace["validation"]
        )
        torch.save({
            "state_dict": model.state_dict(), "seed": seed, "config": config,
            "fit_metadata": fit_record, "fit_train_ids": train_ids,
            "runtime_versions": runtime_versions, "code_sha256": code_sha256,
            "sampled_pairs_csv": str(sampled_pairs_path),
        }, checkpoint_dir / f"n0_seed_{seed}.pt")

    # Test features and relevance are deliberately kept outside the training call.
    test_modalities, _ = build_train_fitted_modalities(
        tracks, lyrics, genres, train_ids, test_ids
    )
    relevance = build_test_relevance(edges, test_ids)
    for seed in seeds:
        checkpoint = torch.load(
            checkpoint_dir / f"n0_seed_{seed}.pt", map_location="cpu", weights_only=False
        )
        model = SiameseEncoder(num_genres=len(fit["genre_vocabulary"]))
        model.load_state_dict(checkpoint["state_dict"])
        embeddings = _encode(model, test_modalities)
        rankings = {
            query_id: rank_top_k(embeddings, test_ids, query_id, k=K)
            for query_id in test_ids
        }
        per_seed[str(seed)]["metrics"] = evaluate_rankings(rankings, relevance, k=K)

    metric_names = [f"Recall@{K}", f"HitRate@{K}", f"MRR@{K}", f"NDCG@{K}"]
    summary = {}
    for name in metric_names:
        values = [per_seed[str(seed)]["metrics"][name] for seed in seeds]
        summary[name] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    unique_edges = len({frozenset((left, right)) for left, targets in relevance.items() for right in targets})
    warning = (
        f"Exploratory only: this run's measured test graph contains only {unique_edges} "
        f"positive edge{'s' if unique_edges != 1 else ''}; these metrics do not support "
        "a statistical-significance claim."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path = output_path.parent / "n0_training_curves.csv"
    with curves_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        for seed in seeds:
            for row in per_seed[str(seed)]["history"]:
                writer.writerow({"seed": seed, **row})

    with sampled_pairs_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "seed", "partition", "epoch", "left_track_id", "right_track_id", "label"
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_pair_rows)

    source_paths = {
        "data_audit": audit_path, "msd_tracks": tracks_path,
        "mxm_counts": mxm_path, "tagtraum_genres": genres_path,
        "lastfm_similar": lastfm_path,
    }
    report = {
        "experiment": "N0_siamese_encoder", "exploratory": True,
        "config": config,
        "split_counts": {name: len(ids) for name, ids in partitions.items()},
        "training_input_ids": train_validation_ids,
        "test": {"gallery_count": len(test_ids), "query_count": len(relevance), "unique_edge_count": unique_edges},
        "architecture": {
            "bpm": "Linear(1, 8) + ReLU", "lyrics": "Linear(5000, 32) + ReLU",
            "genre": f"Linear({len(fit['genre_vocabulary'])}, 16) + ReLU",
            "projection": "Linear(56, 64) + L2 normalization",
        },
        "fit_metadata": fit_record, "per_seed": per_seed,
        "summary": summary,
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "runtime_versions": runtime_versions,
        "code_sha256": code_sha256,
        "paths": {
            "data_dir": str(data_dir), "audit": str(audit_path),
            "output": str(output_path), "checkpoint_dir": str(checkpoint_dir),
        },
        "training_curves_csv": str(curves_path),
        "sampled_pairs_csv": str(sampled_pairs_path),
        "checkpoint_dir": str(checkpoint_dir),
        "warning": warning,
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--audit", type=Path, default=ROOT / "results" / "data_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "n0_experiment.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "results" / "checkpoints")
    args = parser.parse_args(argv)
    run(args.data_dir, args.audit, args.output, args.checkpoint_dir, args.epochs, args.seeds)


if __name__ == "__main__":
    main()

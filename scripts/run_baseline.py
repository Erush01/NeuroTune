"""Run the fixed, train-only B0 retrieval baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.parse_lastfm import parse_lastfm
from src.data.parse_msd import parse_msd
from src.data.parse_mxm import parse_mxm
from src.data.parse_tagtraum import parse_tagtraum
from src.evaluate import build_test_relevance, evaluate_rankings
from src.features import build_b0_vectors
from src.retrieve import rank_top_k


ROOT = Path(__file__).resolve().parents[1]


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


def run(data_dir, audit_path, output_path, k=10):
    data_dir = Path(data_dir)
    audit_path = Path(audit_path)
    output_path = Path(output_path)
    with audit_path.open(encoding="utf-8") as stream:
        audit = json.load(stream)

    split = audit["splits"]["artist_disjoint"]["track_ids"]
    train_ids = split["train"]
    validation_ids = split["validation"]
    test_ids = split["test"]
    partitions = {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }
    for name, ids in partitions.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate track IDs in {name} partition")
    partition_names = list(partitions)
    for index, first_name in enumerate(partition_names):
        for second_name in partition_names[index + 1 :]:
            overlap = set(partitions[first_name]) & set(partitions[second_name])
            if overlap:
                raise ValueError(
                    f"overlap between {first_name} and {second_name} partitions: "
                    f"{sorted(overlap)}"
                )
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
    vectors, fit = build_b0_vectors(tracks, lyrics, genres, train_ids, test_ids)
    rankings = {query_id: rank_top_k(vectors, test_ids, query_id, k=k) for query_id in test_ids}
    relevance = build_test_relevance(edges, test_ids)
    metrics = evaluate_rankings(rankings, relevance, k=k)
    edge_count = metrics["eligible_positive_edges"]
    warning = (
        f"Exploratory only: this run's measured test graph contains only {edge_count} "
        f"positive edge{'s' if edge_count != 1 else ''}; these metrics do not support "
        "a statistical-significance claim."
    )

    report = {
        "experiment": "fixed_b0",
        "exploratory": True,
        "seed": audit["seed"],
        "split_counts": {name: len(ids) for name, ids in split.items()},
        "catalog_count": len(test_ids),
        "k": k,
        "metrics": metrics,
        "preprocessing": {
            "fit_train_ids": train_ids,
            "bpm_mean": fit["bpm_mean"],
            "bpm_std": fit["bpm_std"],
            "idf_method": "log((1 + train_document_count) / (1 + train_document_frequency)) + 1",
            "idf_vocabulary_size": int(fit["idf"].size),
            "genre_vocabulary": fit["genre_vocabulary"],
            "feature_weights": {"bpm": 1 / np.sqrt(3), "lyrics": 1 / np.sqrt(3), "genre": 1 / np.sqrt(3)},
        },
        "source_sha256": {
            "data_audit": _sha256(audit_path),
            "msd_tracks": _sha256(tracks_path),
            "mxm_counts": _sha256(mxm_path),
            "tagtraum_genres": _sha256(genres_path),
            "lastfm_similar": _sha256(lastfm_path),
        },
        "warning": warning,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--audit", type=Path, default=ROOT / "results" / "data_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "baseline_b0.json")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)
    run(args.data_dir, args.audit, args.output, args.k)


if __name__ == "__main__":
    main()

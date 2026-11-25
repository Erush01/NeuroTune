"""Validation-only score-fusion analysis for the NeuroTune v1 pilot."""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
from src.evaluate import evaluate_rankings
from src.evaluate_neurotune_v1 import (
    build_neurotune_v1_cosine_baseline, evaluate_neurotune_v1_retrieval,
)
from src.features_neurotune_v1 import (
    _canonical_hash, _sha256_file, build_neurotune_v1_features,
)
from src.data.parse_mxm import parse_mxm
from src.data.table import read_rows
from src.model_v1 import NeuroTuneV1Encoder

SEEDS = (684, 685, 686)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
METRIC_NAMES = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10")
ENDPOINT_TOLERANCE = 1e-10
FIT_TRANSFORM_FIELDS = (
    "audio_mean", "audio_std", "audio_fit_rows", "bpm_mean", "bpm_std", "idf",
    "genre_vocabulary", "fit_ids_hash", "split_hash", "code_sha256",
)


def _score_matrix(scores, name):
    try:
        raw_matrix = np.asarray(scores)
        if np.iscomplexobj(raw_matrix):
            raise ValueError(f"{name} must be a finite square matrix")
        matrix = np.asarray(raw_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite square matrix") from error
    if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()):
        raise ValueError(f"{name} must be a finite square matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return matrix


def fuse_score_matrices(model_scores, baseline_scores, model_weight):
    """Return the convex blend of aligned, finite symmetric score matrices."""
    if (isinstance(model_weight, bool) or not isinstance(model_weight, Real)
            or not np.isfinite(model_weight) or not 0 <= model_weight <= 1):
        raise ValueError("model_weight must be finite and between 0 and 1")
    model = _score_matrix(model_scores, "model_scores")
    baseline = _score_matrix(baseline_scores, "baseline_scores")
    if model.shape != baseline.shape:
        raise ValueError("score matrices must have the same shape")
    weight = float(model_weight)
    return weight * model + (1.0 - weight) * baseline


def rank_score_matrix(scores, song_ids):
    """Rank each query against the full gallery, excluding itself and ID-tie-breaking."""
    matrix = _score_matrix(scores, "scores")
    song_ids = list(song_ids)
    if (len(song_ids) != matrix.shape[0] or len(set(song_ids)) != len(song_ids)
            or any(not isinstance(song_id, str) for song_id in song_ids)):
        raise ValueError("song_ids must be unique strings aligned with matrix rows")
    rankings = {}
    for query_index, query_id in enumerate(song_ids):
        candidates = (index for index in range(len(song_ids)) if index != query_index)
        ordered = sorted(candidates, key=lambda index: (-matrix[query_index, index], song_ids[index]))
        rankings[query_id] = [
            {"song_id": song_ids[index], "score": float(matrix[query_index, index])}
            for index in ordered
        ]
    return rankings


def build_validation_relevance(edges, validation_ids):
    """Mirror unique undirected validation edges into query-to-positive sets."""
    song_ids = list(validation_ids)
    if len(set(song_ids)) != len(song_ids) or any(not isinstance(song_id, str) for song_id in song_ids):
        raise ValueError("validation_ids must be unique strings")
    song_set = set(song_ids)
    relevance = {}
    seen_edges = set()
    try:
        iterator = iter(edges)
    except TypeError as error:
        raise ValueError("edges must be iterable") from error
    for edge in iterator:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("each edge must be a pair of song IDs")
        source, target = edge
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("edge endpoints must be strings")
        if source not in song_set or target not in song_set:
            raise ValueError("edge endpoint is not in validation_ids")
        if source == target:
            continue
        canonical = frozenset((source, target))
        if canonical in seen_edges:
            continue
        seen_edges.add(canonical)
        relevance.setdefault(source, set()).add(target)
        relevance.setdefault(target, set()).add(source)
    return relevance


def evaluate_fused_rankings(scores, song_ids, positive_edges, k=10):
    """Rank a fused score matrix and evaluate its unique undirected positives."""
    relevance = build_validation_relevance(positive_edges, song_ids)
    rankings = rank_score_matrix(scores, song_ids)
    metrics = evaluate_rankings(rankings, relevance, k=k)
    metrics["unique_undirected_edge_count"] = sum(map(len, relevance.values())) // 2
    return metrics


def _validation_inputs(data_dir):
    """Load only the train/validation partition values and approved sources."""
    data_dir = Path(data_dir)
    split_path = data_dir / "neurotune_v1_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = split.get("track_ids", split.get("ids", {}))
    train_ids, validation_ids = ids.get("train"), ids.get("validation")
    if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
        raise ValueError("split must provide train and validation track IDs")
    if (not train_ids or not validation_ids or len(set(train_ids)) != len(train_ids)
            or len(set(validation_ids)) != len(validation_ids)
            or set(train_ids) & set(validation_ids)):
        raise ValueError("train and validation IDs must be unique and disjoint")
    tracks = read_rows(data_dir / "ab_catalog_tracks.csv")
    tracks = [{"track_id": row.get("track_id"), "tempo": row.get("tempo", row.get("bpm"))}
              for row in tracks]
    genres = {row["track_id"]: row.get("genre", "") for row in read_rows(data_dir / "ab_catalog_tracks.csv")
              if row.get("track_id")}
    lyrics = parse_mxm(data_dir.parent / "processed" / "mxm_counts")
    edge_path = data_dir / "neurotune_v1_validation_edges.csv"
    with edge_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"track_id_a", "track_id_b"}.issubset(reader.fieldnames):
            raise ValueError("invalid validation-edge file")
        edges = [(row["track_id_a"].strip(), row["track_id_b"].strip()) for row in reader]
    return split, train_ids, validation_ids, tracks, lyrics, genres, edges


def _embed_validation(features, validation_ids, checkpoint, seed):
    if checkpoint.get("variant") != "C" or checkpoint.get("seed") != seed:
        raise ValueError(f"checkpoint does not match C seed {seed}")
    model = NeuroTuneV1Encoder("C", features["genre"].shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = NeuroTuneV1Dataset(validation_ids, features)
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=neurotune_v1_collate):
            outputs.append(model(batch["audio"], batch["mask"], bpm=batch["bpm"], genre=batch["genre"]).cpu().numpy())
    embedding = np.concatenate(outputs, axis=0)
    if not np.isfinite(embedding).all():
        raise ValueError("checkpoint produced non-finite validation embeddings")
    return embedding


def _stage_report(output_dir, report):
    """Publish a one-file namespace without replacing any concurrent destination."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"output namespace already exists: {output_dir}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    staged_report = stage / "experiment.json"
    output_report = output_dir / "experiment.json"
    owned_output_dir = False
    try:
        staged_report.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        # Exclusive mkdir avoids replacing a destination created after the check above.
        os.mkdir(output_dir)
        owned_output_dir = True
        # A hard link fails rather than replacing an existing report file.
        os.link(staged_report, output_report)
    except BaseException:
        if owned_output_dir:
            try:
                if output_report.exists() and os.path.samefile(staged_report, output_report):
                    output_report.unlink()
            except FileNotFoundError:
                pass
            try:
                output_dir.rmdir()
            except OSError:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise
    shutil.rmtree(stage)


def _source_hash(path):
    path = Path(path)
    if path.is_file():
        return _sha256_file(path)
    entries = [(child.relative_to(path).as_posix(), _sha256_file(child))
               for child in sorted(path.rglob("*")) if child.is_file()]
    return _canonical_hash(entries)


def run_validation_fusion(data_dir="data/msd_hf_pilot",
                          checkpoint_dir="results/neurotune_v1_ab/checkpoints",
                          reference_report="results/neurotune_v1_ab/experiment.json",
                          output_dir="results/neurotune_v1_validation_score_fusion"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output namespace already exists: {output_dir}")
    reference = json.loads(Path(reference_report).read_text(encoding="utf-8"))
    c_runs = {run["seed"]: run for run in reference["runs"] if run.get("variant") == "C"}
    if set(c_runs) != set(SEEDS):
        raise ValueError("reference report must contain all three C validation runs")
    split, train_ids, validation_ids, tracks, lyrics, genres, edges = _validation_inputs(data_dir)
    split_path = Path(data_dir) / "neurotune_v1_split.json"
    cache_path = Path(data_dir) / "ab_audio_30s_26ch.h5"
    split_hash = _sha256_file(split_path)
    all_ids = train_ids + validation_ids
    features, manifest = build_neurotune_v1_features(
        cache_path, tracks, lyrics, genres, train_ids=train_ids, song_ids=all_ids,
        split_hash=split_hash)
    expected_fit_hash = _canonical_hash(sorted(train_ids))
    if manifest.get("fit_ids_hash") != expected_fit_hash:
        raise ValueError("feature transforms were not fitted on the declared train IDs")
    positions = {track_id: index for index, track_id in enumerate(all_ids)}
    if len(positions) != len(all_ids) or any(track_id not in positions for track_id in validation_ids):
        raise ValueError("validation feature row lookup failed")
    val_rows = [positions[track_id] for track_id in validation_ids]
    val_features = {name: ([values[index] for index in val_rows] if name == "audio" else values[val_rows])
                    for name, values in features.items()}
    baseline_embeddings = build_neurotune_v1_cosine_baseline(
        val_features, validation_ids, modalities=("audio", "bpm", "genre"))
    baseline_scores = baseline_embeddings @ baseline_embeddings.T
    relevance = build_validation_relevance(edges, validation_ids)
    if len(edges) != 185 or len(relevance) != 238:
        raise ValueError("validation edge/query counts differ from the approved contract")
    expected_baseline = evaluate_fused_rankings(baseline_scores, validation_ids, edges)
    c_matched_baseline = evaluate_neurotune_v1_retrieval(baseline_embeddings, validation_ids, edges)
    for metric in METRIC_NAMES:
        if not np.isclose(expected_baseline[metric], c_matched_baseline[metric], rtol=0,
                          atol=ENDPOINT_TOLERANCE):
            raise ValueError(f"lambda=0 baseline endpoint mismatch: {metric}")
    results, checkpoint_hashes = {}, {}
    for seed in SEEDS:
        checkpoint_path = Path(checkpoint_dir) / f"C_seed_{seed}.pt"
        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
        if checkpoint.get("variant") != "C" or checkpoint.get("seed") != seed:
            raise ValueError(f"checkpoint identity mismatch for seed {seed}")
        saved_manifest = checkpoint.get("transform_manifest", {})
        if any(saved_manifest.get(field) != manifest.get(field) for field in FIT_TRANSFORM_FIELDS):
            raise ValueError(f"checkpoint fit-transform contract mismatch for seed {seed}")
        if checkpoint.get("provenance", {}).get("split") != split_hash or checkpoint.get("provenance", {}).get("cache") != manifest["source_sha256"]["audio_cache"]:
            raise ValueError(f"checkpoint source provenance mismatch for seed {seed}")
        if c_runs[seed].get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"reference checkpoint hash mismatch for seed {seed}")
        checkpoint_hashes[str(seed)] = checkpoint_hash
        model_embedding = _embed_validation(val_features, validation_ids, checkpoint, seed)
        model_scores = model_embedding @ model_embedding.T
        seed_results = {}
        for weight in WEIGHTS:
            metrics = evaluate_fused_rankings(
                fuse_score_matrices(model_scores, baseline_scores, weight), validation_ids, edges)
            seed_results[str(weight)] = {name: metrics[name] for name in METRIC_NAMES} | {
                "query_count": metrics["query_count"], "eligible_positive_edges": metrics["eligible_positive_edges"]}
        for metric in METRIC_NAMES:
            if not np.isclose(seed_results["1.0"][metric], c_runs[seed]["validation"][metric], rtol=0, atol=ENDPOINT_TOLERANCE):
                raise ValueError(f"lambda=1 checkpoint endpoint mismatch for seed {seed}: {metric}")
        results[str(seed)] = seed_results
    means = {str(weight): {metric: float(np.mean([results[str(seed)][str(weight)][metric] for seed in SEEDS]))
                           for metric in METRIC_NAMES} for weight in WEIGHTS}
    selected = max(WEIGHTS, key=lambda weight: (means[str(weight)]["NDCG@10"], -weight))
    root = Path(__file__).resolve().parents[1]
    code_paths = ["scripts/evaluate_neurotune_v1_validation_fusion.py", "src/features_neurotune_v1.py",
                  "src/model_v1.py", "src/data/neurotune_v1_dataset.py", "src/evaluate_neurotune_v1.py"]
    report = {
        "analysis": "validation-only score fusion", "baseline_modalities": ["audio", "bpm", "genre"],
        "seeds": list(SEEDS), "weight_grid": list(WEIGHTS),
        "selection": {"metric": "mean NDCG@10 across seeds", "tie_break": "lower model weight", "weight": selected},
        "counts": {"train_ids": len(train_ids), "validation_tracks": len(validation_ids),
                   "validation_edges": len(edges), "positive_queries": len(relevance)},
        "metrics_by_seed_and_weight": results, "mean_metrics_by_weight": means,
        "endpoint_tolerance": ENDPOINT_TOLERANCE,
        "hashes": {"split": split_hash, "cache": _sha256_file(cache_path),
                   "catalog": _sha256_file(Path(data_dir) / "ab_catalog_tracks.csv"),
                   "lyrics_source": _source_hash(Path(data_dir).parent / "processed" / "mxm_counts"),
                   "validation_edges": _sha256_file(Path(data_dir) / "neurotune_v1_validation_edges.csv"),
                   "train_fit_ids": expected_fit_hash, "checkpoint_sha256": checkpoint_hashes,
                   "source_code": {path: _sha256_file(root / path) for path in code_paths}},
        "provenance_caveats": [
            "Post-hoc validation analysis; the validation partition was already used to select the C checkpoints.",
            "These results are not independent performance evidence and do not support generalization claims.",
            "Third-party A/B source provenance and licensing are unverified; unobserved links are unverified negatives.",
            "No test IDs, edges, or metrics are used for features, fusion, or the report; only train and validation split lists and validation edges are used.",
        ],
    }
    _stage_report(output_dir, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/msd_hf_pilot")
    parser.add_argument("--checkpoint-dir", default="results/neurotune_v1_ab/checkpoints")
    parser.add_argument("--reference-report", default="results/neurotune_v1_ab/experiment.json")
    parser.add_argument("--output-dir", default="results/neurotune_v1_validation_score_fusion")
    args = parser.parse_args(argv)
    run_validation_fusion(args.data_dir, args.checkpoint_dir, args.reference_report, args.output_dir)


if __name__ == "__main__":
    main()

"""Run the fixed three-fold MSD audio-only Siamese retrieval experiment."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.audit_msd_audio_cv import _source_sha256
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


def _runtime_versions():
    return {"python": platform.python_version(), "numpy": str(np.__version__), "pytorch": str(torch.__version__)}


def _encode(model, audio):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(audio, dtype=torch.float32)).cpu().numpy()


def _canonical_id(value):
    return str(value).strip().upper()


def _validate_audit(audit):
    if audit.get("status") != "ready":
        raise ValueError("CV audit status must be ready")
    folds = audit.get("folds")
    if not isinstance(folds, dict) or len(folds) != 3:
        raise ValueError("CV audit must contain exactly three folds")
    ids_by_fold = {}
    all_ids = []
    assigned = {}
    for key, record in folds.items():
        try:
            index = int(key)
            ids = [_canonical_id(track_id) for track_id in record["track_ids"]]
            edges = record["positive_edges"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed CV fold") from exc
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate IDs in fold {key}")
        ids_by_fold[index] = ids
        all_ids.extend(ids)
        assigned.update({track_id: index for track_id in ids})
        for edge in edges:
            if (not isinstance(edge, list) or len(edge) != 2 or
                    edge[0] != _canonical_id(edge[0]) or edge[1] != _canonical_id(edge[1]) or
                    edge[0] >= edge[1] or edge[0] not in set(ids) or edge[1] not in set(ids)):
                raise ValueError(f"noncanonical or out-of-fold positive edge in fold {key}")
    if sorted(ids_by_fold) != [0, 1, 2]:
        raise ValueError("CV fold IDs must be 0, 1, and 2")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("CV folds must be disjoint")
    expected_count = audit.get("catalog_counts", {}).get("eligible_tracks")
    if expected_count != len(all_ids):
        raise ValueError("CV folds do not cover the audited catalog")
    return ids_by_fold, assigned, all_ids


def _check_targets(targets, inputs):
    normalized = [Path(path).resolve() for path in targets]
    if len(normalized) != len(set(normalized)):
        raise ValueError("output paths must be distinct")
    input_paths = {Path(path).resolve() for path in inputs}
    if set(normalized) & input_paths:
        raise ValueError("output paths must not collide with input paths")
    existing = [path for path in normalized if path.exists()]
    if existing:
        raise FileExistsError(f"output target already exists: {existing[0]}")


def _validate_fold_cache(cache_path, metadata_path, all_ids):
    load_audio_sequence_cache(cache_path, metadata_path, expected_ids=all_ids)


def _select_fold_audio(cache_path, metadata_path, all_ids, selected_ids):
    cache_ids, sequences, _ = load_audio_sequence_cache(
        cache_path, metadata_path, expected_ids=all_ids)
    indices = {track_id: index for index, track_id in enumerate(cache_ids)}
    return np.ascontiguousarray(np.asarray([sequences[indices[track_id]] for track_id in selected_ids],
                                           dtype=np.float32))


def _run_impl(audit_path, cache_dir, metadata_dir, lastfm_path, output_path, checkpoint_dir,
              epochs=EPOCHS, seeds=None):
    audit_path, cache_dir, metadata_dir, lastfm_path, output_path, checkpoint_dir = map(
        Path, (audit_path, cache_dir, metadata_dir, lastfm_path, output_path, checkpoint_dir))
    seeds = list(SEEDS if seeds is None else seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    with audit_path.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    audit_sha256 = _sha256(audit_path)
    audited_lastfm_sha256 = audit.get("source_sha256", {}).get("lastfm")
    if not isinstance(audited_lastfm_sha256, str):
        raise ValueError("Last.fm source provenance is missing from the CV audit")
    lastfm_sha256 = _source_sha256(lastfm_path)
    if audited_lastfm_sha256 != lastfm_sha256:
        raise ValueError("Last.fm source provenance does not match the current Last.fm tree")
    ids_by_fold, assigned, all_ids = _validate_audit(audit)
    curves_path = output_path.parent / "msd_audio_only_cv_training_curves.csv"
    pairs_path = output_path.parent / "msd_audio_only_cv_sampled_pairs.csv"
    checkpoint_paths = {fold: {seed: checkpoint_dir / f"msd_audio_only_cv_fold_{fold}_seed_{seed}.pt"
                               for seed in seeds} for fold in ids_by_fold}
    targets = [output_path, curves_path, pairs_path] + [path for group in checkpoint_paths.values() for path in group.values()]
    cache_paths = {fold: cache_dir / f"msd_audio_only_cv_fold_{fold}.h5" for fold in ids_by_fold}
    metadata_paths = {fold: metadata_dir / f"msd_audio_only_cv_fold_{fold}_meta.json" for fold in ids_by_fold}
    _check_targets(targets, [audit_path, lastfm_path, *cache_paths.values(), *metadata_paths.values()])

    normalized_by_fold = {}
    for fold, ids in ids_by_fold.items():
        expected_train = [track_id for other in sorted(ids_by_fold) if other != fold for track_id in ids_by_fold[other]]
        expected_test = ids
        with metadata_paths[fold].open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("audit_sha256") != audit_sha256:
            raise ValueError(f"fold {fold} metadata audit_sha256 does not match current CV audit")
        split_ids = metadata.get("split_ids", {})
        if split_ids.get("train") != expected_train or split_ids.get("validation") != [] or split_ids.get("test") != expected_test:
            raise ValueError(f"fold {fold} metadata split_ids do not match audit")
        normalization = metadata.get("normalization")
        if not isinstance(normalization, dict) or not isinstance(normalization.get("train_ids"), list):
            raise ValueError(f"fold {fold} normalization metadata must contain train_ids")
        if any(not isinstance(track_id, str) for track_id in normalization["train_ids"]):
            raise ValueError(f"fold {fold} normalization train_ids must be strings")
        normalized = list(normalization["train_ids"])
        if normalized != expected_train:
            raise ValueError(f"fold {fold} normalization train_ids do not match expected train IDs")
        _validate_fold_cache(cache_paths[fold], metadata_paths[fold], all_ids)
        normalized_by_fold[fold] = normalized

    edges_by_fold = {fold: [(left, right) for edge in audit["folds"][str(fold)]["positive_edges"]
                            for left, right in [tuple(edge)]] for fold in ids_by_fold}
    config = {"epochs": int(epochs), "seeds": seeds, "batch_size": BATCH_SIZE,
              "learning_rate": LEARNING_RATE, "optimizer": "Adam", "margin": MARGIN,
              "embedding_dim": EMBEDDING_DIM, "k": K,
              "negative_policy": {"negatives_per_positive": 1, "same_genre_fraction": 0.0,
                                  "allocation": "uniform random negatives only", "resample_train_each_epoch": True}}
    source_sha256 = {"audit": audit_sha256, "lastfm_similar": lastfm_sha256}
    for fold in ids_by_fold:
        source_sha256[f"fold_{fold}_cache"] = _sha256(cache_paths[fold])
        source_sha256[f"fold_{fold}_metadata"] = _sha256(metadata_paths[fold])
    code_files = (
        "scripts/run_msd_audio_cv.py", "scripts/audit_msd_audio_cv.py",
        "scripts/preprocess_audio_cv.py", "src/data/audio_only_cv.py",
        "src/data/audio_only_audit.py", "src/data/parse_msd.py",
        "src/data/parse_msd_duplicates.py", "src/data/parse_lastfm.py",
        "src/data/table.py", "src/data/preprocess_audio.py",
        "src/train_audio_only.py", "src/model.py", "src/pairs.py",
        "src/evaluate.py", "src/retrieve.py", "requirements.txt",
    )
    code_sha256 = {name: _sha256(ROOT / name) for name in code_files}
    runtime_versions = _runtime_versions()
    per_fold = {}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("w", encoding="utf-8", newline="") as pairs_stream, curves_path.open(
            "w", encoding="utf-8", newline="") as curves_stream:
        pair_writer = csv.DictWriter(pairs_stream, fieldnames=["fold", "seed", "partition", "epoch", "left_track_id", "right_track_id", "label"])
        curve_writer = csv.DictWriter(curves_stream, fieldnames=["fold", "seed", "epoch", "train_loss", "validation_loss"])
        pair_writer.writeheader()
        curve_writer.writeheader()
        for fold in sorted(ids_by_fold):
            test_ids = ids_by_fold[fold]
            train_ids = [track_id for other in sorted(ids_by_fold) if other != fold for track_id in ids_by_fold[other]]
            train_edges = [edge for other in sorted(ids_by_fold) if other != fold for edge in edges_by_fold[other]]
            train_pools = build_pair_pools([{"source": a, "target": b} for a, b in train_edges], train_ids, {})
            per_fold[str(fold)] = {"train_ids": train_ids, "test_ids": test_ids, "per_seed": {}}
            # Load only this fold's cache, align by ID, then drop the full cache
            # before training on the contiguous training-only copy.
            training_audio = _select_fold_audio(cache_paths[fold], metadata_paths[fold], all_ids, train_ids)
            for seed in seeds:
                model, history, trace = train_audio_only_seed(
                    training_audio, train_ids, train_pools, None, seed, epochs=epochs,
                    batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, margin=MARGIN)
                per_fold[str(fold)]["per_seed"][str(seed)] = {"history": history}
                for epoch, rows in enumerate(trace["train_by_epoch"], 1):
                    for left, right, label in rows:
                        pair_writer.writerow({"fold": fold, "seed": seed, "partition": "train", "epoch": epoch,
                                              "left_track_id": left, "right_track_id": right, "label": label})
                for row in history:
                    curve_writer.writerow({"fold": fold, "seed": seed, **row})
                for stream in (pairs_stream, curves_stream):
                    stream.flush()
                torch.save({"state_dict": model.state_dict(), "fold": fold, "seed": seed, "config": config,
                            "fold_train_ids": train_ids, "fold_test_ids": test_ids,
                            "fit_train_ids": train_ids, "normalization_train_ids": normalized_by_fold[fold],
                            "runtime_versions": runtime_versions, "code_sha256": code_sha256,
                            "source_sha256": source_sha256, "sampled_pairs_csv": str(pairs_path)},
                           checkpoint_paths[fold][seed])
                del trace, model
            del training_audio
    trace_hash = _sha256(pairs_path)
    for fold in sorted(ids_by_fold):
        for seed in seeds:
            path = checkpoint_paths[fold][seed]
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["sampled_pairs_csv_sha256"] = trace_hash
            torch.save(checkpoint, path)

    # All fold/seed checkpoints are complete before any held-out labels or features are touched.
    for fold in sorted(ids_by_fold):
        test_ids = ids_by_fold[fold]
        # Held-out relevance and rows are touched only after all checkpoints have
        # been written and their completed training-trace hash has been added.
        test_edges = [{"source": a, "target": b} for a, b in edges_by_fold[fold]]
        relevance = build_test_relevance(test_edges, test_ids)
        test_audio = _select_fold_audio(cache_paths[fold], metadata_paths[fold], all_ids, test_ids)
        for seed in seeds:
            checkpoint = torch.load(checkpoint_paths[fold][seed], map_location="cpu", weights_only=False)
            model = AudioOnlySiameseEncoder().cpu()
            model.load_state_dict(checkpoint["state_dict"])
            embeddings = _encode(model, test_audio)
            rankings = {track_id: rank_top_k(embeddings, test_ids, track_id, k=K) for track_id in test_ids}
            per_fold[str(fold)]["per_seed"][str(seed)]["metrics"] = evaluate_rankings(rankings, relevance, k=K)
            del embeddings, rankings, model, checkpoint
        per_fold[str(fold)]["test"] = {"gallery_count": len(test_ids), "query_count": len(relevance),
                                        "unique_edge_count": sum(map(len, relevance.values())) // 2}
        del test_audio, relevance, test_edges
    metrics = [f"Recall@{K}", f"HitRate@{K}", f"MRR@{K}", f"NDCG@{K}"]
    observations = [record["per_seed"][str(seed)]["metrics"] for record in per_fold.values() for seed in seeds]
    summary = {metric: {"mean": float(np.mean([row[metric] for row in observations])),
                       "sample_std": float(np.std([row[metric] for row in observations], ddof=1)) if len(observations) > 1 else 0.0}
               for metric in metrics}
    report = {"experiment": "MSD_audio_only_siamese_cross_validation", "exploratory": True,
              "config": config, "fold_count": 3, "per_fold": per_fold, "summary": summary,
              "source_sha256": source_sha256, "code_sha256": code_sha256,
              "runtime_versions": runtime_versions, "sampled_pairs_csv_sha256": trace_hash,
              "paths": {"audit": str(audit_path), "cache_dir": str(cache_dir), "metadata_dir": str(metadata_dir),
                        "lastfm_similar": str(lastfm_path), "output": str(output_path), "checkpoint_dir": str(checkpoint_dir)},
              "training_curves_csv": str(curves_path), "sampled_pairs_csv": str(pairs_path),
              "limitations": ["Three artist-disjoint folds have differing held-out galleries; results are exploratory."],
              "warning": "Fold galleries differ from the old single-split run; these CV metrics are exploratory and not directly comparable."}
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def run(audit_path, cache_dir, metadata_dir, lastfm_path, output_path, checkpoint_dir,
        epochs=EPOCHS, seeds=None):
    """Run CV and remove only newly-created outputs if execution fails."""
    output_path, checkpoint_dir = Path(output_path), Path(checkpoint_dir)
    run_seeds = list(SEEDS if seeds is None else seeds)
    if not run_seeds:
        raise ValueError("seeds must not be empty")
    targets = [output_path,
               output_path.parent / "msd_audio_only_cv_training_curves.csv",
               output_path.parent / "msd_audio_only_cv_sampled_pairs.csv"]
    targets.extend(checkpoint_dir / f"msd_audio_only_cv_fold_{fold}_seed_{seed}.pt"
                   for fold in range(3) for seed in run_seeds)
    existed_before = {path.resolve() for path in targets if path.exists()}
    try:
        return _run_impl(audit_path, cache_dir, metadata_dir, lastfm_path, output_path,
                         checkpoint_dir, epochs, run_seeds)
    except BaseException:
        for path in targets:
            resolved = path.resolve()
            if resolved not in existed_before and path.is_file():
                try:
                    path.unlink()
                except BaseException:
                    # Cleanup is best-effort; preserve the original run failure
                    # and continue removing the remaining targets.
                    pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--audit", type=Path, default=ROOT / "results/msd_audio_only_cv_audit.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--lastfm", type=Path, default=ROOT / "data/processed/lastfm_similar")
    parser.add_argument("--output", type=Path, default=ROOT / "results/msd_audio_only_cv_experiment.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "results/checkpoints")
    args = parser.parse_args(argv)
    run(args.audit, args.cache_dir, args.metadata_dir, args.lastfm, args.output, args.checkpoint_dir, args.epochs, args.seeds)


if __name__ == "__main__":
    main()

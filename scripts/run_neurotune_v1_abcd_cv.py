"""Run fixed, source-separated C-model CV on the verified A-D artist folds."""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from src.data.neurotune_v1_abcd_cv_training import (
    build_abcd_cv_fold_features, build_abcd_cv_fold_schedules, load_abcd_cv_training,
)
from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
from src.evaluate_neurotune_v1_cv import (
    build_neurotune_v1_c_cosine_baseline,
    evaluate_neurotune_v1_source_separated_retrieval,
)
from src.train_neurotune_v1 import train_neurotune_v1
from src.data.neurotune_v1_split import sha256_file

ROOT = Path(__file__).resolve().parents[1]
FOLDS, SEEDS = (0, 1, 2), (684, 685, 686)
EPOCHS, BATCH_SIZE, LEARNING_RATE, MARGIN = 50, 64, 0.001, 0.2
CAVEATS = [
    "The two held-out Last.fm edge sources are incomplete positive-label sets; unobserved links are not verified negatives.",
    "Train-source test labels are held-out edges from a source also used for training in other folds; the official test source is reported separately.",
    "Three fold-level baseline observations are distinct from nine model fold/seed observations; no significance or generalization claims are warranted.",
    "The third-party A-D archive's public provenance and license are unverified; project use was user-authorized.",
]
INPUTS = {
    "audit_path": "data/msd_hf_abcd_pilot/abcd_catalog_audit.json",
    "catalog_path": "data/msd_hf_abcd_pilot/abcd_catalog_tracks.csv",
    "train_edges_path": "data/msd_hf_abcd_pilot/abcd_lastfm_train_edges.csv",
    "test_edges_path": "data/msd_hf_abcd_pilot/abcd_lastfm_test_edges.csv",
    "cache_path": "data/msd_hf_abcd_pilot/abcd_audio_30s_26ch.h5",
    "cache_metadata_path": "data/msd_hf_abcd_pilot/abcd_audio_30s_26ch_meta.json",
    "rejection_path": "data/msd_hf_abcd_pilot/abcd_audio_30s_26ch_rejections.json",
    "ab_catalog_path": "data/msd_hf_pilot/ab_catalog_tracks.csv",
    "ab_split_path": "data/msd_hf_pilot/neurotune_v1_split.json",
    "ab_test_edges_path": "data/msd_hf_pilot/ab_lastfm_test_edges.csv",
    "cv_dir": "data/msd_hf_abcd_pilot/cv",
}
CODE_PATHS = (
    "scripts/run_neurotune_v1_abcd_cv.py", "src/data/neurotune_v1_abcd_cv_training.py",
    "src/data/neurotune_v1_abcd_cv.py", "src/data/neurotune_v1_split.py",
    "src/data/neurotune_v1_dataset.py", "src/features_neurotune_v1.py", "src/triplets.py",
    "src/train_neurotune_v1.py", "src/evaluate_neurotune_v1_cv.py",
    "src/evaluate_neurotune_v1.py", "src/evaluate.py", "src/retrieve.py", "src/model_v1.py",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _path_hash(path):
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return _digest([(child.relative_to(path).as_posix(), sha256_file(child))
                        for child in sorted(path.rglob("*")) if child.is_file()])
    raise FileNotFoundError(path)


def _hash_snapshot(paths):
    return {name: _path_hash(path) for name, path in paths.items()}


def _slice(features, source_ids, target_ids):
    positions = {track_id: i for i, track_id in enumerate(source_ids)}
    selected = [positions[x] for x in target_ids]
    return {key: ([value[i] for i in selected] if isinstance(value, (list, tuple)) else np.asarray(value)[selected])
            for key, value in features.items()}


def _embed(model, dataset):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            batch = neurotune_v1_collate([dataset[i] for i in range(start, min(start + BATCH_SIZE, len(dataset)))])
            chunks.append(model(**{k: batch[k] for k in ("audio", "mask", *model.active_modalities)}).cpu().numpy())
    return np.concatenate(chunks)


def _curve_file(path, rows):
    fields = ("fold", "variant", "seed", "epoch", "train_triplet_loss", "validation_triplet_loss",
              "Recall@10", "NDCG@10", "MRR@10", "query_count", "eligible_positive_edges")
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run(data_root=ROOT, results_root=ROOT / "results", *, _test_only=False, _dependencies=None):
    started = time.monotonic()
    deps = _dependencies or {}
    loader = deps.get("loader", load_abcd_cv_training)
    feature_builder = deps.get("feature_builder", build_abcd_cv_fold_features)
    schedule_builder = deps.get("schedule_builder", build_abcd_cv_fold_schedules)
    trainer = deps.get("trainer", train_neurotune_v1)
    evaluator = deps.get("evaluator", evaluate_neurotune_v1_source_separated_retrieval)
    baseline_builder = deps.get("baseline_builder", build_neurotune_v1_c_cosine_baseline)
    if not _test_only and (deps or loader is not load_abcd_cv_training or feature_builder is not build_abcd_cv_fold_features
                           or schedule_builder is not build_abcd_cv_fold_schedules or trainer is not train_neurotune_v1):
        raise ValueError("production dependencies and protocol are fixed")
    final = Path(results_root) / "neurotune_v1_abcd_cv"
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    input_paths = {name: Path(data_root) / rel for name, rel in INPUTS.items()}
    initial_input_hashes = _hash_snapshot(input_paths)
    initial_manifest_sha256 = sha256_file(input_paths["cv_dir"] / "manifest.json")
    initial_code_hashes = {path: sha256_file(ROOT / path) for path in CODE_PATHS}
    # Loader receives only the verified, fixed input locations; no lyric path is opened or parsed.
    loaded = loader(**input_paths)
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".neurotune_v1_abcd_cv.stage-", dir=parent))
    try:
        (stage / "checkpoints").mkdir()
        records, curves, schedule_hashes, transforms = [], [], {}, {}
        fold_summary, fold_rebuild = [], []
        for fold in loaded["folds"]:
            fold_id = fold["fold"]
            features, transform = feature_builder(loaded, fold)
            if set(features) != {"audio", "bpm", "genre"}:
                raise ValueError("C input must contain audio, BPM, and genre only")
            feature_ids = sorted(fold["train"] + fold["validation"] + fold["test"])
            train_ds = NeuroTuneV1Dataset(fold["train"], _slice(features, feature_ids, fold["train"]), modalities=("bpm", "genre"))
            val_ds = NeuroTuneV1Dataset(fold["validation"], _slice(features, feature_ids, fold["validation"]), modalities=("bpm", "genre"))
            transforms[str(fold_id)] = _digest(transform)
            val_edges = fold["validation_edges"]
            fold_seeds = []
            for seed in SEEDS:
                train_schedule, val_schedule = schedule_builder(loaded, fold, seed)
                schedule_path = stage / f"triplet_schedules/fold_{fold_id}_seed_{seed}.npz"
                schedule_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(schedule_path,
                    train_ids=np.asarray(fold["train"], dtype="U"), validation_ids=np.asarray(fold["validation"], dtype="U"),
                    train_schedule=train_schedule.astype(np.int32, copy=False), validation_schedule=val_schedule.astype(np.int32, copy=False))
                schedule_hash = sha256_file(schedule_path)
                schedule_hashes[f"{fold_id}/{seed}"] = schedule_hash
                checkpoint_rel = f"checkpoints/fold_{fold_id}_C_seed_{seed}.pt"
                split_manifest_sha256 = initial_manifest_sha256
                provenance = {
                    "fold": fold_id, "seed": seed, "variant": "C",
                    "input_sha256": initial_input_hashes, "code_sha256": initial_code_hashes,
                    "split_manifest_sha256": split_manifest_sha256,
                    "transform_sha256": transforms[str(fold_id)], "schedule_sha256": schedule_hash,
                    "data_source_identities": {
                        "audit_sha256": initial_input_hashes["audit_path"],
                        "catalog_sha256": initial_input_hashes["catalog_path"],
                        "cache_sha256": initial_input_hashes["cache_path"],
                        "train_source_sha256": initial_input_hashes["train_edges_path"],
                        "official_test_source_sha256": initial_input_hashes["test_edges_path"],
                        "fold_train_ids_sha256": _digest(fold["train"]),
                        "fold_validation_ids_sha256": _digest(fold["validation"]),
                        "fold_test_ids_sha256": _digest(fold["test"]),
                        "split_seed": loaded["manifest"]["seed"],
                    },
                }
                t0 = time.monotonic()
                result = trainer(train_ds, val_ds, train_schedule, val_schedule,
                    [{"source": a, "target": b} for a, b in val_edges], "C", seed,
                    stage / checkpoint_rel, transform, provenance, epochs=EPOCHS, batch_size=BATCH_SIZE)
                elapsed = time.monotonic() - t0
                if len(result["history"]) != EPOCHS:
                    raise ValueError("trainer must return exactly 50 epoch records")
                curves.extend({"fold": fold_id, "variant": "C", "seed": seed, **row} for row in result["history"])
                report = {"fold": fold_id, "variant": "C", "seed": seed, "selected_epoch": result["best_epoch"],
                          "validation": result["best_metrics"], "runtime_seconds": elapsed,
                          "checkpoint_path": checkpoint_rel, "checkpoint_sha256": sha256_file(stage / checkpoint_rel),
                          "schedule_path": schedule_path.relative_to(stage).as_posix(), "schedule_sha256": schedule_hash,
                          "transform_sha256": transforms[str(fold_id)], "provenance": provenance}
                records.append(report); fold_seeds.append(report)
                del result, train_schedule, val_schedule
            fold_summary.append({"fold": fold_id, "track_counts": {
                "train": len(fold["train"]), "validation": len(fold["validation"]), "test": len(fold["test"])},
                "source_edge_counts": {
                    "train_source_train": len(fold["train_edges"]),
                    "train_source_validation": len(fold["validation_edges"]),
                    "train_source_test": len(fold["test_train_source_edges"]),
                    "official_test_source_test": len(fold["test_official_source_edges"])}})
            fold_rebuild.append(fold)
            del features, train_ds, val_ds

        _curve_file(stage / "training_curves.csv", curves)
        expected = [stage / f"checkpoints/fold_{fold}_C_seed_{seed}.pt" for fold in FOLDS for seed in SEEDS]
        schedules = [stage / f"triplet_schedules/fold_{fold}_seed_{seed}.npz" for fold in FOLDS for seed in SEEDS]
        with (stage / "training_curves.csv").open(encoding="utf-8") as stream:
            curve_file_rows = sum(1 for _ in stream)
        if len(curves) != 450 or not all(p.is_file() for p in expected + schedules) or curve_file_rows != 451:
            raise RuntimeError("test construction barrier failed: checkpoints, schedules, and 450 curves required")

        # The helper's train-fitted transform includes rows for the full fold; do not build test datasets,
        # embeddings, or relevance maps until the complete checkpoint/schedule/curve barrier above.
        for fold in fold_rebuild:
            fold_id = fold["fold"]
            features, transform = feature_builder(loaded, fold)
            if _digest(transform) != transforms[str(fold_id)]:
                raise ValueError("test transform manifest differs from training transform")
            ids = sorted(fold["train"] + fold["validation"] + fold["test"])
            test_features = _slice(features, ids, fold["test"])
            test_ds = NeuroTuneV1Dataset(fold["test"], test_features, modalities=("bpm", "genre"))
            checked_checkpoints = []
            for record in [r for r in records if r["fold"] == fold_id]:
                checkpoint_path = stage / record["checkpoint_path"]
                if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
                    raise ValueError("checkpoint hash differs from training record")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if (checkpoint.get("variant") != "C" or checkpoint.get("seed") != record["seed"]
                        or checkpoint.get("provenance") != record["provenance"]
                        or checkpoint.get("transform_manifest") is None
                        or _digest(checkpoint["transform_manifest"]) != record["transform_sha256"]
                        or _digest(checkpoint["transform_manifest"]) != _digest(transform)):
                    raise ValueError("checkpoint variant, seed, provenance, or transform differs from training record")
                checked_checkpoints.append((record, checkpoint))
            base_embeddings = baseline_builder(test_features, fold["test"])
            baseline = evaluator(base_embeddings, fold["test"], fold["test_train_source_edges"], fold["test_official_source_edges"])
            fold_summary[fold_id]["baseline_test"] = baseline
            for record, checkpoint in checked_checkpoints:
                from src.model_v1 import NeuroTuneV1Encoder
                model = NeuroTuneV1Encoder("C", test_features["genre"].shape[1]).cpu()
                model.load_state_dict(checkpoint["model_state_dict"])
                embeddings = _embed(model, test_ds)
                record["test"] = evaluator(embeddings, fold["test"], fold["test_train_source_edges"], fold["test_official_source_edges"])
                record["test_source_counts"] = {
                    "held_out_train_source_edges": len(fold["test_train_source_edges"]),
                    "official_test_source_edges": len(fold["test_official_source_edges"]),
                    "held_out_train_source_queries": len({x for edge in fold["test_train_source_edges"] for x in edge}),
                    "official_test_source_queries": len({x for edge in fold["test_official_source_edges"] for x in edge}),
                }
                del model, checkpoint, embeddings
            fold_runs = [r for r in records if r["fold"] == fold_id]
            fold_summary[fold_id]["model_seed_mean_test"] = {
                source: {metric: float(np.mean([r["test"][source][metric] for r in fold_runs]))
                         for metric in ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10")
                         if metric in fold_runs[0]["test"][source]}
                for source in ("held_out_train_source", "official_test_source")}
            fold_summary[fold_id]["test_source_counts"] = {
                "held_out_train_source_edges": len(fold["test_train_source_edges"]),
                "official_test_source_edges": len(fold["test_official_source_edges"]),
                "held_out_train_source_queries": len({x for edge in fold["test_train_source_edges"] for x in edge}),
                "official_test_source_queries": len({x for edge in fold["test_official_source_edges"] for x in edge}),
            }
            del features, test_features, test_ds, base_embeddings
            del checked_checkpoints

        final_input_hashes = _hash_snapshot(input_paths)
        final_code_hashes = {path: sha256_file(ROOT / path) for path in CODE_PATHS}
        if final_input_hashes != initial_input_hashes or final_code_hashes != initial_code_hashes:
            raise RuntimeError("input or code files changed during run")
        if sha256_file(input_paths["cv_dir"] / "manifest.json") != initial_manifest_sha256:
            raise RuntimeError("CV manifest bytes changed during run")
        input_hashes, code_hashes = initial_input_hashes, initial_code_hashes
        fold_edge_hashes = {p.name: sha256_file(p) for p in sorted(input_paths["cv_dir"].glob("fold_*_edges.csv"))}
        observations = [r["test"] for r in records]
        metrics = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10")
        aggregate = {source: {metric: {"mean": float(np.mean([r[source][metric] for r in observations])),
                                           "sd": float(np.std([r[source][metric] for r in observations]))}
                              for metric in metrics if metric in observations[0][source]}
                     for source in ("held_out_train_source", "official_test_source")}
        baseline_observations = [fold["baseline_test"] for fold in fold_summary]
        aggregate_baseline = {source: {metric: {"mean": float(np.mean([r[source][metric] for r in baseline_observations])),
                                                        "sd": float(np.std([r[source][metric] for r in baseline_observations]))}
                                           for metric in metrics if metric in baseline_observations[0][source]}
                              for source in ("held_out_train_source", "official_test_source")}
        report = {"protocol": {"variant": "C", "modalities": ["audio", "bpm", "genre"], "folds": list(FOLDS), "seeds": list(SEEDS), "epochs": EPOCHS,
                    "split_seed": loaded["manifest"]["seed"],
                    "batch_size": BATCH_SIZE, "margin": MARGIN, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "device": "cpu"},
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "total_seconds": time.monotonic()-started},
                  "hashes": {"inputs": input_hashes, "code": code_hashes,
                             "audit": input_hashes["audit_path"], "catalog": input_hashes["catalog_path"],
                             "sources": {"train": input_hashes["train_edges_path"], "official_test": input_hashes["test_edges_path"]},
                             "cache": input_hashes["cache_path"], "cache_metadata": input_hashes["cache_metadata_path"],
                             "rejection": input_hashes["rejection_path"],
                              "cv_manifest": initial_manifest_sha256,
                             "known_positive_edges": sha256_file(input_paths["cv_dir"] / "known_positive_edges.csv"),
                             "fold_edges": fold_edge_hashes, "transforms": transforms,
                             "schedules": schedule_hashes, "checkpoints": {r["checkpoint_path"]: r["checkpoint_sha256"] for r in records}},
                  "counts": {"model_observations": 9, "fold_level_baseline_observations": 3, "training_curve_rows": len(curves),
                              "catalog_candidate_tracks": len(loaded["catalog"]),
                              "cache_valid_tracks": len(loaded["cache_ids"]),
                              "rejected_tracks": len(loaded["catalog"]) - len(loaded["cache_ids"]),
                             "known_positive_edges": len(loaded["known_positive_edges"]),
                             "legacy_ab_test_source_exclusions": len(loaded["ab_test_source_edges"])},
                  "aggregate_model_test": aggregate, "aggregate_baseline_test": aggregate_baseline,
                  "folds": fold_summary, "runs": records, "caveats": CAVEATS}
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if (_hash_snapshot(input_paths) != initial_input_hashes
                or {path: sha256_file(ROOT / path) for path in CODE_PATHS} != initial_code_hashes):
            raise RuntimeError("input or code files changed during run")
        if sha256_file(input_paths["cv_dir"] / "manifest.json") != initial_manifest_sha256:
            raise RuntimeError("CV manifest bytes changed during run")
        published_files, published_dirs = [], []
        try:
            final.mkdir()
            created = final.stat()
            published_dirs.append((final, created.st_dev, created.st_ino))
            for directory in (final / "checkpoints", final / "triplet_schedules"):
                directory.mkdir()
                created = directory.stat()
                published_dirs.append((directory, created.st_dev, created.st_ino))
            files = sorted(p for p in stage.rglob("*") if p.is_file())
            completion = stage / "experiment.json"
            for source in [p for p in files if p != completion] + [completion]:
                destination = final / source.relative_to(stage)
                os.link(source, destination)
                created = source.stat()
                published_files.append((destination, created.st_dev, created.st_ino))
        except BaseException:
            for path, device, inode in reversed(published_files):
                try:
                    current = path.stat()
                    if (current.st_dev, current.st_ino) == (device, inode):
                        path.unlink()
                except FileNotFoundError:
                    pass
            for path, device, inode in reversed(published_dirs):
                try:
                    current = path.stat()
                    if (current.st_dev, current.st_ino) == (device, inode):
                        path.rmdir()
                except OSError:
                    pass
            raise
        shutil.rmtree(stage)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _run_synthetic(*args, **kwargs):
    """Private dependency-injection seam reserved for synthetic tests."""
    kwargs["_test_only"] = True
    return _run(*args, **kwargs)


def run_experiment():
    return _run(results_root=ROOT / "results")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_experiment()


if __name__ == "__main__":
    main()

"""Run source-verified, artist-grouped official-10K exploratory CV."""

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

import h5py
import numpy as np
import torch

from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
from src.data.parse_msd import parse_msd
from src.data.parse_mxm import parse_mxm
from src.data.parse_tagtraum import parse_tagtraum
from src.evaluate_neurotune_v1 import build_neurotune_v1_cosine_baseline, evaluate_neurotune_v1_retrieval
from src.features_neurotune_v1 import build_neurotune_v1_features
from src.train_neurotune_v1_cv import train_neurotune_v1_cv
from src.triplets import sample_triplet_schedule

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (684, 685, 686)
EPOCHS, BATCH_SIZE = 50, 64
CACHE_VALIDATION_CHUNK_ROWS = 65536
INPUTS = {
    "audit": "results/neurotune_v1_official10k_cv_audit.json",
    "cache": "data/processed/neurotune_v1_official10k_audio_30s_26ch.h5",
    "cache_meta": "data/processed/neurotune_v1_official10k_audio_30s_26ch_meta.json",
    "tracks": "data/processed/msd_tracks.csv", "mxm": "data/processed/mxm_counts",
    "genres": "data/processed/tagtraum_genres.csv", "lastfm_train": "lastfm_train.zip",
    "data_audit": "results/data_audit.json",
}
CODE_PATHS = (
    "scripts/run_neurotune_v1_official10k_cv.py", "scripts/audit_neurotune_v1_official10k_cv.py",
    "scripts/audit_msd_hf_ab.py", "src/data/neurotune_v1_official10k_cv.py",
    "src/data/preprocess_neurotune_v1_official10k.py", "src/data/preprocess_audio_window.py",
    "src/data/parse_msd.py", "src/data/parse_mxm.py", "src/data/parse_tagtraum.py",
    "src/data/neurotune_v1_dataset.py", "src/features_neurotune_v1.py",
    "src/triplets.py", "src/train_neurotune_v1_cv.py", "src/evaluate_neurotune_v1.py",
    "src/evaluate.py", "src/retrieve.py", "src/model_v1.py",
)
CAVEATS = [
    "Exploratory artist-grouped CV on previously used official Last.fm train labels, not an independent external test.",
    "380 cross-fold edges are excluded from every outer test set.",
    "Missing links are not verified negatives.",
    "No significance or generalization claims are warranted.",
]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha(path):
    path = Path(path)
    if path.is_file():
        return _sha(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = child.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _transform_contract(manifest):
    """Compare fitted transforms, excluding song-view-specific provenance hashes."""
    return {key: manifest[key] for key in (
        "audio_mean", "audio_std", "audio_fit_rows", "bpm_mean", "bpm_std", "idf",
        "genre_vocabulary", "fit_ids_hash", "split_hash", "code_sha256")}


def _slice_feature_rows(features, source_ids, target_ids):
    """Align every row-wise modality to target IDs using explicit ID indexing."""
    source_ids, target_ids = list(source_ids), list(target_ids)
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("feature source and target IDs must be unique")
    if not set(target_ids) <= set(source_ids):
        raise ValueError("target feature IDs must be present in source IDs")
    row_index = {track_id: index for index, track_id in enumerate(source_ids)}
    selected = [row_index[track_id] for track_id in target_ids]
    result = {}
    for name, values in features.items():
        if len(values) != len(source_ids):
            raise ValueError(f"feature modality {name!r} does not align with source IDs")
        if isinstance(values, (list, tuple)):
            rows = [values[index] for index in selected]
            result[name] = tuple(rows) if isinstance(values, tuple) else rows
        else:
            result[name] = np.asarray(values)[selected]
    return result


def _fold_split(audit, fold):
    all_ids = sorted(audit["track_ids"])
    test_ids = sorted(fold["track_ids"])
    if len(set(all_ids)) != len(all_ids) or not set(test_ids) <= set(all_ids):
        raise ValueError("audit IDs do not describe a valid catalog")
    pool_ids = sorted(set(all_ids) - set(test_ids))
    if not pool_ids or set(pool_ids) & set(test_ids):
        raise ValueError("invalid fold test/pool partition")
    split = {"fold": fold["fold"], "all_ids": all_ids, "pool_ids": pool_ids, "test_ids": test_ids,
             "train_edges": sorted(map(list, fold["train_edges"])),
             "validation_edges": sorted(map(list, fold["validation_edges"])),
             "test_edges": sorted(map(list, fold["test_edges"])),
             "known_positive_edges": sorted(map(list, fold["non_test_pool_edges"]))}
    for name in ("train_edges", "validation_edges"):
        if any(a not in pool_ids or b not in pool_ids for a, b in split[name]):
            raise ValueError(f"{name} contains held-out test IDs")
    if any(a not in test_ids or b not in test_ids for a, b in split["test_edges"]):
        raise ValueError("test edges must have both endpoints in the held-out fold")
    if any(a not in pool_ids or b not in pool_ids for a, b in split["known_positive_edges"]):
        raise ValueError("non-test pool positives contain held-out test IDs")
    if set(map(tuple, split["train_edges"])) & set(map(tuple, split["validation_edges"])):
        raise ValueError("train and validation labels overlap")
    if set(map(tuple, split["train_edges"])) | set(map(tuple, split["validation_edges"])) != set(map(tuple, split["known_positive_edges"])):
        raise ValueError("train/validation labels do not partition non-test positives")
    return split, _digest_json(split)


def _validate_cache(meta, all_ids, cache_path=None):
    schema = [*(f"pitch_{i}" for i in range(12)), *(f"timbre_{i}" for i in range(12)),
              "loudness_max", "clipped_segment_duration"]
    if (meta.get("track_count") != len(all_ids) or meta.get("interval") != "[0,30)"
            or meta.get("window_seconds") != 30.0 or meta.get("channel_count") != 26
            or meta.get("channel_schema") != schema
            or meta.get("storage", {}).get("features") != "raw unnormalized float32 (T, 26)"):
        raise ValueError("cache metadata violates 30s/26-channel catalog contract")
    source_hashes = meta.get("source_sha256", {})
    if (set(source_hashes) != set(all_ids)
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in source_hashes.values())):
        raise ValueError("cache metadata source hashes must cover all catalog IDs exactly")
    if cache_path is not None:
        with h5py.File(cache_path, "r") as h5:
            feats, offsets = h5["features"], h5["offsets"][:]
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["track_ids"][:]]
            if ids != sorted(all_ids) or feats.shape[1:] != (26,) or feats.dtype != np.dtype("float32"):
                raise ValueError("raw cache HDF5 schema or IDs mismatch")
            if offsets.dtype != np.dtype("int64") or len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(feats) or np.any(np.diff(offsets) <= 0):
                raise ValueError("raw cache ragged offsets mismatch")
            for start in range(0, feats.shape[0], CACHE_VALIDATION_CHUNK_ROWS):
                if not np.isfinite(feats[start:start + CACHE_VALIDATION_CHUNK_ROWS]).all():
                    raise ValueError("raw cache contains nonfinite values")
            if meta.get("segment_count") != len(feats):
                raise ValueError("cache metadata offsets/segment counts mismatch")


def _preflight(data_dir):
    paths = {name: Path(data_dir) / rel for name, rel in INPUTS.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing official inputs: " + ", ".join(missing))
    audit = json.loads(paths["audit"].read_text())
    if len(audit.get("track_ids", [])) != 939 or audit.get("counts", {}).get("source_edges") != 600 or audit.get("counts", {}).get("cross_fold_edges") != 380:
        raise ValueError("audit violates official 939-track/600-edge/380-exclusion counts")
    expected_counts = {
        "test_tracks": [323, 310, 306], "test_edges": [63, 78, 79],
        "non_test_pool_edges": [289, 262, 269], "train_edges": [232, 210, 216],
        "validation_edges": [57, 52, 53],
    }
    folds = audit.get("folds", [])
    if len(folds) != 3 or [fold.get("fold") for fold in folds] != [0, 1, 2]:
        raise ValueError("audit must contain exactly the three canonical folds")
    for key, expected in expected_counts.items():
        if [fold.get("counts", {}).get(key) for fold in folds] != expected:
            raise ValueError(f"audit {key} fold counts differ from approved production counts")
    meta = json.loads(paths["cache_meta"].read_text())
    expected = audit.get("source_sha256", {})
    mapping = {"data_audit": "data_audit", "tracks": "tracks", "mxm": "mxm", "genres": "genres", "lastfm_train": "lastfm_train"}
    if set(expected) != set(mapping) or any(_tree_sha(paths[key]) != expected[source] for key, source in mapping.items()):
        raise ValueError("official input source hashes differ from audit")
    if _sha(paths["audit"]) != meta.get("audit_sha256"):
        raise ValueError("cache metadata audit hash mismatch")
    metadata_code = meta.get("code_sha256", {})
    if set(metadata_code) != {"preprocess_audio_window", "preprocess_neurotune_v1_official10k"}:
        raise ValueError("cache metadata must contain both preprocessing code hashes")
    for logical, digest in metadata_code.items():
        code_path = ROOT / "src/data" / ("preprocess_audio_window.py" if logical == "preprocess_audio_window" else "preprocess_neurotune_v1_official10k.py")
        if _sha(code_path) != digest:
            raise ValueError("cache metadata code hash mismatch")
    _validate_cache(meta, audit.get("track_ids", []), paths["cache"])
    hashes = {"audit": _sha(paths["audit"]), "cache": _sha(paths["cache"]),
              "cache_meta": _sha(paths["cache_meta"]), "sources": {k: _tree_sha(v) for k, v in paths.items()},
              "code": {p: _sha(ROOT / p) for p in CODE_PATHS}}
    return paths, audit, meta, hashes


def _embed(model, dataset):
    model.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            batch = neurotune_v1_collate([dataset[i] for i in range(start, min(start + BATCH_SIZE, len(dataset)))])
            values.append(model(**{k: batch[k] for k in ("audio", "mask", *model.active_modalities)}).cpu().numpy())
    return np.concatenate(values)


def _run_experiment_reserved(data_dir=ROOT, results_dir="results", *, feature_builder=build_neurotune_v1_features,
                            trainer=train_neurotune_v1_cv, _test_only=False):
    if not isinstance(_test_only, bool):
        raise ValueError("_test_only must be boolean")
    if not _test_only and (feature_builder is not build_neurotune_v1_features or trainer is not train_neurotune_v1_cv):
        raise ValueError("production uses fixed official dependencies and settings")
    namespace = "neurotune_v1_official10k_cv_test" if _test_only else "neurotune_v1_official10k_cv"
    final = Path(results_dir) / namespace
    if final.exists():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    paths, audit, meta, hashes = _preflight(data_dir)
    tracks = parse_msd(paths["tracks"])
    lyrics = parse_mxm(paths["mxm"])
    genres = parse_tagtraum(paths["genres"])
    idset = set(audit["track_ids"])
    if ((not _test_only and len(idset) != 939) or not idset <= {t["track_id"] for t in tracks}
            or not idset <= lyrics.keys() or not idset <= genres.keys()):
        raise ValueError("parsed official inputs do not cover exact 939-track catalog")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{namespace}.stage-", dir=final.parent))
    started = time.monotonic()
    try:
        reports, curves, schedule_hashes, split_hashes, transform_hashes = [], [], {}, {}, {}
        fold_payload = []
        for fold in audit["folds"]:
            split, fold_hash = _fold_split(audit, fold)
            fold_index = fold["fold"]
            split_hashes[str(fold_index)] = fold_hash
            features, transform = feature_builder(paths["cache"], tracks, lyrics, genres,
                                                  train_ids=split["pool_ids"], song_ids=split["pool_ids"], split_hash=fold_hash)
            transform_hashes[str(fold_index)] = _digest_json(transform)
            dataset = NeuroTuneV1Dataset(split["pool_ids"], features)
            train_edges, val_edges, known = (split[k] for k in ("train_edges", "validation_edges", "known_positive_edges"))
            validation_schedule = sample_triplet_schedule(split["pool_ids"], val_edges, 1, 684 + fold_index, known_positive_edges=known)
            fold_payload.append((split, fold_hash, transform, dataset, train_edges, val_edges, known, validation_schedule))

        for split, fold_hash, transform, dataset, train_edges, val_edges, known, validation_schedule in fold_payload:
            fold = split["fold"]
            for seed in SEEDS:
                train_schedule = sample_triplet_schedule(split["pool_ids"], train_edges, EPOCHS, seed, known_positive_edges=known)
                filename = f"fold_{fold}_seed_{seed}_schedule.npz"
                schedule_path = stage / filename
                np.savez_compressed(schedule_path, train_ids=np.asarray(split["pool_ids"], dtype="U"),
                                    train_schedule=train_schedule.astype(np.int32),
                                    validation_schedule=validation_schedule.astype(np.int32))
                schedule_hash = _sha(schedule_path)
                schedule_hashes[f"{fold}:{seed}"] = schedule_hash
                checkpoint = stage / "checkpoints" / f"fold_{fold}_seed_{seed}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                provenance_sources = dict(audit["source_sha256"])
                provenance_sources.update({f"raw_hdf5:{track_id}": digest
                                           for track_id, digest in meta["source_sha256"].items()})
                provenance = {"split_sha256": fold_hash, "source_sha256": provenance_sources,
                              "code_sha256": hashes["code"], "cache_sha256": hashes["cache"],
                              "schedule_sha256": schedule_hash}
                result = trainer(dataset, train_edges, val_edges, known, train_schedule, validation_schedule,
                                 seed, checkpoint, transform, provenance,
                                 {"fold": fold, "epochs": EPOCHS, "batch_size": BATCH_SIZE})
                for row in result["history"]:
                    curves.append({"fold": fold, "seed": seed, **row})
                reports.append({"fold": fold, "seed": seed, "selected_epoch": result["best_epoch"],
                                "validation": result["best_metrics"], "checkpoint": f"checkpoints/{checkpoint.name}",
                                "checkpoint_sha256": _sha(checkpoint), "schedule": filename,
                                "schedule_sha256": schedule_hash})
        # Critical leakage barrier: all checkpoints and schedules/curves are persisted first.
        checkpoints = list((stage / "checkpoints").glob("*.pt"))
        if len(checkpoints) != 9:
            raise RuntimeError("all nine selected checkpoints must exist before test evaluation")
        with (stage / "training_curves.csv").open("w", newline="", encoding="utf-8") as f:
            fields = sorted({key for row in curves for key in row})
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(curves)
        curves_sha256 = _sha(stage / "training_curves.csv")

        test_baseline = {}
        for split, fold_hash, transform, _, _, _, _, _ in fold_payload:
            fold = split["fold"]
            full_features, test_transform = feature_builder(paths["cache"], tracks, lyrics, genres,
                train_ids=split["pool_ids"], song_ids=split["all_ids"], split_hash=fold_hash)
            if _transform_contract(test_transform) != _transform_contract(transform):
                raise ValueError("fold transform manifest changed when building held-out test features")
            test_features = _slice_feature_rows(full_features, split["all_ids"], split["test_ids"])
            test_dataset = NeuroTuneV1Dataset(split["test_ids"], test_features)
            test_baseline[str(fold)] = evaluate_neurotune_v1_retrieval(
                build_neurotune_v1_cosine_baseline(test_features, split["test_ids"], modalities=("audio", "bpm", "genre")),
                split["test_ids"], split["test_edges"])
            for report in [r for r in reports if r["fold"] == fold]:
                checkpoint = torch.load(stage / report["checkpoint"], map_location="cpu", weights_only=False)
                from src.model_v1 import NeuroTuneV1Encoder
                model = NeuroTuneV1Encoder("C", test_dataset.features["genre"].shape[1]).cpu()
                model.load_state_dict(checkpoint["model_state_dict"])
                report["test"] = evaluate_neurotune_v1_retrieval(_embed(model, test_dataset), split["test_ids"], split["test_edges"])
        metrics = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10")
        model_aggregate = {metric: float(np.mean([r["test"][metric] for r in reports])) for metric in metrics}
        baseline_aggregate = {metric: float(np.mean([value[metric] for value in test_baseline.values()]))
                              for metric in metrics}
        fold_aggregate = {str(fold): {metric: float(np.mean([r["test"][metric] for r in reports if r["fold"] == fold]))
                                      for metric in metrics} for fold in range(3)}
        seed_aggregate = {str(seed): {metric: float(np.mean([r["test"][metric] for r in reports if r["seed"] == seed]))
                                      for metric in metrics} for seed in SEEDS}
        report = {"run_type": "synthetic_test" if _test_only else "official_exploratory_cv",
                  "protocol": {"variant": "C", "modalities": ["audio", "bpm", "genre"], "seeds": list(SEEDS),
                               "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": 0.001,
                               "margin": 0.2, "device": "cpu"},
                  "counts": audit["counts"], "fold_counts": [f["counts"] for f in audit["folds"]],
                  "runtime": {"seconds": time.monotonic() - started, "python": platform.python_version(),
                              "numpy": np.__version__, "torch": torch.__version__},
                  "source_sha256": audit["source_sha256"],
                  "raw_hdf5_source_sha256": meta["source_sha256"], "hashes": hashes,
                  "split_sha256": split_hashes, "transform_sha256": transform_hashes,
                  "schedule_sha256": schedule_hashes, "cache_metadata_sha256": hashes["cache_meta"],
                  "artifact_sha256": {"training_curves.csv": curves_sha256,
                                      "checkpoints": {r["checkpoint"]: r["checkpoint_sha256"] for r in reports},
                                      "schedules": {r["schedule"]: r["schedule_sha256"] for r in reports}},
                  "baseline_test": test_baseline, "runs": reports,
                  "aggregates": {"description": "unweighted descriptive means; model has nine fold/seed observations, baseline has three folds",
                                 "model_nine_fold_seed_observations": model_aggregate,
                                 "baseline_three_fold_observations": baseline_aggregate,
                                 "model_by_fold": fold_aggregate, "model_by_seed": seed_aggregate},
                  "caveats": CAVEATS}
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if final.exists():
            raise FileExistsError(final)
        os.replace(stage, final)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _reserve_output_namespace(final):
    """Exclusively reserve a result namespace until the caller releases the lock."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    lock = final.with_name(f".{final.name}.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    return lock


def run_experiment(data_dir=ROOT, results_dir="results", *, feature_builder=build_neurotune_v1_features,
                   trainer=train_neurotune_v1_cv, _test_only=False):
    if not isinstance(_test_only, bool):
        raise ValueError("_test_only must be boolean")
    namespace = "neurotune_v1_official10k_cv_test" if _test_only else "neurotune_v1_official10k_cv"
    final = Path(results_dir) / namespace
    lock = _reserve_output_namespace(final)
    try:
        return _run_experiment_reserved(data_dir, results_dir, feature_builder=feature_builder,
                                        trainer=trainer, _test_only=_test_only)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run_experiment(args.data_dir, args.results_dir)


if __name__ == "__main__":
    main()

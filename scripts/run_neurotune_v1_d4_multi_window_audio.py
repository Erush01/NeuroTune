"""Run the isolated, validation-only D4 multi-window audio experiment."""

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
import h5py

from src.data.neurotune_v1_abcd_cv_training import (
    build_abcd_cv_development_features, build_abcd_cv_d4_multi_window_features,
    build_abcd_cv_fold_mxm, load_abcd_cv_training,
)
from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset
from src.data.neurotune_v1_split import sha256_file
from src.evaluate_neurotune_v1 import build_d4_neurotune_v1_cosine_baseline, evaluate_neurotune_v1_retrieval
from src.train_neurotune_v1 import train_neurotune_v1

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/neurotune_v1_d4_multi_window_audio.json"
FOLDS, SEEDS, EPOCHS, BATCH_SIZE = (0, 1, 2), (684, 685, 686), 50, 64
LR, MARGIN, MODALITIES = 0.001, 0.2, ("audio", "bpm", "lyrics", "genre")
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
    "d4_cache_path": "data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch.h5",
    "d4_metadata_path": "data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch_meta.json",
    "d4_rejection_path": "data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch_rejections.json",
    "mxm_path": "data/processed/mxm_counts",
}
EXPECTED_TRACK_COUNT = 13822
D4_CACHE_CODE_PATHS = (
    "src/data/preprocess_msd_hf_abcd.py", "scripts/preprocess_msd_hf_abcd.py",
    "scripts/preprocess_neurotune_v1_d4_audio_windows.py", "src/data/preprocess_audio_window.py",
)
CODE_PATHS = (
    "scripts/run_neurotune_v1_d4_multi_window_audio.py", "config/neurotune_v1_d4_multi_window_audio.json",
    "src/data/neurotune_v1_abcd_cv_training.py", "src/data/neurotune_v1_dataset.py",
    "src/data/neurotune_v1_split.py", "src/features_neurotune_v1.py", "src/features.py",
    "src/train_neurotune_v1.py", "src/model_v1.py", "src/evaluate_neurotune_v1.py",
    "src/evaluate.py", "src/retrieve.py", "src/triplets.py",
    *D4_CACHE_CODE_PATHS,
)
METRICS = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10", "query_count", "eligible_positive_edges")

EXPECTED_CONFIG = {
    "variant": "D", "folds": [0, 1, 2], "seeds": [684, 685, 686], "epochs": 50,
    "batch_size": 64, "optimizer": "Adam", "learning_rate": 0.001, "margin": 0.2, "device": "cpu",
    "modalities": ["audio", "bpm", "lyrics", "genre"], "sampler": "D0_uniform_triplet_schedules",
    "trainer_seed_derivation": "seed",
    "dropout": {"probabilities": {"audio": 0.0, "bpm": 0.0, "lyrics": 0.0, "genre": 0.0}},
    "audio_windows": ["start", "middle", "end"], "window_seconds": 30,
    "window_formula": "start=0; middle=max(0,(duration-30)/2); end=max(0,duration-30)",
    "window_interval": "[start,start+30)", "channel_count": 26,
    "feature_cache_dtype": "float32",
    "feature_cache_normalization": "raw/unnormalized; deferred until train split",
    "pooling": "mean_valid_raw_projected_64d_before_fusion",
    "normalization": "unchanged_fusion_normalization_only_after_fusion",
    "selection": {"primary": "validation NDCG@10", "tie_breakers": ["validation triplet loss", "earliest epoch"]},
    "reference_namespaces": {"D0": "neurotune_v1_full_modal_ablation", "schedules": "neurotune_v1_abcd_cv"},
    "output_namespaces": {"pilot": "neurotune_v1_d4_multi_window_audio_pilot", "full": "neurotune_v1_d4_multi_window_audio"},
    "evaluation_scope": "validation_only_development",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_config(config):
    try:
        same = _canonical(config) == _canonical(EXPECTED_CONFIG)
    except (TypeError, ValueError) as exc:
        raise ValueError("D4 config is not valid canonical JSON") from exc
    if not same:
        raise ValueError("D4 config differs from the frozen protocol")
    return True


def _load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_config(config)
    return config


def _scope(pilot):
    return ((0, 684),) if pilot else tuple((fold, seed) for fold in FOLDS for seed in SEEDS)


def _validate_metrics(metrics):
    if (not isinstance(metrics, dict) or not set(METRICS) <= set(metrics)
            or any(type(value) not in (int, float) or not np.isfinite(value) for value in metrics.values())):
        raise ValueError("validation metrics are malformed or non-finite")


def _select_d0_rows(rows, pairs):
    if not isinstance(rows, list):
        raise ValueError("D0 model_validation rows are missing")
    selected = {}
    full_set = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    wanted = set(pairs)
    if not wanted or not wanted <= full_set:
        raise ValueError("requested D0 pairs are outside the fixed protocol")
    for row in rows:
        if not isinstance(row, dict) or row.get("variant") != "D":
            continue
        if type(row.get("fold")) is not int or type(row.get("seed")) is not int:
            raise ValueError("D0 fold/seed keys must be exact integers")
        pair = (row.get("fold"), row.get("seed"))
        if pair not in full_set:
            raise ValueError("D0 contains an unknown D fold/seed reference")
        if pair in selected:
            raise ValueError("D0 contains duplicate D fold/seed references")
        checkpoint_path = row.get("checkpoint_path", row.get("path"))
        checkpoint_hash = row.get("checkpoint_sha256", row.get("sha256"))
        expected_path = f"checkpoints/fold_{pair[0]}_D_seed_{pair[1]}.pt"
        hashes = (checkpoint_hash, row.get("schedule_sha256"), row.get("transform_sha256"))
        if (type(row.get("selected_epoch")) is not int or not 1 <= row["selected_epoch"] <= EPOCHS
                or checkpoint_path != expected_path
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value) for value in hashes)):
            raise ValueError("D0 reference row has malformed epoch/path/hash")
        row = dict(row, _checkpoint_path=checkpoint_path, _checkpoint_sha256=checkpoint_hash)
        _validate_metrics(row.get("validation"))
        selected[pair] = row
    if set(selected) != full_set:
        raise ValueError("D0 must provide exactly one D reference for every fixed fold/seed pair")
    return {pair: selected[pair] for pair in pairs}


def _hdf5_track_ids(path):
    with h5py.File(path, "r") as cache:
        if "track_ids" not in cache:
            raise ValueError("audio cache is missing track_ids")
        raw = cache["track_ids"][:]
    ids = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("audio cache track_ids must be sorted and unique")
    return ids


def _decode_hdf5_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _validate_d4_cache(inputs):
    metadata = json.loads(inputs["d4_metadata_path"].read_text(encoding="utf-8"))
    rejected = json.loads(inputs["d4_rejection_path"].read_text(encoding="utf-8"))
    audit = json.loads(inputs["audit_path"].read_text(encoding="utf-8"))
    if (metadata.get("status") != "complete" or type(metadata.get("track_count")) is not int
            or metadata.get("track_count") != EXPECTED_TRACK_COUNT
            or type(metadata.get("rejected_track_count")) is not int or metadata.get("rejected_track_count") != 0
            or metadata.get("rejected") != [] or rejected != []):
        raise ValueError("D4 cache metadata must be complete with the expected cohort and zero rejections")
    if (metadata.get("audit_sha256") != sha256_file(inputs["audit_path"])
            or metadata.get("catalog_sha256") != sha256_file(inputs["catalog_path"])
            or metadata.get("source_sha256") != audit.get("source_sha256")):
        raise ValueError("D4 cache sidecar audit/catalog/source provenance mismatch")
    d0_metadata = json.loads(inputs["cache_metadata_path"].read_text(encoding="utf-8"))
    channels = d0_metadata.get("channels")
    if (type(metadata.get("window_seconds")) is not int or metadata["window_seconds"] != 30
            or metadata.get("window_formula") != EXPECTED_CONFIG["window_formula"]
            or metadata.get("interval") != EXPECTED_CONFIG["window_interval"]
            or type(metadata.get("channel_count")) is not int
            or metadata["channel_count"] != EXPECTED_CONFIG["channel_count"]
            or metadata.get("dtype") != EXPECTED_CONFIG["feature_cache_dtype"]
            or metadata.get("normalization") != EXPECTED_CONFIG["feature_cache_normalization"]
            or not isinstance(channels, list) or len(channels) != EXPECTED_CONFIG["channel_count"]
            or metadata.get("channels") != channels):
        raise ValueError("D4 cache sidecar window geometry or schema differs from the frozen contract")
    recorded_code = metadata.get("code_sha256")
    if not isinstance(recorded_code, dict) or set(recorded_code) != set(D4_CACHE_CODE_PATHS):
        raise ValueError("D4 cache code hashes are malformed")
    current_code = {path: sha256_file(ROOT / path) for path in D4_CACHE_CODE_PATHS}
    if recorded_code != current_code:
        raise ValueError("D4 cache preprocessing-code hashes are stale")
    d4_ids = _hdf5_track_ids(inputs["d4_cache_path"])
    d0_ids = _hdf5_track_ids(inputs["cache_path"])
    if (len(d4_ids) != EXPECTED_TRACK_COUNT or metadata.get("valid_track_ids") != d4_ids
            or d4_ids != d0_ids):
        raise ValueError("D4 cache IDs must match metadata and the exact D0 cohort")
    with h5py.File(inputs["d4_cache_path"], "r") as cache:
        if ("features" not in cache or cache["features"].ndim != 2
                or cache["features"].shape[1] != EXPECTED_CONFIG["channel_count"]
                or cache["features"].dtype != np.dtype(EXPECTED_CONFIG["feature_cache_dtype"])):
            raise ValueError("D4 HDF5 feature matrix differs from the frozen channel/dtype contract")
        attrs = cache.attrs
        attr_pairs = {
            "window_seconds": metadata["window_seconds"], "window_formula": metadata["window_formula"],
            "interval": metadata["interval"], "channels": metadata["channels"],
            "normalization": metadata["normalization"], "audit_sha256": metadata["audit_sha256"],
            "catalog_sha256": metadata["catalog_sha256"], "source_sha256": metadata["source_sha256"],
            "code_sha256": metadata["code_sha256"],
        }
        for key, expected in attr_pairs.items():
            if key not in attrs or _canonical(_decode_hdf5_attr(attrs[key])) != _canonical(expected):
                raise ValueError(f"D4 HDF5 {key} attribute differs from its metadata sidecar")
    return d4_ids


def _hash_path(path):
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return hashlib.sha256(_canonical([(p.relative_to(path).as_posix(), sha256_file(p))
                                         for p in sorted(path.rglob("*")) if p.is_file()]).encode()).hexdigest()
    raise FileNotFoundError(path)


def _schedule(path, fold, pair, reference_hash):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != reference_hash:
        raise ValueError(f"D0 schedule hash mismatch for {pair}")
    with np.load(path, allow_pickle=False) as archive:
        required = ("train_ids", "validation_ids", "train_schedule", "validation_schedule")
        if set(archive.files) != set(required):
            raise ValueError("D0 schedule fields are malformed")
        result = {name: archive[name].copy() for name in required}
    if (result["train_ids"].dtype.kind not in "US" or result["validation_ids"].dtype.kind not in "US"
            or list(result["train_ids"].astype(str)) != list(fold["train"])
            or list(result["validation_ids"].astype(str)) != list(fold["validation"])):
        raise ValueError("D0 schedule IDs differ from exact fold order")
    for name, n, epochs in (("train_schedule", len(fold["train"]), EPOCHS),
                            ("validation_schedule", len(fold["validation"]), 1)):
        arr = result[name]
        if (arr.dtype != np.int32 or arr.ndim != 3 or arr.shape[0] != epochs or arr.shape[2] != 3
                or arr.shape[1] == 0 or np.any(arr < 0) or np.any(arr >= n)):
            raise ValueError(f"D0 {name} has invalid dtype, shape, or indices")
    return result, raw


def _take(features, ids, chosen):
    pos = {track_id: i for i, track_id in enumerate(ids)}
    ix = [pos[x] for x in chosen]
    selected = {}
    for key, value in features.items():
        if key == "audio_windows":
            selected[key] = {x: value[x] for x in chosen}
        elif key == "window_valid":
            selected[key] = np.asarray(value)[ix]
        else:
            selected[key] = np.asarray(value)[ix]
    return selected


def _run(data_root=ROOT, results_root=ROOT / "results", *, pilot=False, _test_only=False, _dependencies=None):
    _load_config()
    deps = _dependencies or {}
    if deps and not _test_only:
        raise ValueError("production dependencies and protocol are fixed")
    loader = deps.get("loader", load_abcd_cv_training)
    mxm_builder = deps.get("mxm_builder", build_abcd_cv_fold_mxm)
    feature_builder = deps.get("feature_builder", build_abcd_cv_d4_multi_window_features)
    d0_transform_builder = deps.get("d0_transform_builder", build_abcd_cv_development_features)
    trainer = deps.get("trainer", train_neurotune_v1)
    baseline_builder = deps.get("baseline_builder", build_d4_neurotune_v1_cosine_baseline)
    evaluator = deps.get("evaluator", evaluate_neurotune_v1_retrieval)
    pairs = _scope(pilot)
    root, outroot = Path(data_root), Path(results_root)
    namespace = EXPECTED_CONFIG["output_namespaces"]["pilot" if pilot else "full"]
    final = outroot / namespace
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    d0root, schedroot = outroot / "neurotune_v1_full_modal_ablation", outroot / "neurotune_v1_abcd_cv"
    report_path = d0root / "experiment.json"
    d0report = json.loads(report_path.read_text(encoding="utf-8"))
    if d0report.get("status") != "complete":
        raise ValueError("D0 reference experiment must be complete")
    rows = _select_d0_rows(d0report.get("model_validation"), pairs)
    cosine = d0report.get("cosine_validation")
    if not isinstance(cosine, list):
        raise ValueError("D0 cosine validation references are missing")
    d0_cosine = {}
    folds_for_cosine = list(FOLDS)
    for fold in folds_for_cosine:
        found = [row for row in cosine if isinstance(row, dict) and row.get("variant") == "D" and row.get("fold") == fold]
        if len(found) != 1:
            raise ValueError("D0 requires exactly one D cosine reference per fold")
        _validate_metrics(found[0].get("validation")); d0_cosine[fold] = found[0]
    cp_paths, sp_paths = {}, {}
    for fold, seed in pairs:
        ref = rows[(fold, seed)]
        cp = d0root / f"checkpoints/fold_{fold}_D_seed_{seed}.pt"
        sp = schedroot / f"triplet_schedules/fold_{fold}_seed_{seed}.npz"
        if ref["_checkpoint_path"] != cp.relative_to(d0root).as_posix():
            raise ValueError("D0 checkpoint path is not the fixed D checkpoint path")
        cp_paths[(fold, seed)] = cp; sp_paths[(fold, seed)] = sp
    protected = {"d0_report": report_path, "config": CONFIG_PATH}
    protected.update({f"checkpoint:{f}/{s}": cp_paths[(f, s)] for f, s in pairs})
    protected.update({f"schedule:{f}/{s}": sp_paths[(f, s)] for f, s in pairs})
    protected.update({f"code:{p}": ROOT / p for p in CODE_PATHS})
    inputs = {k: root / v for k, v in INPUTS.items()}
    protected.update({f"input:{k}": v for k, v in inputs.items()})
    before = {k: _hash_path(v) for k, v in protected.items()}
    for pair in pairs:
        if before[f"checkpoint:{pair[0]}/{pair[1]}"] != rows[pair]["_checkpoint_sha256"]:
            raise ValueError("D0 checkpoint SHA-256 differs from its report; weights are not loaded")
        if before[f"schedule:{pair[0]}/{pair[1]}"] != rows[pair]["schedule_sha256"]:
            raise ValueError("D0 schedule SHA-256 differs from its report")
    loader_inputs = {k: v for k, v in inputs.items()
                     if k not in {"d4_cache_path", "d4_metadata_path", "d4_rejection_path", "mxm_path"}}
    loaded = loader(**loader_inputs)
    if [fold.get("fold") for fold in loaded.get("folds", [])] != list(FOLDS):
        raise ValueError("loaded folds must be fixed folds 0, 1, and 2 in order")
    _validate_d4_cache(inputs)
    lyrics, mxm_provenance = mxm_builder({k: loaded[k] for k in ("manifest", "catalog", "cache_path")}, inputs["mxm_path"])
    outroot.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{namespace}.stage-", dir=outroot))
    started = time.monotonic()
    try:
        (stage / "checkpoints").mkdir(); (stage / "triplet_schedules").mkdir()
        records, curves, baselines, copies, transforms, coverage = [], [], [], {}, {}, {}
        for fold_data in loaded["folds"]:
            f = fold_data["fold"]
            fold_pairs = [(ff, ss) for ff, ss in pairs if ff == f]
            if not fold_pairs:
                continue
            ids = sorted(fold_data["train"] + fold_data["validation"])
            test_ids = set(fold_data.get("test", []))
            if set(ids) & test_ids:
                raise ValueError("D4 feature IDs overlap independent-test IDs")
            development_fold = {k: fold_data[k] for k in ("fold", "train", "validation")}
            d0_features, d0_transform = d0_transform_builder(
                {k: loaded[k] for k in ("manifest", "catalog", "cache_path")}, development_fold, lyrics)
            del d0_features
            features, transform = feature_builder({k: loaded[k] for k in ("manifest", "catalog", "cache_path")},
                                                  development_fold,
                                                  lyrics, inputs["d4_cache_path"])
            if (set(features.get("audio_windows", {})) != set(ids)
                    or np.asarray(features.get("window_valid")).shape != (len(ids), 3)):
                raise ValueError("D4 builder returned incomplete or misaligned train/validation features")
            d4_non_audio = {key: value for key, value in transform.items()
                            if key not in {"audio_mean", "audio_std", "audio_fit_rows"}}
            expected_non_audio = {key: value for key, value in d0_transform.items()
                                  if key not in {"audio_mean", "audio_std", "audio_fit_rows"}}
            d4_non_audio["source_sha256"] = {key: value for key, value in d4_non_audio["source_sha256"].items()
                                               if key != "d4_audio_cache"}
            expected_non_audio["source_sha256"] = {key: value for key, value in expected_non_audio["source_sha256"].items()
                                                     if key != "audio_cache"}
            if _canonical(d4_non_audio) != _canonical(expected_non_audio):
                raise ValueError("D4 non-audio transform values/hashes differ from the unchanged D0 transform")
            if transform.get("source_sha256", {}).get("d4_audio_cache") != before["input:d4_cache_path"]:
                raise ValueError("D4 transform audio-cache hash differs from the protected D4 cache")
            d4_transform_hash = hashlib.sha256(_canonical(transform).encode()).hexdigest()
            for _, seed in fold_pairs:
                if d4_transform_hash == rows[(f, seed)]["transform_sha256"]:
                    raise ValueError("D4 audio transform hash unexpectedly matches the D0 transform")
            transforms[str(f)] = transform
            window_valid = np.asarray(features["window_valid"], dtype=np.bool_)
            coverage[str(f)] = {"train_validation_track_count": len(ids),
                                "valid_windows": [int(x) for x in window_valid.sum(axis=0)],
                                "tracks_with_one_two_three_valid_windows": [
                                    int(np.sum(window_valid.sum(axis=1) == count)) for count in (1, 2, 3)]}
            val_ids = list(fold_data["validation"])
            val_features = _take(features, ids, val_ids)
            baseline_embedding = baseline_builder(val_features, val_ids, modalities=MODALITIES)
            baselines.append({"fold": f, "variant": "D", "validation": evaluator(baseline_embedding, val_ids, fold_data["validation_edges"])})
            for _, seed in fold_pairs:
                pair = (f, seed); ref = rows[pair]
                schedule, raw = _schedule(sp_paths[pair], fold_data, pair, ref["schedule_sha256"])
                dest = stage / f"triplet_schedules/fold_{f}_seed_{seed}.npz"
                dest.write_bytes(raw); copies[f"{f}/{seed}"] = sha256_file(dest)
                if copies[f"{f}/{seed}"] != ref["schedule_sha256"]:
                    raise ValueError("copied D0 schedule differs byte-for-byte")
                train_ids, train_feats = list(fold_data["train"]), _take(features, ids, list(fold_data["train"]))
                val_feats = val_features
                train_ds = NeuroTuneV1Dataset(train_ids, train_feats, modalities=("bpm", "lyrics", "genre"))
                val_ds = NeuroTuneV1Dataset(val_ids, val_feats, modalities=("bpm", "lyrics", "genre"))
                cp_rel = f"checkpoints/fold_{f}_D_seed_{seed}.pt"
                candidate = stage / f".candidate_{f}_{seed}.pt"
                provenance = {"fold": f, "seed": seed, "variant": "D", "initialization": "fresh_from_scratch",
                              "schedule_sha256": ref["schedule_sha256"], "d0_checkpoint_loaded": False,
                              "test_features_created": False}
                result = trainer(train_ds, val_ds, schedule["train_schedule"], schedule["validation_schedule"],
                                 [{"source": a, "target": b} for a, b in fold_data["validation_edges"]],
                                 "D", seed, candidate, transform, provenance, epochs=EPOCHS, batch_size=BATCH_SIZE)
                history, best_epoch, best_metrics = result["history"], result["best_epoch"], result["best_metrics"]
                if (len(history) != EPOCHS
                        or any(not isinstance(r, dict) or type(r.get("epoch")) is not int for r in history)
                        or sorted(r["epoch"] for r in history) != list(range(1, EPOCHS + 1))
                        or type(best_epoch) is not int or not 1 <= best_epoch <= EPOCHS):
                    raise ValueError("D4 trainer must return 50 epoch rows and a selected epoch in range")
                if any(any(type(value) not in (int, float) or not np.isfinite(value)
                           for key, value in row.items() if key != "epoch") for row in history):
                    raise ValueError("D4 training curves contain malformed or non-finite values")
                _validate_metrics(best_metrics)
                dest_cp = stage / cp_rel; os.replace(candidate, dest_cp)
                records.append({"fold": f, "variant": "D", "seed": seed, "selected_epoch": int(best_epoch),
                                "validation": best_metrics, "checkpoint_path": cp_rel,
                                "checkpoint_sha256": sha256_file(dest_cp), "schedule_sha256": ref["schedule_sha256"],
                                "transform_sha256": d4_transform_hash,
                                "d0_reference": {"selected_epoch": ref["selected_epoch"], "validation": ref["validation"]}})
                curves.extend({"fold": f, "variant": "D", "seed": seed, **r} for r in history)
        expected_n = len(pairs)
        if len(records) != expected_n or len(curves) != expected_n * EPOCHS or len(copies) != expected_n:
            raise RuntimeError("D4 output count barrier failed")
        with (stage / "training_curves.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = sorted({key for row in curves for key in row}); writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(curves)
        # Store D0 cosine references separately; the only new cosine rows are D4-matched.
        d0_cosine_rows = [dict(d0_cosine[f]) for f in folds_for_cosine]
        report = {"status": "complete", "evaluation_scope": "validation_only_development",
                  "protocol": EXPECTED_CONFIG, "pilot": bool(pilot), "model_validation": records,
                  "cosine_validation": baselines, "d0_cosine_validation_references": d0_cosine_rows,
                  "counts": {"model_validation_observations": expected_n, "new_checkpoints": expected_n,
                             "schedule_copies": expected_n, "training_curve_rows": len(curves),
                             "d4_matched_cosine_observations": len(baselines)},
                  "window_coverage": {"folds": coverage},
                  "hashes": {"protected_before": before, "d4_transform_manifests": transforms,
                      "d4_transform_sha256": {f: hashlib.sha256(_canonical(t).encode()).hexdigest()
                                               for f, t in transforms.items()},
                      "schedule_copies": copies, "new_checkpoints": {r["checkpoint_path"]: r["checkpoint_sha256"] for r in records},
                      "training_curves": sha256_file(stage / "training_curves.csv")},
                  "mxm": mxm_provenance, "runtime": {"seconds": time.monotonic() - started,
                  "python": platform.python_version(), "numpy": np.__version__, "pytorch": torch.__version__},
                  "independent_test_artifacts_created": False,
                  "caveat": "Validation metrics are development/selection evidence only; no independent-test features, embeddings, relevance, or metrics were produced."}
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if {k: _hash_path(v) for k, v in protected.items()} != before:
            raise RuntimeError("protected inputs, references, or code changed during run")
        created_files, created_dirs = [], []
        try:
            final.mkdir()
            stat = final.stat(); created_dirs.append((final, stat.st_dev, stat.st_ino))
            for folder in ("checkpoints", "triplet_schedules"):
                path = final / folder; path.mkdir()
                stat = path.stat(); created_dirs.append((path, stat.st_dev, stat.st_ino))
            files = sorted(p for p in stage.rglob("*") if p.is_file())
            for src in [p for p in files if p.name != "experiment.json"] + [stage / "experiment.json"]:
                destination = final / src.relative_to(stage)
                os.link(src, destination)
                stat = destination.stat(); created_files.append((destination, stat.st_dev, stat.st_ino))
        except BaseException:
            for path, device, inode in reversed(created_files):
                try:
                    stat = path.stat()
                    if (stat.st_dev, stat.st_ino) == (device, inode): path.unlink()
                except FileNotFoundError:
                    pass
            for path, device, inode in reversed(created_dirs):
                try:
                    stat = path.stat()
                    if (stat.st_dev, stat.st_ino) == (device, inode): path.rmdir()
                except OSError:
                    pass
            raise
        shutil.rmtree(stage)
        return report
    except BaseException:
        if stage.exists(): shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--pilot", action="store_true", help="run fold 0, seed 684 only (still 50 epochs)")
    args = parser.parse_args()
    _run(args.data_root, args.results_root, pilot=args.pilot)


if __name__ == "__main__":
    main()

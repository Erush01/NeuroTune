"""Run the isolated, validation-only D3 combined experiment."""

import argparse
import csv
import json
import hashlib
import io
import math
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from src.data.neurotune_v1_abcd_cv_training import (
    build_abcd_cv_development_features, build_abcd_cv_fold_mxm, load_abcd_cv_training,
)
from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset
from src.data.neurotune_v1_split import sha256_file
from src.train_neurotune_v1 import train_neurotune_v1
from src.triplets import MixedSemiHardTripletSampler
from scripts.run_neurotune_v1_training_improvements import _make_schedule_provider


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/neurotune_v1_d3_combined.json"
EXPECTED_CONFIG = {
    "variant": "D",
    "folds": [0, 1, 2],
    "seeds": [684, 685, 686],
    "epochs": 50,
    "batch_size": 64,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "margin": 0.2,
    "device": "cpu",
    "trainer_seed_derivation": "seed",
    "sampler": {
        "strategy": "mixed_semi_hard",
        "semi_hard_fraction": 0.25,
        "uniform_warmup_epochs": 3,
        "refresh_interval": 5,
        "miner_version": "strict_cosine_band_v1",
        "rng": {
            "implementation": "python.random.Random",
            "version": 2,
            "epoch_seed_derivation": "seed*1000003 + fold*1009 + epoch",
        },
        "sampling_kind_codes": {"uniform": 0, "semi_hard": 1, "fallback_uniform": 2},
    },
    "dropout": {
        "probabilities": {"audio": 0.0, "bpm": 0.1, "genre": 0.25, "lyrics": 0.2},
        "placement": "post_branch_activation_before_fusion",
        "scaling": "none",
        "scope": "training_only",
        "rng": {
            "implementation": "torch.Generator(device='cpu')",
            "fold_seed_derivation": "seed * 1000003 + fold * 1009",
            "epoch_seed_derivation": "dropout_seed + epoch_index_zero_based",
            "consumption": "deterministic_triplet_batch_order; anchor_positive_negative; bpm_genre_lyrics",
        },
    },
    "initialization": "independent_per_fold_seed",
    "reference_namespaces": {
        "D0": "neurotune_v1_full_modal_ablation",
        "D1": "neurotune_v1_training_improvements",
        "D2": "neurotune_v1_d2_modality_dropout",
    },
    "output_namespace": "neurotune_v1_d3_combined",
    "evaluation_scope": "validation_only_development",
}


def _validate_config(config):
    try:
        actual = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = json.dumps(EXPECTED_CONFIG, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("D3 config is not valid canonical JSON") from exc
    if actual != expected:
        raise ValueError("D3 config differs from the frozen protocol")
    return True


def _load_config():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    _validate_config(config)
    return config


FOLDS, SEEDS, EPOCHS = (0, 1, 2), (684, 685, 686), 50
METRICS = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10", "query_count", "eligible_positive_edges")
NAMESPACES = {
    "D0": "neurotune_v1_full_modal_ablation",
    "D1": "neurotune_v1_training_improvements",
    "D2": "neurotune_v1_d2_modality_dropout",
}
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
    "cv_dir": "data/msd_hf_abcd_pilot/cv", "mxm_path": "data/processed/mxm_counts",
}
CODE_PATHS = (
    "scripts/run_neurotune_v1_d3_combined.py", "scripts/run_neurotune_v1_training_improvements.py",
    "scripts/run_neurotune_v1_modality_dropout.py", "src/data/neurotune_v1_abcd_cv_training.py",
    "scripts/run_neurotune_v1_full_modal_ablation.py", "src/data/neurotune_v1_abcd_cv.py",
    "src/data/parse_mxm.py", "src/data/neurotune_v1_split.py", "src/data/neurotune_v1_dataset.py",
    "src/features_neurotune_v1.py", "src/features.py", "src/train_neurotune_v1.py",
    "src/evaluate_neurotune_v1.py", "src/evaluate.py", "src/retrieve.py", "src/data/table.py",
    "src/model_v1.py", "src/triplets.py",
)
DROP_PROBS = {"audio": 0.0, "bpm": 0.1, "genre": 0.25, "lyrics": 0.2}
MODALITIES = ("bpm", "lyrics", "genre")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json_keys(value):
    if isinstance(value, dict):
        return {("/".join(map(str, key)) if isinstance(key, tuple) else str(key)): _json_keys(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_json_keys(item) for item in value]
    return value


def _schedule_digest(triplets, type_codes):
    rows = np.asarray(triplets, dtype=np.int32)
    codes = np.asarray(type_codes, dtype=np.uint8)
    if codes.shape != (len(rows),):
        raise ValueError("D1 sampling codes do not align with its schedule")
    digest = hashlib.sha256()
    digest.update(np.asarray(rows.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(codes.shape, dtype=np.int64).tobytes())
    digest.update(rows.tobytes())
    digest.update(codes.tobytes())
    digest.update(json.dumps({"fallback_uniform": 2, "semi_hard": 1, "uniform": 0},
                             sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _safe_artifact(namespace_root, relative_path):
    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        raise ValueError("reference artifact path is not a safe relative path")
    rel = Path(relative_path)
    if any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("reference artifact path contains traversal")
    root = Path(namespace_root).resolve()
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise ValueError("reference artifact path escapes its namespace")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _metrics_valid(metrics):
    return isinstance(metrics, dict) and all(
        key in metrics and type(metrics[key]) in (int, float) and math.isfinite(metrics[key]) for key in METRICS
    )


def _fixed_protocol(report, protocol, *, status="complete"):
    if not isinstance(report, dict) or report.get("status") != status:
        raise ValueError("reference report is incomplete")
    actual = report.get("protocol")
    if not isinstance(actual, dict):
        raise ValueError("reference report protocol is missing")
    selected = {key: actual.get(key) for key in protocol}
    if _canonical(selected) != _canonical(protocol):
        raise ValueError("reference report fixed protocol mismatch")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _select_rows(rows, *, name, by_metadata=False):
    requested = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if not isinstance(rows, list):
        raise ValueError(f"{name} validation rows are missing")
    selected = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if by_metadata:
            if not isinstance(row.get("schedule_file_sha256"), str):
                continue
            if row.get("variant") != "D":
                raise ValueError(f"{name} schedule metadata is attached to a non-D row")
        elif row.get("variant") != "D":
            continue
        fold, seed = row.get("fold"), row.get("seed")
        if type(fold) is not int or type(seed) is not int:
            raise ValueError(f"{name} fold/seed IDs must be exact integers")
        key = (fold, seed)
        if key not in requested:
            raise ValueError(f"{name} contains an unexpected fold/seed row")
        if key in selected:
            raise ValueError(f"{name} contains duplicate fold/seed rows")
        selected[key] = row
    if set(selected) != requested:
        raise ValueError(f"{name} does not contain exactly the fixed fold/seed rows")
    return selected


def _validate_references(results_root):
    """Validate immutable D0/D1/D2 reports and all checkpoint/schedule references."""
    results_root = Path(results_root)
    roots = {name: results_root / namespace for name, namespace in NAMESPACES.items()}
    reports = {}
    report_paths = {}
    for name, root in roots.items():
        report_path = _safe_artifact(root, "experiment.json")
        report_paths[name] = report_path
        reports[name] = json.loads(report_path.read_text(encoding="utf-8"))

    d0_protocol = {"folds": list(FOLDS), "seeds": list(SEEDS), "epochs": 50, "batch_size": 64,
        "optimizer": "Adam", "learning_rate": .001, "triplet_margin": .2, "device": "cpu",
        "trained_variants": ["A", "B", "D"], "modalities": {"A": [], "B": ["bpm"],
        "C": ["bpm", "genre"], "D": ["bpm", "lyrics", "genre"]}}
    _fixed_protocol(reports["D0"], d0_protocol)
    d0_rows = _select_rows(reports["D0"].get("model_validation"), name="D0")
    cosine_rows = reports["D0"].get("cosine_validation")
    if not isinstance(cosine_rows, list):
        raise ValueError("D0 cosine validation rows are missing")
    cosine = {}
    for row in cosine_rows:
        if isinstance(row, dict) and row.get("variant") == "D":
            fold = row.get("fold")
            if type(fold) is not int or fold not in FOLDS or fold in cosine:
                raise ValueError("D0 cosine validation rows are duplicate or unexpected")
            cosine[fold] = row
    if set(cosine) != set(FOLDS):
        raise ValueError("D0 cosine validation must contain exactly one row per fold")

    d1_config_path = _safe_artifact(roots["D1"], "d1_config.json")
    frozen_d1_path = ROOT / "config/neurotune_v1_d1_semi_hard.json"
    frozen_d1 = json.loads(frozen_d1_path.read_text(encoding="utf-8"))
    d1_config = json.loads(d1_config_path.read_text(encoding="utf-8"))
    if d1_config != {"frozen_config": frozen_d1, "run_scope": "full", "actual_folds": list(FOLDS), "actual_seeds": list(SEEDS)}:
        raise ValueError("D1 full-run config wrapper differs from the frozen protocol")
    d1_protocol = {"folds": list(FOLDS), "seeds": list(SEEDS), "epochs": 50, "batch_size": 64,
        "learning_rate": .001, "triplet_margin": .2, "optimizer": "Adam", "device": "cpu",
        "trained_variants": ["D"], "negative_strategy": "mixed_semi_hard",
        "modalities": {"D": ["bpm", "lyrics", "genre"]}, "semi_hard_fraction": .25,
        "uniform_warmup_epochs": 3, "refresh_interval": 5, "python_random_version": 2,
        "epoch_seed_derivation": "seed*1000003 + fold*1009 + epoch", "miner_version": "strict_cosine_band_v1",
        "selection": "validation NDCG@10, then validation loss"}
    _fixed_protocol(reports["D1"], d1_protocol)
    if reports["D1"].get("run_scope") != "full" or reports["D1"].get("evaluation_scope") != "validation_only_development" or reports["D1"].get("independent_test_completed") is not False:
        raise ValueError("D1 report is not complete/full/validation-only")
    d1_hashes = reports["D1"].get("hashes", {})
    if d1_hashes.get("d1_config_sha256") != _sha256(frozen_d1_path):
        raise ValueError("D1 report config hash mismatch")
    if d1_hashes.get("d0_experiment_sha256") != _sha256(report_paths["D0"]):
        raise ValueError("D1 D0 report reference hash mismatch")
    d1_all_rows = reports["D1"].get("model_validation")
    if not isinstance(d1_all_rows, list):
        raise ValueError("D1 model validation rows are missing")
    d1_d0 = [row for row in d1_all_rows if isinstance(row, dict) and row.get("variant") == "D" and not row.get("schedule_file_sha256")]
    d1_ref_map = {}
    for row in d1_d0:
        fold, seed = row.get("fold"), row.get("seed")
        if type(fold) is not int or type(seed) is not int:
            raise ValueError("D1 embedded D0 fold/seed IDs must be exact integers")
        key = (fold, seed)
        if key in d1_ref_map:
            raise ValueError("D1 embedded D0 validation rows are duplicated")
        d1_ref_map[key] = row
    if len(d1_d0) != len(d0_rows) or set(d1_ref_map) != set(d0_rows):
        raise ValueError("D1 D0 model reference rows are incomplete or duplicated")
    for key, row in d0_rows.items():
        if d1_ref_map[key].get("checkpoint_sha256") != row.get("checkpoint_sha256") or d1_ref_map[key].get("schedule_sha256") != row.get("schedule_sha256"):
            raise ValueError("D1 report D0 model reference hashes mismatch")
        pair = f"{key[0]}/{key[1]}"
        if d1_hashes.get("d0_checkpoints", {}).get(pair) != row.get("checkpoint_sha256") or d1_hashes.get("d0_schedules", {}).get(pair) != row.get("schedule_sha256"):
            raise ValueError("D1 report D0 checkpoint/schedule hash manifest mismatch")
    d1_rows = _select_rows(d1_all_rows, name="D1", by_metadata=True)

    d2_config_path = _safe_artifact(roots["D2"], "d2_config.json")
    d2_frozen_path = ROOT / "config/neurotune_v1_d2_modality_dropout.json"
    if d2_config_path.read_bytes() != d2_frozen_path.read_bytes():
        raise ValueError("D2 config differs from the frozen config")
    d2_protocol = json.loads(d2_frozen_path.read_text(encoding="utf-8"))
    _fixed_protocol(reports["D2"], d2_protocol)
    d2_hashes = reports["D2"].get("hashes", {})
    if d2_hashes.get("config") != _sha256(d2_config_path):
        raise ValueError("D2 report config hash mismatch")
    if d2_hashes.get("d0_experiment") != _sha256(report_paths["D0"]):
        raise ValueError("D2 D0 report reference hash mismatch")
    for flag in ("test_evaluation_deferred",):
        if reports["D2"].get(flag) is not True:
            raise ValueError("D2 report is not validation-only")
    for flag in ("test_features_constructed", "test_embeddings_constructed", "test_relevance_constructed", "test_metrics_computed"):
        if reports["D2"].get(flag) is not False:
            raise ValueError("D2 report contains test-derived artifacts")
    d2_rows = _select_rows(reports["D2"].get("records"), name="D2")

    d1_output = d1_hashes.get("outputs", {})
    d2_output = d2_hashes.get("output_files", {})
    if not isinstance(d1_output, dict) or not isinstance(d2_output, dict):
        raise ValueError("D1/D2 output hash manifests are missing")
    pair_keys = {f"{fold}/{seed}" for fold in FOLDS for seed in SEEDS}
    checkpoint_paths = {f"checkpoints/fold_{fold}_D_seed_{seed}.pt" for fold in FOLDS for seed in SEEDS}
    schedule_paths = {f"triplet_schedules/fold_{fold}_seed_{seed}.npz" for fold in FOLDS for seed in SEEDS}
    d1_expected_files = {"d1_config.json", "training_curves.csv", "mining_diagnostics.csv", "validation_metrics.csv"}
    d1_expected_files |= checkpoint_paths | schedule_paths
    if (set(d1_output) != pair_keys | {"checkpoints", "files"}
            or not isinstance(d1_output.get("checkpoints"), dict)
            or set(d1_output["checkpoints"]) != checkpoint_paths
            or not isinstance(d1_output.get("files"), dict)
            or set(d1_output["files"]) != d1_expected_files):
        raise ValueError("D1 output manifests do not cover exactly the fixed artifacts")
    d2_expected_files = {"d2_config.json", "training_curves.csv", "modality_dropout_diagnostics.csv"}
    d2_expected_files |= checkpoint_paths | schedule_paths
    if set(d2_output) != d2_expected_files:
        raise ValueError("D2 output manifest does not cover exactly the fixed artifacts")
    for name, manifest in (("D1", d1_output.get("files")), ("D2", d2_output)):
        if not isinstance(manifest, dict):
            raise ValueError(f"{name} output-file hash manifest is missing")
        for relpath, expected_hash in manifest.items():
            artifact = _safe_artifact(roots[name], relpath)
            if not isinstance(expected_hash, str) or _sha256(artifact) != expected_hash:
                raise ValueError(f"{name} output-file hash mismatch")
    schedule_copies = reports["D0"].get("hashes", {}).get("schedule_copies")
    d0_hashes = {}
    for key, row in d0_rows.items():
        fold, seed = key
        cp_rel = f"checkpoints/fold_{fold}_D_seed_{seed}.pt"
        sch_rel = f"triplet_schedules/fold_{fold}_seed_{seed}.npz"
        cp = _safe_artifact(roots["D0"], row.get("checkpoint_path"))
        sch = _safe_artifact(roots["D0"], sch_rel)
        cp_hash, sch_hash = _sha256(cp), _sha256(sch)
        if row.get("checkpoint_path") != cp_rel or row.get("checkpoint_sha256") != cp_hash or row.get("schedule_sha256") != sch_hash:
            raise ValueError("D0 checkpoint/schedule path or hash mismatch")
        if not isinstance(schedule_copies, dict) or schedule_copies.get(f"{fold}/{seed}") != sch_hash:
            raise ValueError("D0 schedule-copy hash mismatch")
        if type(row.get("selected_epoch")) is not int or not 1 <= row["selected_epoch"] <= EPOCHS or not _metrics_valid(row.get("validation")):
            raise ValueError("D0 selected epoch or validation metrics are invalid")
        d0_hashes[key] = {"report": _sha256(report_paths["D0"]), "checkpoint": cp_hash, "schedule": sch_hash}

    validated_hashes = {"reports": {name: _sha256(path) for name, path in report_paths.items()},
                        "configs": {"D1": _sha256(d1_config_path), "D2": _sha256(d2_config_path)},
                        "models": {}}
    model_maps = {"D0": d0_rows, "D1": d1_rows, "D2": d2_rows}
    for name, rows in (("D1", d1_rows), ("D2", d2_rows)):
        for key, row in rows.items():
            fold, seed = key
            cp_rel = f"checkpoints/fold_{fold}_D_seed_{seed}.pt"
            sch_rel = f"triplet_schedules/fold_{fold}_seed_{seed}.npz"
            if row.get("checkpoint_path") != cp_rel:
                raise ValueError(f"{name} checkpoint path mismatch")
            cp_hash = _sha256(_safe_artifact(roots[name], row["checkpoint_path"]))
            if row.get("checkpoint_sha256") != cp_hash:
                raise ValueError(f"{name} checkpoint hash mismatch")
            if name == "D1" and d1_output.get("checkpoints", {}).get(row["checkpoint_path"]) != cp_hash:
                raise ValueError("D1 output checkpoint hash mismatch")
            if type(row.get("selected_epoch")) is not int or not 1 <= row["selected_epoch"] <= EPOCHS or not _metrics_valid(row.get("validation")):
                raise ValueError(f"{name} selected epoch or validation metrics are invalid")
            sch = _safe_artifact(roots[name], sch_rel)
            sch_hash = _sha256(sch)
            if name == "D1":
                manifest_row = d1_output.get(f"{fold}/{seed}", {})
                if row.get("schedule_file_sha256") != sch_hash or manifest_row.get("archive_sha256") != sch_hash or manifest_row.get("schedule_sha256") != row.get("schedule_sha256"):
                    raise ValueError("D1 schedule archive hashes mismatch")
                with np.load(sch, allow_pickle=False) as archive:
                    triplets, codes, digests = archive["train_schedule"], archive["sampling_type_codes"], archive["schedule_sha256"]
                    if triplets.shape[0] != EPOCHS or codes.shape[0] != EPOCHS or len(digests) != EPOCHS:
                        raise ValueError("D1 inner schedule arrays have invalid dimensions")
                    calculated = [_schedule_digest(t, c) for t, c in zip(triplets, codes)]
                if list(map(str, digests.tolist())) != calculated or _digest(calculated) != row.get("schedule_sha256"):
                    raise ValueError("D1 inner schedule digests mismatch")
            else:
                expected_d0 = d0_hashes[key]["schedule"]
                if sch_hash != expected_d0 or row.get("schedule_sha256") != expected_d0:
                    raise ValueError("D2 schedule does not match the referenced D0 schedule")
                if d2_hashes.get("d0_schedules", {}).get(f"{fold}/{seed}") != expected_d0 or d2_hashes.get("schedule_copies", {}).get(f"{fold}/{seed}") != expected_d0:
                    raise ValueError("D2 D0 schedule reference hash mismatch")
                if d2_output.get(sch_rel) != sch_hash or d2_output.get(row["checkpoint_path"]) != cp_hash:
                    raise ValueError("D2 output-file hash mismatch")
            validated_hashes.setdefault("models", {}).setdefault(name, {})[key] = {"checkpoint": cp_hash, "schedule": sch_hash}

    counts = {}
    for key in d0_rows:
        metric_rows = [d0_rows[key], d1_rows[key], d2_rows[key]]
        if any(not _metrics_valid(row.get("validation")) for row in metric_rows):
            raise ValueError("reference validation metrics are invalid")
        pair_counts = {(row["validation"]["query_count"], row["validation"]["eligible_positive_edges"]) for row in metric_rows}
        if len(pair_counts) != 1:
            raise ValueError("D0/D1/D2 validation query/eligible counts differ")
        counts[key] = next(iter(pair_counts))
    for fold in FOLDS:
        cosine_metrics = cosine[fold].get("validation")
        expected_counts = {(counts[(fold, seed)]) for seed in SEEDS}
        if (not _metrics_valid(cosine_metrics) or len(expected_counts) != 1
                or (cosine_metrics["query_count"], cosine_metrics["eligible_positive_edges"]) != next(iter(expected_counts))):
            raise ValueError("D0 cosine validation counts or metrics mismatch")
    for key, row in d0_rows.items():
        validated_hashes["models"].setdefault("D0", {})[key] = d0_hashes[key]
    return {"models": model_maps, "cosine": cosine, "hashes": validated_hashes}


def _hash_path(path):
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return _digest([(p.relative_to(path).as_posix(), sha256_file(p))
                        for p in sorted(path.rglob("*")) if p.is_file()])
    raise FileNotFoundError(path)


def _slice(features, ids, selected_ids):
    positions = {track_id: i for i, track_id in enumerate(ids)}
    ix = [positions[track_id] for track_id in selected_ids]
    return {key: ([value[i] for i in ix] if isinstance(value, (list, tuple)) else np.asarray(value)[ix])
            for key, value in features.items()}


def _run(data_root=ROOT, results_root=ROOT / "results", *, _test_only=False,
         _dependencies=None, _protocol=None):
    started = time.monotonic()
    deps = _dependencies or {}
    if not _test_only and (deps or _protocol):
        raise ValueError("production dependencies and protocol are fixed")
    config_path = ROOT / "config/neurotune_v1_d3_combined.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    _validate_config(config)
    protocol = {"folds": FOLDS, "seeds": SEEDS, "epochs": EPOCHS, "batch_size": 64}
    if _test_only and _protocol:
        protocol.update(_protocol)
    if set(protocol) != {"folds", "seeds", "epochs", "batch_size"} or not protocol["folds"] or not protocol["seeds"]:
        raise ValueError("invalid test protocol")
    folds, seeds = tuple(protocol["folds"]), tuple(protocol["seeds"])
    epochs, batch_size = protocol["epochs"], protocol["batch_size"]
    if any(type(x) is not int or x <= 0 for x in (epochs, batch_size)):
        raise ValueError("invalid test protocol dimensions")
    loader = deps.get("loader", load_abcd_cv_training)
    mxm_builder = deps.get("mxm_builder", build_abcd_cv_fold_mxm)
    feature_builder = deps.get("feature_builder", build_abcd_cv_development_features)
    trainer = deps.get("trainer", train_neurotune_v1)
    sampler_factory = deps.get("sampler_factory", MixedSemiHardTripletSampler)
    dataset_class = deps.get("Dataset", NeuroTuneV1Dataset)
    root, result_root = Path(data_root), Path(results_root)
    result_root.mkdir(parents=True, exist_ok=True)
    final = result_root / "neurotune_v1_d3_combined"
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    d0root = result_root / NAMESPACES["D0"]
    inputs = {k: root / v for k, v in INPUTS.items()}
    model_paths = {f"{name}:{f}/{s}": result_root / namespace / f"checkpoints/fold_{f}_D_seed_{s}.pt"
                   for name, namespace in NAMESPACES.items() for f in folds for s in seeds}
    schedule_paths = {f"{f}/{s}": d0root / f"triplet_schedules/fold_{f}_seed_{s}.npz"
                      for f in folds for s in seeds}
    protected = {**{f"input:{k}": p for k, p in inputs.items()},
                 **{f"code:{p}": ROOT / p for p in CODE_PATHS},
                 **{f"reference_checkpoint:{k}": p for k, p in model_paths.items()},
                 **{f"d0_schedule:{k}": p for k, p in schedule_paths.items()},
                 "config": config_path}
    # Reference artifact manifests are verified by Task 2. Include complete namespaces and frozen source configs.
    for name, namespace in NAMESPACES.items():
        protected[f"namespace:{name}"] = result_root / namespace
    protected["d1_frozen_config"] = ROOT / "config/neurotune_v1_d1_semi_hard.json"
    protected["d2_frozen_config"] = ROOT / "config/neurotune_v1_d2_modality_dropout.json"
    before = {k: _hash_path(p) for k, p in protected.items()}
    references = _validate_references(result_root)
    manifest_hash = sha256_file(inputs["cv_dir"] / "manifest.json")
    known_hash = sha256_file(inputs["cv_dir"] / "known_positive_edges.csv")
    fold_edge_hashes = {p.name: sha256_file(p) for p in sorted(inputs["cv_dir"].glob("fold_*_edges.csv"))}
    loaded = loader(**{k: v for k, v in inputs.items() if k != "mxm_path"})
    if [f.get("fold") for f in loaded.get("folds", [])] != list(folds):
        raise ValueError("loaded folds differ from requested protocol")
    stage = Path(tempfile.mkdtemp(prefix=".neurotune_v1_d3_combined.stage-", dir=result_root))
    try:
        (stage / "checkpoints").mkdir()
        (stage / "triplet_schedules").mkdir()
        (stage / "d3_config.json").write_bytes(config_bytes)
        records, curves, mining, dropout, comparisons, mxm_by_fold = [], [], [], [], [], {}
        for fold in loaded["folds"]:
            f = fold["fold"]
            ids = sorted(fold["train"] + fold["validation"])
            # Restrict the catalog presented to the feature builder to train/validation only.
            fold_data = {k: loaded[k] for k in ("manifest", "catalog", "cache_path") if k in loaded}
            fold_data["catalog"] = {track_id: loaded["catalog"][track_id] for track_id in ids}
            fold_info = {k: fold[k] for k in ("fold", "train", "validation")}
            lyrics, mxm_provenance = mxm_builder(fold_data, inputs["mxm_path"])
            mxm_by_fold[str(f)] = mxm_provenance
            features, transform = feature_builder(fold_data, fold_info, lyrics)
            if set(features) != {"audio", "bpm", "genre", "lyrics"}:
                raise ValueError("development feature builder must return train/validation modalities only")
            transform_hash = _digest(transform)
            train_ids, val_ids = list(fold["train"]), list(fold["validation"])
            train_ds = dataset_class(train_ids, _slice(features, ids, train_ids), modalities=MODALITIES)
            val_ds = dataset_class(val_ids, _slice(features, ids, val_ids), modalities=MODALITIES)
            positive_edges = [edge for edge in fold["train_edges"] if set(edge) <= set(train_ids)]
            known_edges = [edge for edge in loaded["known_positive_edges"] if set(edge) <= set(train_ids)]
            for seed in seeds:
                key = (f, seed)
                d0row = references["models"]["D0"][key]
                d0_checkpoint = torch.load(model_paths[f"D0:{f}/{seed}"], map_location="cpu", weights_only=False)
                old_transform = d0_checkpoint.get("transform_manifest")
                if not isinstance(old_transform, dict):
                    raise ValueError("D0 checkpoint transform manifest is missing")
                active = ("fit_ids_hash", "audio_mean", "audio_std", "audio_fit_rows", "bpm_mean",
                          "bpm_std", "genre_vocabulary", "split_hash")
                if ({name: old_transform.get(name) for name in active} != {name: transform.get(name) for name in active}
                        or {name: old_transform.get("source_sha256", {}).get(name) for name in ("audio_cache", "split_manifest")}
                        != {name: transform.get("source_sha256", {}).get(name) for name in ("audio_cache", "split_manifest")}):
                    raise ValueError("D0 active transform fields differ from current development transform")
                d0_path = schedule_paths[f"{f}/{seed}"]
                raw = d0_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != d0row["schedule_sha256"]:
                    raise ValueError("verified D0 train schedule hash changed")
                with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                    schedule = {name: archive[name].copy() for name in
                                ("train_ids", "validation_ids", "train_schedule", "validation_schedule")}
                if (list(schedule["train_ids"].astype(str)) != train_ids or
                        list(schedule["validation_ids"].astype(str)) != val_ids):
                    raise ValueError("D0 schedule IDs differ from current fold")
                if schedule["train_schedule"].dtype != np.int32:
                    raise ValueError("D0 train schedule must be int32")
                provider = _make_schedule_provider(train_ids, schedule["train_schedule"], positive_edges,
                    known_edges, sampler_factory, fold=f, seed=seed)
                dropout_seed = seed * 1000003 + f * 1009
                initialization_seed = seed
                cp_rel = f"checkpoints/fold_{f}_D_seed_{seed}.pt"
                candidate = stage / f".candidate_fold_{f}_D_seed_{seed}.pt"
                provenance = {"fold": f, "seed": seed, "variant": "D", "initialization": "independent",
                    "initialization_seed": initialization_seed,
                    "trainer_seed_derivation": "seed",
                    "dropout_seed_derivation": "seed * 1000003 + fold * 1009",
                    "transform_sha256": transform_hash, "d0_schedule_sha256": d0row["schedule_sha256"],
                    "known_positive_edges_sha256_exclusion_only": known_hash,
                    "official_test_edge_source_sha256_exclusion_only": before["input:test_edges_path"],
                    "legacy_test_edge_source_sha256_exclusion_only": before["input:ab_test_edges_path"],
                    "test_source_edges_used_as_targets": False, "dropout_seed": dropout_seed,
                    "d0_reference_checkpoint_sha256": before[f"reference_checkpoint:D0:{f}/{seed}"],
                    "mxm": mxm_provenance}
                result = trainer(train_ds, val_ds, None, schedule["validation_schedule"],
                    [{"source": a, "target": b} for a, b in fold["validation_edges"]], "D", initialization_seed,
                    candidate, transform, provenance, epochs=epochs, batch_size=batch_size,
                    train_schedule_provider=provider, record_epoch_timing=True,
                    bpm_dropout_probability=DROP_PROBS["bpm"], genre_dropout_probability=DROP_PROBS["genre"],
                    lyrics_dropout_probability=DROP_PROBS["lyrics"], dropout_seed=dropout_seed)
                history, best_epoch, metrics = result["history"], result["best_epoch"], result["best_metrics"]
                if len(history) != epochs or not _metrics_valid(metrics):
                    raise ValueError("trainer returned incomplete history or invalid validation metrics")
                for name in ("D0", "D1", "D2"):
                    ref_metrics = references["models"][name][key]["validation"]
                    if (metrics["query_count"], metrics["eligible_positive_edges"]) != (
                            ref_metrics["query_count"], ref_metrics["eligible_positive_edges"]):
                        raise ValueError("D3 validation query/eligible counts differ from references")
                os.replace(candidate, stage / cp_rel)
                archive_path = stage / f"triplet_schedules/fold_{f}_seed_{seed}.npz"
                samples = [entry["sample"] for entry in provider.state["history"]]
                epoch_offsets = [0]
                schedule_parts, code_parts, inner_hashes = [], [], []
                for sample in samples:
                    triplets = np.asarray(sample.triplets, dtype=np.int32)
                    codes = np.asarray([0 if kind == "uniform" else 1 if kind == "semi_hard" else 2
                                        for kind in sample.sampling_kinds], dtype=np.uint8)
                    schedule_parts.append(triplets)
                    code_parts.append(codes)
                    inner_hashes.append(_schedule_digest(triplets, codes))
                    epoch_offsets.append(epoch_offsets[-1] + len(triplets))
                schedule_rows = np.concatenate(schedule_parts, axis=0)
                code_rows = np.concatenate(code_parts)
                np.savez_compressed(archive_path, train_ids=np.asarray(train_ids, dtype="U"),
                    validation_ids=np.asarray(val_ids, dtype="U"), train_triplets=schedule_rows,
                    epoch_offsets=np.asarray(epoch_offsets, dtype=np.int64),
                    sampling_type_codes=code_rows, schedule_sha256=np.asarray(inner_hashes),
                    validation_schedule=schedule["validation_schedule"])
                schedule_hash = _digest(inner_hashes)
                row = {"fold": f, "seed": seed, "variant": "D", "selected_epoch": int(best_epoch),
                    "validation": metrics, "checkpoint_path": cp_rel, "checkpoint_sha256": sha256_file(stage / cp_rel),
                    "schedule_path": f"triplet_schedules/fold_{f}_seed_{seed}.npz",
                    "schedule_file_sha256": sha256_file(archive_path), "schedule_sha256": schedule_hash,
                    "transform_sha256": transform_hash, "initialization": "independent_per_fold_seed"}
                records.append(row)
                comparisons.append({"fold": f, "seed": seed, "D3": metrics,
                    **{name: references["models"][name][key]["validation"] for name in ("D0", "D1", "D2")},
                    "D_cosine": references["cosine"][f]["validation"]})
                for item in history:
                    curves.append({"fold": f, "seed": seed, **item})
                    sample = samples[item["epoch"] - 1]
                    kinds = list(sample.sampling_kinds)
                    mining.append({"fold": f, "seed": seed, "epoch": item["epoch"],
                        "uniform_count": kinds.count("uniform"), "semi_hard_count": kinds.count("semi_hard"),
                        "fallback_uniform_count": kinds.count("fallback_uniform"),
                        "schedule_sha256": provider.state["history"][item["epoch"]-1]["schedule_sha256"]})
                    dropout.append({"fold": f, "seed": seed, "epoch": item["epoch"], **{
                        f"{modality}_{suffix}": item.get(f"{modality}_dropout_{source}", 0)
                        for modality in MODALITIES for suffix, source in (("attempted", "attempted_count"),
                            ("dropped", "dropped_count"), ("realized_rate", "realized_rate"))}})
        comparisons_csv = []
        for comparison in comparisons:
            flat = {"fold": comparison["fold"], "seed": comparison["seed"]}
            for label in ("D3", "D0", "D1", "D2", "D_cosine"):
                flat.update({f"{label}_{metric}": value for metric, value in comparison[label].items()})
            comparisons_csv.append(flat)
        for filename, rows in (("training_curves.csv", curves), ("mining_diagnostics.csv", mining),
                ("modality_dropout_diagnostics.csv", dropout), ("validation_metrics.csv", comparisons_csv)):
            fields = sorted({k for row in rows for k in row})
            with (stage / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        if len(records) != len(folds) * len(seeds):
            raise RuntimeError("D3 output count barrier failed")
        output_hashes = {p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        report = {"status": "complete", "protocol": {**config, "folds": list(folds), "seeds": list(seeds),
            "epochs": epochs, "batch_size": batch_size}, "records": records, "validation_comparisons": comparisons,
            "hashes": {"references": _json_keys(references["hashes"]), "protected_before": before,
                "data_sources": {k: before[f"input:{k}"] for k in INPUTS},
                "code": {p: before[f"code:{p}"] for p in CODE_PATHS}, "split_manifest": manifest_hash,
                "fold_edge_csv_sha256": fold_edge_hashes, "d0_schedules": {k: before[f"d0_schedule:{k}"] for k in schedule_paths},
                "output_files": output_hashes}, "mxm_by_fold": mxm_by_fold,
            "rng_mining": "seed*1000003 + fold*1009 + one-based epoch",
            "rng_dropout": "seed * 1000003 + fold * 1009; trainer adds zero-based epoch index",
            "negative_exclusion_provenance": {"known_positive_edges_sha256": known_hash,
                "official_test_edge_source_sha256_exclusion_only": before["input:test_edges_path"],
                "legacy_test_edge_source_sha256_exclusion_only": before["input:ab_test_edges_path"],
                    "test_source_edges_used_as_targets": False},
            "initialization": "independent_per_fold_seed", "evaluation_scope": "validation_only_development",
            "counts": {"training_runs": len(records), "training_curve_rows": len(curves),
                "mining_diagnostic_rows": len(mining), "modality_dropout_diagnostic_rows": len(dropout),
                "comparison_rows": len(comparisons)}, "runtime": {"seconds": time.monotonic()-started,
                "python": platform.python_version(), "numpy": np.__version__, "pytorch": torch.__version__},
            "test_evaluation_deferred": True, "test_features_constructed": False,
            "test_embeddings_constructed": False, "test_relevance_constructed": False, "test_metrics_computed": False}
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if {k: _hash_path(p) for k, p in protected.items()} != before:
            raise RuntimeError("protected inputs, code, config, or reference namespaces changed during run")
        # Exclusive namespace creation, then hard-link staged files; rollback touches only our inodes.
        created_files, created_dirs = [], []
        try:
            final.mkdir()
            st = final.stat(); created_dirs.append((final, st.st_dev, st.st_ino))
            for name in ("checkpoints", "triplet_schedules"):
                dest = final / name
                dest.mkdir(); st = dest.stat(); created_dirs.append((dest, st.st_dev, st.st_ino))
            for source in sorted(p for p in stage.rglob("*") if p.is_file()):
                dest = final / source.relative_to(stage)
                os.link(source, dest)
                st = dest.stat(); created_files.append((dest, st.st_dev, st.st_ino))
        except BaseException:
            for path, dev, ino in reversed(created_files):
                try:
                    st = path.stat()
                    if (st.st_dev, st.st_ino) == (dev, ino): path.unlink()
                except FileNotFoundError: pass
            for path, dev, ino in reversed(created_dirs):
                try:
                    st = path.stat()
                    if (st.st_dev, st.st_ino) == (dev, ino): path.rmdir()
                except OSError: pass
            raise
        shutil.rmtree(stage)
        return report
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    _run(args.data_root, args.results_root)


if __name__ == "__main__":
    main()

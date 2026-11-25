"""Train isolated D1 semi-hard variant D and publish validation-only evidence."""

import argparse
import csv
import hashlib
import io
import json
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
from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
from src.data.neurotune_v1_split import sha256_file
from src.evaluate_neurotune_v1 import build_neurotune_v1_cosine_baseline, evaluate_neurotune_v1_retrieval
from src.train_neurotune_v1 import train_neurotune_v1
from src.triplets import MixedSemiHardTripletSampler, MixedTripletSample

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/neurotune_v1_d1_semi_hard.json"
CODE_EPOCH_SEED_DERIVATION = "seed*1000003 + fold*1009 + epoch"
CODE_SAMPLING_KIND_CODES = {"uniform": 0, "semi_hard": 1, "fallback_uniform": 2}
CODE_MINER_VERSION = "strict_cosine_band_v1"


def _validate_protocol_config(config):
    required = {"variant", "folds", "seeds", "epochs", "batch_size", "optimizer", "learning_rate", "margin",
                "device", "negative_strategy", "semi_hard_fraction", "uniform_warmup_epochs", "refresh_interval",
                "rng", "miner_version", "sampling_kind_codes", "pilot", "full", "evaluation_scope"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("D1 protocol config has unexpected or missing fields")
    for name in ("rng", "pilot", "full", "sampling_kind_codes"):
        if not isinstance(config[name], dict):
            raise ValueError(f"D1 protocol config {name} must be an object")
    if (set(config["rng"]) != {"implementation", "version", "epoch_seed_derivation"}
            or set(config["pilot"]) != {"folds", "seeds", "output_namespace"}
            or set(config["full"]) != {"folds", "seeds", "output_namespace"}):
        raise ValueError("D1 protocol config has an invalid nested object shape")

    def integer_list(value, label, *, minimum=0):
        if (not isinstance(value, list) or not value
                or any(type(item) is not int or item < minimum for item in value)
                or len(value) != len(set(value))):
            raise ValueError(f"D1 protocol config {label} must be a nonempty unique integer list")

    integer_list(config["folds"], "folds")
    integer_list(config["seeds"], "seeds", minimum=1)
    for field in ("epochs", "batch_size", "uniform_warmup_epochs", "refresh_interval"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"D1 protocol config {field} must be a positive integer")
    for field in ("learning_rate", "margin"):
        value = config[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"D1 protocol config {field} must be a positive finite number")
    fraction = config["semi_hard_fraction"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("D1 protocol config semi_hard_fraction must be in [0, 1]")
    if config["uniform_warmup_epochs"] >= config["epochs"]:
        raise ValueError("D1 protocol config warm-up epochs must be less than total epochs")
    if config["refresh_interval"] > config["epochs"]:
        raise ValueError("D1 protocol config refresh_interval cannot exceed total epochs")
    if config["variant"] != "D" or config["optimizer"] != "Adam" or config["device"] != "cpu":
        raise ValueError("D1 protocol config must specify variant D, Adam, and CPU")
    if config["negative_strategy"] != "mixed_semi_hard":
        raise ValueError("D1 protocol config negative_strategy must be mixed_semi_hard")
    rng = config["rng"]
    if (rng["implementation"] != "python.random.Random" or type(rng["version"]) is not int or rng["version"] != 2
            or rng["epoch_seed_derivation"] != CODE_EPOCH_SEED_DERIVATION):
        raise ValueError("D1 protocol config RNG implementation/version/seed formula is unsupported")
    if config["miner_version"] != CODE_MINER_VERSION:
        raise ValueError("D1 protocol config miner_version is unsupported")
    codes = config["sampling_kind_codes"]
    if (set(codes) != set(CODE_SAMPLING_KIND_CODES)
            or any(type(value) is not int for value in codes.values()) or codes != CODE_SAMPLING_KIND_CODES):
        raise ValueError("D1 protocol config sampling_kind_codes do not match the runner encoding")
    pilot, full = config["pilot"], config["full"]
    integer_list(full["folds"], "full.folds")
    integer_list(full["seeds"], "full.seeds", minimum=1)
    integer_list(pilot["folds"], "pilot.folds")
    integer_list(pilot["seeds"], "pilot.seeds", minimum=1)
    for scope_name, scope in (("pilot", pilot), ("full", full)):
        if not isinstance(scope["output_namespace"], str) or not scope["output_namespace"].strip():
            raise ValueError(f"D1 protocol config {scope_name}.output_namespace must be a nonempty string")
    if (pilot["output_namespace"] != "neurotune_v1_training_improvements_pilot"
            or full["output_namespace"] != "neurotune_v1_training_improvements"):
        raise ValueError("D1 protocol config output_namespace values must be the fixed safe names")
    if (full["folds"] != config["folds"] or full["seeds"] != config["seeds"]
            or pilot["seeds"] != config["seeds"] or not set(pilot["folds"]) <= set(full["folds"])
            or pilot["folds"] != [0] or pilot["output_namespace"] == full["output_namespace"]):
        raise ValueError("D1 protocol config pilot/full scopes are inconsistent")
    if config["evaluation_scope"] != "validation_only_development":
        raise ValueError("D1 protocol config evaluation_scope must be validation_only_development")


def _load_protocol_config(path=CONFIG_PATH, *, expected=None):
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load frozen D1 protocol config: {path}") from error
    _validate_protocol_config(config)
    if expected is None:
        expected = globals().get("PROTOCOL_CONFIG")
    if expected is not None and config != expected:
        raise ValueError("D1 protocol config differs from the frozen protocol loaded at startup")
    return config


PROTOCOL_CONFIG = _load_protocol_config(CONFIG_PATH)
FOLDS, SEEDS = tuple(PROTOCOL_CONFIG["folds"]), tuple(PROTOCOL_CONFIG["seeds"])
EPOCHS, BATCH_SIZE = PROTOCOL_CONFIG["epochs"], PROTOCOL_CONFIG["batch_size"]
LEARNING_RATE, MARGIN = PROTOCOL_CONFIG["learning_rate"], PROTOCOL_CONFIG["margin"]
WARMUP_EPOCHS, REFRESH_INTERVAL = PROTOCOL_CONFIG["uniform_warmup_epochs"], PROTOCOL_CONFIG["refresh_interval"]
SEMI_HARD_FRACTION = PROTOCOL_CONFIG["semi_hard_fraction"]
PYTHON_RANDOM_VERSION = PROTOCOL_CONFIG["rng"]["version"]
EPOCH_SEED_DERIVATION = PROTOCOL_CONFIG["rng"]["epoch_seed_derivation"]
MINER_VERSION = PROTOCOL_CONFIG["miner_version"]
SAMPLING_KIND_CODES = PROTOCOL_CONFIG["sampling_kind_codes"]
MODALITIES = ("bpm", "lyrics", "genre")
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
    "scripts/run_neurotune_v1_training_improvements.py", "src/data/neurotune_v1_abcd_cv_training.py",
    "src/data/neurotune_v1_abcd_cv.py", "src/data/parse_mxm.py", "src/data/neurotune_v1_split.py",
    "src/data/neurotune_v1_dataset.py", "src/features_neurotune_v1.py", "src/features.py",
    "src/train_neurotune_v1.py", "src/evaluate_neurotune_v1.py", "src/evaluate.py", "src/retrieve.py",
    "src/data/table.py", "src/model_v1.py", "src/triplets.py",
)
METRICS = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10")
ACTIVE_TRANSFORM_FIELDS = ("fit_ids_hash", "audio_mean", "audio_std", "audio_fit_rows", "bpm_mean",
                           "bpm_std", "genre_vocabulary", "split_hash")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hash_path(path):
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return _digest([(p.relative_to(path).as_posix(), sha256_file(p)) for p in sorted(path.rglob("*")) if p.is_file()])
    raise FileNotFoundError(path)


def _schedule_digest(triplets, type_codes, mapping=None):
    rows = np.asarray(triplets, dtype=np.int32)
    codes = np.asarray(type_codes, dtype=np.uint8)
    if codes.shape != (len(rows),):
        raise ValueError("sampling type codes must have one value per triplet")
    mapping = SAMPLING_KIND_CODES if mapping is None else mapping
    digest = hashlib.sha256()
    digest.update(np.asarray(rows.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(codes.shape, dtype=np.int64).tobytes())
    digest.update(rows.tobytes())
    digest.update(codes.tobytes())
    digest.update(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _uniform_sample(triplets, train_ids, known_positive_edges):
    rows = np.asarray(triplets, dtype=np.int32).copy()
    if rows.ndim != 2 or rows.shape[1] != 3 or np.any(rows < 0) or np.any(rows >= len(train_ids)):
        raise ValueError("uniform schedule contains an index outside train IDs")
    neighbors = {track_id: set() for track_id in train_ids}
    for left, right in known_positive_edges:
        if left in neighbors and right in neighbors:
            neighbors[left].add(right)
            neighbors[right].add(left)
    counts = np.asarray([len(train_ids) - len(neighbors[train_ids[int(anchor)]]) - 1 for anchor in rows[:, 0]], dtype=np.int32)
    return MixedTripletSample(rows, tuple("uniform" for _ in rows), counts,
                              np.full(len(rows), -1, dtype=np.int32), np.zeros(len(rows), dtype=np.bool_),
                              0, 0, 1.0, 0.0)


def _model_embeddings(model, dataset):
    values = []
    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            rows = [dataset[i] for i in range(start, min(len(dataset), start + BATCH_SIZE))]
            batch = neurotune_v1_collate(rows)
            args = {key: batch[key] for key in ("audio", "mask", *model.active_modalities)}
            values.append(model(**args).detach().cpu().numpy())
    return np.concatenate(values, axis=0)


def _make_schedule_provider(train_ids, d0_train_schedule, positive_edges, known_positive_edges,
                            sampler_factory=MixedSemiHardTripletSampler, *, fold=0, seed=684):
    if np.asarray(d0_train_schedule).dtype != np.int32 or np.asarray(d0_train_schedule).shape[0] < WARMUP_EPOCHS:
        raise ValueError("verified D0 train schedule must provide three int32 warm-up epochs")
    state = {"sampler": None, "refresh_epoch": None, "history": []}

    def provider(model, train_dataset, epoch_number):
        if list(train_dataset.track_ids) != list(train_ids):
            raise ValueError("provider dataset IDs differ from the verified train partition")
        if epoch_number <= WARMUP_EPOCHS:
            sample = _uniform_sample(d0_train_schedule[epoch_number - 1], train_ids, known_positive_edges)
        else:
            refresh = epoch_number == WARMUP_EPOCHS + 1 or (epoch_number - WARMUP_EPOCHS - 1) % REFRESH_INTERVAL == 0
            if refresh:
                state["sampler"] = None
                embeddings = _model_embeddings(model, train_dataset)
                state["sampler"] = sampler_factory(
                    list(train_ids), list(positive_edges), embeddings, margin=MARGIN,
                    semi_hard_fraction=SEMI_HARD_FRACTION, known_positive_edges=list(known_positive_edges),
                )
                state["refresh_epoch"] = epoch_number
            epoch_seed = int(seed) * 1000003 + int(fold) * 1009 + int(epoch_number)
            sample = state["sampler"].sample(epoch_seed)
        codes = np.asarray([SAMPLING_KIND_CODES[kind] for kind in sample.sampling_kinds], dtype=np.uint8)
        schedule_hash = _schedule_digest(sample.triplets, codes)
        state["history"].append({"epoch": epoch_number, "sample": sample, "schedule_sha256": schedule_hash,
                                  "refresh_epoch": state["refresh_epoch"],
                                  "epoch_seed": None if epoch_number <= WARMUP_EPOCHS else epoch_seed,
                                  "schedule_source": "verified_d0_uniform_schedule" if epoch_number <= WARMUP_EPOCHS else MINER_VERSION})
        return sample

    provider.state = state
    return provider


def _build_report(scope, d1_records, d0_records, baselines, *, folds=FOLDS, **fields):
    return {"status": "complete", "run_scope": scope, "evaluation_scope": "validation_only_development",
            "independent_test_completed": False, "protocol": {"folds": list(folds), "seeds": list(SEEDS),
             "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
             "triplet_margin": MARGIN, "optimizer": "Adam", "device": "cpu", "trained_variants": ["D"],
             "negative_strategy": PROTOCOL_CONFIG["negative_strategy"],
            "modalities": {"D": list(MODALITIES)}, "semi_hard_fraction": SEMI_HARD_FRACTION,
            "uniform_warmup_epochs": WARMUP_EPOCHS, "refresh_interval": REFRESH_INTERVAL,
            "python_random_version": PYTHON_RANDOM_VERSION,
            "epoch_seed_derivation": EPOCH_SEED_DERIVATION, "miner_version": MINER_VERSION,
            "selection": "validation NDCG@10, then validation loss"},
            "rng_mining_provenance": {"python_random_version": PYTHON_RANDOM_VERSION,
                "epoch_seed_derivation": EPOCH_SEED_DERIVATION, "miner_version": MINER_VERSION},
            "negative_exclusion_provenance": {"policy": "Inherited from D0; known-positive edges from train, validation, and official/legacy test sources are exclusion-only.",
            "test_source_edges_used_as_targets": False, "test_features_embeddings_relevance_or_metrics_built": False},
            "model_validation": d0_records + d1_records, "cosine_validation": baselines,
            "counts": {"d1_training_runs": len(d1_records), "d0_validation_references": len(d0_records),
                       "cosine_validation_observations": len(baselines)}, **fields}


def _validate_metrics(metrics, label):
    required = {*METRICS, "query_count", "eligible_positive_edges"}
    if not isinstance(metrics, dict) or not required <= metrics.keys() or any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) for v in metrics.values()):
        raise ValueError(f"{label} must contain finite validation metrics and counts")


def _verify_d0_record(record, checkpoint, schedule, fold, seed, checkpoint_hash, schedule_hash,
                      manifest_hash, input_hashes):
    """Validate the immutable D0 record/checkpoint/schedule triplet."""
    provenance = checkpoint.get("provenance", {})
    config = checkpoint.get("config", {})
    if (record.get("fold") != fold["fold"] or record.get("seed") != seed or record.get("variant") != "D"
            or record.get("checkpoint_sha256") != checkpoint_hash or record.get("schedule_sha256") != schedule_hash
            or record.get("checkpoint_path") != f"checkpoints/fold_{fold['fold']}_D_seed_{seed}.pt"):
        raise ValueError("D0 report record/hash mismatch")
    if (checkpoint.get("fold", fold["fold"]) != fold["fold"] or checkpoint.get("seed") != seed
            or checkpoint.get("variant") != "D" or provenance.get("fold") != fold["fold"]
            or provenance.get("seed") != seed or provenance.get("variant") != "D"
            or config.get("device") != "cpu" or config.get("epochs") != EPOCHS
            or config.get("batch_size") != BATCH_SIZE or config.get("optimizer") != "Adam"
            or config.get("learning_rate") != LEARNING_RATE or config.get("margin") != MARGIN):
        raise ValueError("D0 checkpoint fold/seed/variant/config mismatch")
    if (list(schedule["train_ids"].astype(str)) != list(fold["train"])
            or list(schedule["validation_ids"].astype(str)) != list(fold["validation"])):
        raise ValueError("D0 schedule IDs differ from current fold partitions")
    old_transform = checkpoint.get("transform_manifest")
    if (not isinstance(old_transform, dict) or _digest(old_transform) != provenance.get("transform_sha256")
            or provenance.get("transform_sha256") != record.get("transform_sha256")):
        raise ValueError("D0 checkpoint transform manifest hash mismatch")
    if (record.get("validation") != checkpoint.get("selection_metrics")
            or provenance.get("input_sha256") != input_hashes
            or provenance.get("split_manifest_sha256") != manifest_hash):
        raise ValueError("D0 validation/input/split record mismatch")
    _validate_metrics(record.get("validation"), "D0 validation")
    return old_transform


def _slice(features, ids, selected_ids):
    pos = {track_id: i for i, track_id in enumerate(ids)}
    ix = [pos[i] for i in selected_ids]
    return {key: ( [value[i] for i in ix] if isinstance(value, (tuple, list)) else np.asarray(value)[ix])
            for key, value in features.items()}


def _run(data_root=ROOT, results_root=ROOT / "results", *, pilot=False, _test_only=False, _dependencies=None):
    started = time.monotonic()
    deps = _dependencies or {}
    if deps and not _test_only:
        raise ValueError("production dependencies and protocol are fixed")
    root, result_root = Path(data_root), Path(results_root)
    frozen_config = _load_protocol_config(CONFIG_PATH)
    scope_config = frozen_config["pilot" if pilot else "full"]
    namespace = scope_config["output_namespace"]
    folds = tuple(scope_config["folds"])
    final = result_root / namespace
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    inputs = {key: root / rel for key, rel in INPUTS.items()}
    d0_root = result_root / "neurotune_v1_full_modal_ablation"
    d0_report_path = d0_root / "experiment.json"
    all_pairs = [(f, s) for f in FOLDS for s in SEEDS]
    d0_cps = {f"checkpoint_{f}_{s}": d0_root / f"checkpoints/fold_{f}_D_seed_{s}.pt" for f, s in all_pairs}
    d0_schedules = {f"schedule_{f}_{s}": d0_root / f"triplet_schedules/fold_{f}_seed_{s}.npz" for f, s in all_pairs}
    protected = {**{f"input:{k}": p for k, p in inputs.items()},
                 **{f"code:{p}": ROOT / p for p in CODE_PATHS},
                 "d1_config": CONFIG_PATH,
                 "d0_report": d0_report_path, **{f"d0:{k}": p for k, p in {**d0_cps, **d0_schedules}.items()}}
    before = {key: _hash_path(path) for key, path in protected.items()}
    d0_report_bytes = d0_report_path.read_bytes()
    d0_report = json.loads(d0_report_bytes)
    d0_records = [row for row in d0_report.get("model_validation", []) if row.get("variant") == "D"]
    d0_by_pair = {(row.get("fold"), row.get("seed")): row for row in d0_records}
    baseline_rows = [row for row in d0_report.get("cosine_validation", []) if row.get("variant") == "D"]
    if len(d0_by_pair) != len(d0_records) or set(d0_by_pair) != set(all_pairs):
        raise ValueError("D0 report must contain exactly one D validation record per fixed fold/seed")
    if len(baseline_rows) != len(FOLDS) or {row.get("fold") for row in baseline_rows} != set(FOLDS):
        raise ValueError("D0 cosine validation baselines do not cover the fixed folds")
    manifest_hash = sha256_file(inputs["cv_dir"] / "manifest.json")
    loaded = deps.get("loader", load_abcd_cv_training)(**{k: v for k, v in inputs.items() if k != "mxm_path"})
    if [fold.get("fold") for fold in loaded.get("folds", [])] != list(FOLDS):
        raise ValueError("loaded folds must be fixed folds 0, 1, and 2")
    mxm_provenance_by_fold = {}
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{namespace}.stage-", dir=final.parent))
    try:
        for folder in ("checkpoints", "triplet_schedules"):
            (stage / folder).mkdir()
        d1_records, curves, mining_rows, paired_rows, baselines_out = [], [], [], [], []
        schedule_files = {}
        fold_by_no = {entry["fold"]: entry for entry in loaded["folds"]}
        for fold_no in folds:
            fold = fold_by_no[fold_no]
            development_ids = sorted(fold["train"] + fold["validation"])
            fold_loaded = dict(loaded)
            fold_loaded["catalog"] = {track_id: loaded["catalog"][track_id] for track_id in development_ids}
            lyrics, mxm_provenance = deps.get("mxm_builder", build_abcd_cv_fold_mxm)(
                {k: fold_loaded[k] for k in ("manifest", "catalog", "cache_path") if k in fold_loaded}, inputs["mxm_path"])
            mxm_provenance_by_fold[str(fold_no)] = mxm_provenance
            d0_schedules_by_seed = {}
            for seed in SEEDS:
                pair_key = (fold_no, seed)
                record = d0_by_pair.get(pair_key)
                if record is None:
                    raise ValueError(f"D0 report is missing D validation record for {pair_key}")
                cp_path, sch_path = d0_cps[f"checkpoint_{fold_no}_{seed}"], d0_schedules[f"schedule_{fold_no}_{seed}"]
                if record.get("checkpoint_sha256") != before[f"d0:checkpoint_{fold_no}_{seed}"] or record.get("schedule_sha256") != before[f"d0:schedule_{fold_no}_{seed}"]:
                    raise ValueError("D0 checkpoint or schedule bytes do not match the report hashes")
                checkpoint_bytes = cp_path.read_bytes()
                checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
                with np.load(io.BytesIO(sch_path.read_bytes()), allow_pickle=False) as archive:
                    schedule = {key: archive[key].copy() for key in ("train_ids", "validation_ids", "train_schedule", "validation_schedule")}
                expected_inputs = {k: before[f"input:{k}"] for k in INPUTS if k not in {"mxm_path", "test_edges_path", "ab_test_edges_path", "cv_dir"}}
                _verify_d0_record(record, checkpoint, schedule, fold, seed,
                    before[f"d0:checkpoint_{fold_no}_{seed}"], before[f"d0:schedule_{fold_no}_{seed}"],
                    manifest_hash, expected_inputs)
                config, provenance = checkpoint.get("config", {}), checkpoint.get("provenance", {})
                expected_cp = f"checkpoints/fold_{fold_no}_D_seed_{seed}.pt"
                if (record.get("checkpoint_path") != expected_cp
                    or checkpoint.get("fold", fold_no) != fold_no or checkpoint.get("seed") != seed or checkpoint.get("variant") != "D"
                    or provenance.get("fold") != fold_no or provenance.get("seed") != seed or provenance.get("variant") != "D"
                    or config.get("device") != "cpu" or config.get("epochs") != EPOCHS or config.get("batch_size") != BATCH_SIZE
                    or config.get("optimizer") != "Adam" or config.get("learning_rate") != LEARNING_RATE or config.get("margin") != MARGIN):
                    raise ValueError("D0 checkpoint fold/seed/variant/config mismatch")
                if list(schedule["train_ids"].astype(str)) != list(fold["train"]) or list(schedule["validation_ids"].astype(str)) != list(fold["validation"]):
                    raise ValueError("D0 schedule IDs differ from current fold partitions")
                if schedule["train_schedule"].dtype != np.int32 or schedule["train_schedule"].shape != (EPOCHS, len(fold["train_edges"]) * 2, 3):
                    raise ValueError("D0 train schedule shape/dtype mismatch")
                if schedule["validation_schedule"].dtype != np.int32 or schedule["validation_schedule"].shape != (1, len(fold["validation_edges"]) * 2, 3):
                    raise ValueError("D0 validation schedule shape/dtype mismatch")
                for name, size in (("train_schedule", len(fold["train"])), ("validation_schedule", len(fold["validation"]))):
                    if np.any(schedule[name] < 0) or np.any(schedule[name] >= size):
                        raise ValueError(f"D0 {name} contains a partition-external index")
                if provenance.get("input_sha256") != expected_inputs or provenance.get("split_manifest_sha256") != manifest_hash:
                    raise ValueError("D0 checkpoint input or split hashes differ from current sources")
                if provenance.get("transform_sha256") != record.get("transform_sha256"):
                    raise ValueError("D0 report/checkpoint transform hash mismatch")
                old_transform = checkpoint.get("transform_manifest")
                if not isinstance(old_transform, dict) or _digest(old_transform) != provenance.get("transform_sha256"):
                    raise ValueError("D0 checkpoint transform manifest hash mismatch")
                if record.get("validation") != checkpoint.get("selection_metrics"):
                    raise ValueError("D0 report and checkpoint validation records mismatch")
                _validate_metrics(record.get("validation"), "D0 validation")
                d0_schedules_by_seed[seed] = schedule
                d0_schedules_by_seed[seed]["_transform_manifest"] = old_transform
                del checkpoint

            ids = development_ids
            feature_loaded = {k: fold_loaded[k] for k in ("manifest", "catalog", "cache_path") if k in fold_loaded}
            development_fold = {k: fold[k] for k in ("fold", "train", "validation")}
            features, transform = deps.get("feature_builder", build_abcd_cv_development_features)(feature_loaded, development_fold, lyrics)
            transform_hash = _digest(transform)
            for seed in SEEDS:
                old_transform = d0_schedules_by_seed[seed]["_transform_manifest"]
                active_old = {key: old_transform.get(key) for key in ACTIVE_TRANSFORM_FIELDS}
                active_now = {key: transform.get(key) for key in ACTIVE_TRANSFORM_FIELDS}
                old_sources = {key: old_transform.get("source_sha256", {}).get(key) for key in ("audio_cache", "split_manifest")}
                new_sources = {key: transform.get("source_sha256", {}).get(key) for key in ("audio_cache", "split_manifest")}
                if active_old != active_now or old_sources != new_sources:
                    raise ValueError("D0 active transform fields differ from current development transform")
            val_ids = list(fold["validation"])
            val_features = _slice(features, ids, val_ids)
            val_embedding = deps.get("baseline_builder", build_neurotune_v1_cosine_baseline)(
                {key: val_features[key] for key in ("audio", *MODALITIES)}, val_ids, modalities=("audio", *MODALITIES))
            baseline_metrics = deps.get("evaluator", evaluate_neurotune_v1_retrieval)(val_embedding, val_ids, fold["validation_edges"])
            d0_baseline = next(row for row in baseline_rows if row["fold"] == fold_no)
            if baseline_metrics != d0_baseline["validation"]:
                raise ValueError("current D cosine validation baseline differs from D0")
            baselines_out.append(d0_baseline)
            train_ids = list(fold["train"])
            train_ds = NeuroTuneV1Dataset(train_ids, _slice(features, ids, train_ids), modalities=MODALITIES)
            val_ds = NeuroTuneV1Dataset(val_ids, val_features, modalities=MODALITIES)
            known_train = [edge for edge in loaded["known_positive_edges"] if set(edge) <= set(train_ids)]
            for seed in SEEDS:
                d0_schedule = d0_schedules_by_seed[seed]
                provider = _make_schedule_provider(train_ids, d0_schedule["train_schedule"], fold["train_edges"],
                    known_train, deps.get("sampler_factory", MixedSemiHardTripletSampler), fold=fold_no, seed=seed)
                schedule_path = stage / f"triplet_schedules/fold_{fold_no}_seed_{seed}.npz"
                cp_rel = f"checkpoints/fold_{fold_no}_D_seed_{seed}.pt"
                provenance = {"fold": fold_no, "seed": seed, "variant": "D", "independent_initialization": True,
                    "python_random_version": PYTHON_RANDOM_VERSION, "epoch_seed_derivation": EPOCH_SEED_DERIVATION,
                    "miner_version": MINER_VERSION,
                    "input_sha256": {k: before[f"input:{k}"] for k in INPUTS if k not in {"mxm_path", "test_edges_path", "ab_test_edges_path", "cv_dir"}},
                    "split_manifest_sha256": manifest_hash, "transform_sha256": transform_hash,
                    "d0_checkpoint_sha256": before[f"d0:checkpoint_{fold_no}_{seed}"],
                    "d0_schedule_sha256": before[f"d0:schedule_{fold_no}_{seed}"],
                    "test_source_edges_exclusion_only": True}
                result = deps.get("trainer", train_neurotune_v1)(train_ds, val_ds, None,
                    d0_schedule["validation_schedule"], [{"source": a, "target": b} for a, b in fold["validation_edges"]],
                    "D", seed, stage / f".candidate_{fold_no}_{seed}.pt", transform, provenance,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, train_schedule_provider=provider,
                    record_epoch_timing=True)
                history, best_epoch, best_metrics = result["history"], result["best_epoch"], result["best_metrics"]
                if len(history) != EPOCHS or sorted(row.get("epoch") for row in history) != list(range(1, EPOCHS + 1)):
                    raise ValueError("trainer must return 50 epoch records")
                _validate_metrics(best_metrics, "D1 selected validation")
                samples = provider.state["history"]
                all_triplets = np.stack([item["sample"].triplets for item in samples]).astype(np.int32)
                all_kind_codes = np.stack([
                    np.fromiter((SAMPLING_KIND_CODES[kind] for kind in item["sample"].sampling_kinds),
                                dtype=np.uint8, count=len(item["sample"].sampling_kinds))
                    for item in samples
                ])
                digests = [_schedule_digest(row, codes) for row, codes in zip(all_triplets, all_kind_codes)]
                np.savez_compressed(schedule_path, train_ids=np.asarray(train_ids), validation_ids=np.asarray(val_ids),
                    train_schedule=all_triplets, validation_schedule=d0_schedule["validation_schedule"],
                    sampling_type_codes=all_kind_codes,
                    sampling_kind_code_mapping=np.asarray(json.dumps(SAMPLING_KIND_CODES, sort_keys=True)),
                    schedule_sha256=np.asarray(digests))
                schedule_file_hash = sha256_file(schedule_path)
                candidate = stage / f".candidate_{fold_no}_{seed}.pt"
                candidate_checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
                candidate_provenance = candidate_checkpoint.setdefault("provenance", {})
                candidate_provenance["schedule_sha256"] = _digest(digests)
                candidate_provenance["schedule_file_sha256"] = schedule_file_hash
                candidate_provenance["schedule_kind_encoding"] = SAMPLING_KIND_CODES
                torch.save(candidate_checkpoint, candidate)
                final_checkpoint = stage / cp_rel
                os.replace(candidate, final_checkpoint)
                curves.extend({"fold": fold_no, "variant": "D", "seed": seed, **row} for row in history)
                for item, kind_codes in zip(samples, all_kind_codes):
                    sample = item["sample"]
                    uniform_count, semi_hard_count, fallback_count = np.bincount(kind_codes, minlength=3)
                    kind_count = len(kind_codes)
                    mining_rows.append({"fold": fold_no, "seed": seed, "epoch": item["epoch"],
                        "schedule_sha256": item["schedule_sha256"], "refresh_epoch": item["refresh_epoch"],
                        "python_random_version": PYTHON_RANDOM_VERSION, "epoch_seed_derivation": EPOCH_SEED_DERIVATION,
                        "miner_version": MINER_VERSION if item["epoch_seed"] is not None else "d0_uniform_schedule",
                        "schedule_source": item["schedule_source"], "epoch_seed": item["epoch_seed"],
                        "attempted_semi_hard_count": sample.attempted_semi_hard_count,
                        "fallback_count": sample.fallback_count, "realized_uniform_fraction": sample.realized_uniform_fraction,
                        "realized_semi_hard_fraction": sample.realized_semi_hard_fraction,
                        "mean_eligible_uniform_pool": float(np.mean(sample.eligible_uniform_counts)),
                        "mean_eligible_semi_hard_pool": (float(np.mean(sample.eligible_semi_hard_counts[sample.eligible_semi_hard_counts >= 0]))
                            if np.any(sample.eligible_semi_hard_counts >= 0) else None),
                        "uniform_count": int(uniform_count),
                        "semi_hard_count": int(semi_hard_count),
                        "fallback_uniform_count": int(fallback_count),
                        "uniform_fraction": float(uniform_count / kind_count),
                        "semi_hard_fraction": float(semi_hard_count / kind_count),
                        "fallback_uniform_fraction": float(fallback_count / kind_count)})
                d1 = {"fold": fold_no, "variant": "D", "seed": seed, "selected_epoch": best_epoch,
                    "validation": best_metrics, "checkpoint_path": cp_rel, "checkpoint_sha256": sha256_file(final_checkpoint),
                    "schedule_sha256": _digest(digests), "schedule_file_sha256": schedule_file_hash,
                    "transform_sha256": transform_hash}
                d1_records.append(d1)
                paired_rows.append({"fold": fold_no, "seed": seed, "D0_validation": d0_by_pair[(fold_no, seed)]["validation"],
                                    "D1_validation": best_metrics})
                schedule_files[f"{fold_no}/{seed}"] = {"archive_sha256": schedule_file_hash, "schedule_sha256": d1["schedule_sha256"]}
            del features, train_ds, val_ds
        for filename, rows in (("training_curves.csv", curves), ("mining_diagnostics.csv", mining_rows),
                               ("validation_metrics.csv", paired_rows)):
            if rows:
                with (stage / filename).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
                    writer.writeheader(); writer.writerows(rows)
        output_config = {"frozen_config": frozen_config, "run_scope": "pilot" if pilot else "full",
                         "actual_folds": list(folds), "actual_seeds": list(SEEDS)}
        (stage / "d1_config.json").write_text(json.dumps(output_config, indent=2) + "\n", encoding="utf-8")
        d0_selected = [row for row in d0_records if row["fold"] in folds]
        source_hashes = {k: before[f"input:{k}"] for k in INPUTS}
        code_hashes = {p: before[f"code:{p}"] for p in CODE_PATHS}
        report = _build_report("pilot" if pilot else "full", d1_records, d0_selected, baselines_out, folds=folds,
            paired_validation=paired_rows, hashes={"data_sources": source_hashes, "current_d1_code": code_hashes,
            "historical_d0_code": d0_report.get("hashes", {}).get("code", {}),
            "d1_config_sha256": before["d1_config"],
            "d0_experiment_sha256": before["d0_report"], "d0_checkpoints": {f"{f}/{s}": before[f"d0:checkpoint_{f}_{s}"] for f, s in all_pairs if f in folds},
            "d0_schedules": {f"{f}/{s}": before[f"d0:schedule_{f}_{s}"] for f, s in all_pairs if f in folds},
            "split_manifest": manifest_hash, "outputs": {**schedule_files,
            "checkpoints": {r["checkpoint_path"]: r["checkpoint_sha256"] for r in d1_records},
            "files": {p.relative_to(stage).as_posix(): sha256_file(p) for p in stage.rglob("*") if p.is_file()}}},
            mxm=mxm_provenance_by_fold, runtime={"seconds": time.monotonic() - started, "python": platform.python_version(),
            "numpy": np.__version__, "pytorch": torch.__version__}, no_test_metrics=True)
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        after = {key: _hash_path(path) for key, path in protected.items()}
        if before != after:
            raise RuntimeError("protected inputs, code, D0 artifacts changed during run")
        made_dirs, made_files = [], []
        try:
            final.mkdir()
            made_dirs.append((final, final.stat().st_dev, final.stat().st_ino))
            for folder in ("checkpoints", "triplet_schedules"):
                path = final / folder; path.mkdir(); made_dirs.append((path, path.stat().st_dev, path.stat().st_ino))
            staged = sorted(p for p in stage.rglob("*") if p.is_file())
            complete = stage / "experiment.json"
            for src in [p for p in staged if p != complete] + [complete]:
                dst = final / src.relative_to(stage); os.link(src, dst)
                stat = dst.stat(); made_files.append((dst, stat.st_dev, stat.st_ino))
        except BaseException:
            for path, dev, ino in reversed(made_files):
                try:
                    stat = path.stat()
                    if (stat.st_dev, stat.st_ino) == (dev, ino): path.unlink()
                except FileNotFoundError: pass
            for path, dev, ino in reversed(made_dirs):
                try:
                    stat = path.stat()
                    if (stat.st_dev, stat.st_ino) == (dev, ino): path.rmdir()
                except OSError: pass
            raise
        shutil.rmtree(stage)
        return report
    except BaseException:
        try:
            changed = {key: _hash_path(path) for key, path in protected.items()} != before
        except FileNotFoundError:
            changed = True
        if stage.exists(): shutil.rmtree(stage)
        if changed: raise RuntimeError("protected data, code, or D0 artifacts changed or disappeared")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    _run(args.data_root, args.results_root, pilot=args.pilot)


if __name__ == "__main__":
    main()

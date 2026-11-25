"""Run the frozen validation-only D2 modality-dropout experiment."""

import argparse
import csv
import hashlib
import io
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
    build_abcd_cv_development_features, build_abcd_cv_fold_mxm, load_abcd_cv_training,
)
from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset
from src.data.neurotune_v1_split import sha256_file
from src.train_neurotune_v1 import train_neurotune_v1

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "config/neurotune_v1_d2_modality_dropout.json"
FOLDS, SEEDS = (0, 1, 2), (684, 685, 686)
EPOCHS, BATCH_SIZE, LEARNING_RATE, MARGIN = 50, 64, 0.001, 0.2
NAMESPACE = "neurotune_v1_d2_modality_dropout"
DROP_PROBS = {"audio": 0.0, "bpm": 0.1, "genre": 0.25, "lyrics": 0.2}
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
    "cv_dir": "data/msd_hf_abcd_pilot/cv",
    "mxm_path": "data/processed/mxm_counts",
}
CODE_PATHS = (
    "scripts/run_neurotune_v1_modality_dropout.py", "src/data/neurotune_v1_abcd_cv_training.py",
    "scripts/run_neurotune_v1_full_modal_ablation.py",
    "src/data/neurotune_v1_abcd_cv.py", "src/data/parse_mxm.py", "src/data/neurotune_v1_split.py",
    "src/data/neurotune_v1_dataset.py", "src/features_neurotune_v1.py", "src/features.py",
    "src/train_neurotune_v1.py", "src/evaluate_neurotune_v1.py", "src/evaluate.py",
    "src/retrieve.py", "src/data/table.py", "src/model_v1.py", "src/triplets.py",
)
METRICS = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10", "query_count", "eligible_positive_edges")
EXPECTED_CONFIG = {
    "variant": "D", "folds": [0, 1, 2], "seeds": [684, 685, 686], "epochs": 50,
    "batch_size": 64, "optimizer": "Adam", "learning_rate": .001, "margin": .2,
    "device": "cpu", "dropout_probabilities": DROP_PROBS,
    "dropout_placement": "post_branch_activation_before_fusion", "dropout_scaling": "none",
    "dropout_scope": "training_only",
    "dropout_rng": {"implementation": "torch.Generator(device='cpu')",
        "fold_seed_derivation": "seed * 1000003 + fold * 1009",
        "epoch_seed_derivation": "dropout_seed + epoch_index_zero_based",
        "consumption": "deterministic_triplet_batch_order; anchor_positive_negative; bpm_genre_lyrics"},
    "output_namespace": NAMESPACE, "evaluation_scope": "validation_only_development",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hash_path(path):
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return _digest([(p.relative_to(path).as_posix(), sha256_file(p)) for p in sorted(path.rglob("*")) if p.is_file()])
    raise FileNotFoundError(path)


def _validate_config(config):
    try:
        actual = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = json.dumps(EXPECTED_CONFIG, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("D2 config is not valid canonical JSON") from exc
    if actual != expected:
        raise ValueError("D2 config differs from the frozen protocol")
    return True


def _validate_d0_report(report, folds, seeds):
    expected_protocol = {"folds": [0, 1, 2], "seeds": [684, 685, 686], "epochs": 50,
                         "batch_size": 64, "optimizer": "Adam", "learning_rate": .001,
                         "triplet_margin": .2, "device": "cpu",
                         "trained_variants": ["A", "B", "D"],
                         "modalities": {"A": [], "B": ["bpm"], "C": ["bpm", "genre"],
                                        "D": ["bpm", "lyrics", "genre"]}}
    try:
        reported_protocol = report.get("protocol")
        if not isinstance(reported_protocol, dict) or not set(expected_protocol) <= set(reported_protocol):
            raise ValueError("required protocol keys are missing")
        protocol = json.dumps({key: reported_protocol[key] for key in expected_protocol},
                              sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = json.dumps(expected_protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("D0 protocol is invalid") from exc
    if report.get("status") != "complete" or protocol != expected:
        raise ValueError("D0 report status or fixed protocol mismatch")
    requested = {(f, s) for f in folds for s in seeds}
    model_rows = report.get("model_validation")
    cosine_rows = report.get("cosine_validation")
    if not isinstance(model_rows, list) or not isinstance(cosine_rows, list):
        raise ValueError("D0 validation reference rows are invalid")
    d_records = [r for r in model_rows if isinstance(r, dict) and r.get("variant") == "D"]
    selected = {}
    for record in d_records:
        key = (record.get("fold"), record.get("seed"))
        if key in requested:
            if key in selected:
                raise ValueError("duplicate D0 D-model validation reference")
            selected[key] = record
    if set(selected) != requested:
        raise ValueError("D0 D-model validation references are missing")
    cosine_records = [r for r in cosine_rows
                      if isinstance(r, dict) and r.get("variant") == "D" and r.get("fold") in folds]
    cosine = {}
    for record in cosine_records:
        fold = record["fold"]
        if fold in cosine:
            raise ValueError("duplicate D0 D-cosine validation reference")
        cosine[fold] = record
    if set(cosine) != set(folds):
        raise ValueError("D0 D-cosine validation references are missing")
    schedule_copies = report.get("hashes", {}).get("schedule_copies")
    if not isinstance(schedule_copies, dict):
        raise ValueError("D0 schedule-copy hashes are missing")
    return selected, cosine, schedule_copies


def _slice(features, ids, selected_ids):
    positions = {track_id: i for i, track_id in enumerate(ids)}
    ix = [positions[x] for x in selected_ids]
    return {k: ([v[i] for i in ix] if isinstance(v, (list, tuple)) else np.asarray(v)[ix])
            for k, v in features.items()}


def _metrics_valid(metrics):
    return isinstance(metrics, dict) and set(METRICS) <= set(metrics) and all(
        isinstance(metrics[k], (int, float)) and not isinstance(metrics[k], bool) and np.isfinite(metrics[k])
        for k in METRICS)


def _run(data_root=ROOT, results_root=ROOT / "results", *, _test_only=False, _dependencies=None, _protocol=None):
    started = time.monotonic()
    deps = _dependencies or {}
    if not _test_only and (deps or _protocol):
        raise ValueError("production dependencies and protocol are fixed")
    config_path = ROOT / CONFIG_PATH
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    _validate_config(config)
    protocol = {"folds": FOLDS, "seeds": SEEDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE}
    if _test_only and _protocol:
        protocol.update(_protocol)
    if (not set(protocol) == {"folds", "seeds", "epochs", "batch_size"}
            or not protocol["folds"] or not protocol["seeds"] or protocol["epochs"] <= 0 or protocol["batch_size"] <= 0):
        raise ValueError("invalid test protocol")
    folds, seeds, epochs, batch_size = tuple(protocol["folds"]), tuple(protocol["seeds"]), protocol["epochs"], protocol["batch_size"]
    loader = deps.get("loader", load_abcd_cv_training)
    mxm_builder = deps.get("mxm_builder", build_abcd_cv_fold_mxm)
    feature_builder = deps.get("feature_builder", build_abcd_cv_development_features)
    trainer = deps.get("trainer", train_neurotune_v1)
    if _test_only and "Dataset" in deps:
        dataset_class = deps["Dataset"]
    else:
        dataset_class = NeuroTuneV1Dataset
    root, result_root = Path(data_root), Path(results_root)
    final = result_root / NAMESPACE
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    d0root = result_root / "neurotune_v1_full_modal_ablation"
    d0_report_path = d0root / "experiment.json"
    inputs = {k: root / v for k, v in INPUTS.items()}
    schedules = {(f, s): d0root / f"triplet_schedules/fold_{f}_seed_{s}.npz" for f in folds for s in seeds}
    d0_checkpoints = {(f, s): d0root / f"checkpoints/fold_{f}_D_seed_{s}.pt" for f in folds for s in seeds}
    protected = {**{f"input:{k}": v for k, v in inputs.items()},
                 **{f"code:{p}": ROOT / p for p in CODE_PATHS},
                 **{f"schedule:{f}/{s}": p for (f, s), p in schedules.items()},
                 **{f"d0_checkpoint:{f}/{s}": p for (f, s), p in d0_checkpoints.items()},
                 "d0_experiment": d0_report_path, "config": config_path}
    before = {k: _hash_path(p) for k, p in protected.items()}
    d0_report = json.loads(d0_report_path.read_text(encoding="utf-8"))
    d0_models, d0_cosines, d0_schedule_hashes = _validate_d0_report(d0_report, folds, seeds)
    for (f, seed), record in d0_models.items():
        expected_path = f"checkpoints/fold_{f}_D_seed_{seed}.pt"
        if record.get("checkpoint_path") != expected_path:
            raise ValueError("D0 D-model checkpoint path mismatch")
        expected_sha = before[f"d0_checkpoint:{f}/{seed}"]
        if record.get("checkpoint_sha256") != expected_sha:
            raise ValueError("D0 D-model checkpoint hash mismatch")
        selected_epoch = record.get("selected_epoch")
        if (isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int)
                or not 1 <= selected_epoch <= EPOCHS):
            raise ValueError("D0 D-model selected epoch is invalid")
        if not _metrics_valid(record.get("validation")):
            raise ValueError("D0 D-model validation metrics are invalid")
    manifest_hash = sha256_file(inputs["cv_dir"] / "manifest.json")
    known_hash = sha256_file(inputs["cv_dir"] / "known_positive_edges.csv")
    fold_edge_hashes = {p.name: sha256_file(p) for p in sorted(inputs["cv_dir"].glob("fold_*_edges.csv"))}
    loader_inputs = {k: v for k, v in inputs.items() if k != "mxm_path"}
    loaded = loader(**loader_inputs)
    if [x.get("fold") for x in loaded.get("folds", [])] != list(folds):
        raise ValueError("loaded folds differ from requested protocol")
    mxm_loaded = {k: loaded[k] for k in ("manifest", "catalog", "cache_path") if k in loaded}
    lyrics, mxm_provenance = mxm_builder(mxm_loaded, inputs["mxm_path"])
    result_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{NAMESPACE}.stage-", dir=result_root))
    try:
        (stage / "checkpoints").mkdir()
        (stage / "triplet_schedules").mkdir()
        (stage / "d2_config.json").write_bytes(config_bytes)
        records, curves, diagnostics, comparisons, schedule_hashes = [], [], [], [], {}
        for fold in loaded["folds"]:
            f = fold["fold"]
            ids = sorted(fold["train"] + fold["validation"])
            fold_data = {k: loaded[k] for k in ("manifest", "catalog", "cache_path") if k in loaded}
            fold_info = {k: fold[k] for k in ("fold", "train", "validation")}
            features, transform = feature_builder(fold_data, fold_info, lyrics)
            if set(features) != {"audio", "bpm", "genre", "lyrics"}:
                raise ValueError("development feature builder must return train/validation modalities only")
            transform_hash = _digest(transform)
            train_ids, val_ids = list(fold["train"]), list(fold["validation"])
            train_ds = dataset_class(train_ids, _slice(features, ids, train_ids), modalities=MODALITIES)
            val_ds = dataset_class(val_ids, _slice(features, ids, val_ids), modalities=MODALITIES)
            for seed in seeds:
                source = schedules[(f, seed)]
                schedule_data = source.read_bytes()
                schedule_sha = hashlib.sha256(schedule_data).hexdigest()
                d0 = d0_models.get((f, seed))
                if (schedule_sha != before[f"schedule:{f}/{seed}"]
                        or d0.get("schedule_sha256") != schedule_sha
                        or d0_schedule_hashes.get(f"{f}/{seed}") != schedule_sha):
                    raise ValueError("D0 schedule hash differs from its D-model or schedule-copy reference")
                with np.load(io.BytesIO(schedule_data), allow_pickle=False) as archive:
                    schedule = {k: archive[k].copy() for k in ("train_ids", "validation_ids", "train_schedule", "validation_schedule")}
                if list(schedule["train_ids"].astype(str)) != train_ids or list(schedule["validation_ids"].astype(str)) != val_ids:
                    raise ValueError("D0 schedule IDs differ from current fold")
                if schedule["train_schedule"].dtype != np.int32 or schedule["train_schedule"].ndim != 3 or schedule["train_schedule"].shape[2] != 3:
                    raise ValueError("D0 train schedule has invalid dtype or shape")
                dest_rel = f"triplet_schedules/fold_{f}_seed_{seed}.npz"
                (stage / dest_rel).write_bytes(schedule_data)
                copied_hash = sha256_file(stage / dest_rel)
                if copied_hash != before[f"schedule:{f}/{seed}"]:
                    raise ValueError("copied schedule is not byte-identical to D0")
                schedule_hashes[f"{f}/{seed}"] = copied_hash
                cosine = d0_cosines.get(f)
                if d0 is None or cosine is None or not _metrics_valid(d0.get("validation")) or not _metrics_valid(cosine.get("validation")):
                    raise ValueError("D0 matched model/cosine validation reference is unavailable")
                candidate = stage / f".candidate_fold_{f}_D_seed_{seed}.pt"
                provenance = {"fold": f, "seed": seed, "variant": "D",
                    "input_sha256": {k: before[f"input:{k}"] for k in INPUTS if k not in {"mxm_path", "test_edges_path", "ab_test_edges_path", "cv_dir"}},
                    "split_manifest_sha256": manifest_hash, "transform_sha256": transform_hash,
                    "schedule_sha256": copied_hash, "known_positive_exclusion_schedule_sha256": copied_hash,
                    "known_positive_edges_sha256_exclusion_only": known_hash,
                    "official_test_edge_source_sha256_exclusion_only": before["input:test_edges_path"],
                    "legacy_test_edge_source_sha256_exclusion_only": before["input:ab_test_edges_path"],
                    "test_source_edges_used_as_targets": False, "mxm": mxm_provenance,
                    "dropout_seed_derivation": "seed * 1000003 + fold * 1009",
                    "dropout_seed": seed * 1000003 + f * 1009}
                result = trainer(train_ds, val_ds, schedule["train_schedule"], schedule["validation_schedule"],
                    [{"source": a, "target": b} for a, b in fold["validation_edges"]], "D", seed,
                    candidate, transform, provenance, epochs=epochs, batch_size=batch_size,
                    bpm_dropout_probability=DROP_PROBS["bpm"], genre_dropout_probability=DROP_PROBS["genre"],
                    lyrics_dropout_probability=DROP_PROBS["lyrics"], dropout_seed=seed * 1000003 + f * 1009)
                history, best_epoch, best_metrics = result["history"], result["best_epoch"], result["best_metrics"]
                if len(history) != epochs or sorted(row.get("epoch") for row in history) != list(range(1, epochs + 1)):
                    raise ValueError("trainer returned incomplete epoch history")
                if not 1 <= int(best_epoch) <= epochs or not _metrics_valid(best_metrics):
                    raise ValueError("trainer returned invalid validation checkpoint selection")
                cp_rel = f"checkpoints/fold_{f}_D_seed_{seed}.pt"
                os.replace(candidate, stage / cp_rel)
                cp_hash = sha256_file(stage / cp_rel)
                records.append({"fold": f, "variant": "D", "seed": seed, "selected_epoch": int(best_epoch),
                    "validation": best_metrics, "checkpoint_path": cp_rel, "checkpoint_sha256": cp_hash,
                    "schedule_sha256": copied_hash, "transform_sha256": transform_hash,
                    "d0_model_reference": {"path": f"neurotune_v1_full_modal_ablation/{d0['checkpoint_path']}",
                        "sha256": d0["checkpoint_sha256"],
                        "selected_epoch": d0["selected_epoch"], "validation": d0["validation"]},
                    "d0_cosine_reference": {"fold": f, "validation": cosine["validation"]}})
                comparisons.append({"fold": f, "seed": seed, "d2_validation": best_metrics,
                    "d0_model_validation": d0["validation"], "d0_cosine_validation": cosine["validation"]})
                for row in history:
                    curves.append({"fold": f, "variant": "D", "seed": seed, **row})
                    diag = {"fold": f, "seed": seed, "epoch": row["epoch"]}
                    for modality in MODALITIES:
                        attempted = int(row.get(f"{modality}_dropout_attempted_count", 0))
                        dropped = int(row.get(f"{modality}_dropout_dropped_count", 0))
                        diag[f"{modality}_attempted"] = attempted
                        diag[f"{modality}_dropped"] = dropped
                        diag[f"{modality}_realized_rate"] = float(
                            row.get(f"{modality}_dropout_realized_rate", dropped / attempted if attempted else 0.0))
                    diagnostics.append(diag)
                del result, history
            del train_ds, val_ds, features
        curves.sort(key=lambda r: (r["fold"], r["seed"], r["epoch"]))
        for name, rows in (("training_curves.csv", curves), ("modality_dropout_diagnostics.csv", diagnostics)):
            fields = sorted({key for row in rows for key in row})
            with (stage / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        expected_runs = len(folds) * len(seeds)
        if len(records) != expected_runs or len(schedule_hashes) != expected_runs:
            raise RuntimeError("D2 output count barrier failed")
        data_hashes = {k: before[f"input:{k}"] for k in INPUTS if k not in {"test_edges_path", "ab_test_edges_path", "mxm_path", "cv_dir"}}
        code_hashes = {p: before[f"code:{p}"] for p in CODE_PATHS}
        output_hashes = {p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        report = {"status": "complete", "protocol": {**config, "folds": list(folds), "seeds": list(seeds),
                   "epochs": epochs, "batch_size": batch_size,
                   "dropout_seed_derivation": "seed * 1000003 + fold * 1009; trainer epoch stream uses dropout_seed + epoch_index_zero_based"},
            "hashes": {"data_sources": data_hashes, "code": code_hashes,
                "split_manifest": manifest_hash, "cv_edge_artifacts_directory": before["input:cv_dir"],
                "fold_edge_csv_sha256": fold_edge_hashes, "d0_experiment": before["d0_experiment"],
                "d0_schedules": schedule_hashes, "schedule_copies": schedule_hashes,
                "config": before["config"], "new_checkpoints": {r["checkpoint_path"]: r["checkpoint_sha256"] for r in records},
                "output_files": output_hashes},
            "mxm": mxm_provenance,
            "negative_exclusion_provenance": {"known_positive_edges_sha256": known_hash,
                "official_test_edge_source_sha256_exclusion_only": before["input:test_edges_path"],
                "legacy_test_edge_source_sha256_exclusion_only": before["input:ab_test_edges_path"],
                "test_source_edges_used_as_targets": False},
            "records": records, "validation_comparisons": comparisons,
            "counts": {"training_runs": len(records), "validation_curve_rows": len(curves),
                       "dropout_diagnostic_rows": len(diagnostics), "schedule_copies": len(schedule_hashes)},
            "runtime": {"seconds": time.monotonic() - started, "python": platform.python_version(),
                        "numpy": np.__version__, "pytorch": torch.__version__},
            "test_evaluation_deferred": True, "test_features_constructed": False,
            "test_embeddings_constructed": False, "test_relevance_constructed": False, "test_metrics_computed": False}
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if {k: _hash_path(p) for k, p in protected.items()} != before:
            raise RuntimeError("protected inputs, code, config, or D0 artifacts changed during run")
        created_files, created_dirs = [], []
        try:
            final.mkdir()
            st = final.stat(); created_dirs.append((final, st.st_dev, st.st_ino))
            for rel in ("checkpoints", "triplet_schedules"):
                p = final / rel; p.mkdir(); st = p.stat(); created_dirs.append((p, st.st_dev, st.st_ino))
            staged = sorted(p for p in stage.rglob("*") if p.is_file())
            completion = stage / "experiment.json"
            for source in [p for p in staged if p != completion] + [completion]:
                dest = final / source.relative_to(stage)
                os.link(source, dest); st = dest.stat(); created_files.append((dest, st.st_dev, st.st_ino))
        except BaseException:
            for p, dev, ino in reversed(created_files):
                try:
                    st = p.stat()
                    if (st.st_dev, st.st_ino) == (dev, ino): p.unlink()
                except FileNotFoundError: pass
            for p, dev, ino in reversed(created_dirs):
                try:
                    st = p.stat()
                    if (st.st_dev, st.st_ino) == (dev, ino): p.rmdir()
                except OSError: pass
            raise
        shutil.rmtree(stage)
        return report
    except BaseException:
        try:
            mutated = {k: _hash_path(p) for k, p in protected.items()} != before
        except FileNotFoundError:
            mutated = True
        if stage.exists(): shutil.rmtree(stage)
        if mutated: raise RuntimeError("protected inputs, code, config, or D0 artifacts changed or disappeared")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    _run(args.data_root, args.results_root)


if __name__ == "__main__":
    main()

"""Train A/B/D and reuse C validation references without evaluating test folds."""

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
from src.evaluate_neurotune_v1 import build_neurotune_v1_cosine_baseline, evaluate_neurotune_v1_retrieval
from src.train_neurotune_v1 import train_neurotune_v1

ROOT = Path(__file__).resolve().parents[1]
FOLDS, SEEDS = (0, 1, 2), (684, 685, 686)
EPOCHS, BATCH_SIZE, LEARNING_RATE, MARGIN = 50, 64, 0.001, 0.2
MODALITIES = {"A": (), "B": ("bpm",), "C": ("bpm", "genre"), "D": ("bpm", "lyrics", "genre")}
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
CODE_PATHS = ("scripts/run_neurotune_v1_full_modal_ablation.py", "src/data/neurotune_v1_abcd_cv_training.py",
              "src/data/neurotune_v1_abcd_cv.py", "src/data/parse_mxm.py", "src/data/neurotune_v1_split.py",
              "src/data/neurotune_v1_dataset.py", "src/features_neurotune_v1.py", "src/features.py",
              "src/train_neurotune_v1.py", "src/evaluate_neurotune_v1.py", "src/evaluate.py", "src/retrieve.py",
              "src/data/table.py", "src/model_v1.py", "src/triplets.py")
VALIDATION_METRICS = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10")
VALIDATION_SELECTION_FIELDS = (*VALIDATION_METRICS, "query_count", "eligible_positive_edges")
C_CURVE_REQUIRED_FIELDS = ("Recall@10", "NDCG@10", "MRR@10", "query_count", "eligible_positive_edges")
CAVEATS = (
    "Validation metrics are development/selection evidence only, not an independent generalization estimate.",
    "Each model variant has nine fold/seed observations; each cosine baseline has three fold observations.",
    "Unobserved song pairs are not verified negatives.",
    "The public archive provenance and license are unverified; project use was user-authorized.",
    "Evaluation on the previously inspected test cohort is deferred.",
)
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


def _slice(features, ids, selected_ids, keys=None):
    pos = {track_id: i for i, track_id in enumerate(ids)}
    ix = [pos[x] for x in selected_ids]
    selected = features if keys is None else {k: features[k] for k in keys}
    return {k: ([v[i] for i in ix] if isinstance(v, (list, tuple)) else np.asarray(v)[ix]) for k, v in selected.items()}


def _read_c_curves(content):
    rows = []
    source_rows = 0
    if isinstance(content, bytes):
        stream = io.TextIOWrapper(io.BytesIO(content), newline="", encoding="utf-8")
    else:
        stream = Path(content).open(newline="", encoding="utf-8")
    with stream:
        reader = csv.DictReader(stream)
        required = {"fold", "variant", "seed", "epoch", "train_triplet_loss", "validation_triplet_loss",
                    *C_CURVE_REQUIRED_FIELDS}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("C training curves are missing required validation fields")
        for row in reader:
            source_rows += 1
            if row.get("variant") != "C":
                continue
            parsed = {"variant": "C"}
            for key, value in row.items():
                if key == "variant":
                    continue
                try:
                    parsed[key] = int(value) if key in ("fold", "seed", "epoch", "query_count", "eligible_positive_edges") else float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid C curve value for {key}") from exc
                if isinstance(parsed[key], float) and not np.isfinite(parsed[key]):
                    raise ValueError(f"non-finite C curve value for {key}")
            rows.append(parsed)
    expected = {(f, s) for f in FOLDS for s in SEEDS}
    groups = {(r["fold"], r["seed"]) for r in rows}
    if len(rows) != 450 or source_rows != 450 or groups != expected:
        raise ValueError("C training curves must contain 50 rows for each of 9 fold/seed pairs")
    for f, s in expected:
        group = [r for r in rows if (r["fold"], r["seed"]) == (f, s)]
        if sorted(r["epoch"] for r in group) != list(range(1, EPOCHS + 1)):
            raise ValueError("C curve epochs must be unique and cover 1 through 50")
    return rows


def _check_curve_selection(rows, fold, seed, epoch, metrics):
    selected = next((r for r in rows if r["fold"] == fold and r["seed"] == seed and r["epoch"] == epoch), None)
    if selected is None:
        raise ValueError("C selected epoch is absent from training curves")
    for key, value in metrics.items():
        if key in selected and isinstance(value, (int, float)) and not isinstance(value, bool):
            got = selected[key]
            if not np.isclose(float(got), float(value), rtol=1e-6, atol=1e-8):
                raise ValueError(f"C selection metric {key} differs from selected curve (rtol=1e-6, atol=1e-8)")


def _aggregate_validation(rows):
    return {variant: {metric: {"mean": float(np.mean(values)), "population_sd": float(np.std(values))}
                      for metric in VALIDATION_METRICS
                      if (values := [float(row["validation"][metric]) for row in rows
                                     if row["variant"] == variant and metric in row["validation"]])}
            for variant in ("A", "B", "C", "D")}


def _validate_validation_metrics(metrics, label):
    if (not isinstance(metrics, dict) or not set(VALIDATION_SELECTION_FIELDS) <= set(metrics)
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value)
                   for value in metrics.values())):
        raise ValueError(f"{label} must contain all required finite validation metrics and counts")


def _run(data_root=ROOT, results_root=ROOT / "results", *, _test_only=False, _dependencies=None):
    started = time.monotonic()
    deps = _dependencies or {}
    loader = deps.get("loader", load_abcd_cv_training)
    mxm_builder = deps.get("mxm_builder", build_abcd_cv_fold_mxm)
    feature_builder = deps.get("feature_builder", build_abcd_cv_development_features)
    trainer = deps.get("trainer", train_neurotune_v1)
    baseline_builder = deps.get("baseline_builder", build_neurotune_v1_cosine_baseline)
    evaluator = deps.get("evaluator", evaluate_neurotune_v1_retrieval)
    if not _test_only and deps:
        raise ValueError("production dependencies and protocol are fixed")
    root, result_root = Path(data_root), Path(results_root)
    final = result_root / "neurotune_v1_full_modal_ablation"
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    inputs = {k: root / v for k, v in INPUTS.items()}
    croot = result_root / "neurotune_v1_abcd_cv"
    cfiles = {f"checkpoint_{f}_{s}": croot / f"checkpoints/fold_{f}_C_seed_{s}.pt" for f in FOLDS for s in SEEDS}
    cfiles.update({f"schedule_{f}_{s}": croot / f"triplet_schedules/fold_{f}_seed_{s}.npz" for f in FOLDS for s in SEEDS})
    cfiles["curves"] = croot / "training_curves.csv"
    protected = {**inputs, **{f"code:{p}": ROOT / p for p in CODE_PATHS}, **cfiles}
    before = {k: _hash_path(p) for k, p in protected.items()}
    manifest_hash = sha256_file(inputs["cv_dir"] / "manifest.json")
    known_positive_path = inputs["cv_dir"] / "known_positive_edges.csv"
    known_positive_hash = sha256_file(known_positive_path)
    fold_edge_hashes = {path.name: sha256_file(path)
                        for path in sorted(inputs["cv_dir"].glob("fold_*_edges.csv"))}
    loader_inputs = {k: value for k, value in inputs.items() if k != "mxm_path"}
    loaded = loader(**loader_inputs)
    fold_numbers = [fold.get("fold") for fold in loaded.get("folds", [])]
    if fold_numbers != list(FOLDS):
        raise ValueError("loaded folds must be the fixed unique folds 0, 1, and 2 in order")
    mxm_loaded = {k: loaded[k] for k in ("manifest", "catalog", "cache_path") if k in loaded}
    lyrics, mxm_provenance = mxm_builder(mxm_loaded, inputs["mxm_path"])
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".neurotune_v1_full_modal_ablation.stage-", dir=final.parent))
    try:
        (stage / "checkpoints").mkdir(); (stage / "triplet_schedules").mkdir()
        curve_bytes = cfiles["curves"].read_bytes()
        if hashlib.sha256(curve_bytes).hexdigest() != before["curves"]:
            raise ValueError("C training curves changed after pre-run snapshot")
        records, baselines, curves = [], [], _read_c_curves(curve_bytes)
        c_references, schedule_copy_hashes = [], {}
        for fold in loaded["folds"]:
            f = fold["fold"]
            if f not in FOLDS:
                raise ValueError("unexpected fold")
            ids = sorted(fold["train"] + fold["validation"])
            feature_loaded = {k: loaded[k] for k in ("manifest", "catalog", "cache_path") if k in loaded}
            development_fold = {k: fold[k] for k in ("fold", "train", "validation")}
            features, transform = feature_builder(feature_loaded, development_fold, lyrics)
            if set(features) != {"audio", "bpm", "genre", "lyrics"}:
                raise ValueError("development feature builder must return audio/BPM/genre/lyrics")
            transform_hash = _digest(transform)
            fold_schedules, schedule_bytes_by_seed = {}, {}
            for s in SEEDS:
                cp = cfiles[f"checkpoint_{f}_{s}"]; sp = cfiles[f"schedule_{f}_{s}"]
                checkpoint_bytes = cp.read_bytes()
                checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
                if checkpoint_hash != before[f"checkpoint_{f}_{s}"]:
                    raise ValueError("C checkpoint bytes changed after pre-run snapshot")
                checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
                schedule_bytes = sp.read_bytes()
                if hashlib.sha256(schedule_bytes).hexdigest() != before[f"schedule_{f}_{s}"]:
                    raise ValueError("C schedule bytes changed after pre-run snapshot")
                schedule_bytes_by_seed[s] = schedule_bytes
                with np.load(io.BytesIO(schedule_bytes), allow_pickle=False) as archive:
                    schedule = {k: archive[k].copy() for k in ("train_ids", "validation_ids", "train_schedule", "validation_schedule")}
                fold_schedules[s] = schedule
                config = checkpoint.get("config", {})
                provenance = checkpoint.get("provenance", {})
                if (provenance.get("fold") != f or provenance.get("seed") != s
                        or provenance.get("variant") != "C" or checkpoint.get("fold", f) != f
                        or checkpoint.get("seed") != s or checkpoint.get("variant") != "C"
                        or config.get("device") != "cpu" or config.get("epochs") != EPOCHS
                        or config.get("batch_size") != BATCH_SIZE or config.get("optimizer") != "Adam"
                        or config.get("learning_rate") != LEARNING_RATE or config.get("margin") != MARGIN):
                    raise ValueError("C checkpoint fold/seed/variant/config mismatch")
                old_transform = checkpoint.get("transform_manifest")
                if not isinstance(old_transform, dict) or provenance.get("transform_sha256") != _digest(old_transform):
                    raise ValueError("C checkpoint transform manifest hash mismatch")
                if list(schedule["train_ids"].astype(str)) != list(fold["train"]) or list(schedule["validation_ids"].astype(str)) != list(fold["validation"]):
                    raise ValueError("C schedule train/validation IDs differ from current fold")
                for name, partition, epochs in (("train_schedule", "train", EPOCHS),
                                                ("validation_schedule", "validation", 1)):
                    arr = schedule[name]
                    expected_rows = len(fold[partition])
                    if arr.dtype != np.int32 or arr.ndim != 3 or arr.shape[0] != epochs or arr.shape[2] != 3 or arr.shape[1] == 0:
                        raise ValueError(f"C {name} dtype or shape is invalid")
                    if np.any(arr < 0) or np.any(arr >= expected_rows):
                        raise ValueError(f"C {name} indices are outside their partition")
                if schedule["train_ids"].dtype.kind not in "US" or schedule["validation_ids"].dtype.kind not in "US":
                    raise ValueError("C schedule ID arrays must be strings")
                if provenance.get("schedule_sha256") != before[f"schedule_{f}_{s}"]:
                    raise ValueError("C checkpoint schedule hash mismatch")
                if provenance.get("split_manifest_sha256") != manifest_hash:
                    raise ValueError("C checkpoint split manifest hash mismatch")
                if provenance.get("input_sha256") != {k: before[k] for k in INPUTS if k != "mxm_path"}:
                    # Loader records the original CV input set (without MXM).
                    raise ValueError("C checkpoint input hashes differ from current source files")
                active = {k: old_transform.get(k) for k in ACTIVE_TRANSFORM_FIELDS}
                current = {k: transform.get(k) for k in ACTIVE_TRANSFORM_FIELDS}
                active_sources = {k: old_transform.get("source_sha256", {}).get(k)
                                  for k in ("audio_cache", "split_manifest")}
                current_sources = {k: transform.get("source_sha256", {}).get(k)
                                   for k in ("audio_cache", "split_manifest")}
                active_code = {k: old_transform.get("code_sha256", {}).get(k)
                               for k in ("src/features_neurotune_v1.py", "src/features.py")}
                current_code = {k: sha256_file(ROOT / k) for k in active_code}
                if active != current or active_sources != current_sources or active_code != current_code or provenance.get("transform_sha256") is None:
                    raise ValueError("C active transform fields differ from development transform")
                metrics = checkpoint.get("selection_metrics")
                _validate_validation_metrics(metrics, "C checkpoint selection_metrics")
                selected_epoch = checkpoint.get("epoch")
                if not isinstance(selected_epoch, int) or not 1 <= selected_epoch <= EPOCHS:
                    raise ValueError("C checkpoint selected epoch is invalid")
                _check_curve_selection(curves, f, s, selected_epoch, metrics)
                c_references.append({"fold": f, "variant": "C", "seed": s,
                    "path": cp.relative_to(result_root).as_posix(),
                    "sha256": checkpoint_hash, "schedule_sha256": before[f"schedule_{f}_{s}"],
                    "selected_epoch": selected_epoch, "validation": metrics,
                    "transform_sha256": provenance["transform_sha256"],
                    "active_transform_sha256": _digest(active)})
                del checkpoint
            # Copy each verified schedule once into the new namespace.
            for s in SEEDS:
                dest = stage / f"triplet_schedules/fold_{f}_seed_{s}.npz"
                dest.write_bytes(schedule_bytes_by_seed[s])
                schedule_copy_hashes[f"{f}/{s}"] = sha256_file(dest)
                if schedule_copy_hashes[f"{f}/{s}"] != before[f"schedule_{f}_{s}"]:
                    raise ValueError("copied schedule hash differs from read-only C schedule")
            val_ids = list(fold["validation"])
            for variant in ("A", "B", "C", "D"):
                modalities = ("audio", *MODALITIES[variant])
                baseline_features = _slice(features, ids, val_ids, keys=modalities)
                embedding = baseline_builder(baseline_features, val_ids, modalities=modalities)
                metrics = evaluator(embedding, val_ids, fold["validation_edges"])
                baselines.append({"fold": f, "variant": variant, "validation": metrics})
            for variant in ("A", "B", "D"):
                active = MODALITIES[variant]
                train_ids = list(fold["train"])
                keep = ("audio", *active)
                train_features = _slice(features, ids, train_ids, keys=keep)
                vf = _slice(features, ids, val_ids, keys=keep)
                train_ds = NeuroTuneV1Dataset(train_ids, train_features, modalities=active)
                val_ds = NeuroTuneV1Dataset(val_ids, vf, modalities=active)
                for seed in SEEDS:
                    train_schedule = fold_schedules[seed]["train_schedule"]
                    val_schedule = fold_schedules[seed]["validation_schedule"]
                    cp_rel = f"checkpoints/fold_{f}_{variant}_seed_{seed}.pt"
                    provenance = {"fold": f, "seed": seed, "variant": variant,
                                  "input_sha256": {k: before[k] for k in INPUTS
                                                   if k not in {"mxm_path", "test_edges_path", "ab_test_edges_path", "cv_dir"}},
                                   "split_manifest_sha256": manifest_hash, "transform_sha256": transform_hash,
                                   "schedule_sha256": before[f"schedule_{f}_{seed}"], "mxm": mxm_provenance,
                                   "known_positive_exclusion_schedule_sha256": before[f"schedule_{f}_{seed}"],
                                   "known_positive_edges_sha256_exclusion_only": known_positive_hash,
                                   "official_test_edge_source_sha256_exclusion_only": before["test_edges_path"],
                                   "legacy_test_edge_source_sha256_exclusion_only": before["ab_test_edges_path"],
                                   "test_source_edges_used_as_targets": False}
                    candidate_path = stage / f".candidate_fold_{f}_{variant}_seed_{seed}.pt"
                    result = trainer(train_ds, val_ds, train_schedule, val_schedule,
                        [{"source": a, "target": b} for a, b in fold["validation_edges"]], variant, seed,
                        candidate_path, transform, provenance, epochs=EPOCHS, batch_size=BATCH_SIZE)
                    history = result["history"]
                    best_epoch = result["best_epoch"]
                    best_metrics = result["best_metrics"]
                    del result
                    if (len(history) != EPOCHS or sorted(row.get("epoch") for row in history) != list(range(1, EPOCHS + 1))
                            or any(any(not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value)
                                       for key, value in row.items() if key != "epoch") for row in history)):
                        raise ValueError("trainer must return 50 unique finite epoch records")
                    if not 1 <= int(best_epoch) <= EPOCHS:
                        raise ValueError("trainer selected epoch must be in 1..50")
                    _validate_validation_metrics(best_metrics, "trainer best_metrics")
                    final_checkpoint = stage / cp_rel
                    os.replace(candidate_path, final_checkpoint)
                    curves.extend({"fold": f, "variant": variant, "seed": seed, **row} for row in history)
                    records.append({"fold": f, "variant": variant, "seed": seed, "selected_epoch": best_epoch,
                                    "validation": best_metrics, "checkpoint_path": cp_rel,
                                    "checkpoint_sha256": sha256_file(final_checkpoint), "schedule_sha256": before[f"schedule_{f}_{seed}"],
                                    "transform_sha256": transform_hash})
                    del history, best_metrics, train_schedule, val_schedule
                del train_ds, val_ds, train_features, vf
            del features
            del fold_schedules, schedule_bytes_by_seed
        curves.sort(key=lambda r: (int(r["fold"]), r["variant"], int(r["seed"]), int(r["epoch"])))
        fields = sorted({k for row in curves for k in row})
        with (stage / "training_curves.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(curves)
        if len(records) != 27 or len(c_references) != 9 or len(baselines) != 12 or len(curves) != 1800 or len(schedule_copy_hashes) != 9:
            raise RuntimeError("validation output count barrier failed")
        source_hashes = {k: before[k] for k in INPUTS
                         if k not in {"test_edges_path", "ab_test_edges_path", "mxm_path", "cv_dir"}}
        code_hashes = {path: before[f"code:{path}"] for path in CODE_PATHS}
        model_records = c_references + records
        negative_exclusion_hashes = {"known_positive_edges_sha256": known_positive_hash,
                                     "official_test_edge_source_sha256_exclusion_only": before["test_edges_path"],
                                     "legacy_test_edge_source_sha256_exclusion_only": before["ab_test_edges_path"],
                                     "official_test_edge_source_used_as_targets": False,
                                     "legacy_test_edge_source_used_as_targets": False}
        report = {"status": "complete", "protocol": {"folds": FOLDS, "seeds": SEEDS, "epochs": EPOCHS,
                  "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "triplet_margin": MARGIN,
                  "optimizer": "Adam", "device": "cpu", "trained_variants": ["A", "B", "D"], "modalities": MODALITIES},
                   "hashes": {"data_sources": source_hashes, "code": code_hashes,
                              "cv_edge_artifacts_directory": before["cv_dir"],
                              "fold_edge_csv_sha256": fold_edge_hashes,
                             "split_manifest": manifest_hash,
                             "mxm_raw_sources": mxm_provenance.get("source_sha256"),
                             "mxm_directory": mxm_provenance.get("directory_sha256"),
                             "mxm_parsed_mapping": mxm_provenance.get("mapping_sha256"),
                             "c_checkpoints": {f"{r['fold']}/{r['seed']}": r["sha256"] for r in c_references},
                             "c_schedules": {f"{r['fold']}/{r['seed']}": r["schedule_sha256"] for r in c_references},
                             "c_training_curves": before["curves"],
                             "new_training_curves": sha256_file(stage / "training_curves.csv"),
                             "new_checkpoints": {r["checkpoint_path"]: r["checkpoint_sha256"] for r in records},
                             "schedule_copies": schedule_copy_hashes},
                   "negative_exclusion_provenance": negative_exclusion_hashes,
                   "mxm": mxm_provenance, "model_validation": model_records, "cosine_validation": baselines,
                  "aggregate_model_validation": _aggregate_validation(model_records),
                  "aggregate_cosine_validation": _aggregate_validation(baselines),
                  "caveats": list(CAVEATS),
                  "counts": {"model_validation_observations": 36, "new_training_runs": 27,
                             "read_only_c_references": 9, "cosine_validation_observations": 12,
                             "training_curve_rows": 1800, "schedule_copies": 9, "new_checkpoints": 27},
                  "runtime": {"seconds": time.monotonic() - started, "python": platform.python_version(),
                              "numpy": np.__version__, "pytorch": torch.__version__},
                  "test_evaluation_deferred": True}
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        after = {k: _hash_path(p) for k, p in protected.items()}
        if before != after:
            raise RuntimeError("protected inputs, code, or C artifacts changed during run")
        created_files, created_dirs = [], []
        try:
            final.mkdir()
            stat = final.stat(); created_dirs.append((final, stat.st_dev, stat.st_ino))
            for rel in ("checkpoints", "triplet_schedules"):
                path = final / rel; path.mkdir()
                stat = path.stat(); created_dirs.append((path, stat.st_dev, stat.st_ino))
            staged = sorted(p for p in stage.rglob("*") if p.is_file())
            completion = stage / "experiment.json"
            for source in [p for p in staged if p != completion] + [completion]:
                destination = final / source.relative_to(stage)
                os.link(source, destination)
                stat = destination.stat(); created_files.append((destination, stat.st_dev, stat.st_ino))
        except BaseException:
            for path, device, inode in reversed(created_files):
                try:
                    stat = path.stat()
                    if (stat.st_dev, stat.st_ino) == (device, inode): path.unlink()
                except FileNotFoundError: pass
            for path, device, inode in reversed(created_dirs):
                try:
                    stat = path.stat()
                    if (stat.st_dev, stat.st_ino) == (device, inode): path.rmdir()
                except OSError: pass
            raise
        shutil.rmtree(stage)
        return report
    except BaseException:
        try:
            mutated = {k: _hash_path(p) for k, p in protected.items()} != before
        except FileNotFoundError:
            mutated = True
        if stage.exists():
            shutil.rmtree(stage)
        if mutated:
            raise RuntimeError("protected inputs, code, or C artifacts changed or disappeared during run")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    _run(args.data_root, args.results_root)


if __name__ == "__main__":
    main()

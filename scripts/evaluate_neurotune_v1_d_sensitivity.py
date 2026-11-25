"""Evaluate one-at-a-time D input masking on frozen validation galleries only."""

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

from scripts.run_neurotune_v1_full_modal_ablation import (
    CODE_PATHS as ABLATION_CODE_PATHS, FOLDS, INPUTS, ROOT, SEEDS, _digest, _hash_path,
)
from src.data.neurotune_v1_abcd_cv_training import (
    build_abcd_cv_development_features, build_abcd_cv_fold_mxm, load_abcd_cv_training,
)
from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
from src.data.neurotune_v1_split import sha256_file
from src.evaluate_d_modality_sensitivity import mask_one_modality, measure_behavior_change
from src.evaluate_neurotune_v1 import evaluate_neurotune_v1_retrieval
from src.model_v1 import NeuroTuneV1Encoder

CONDITIONS = ("original", "audio", "bpm", "genre", "lyrics")
MASKS = {"audio": "audio", "bpm": "bpm", "genre": "genre", "lyrics": "lyrics"}
METRICS = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10")
CODE_PATHS = tuple(dict.fromkeys((*ABLATION_CODE_PATHS, "scripts/evaluate_neurotune_v1_d_sensitivity.py",
                                  "src/evaluate_d_modality_sensitivity.py")))
P0_RELATIVE = Path("neurotune_v1_full_modal_ablation")
OUTPUT_NAME = "neurotune_v1_d_modality_sensitivity"
ACTIVE_TRANSFORM_FIELDS = ("fit_ids_hash", "audio_mean", "audio_std", "audio_fit_rows", "bpm_mean",
                           "bpm_std", "genre_vocabulary", "split_hash")


def _assert_original_parity(measured, expected, *, atol=1e-6):
    """Fail closed unless original endpoint metrics and counts match P0."""
    required = (*METRICS, "query_count", "eligible_positive_edges")
    if not isinstance(measured, dict) or not isinstance(expected, dict):
        raise ValueError("original validation parity requires metric mappings")
    if any(key not in measured or key not in expected for key in required):
        raise ValueError("original validation parity record is incomplete")
    for key in required:
        left, right = measured[key], expected[key]
        if key in ("query_count", "eligible_positive_edges"):
            if isinstance(left, bool) or isinstance(right, bool) or int(left) != left or int(right) != right or left != right:
                raise ValueError(f"original validation parity mismatch for {key}")
        elif not np.isfinite(left) or not np.isfinite(right) or not np.isclose(left, right, rtol=0, atol=atol):
            raise ValueError(f"original validation parity mismatch for {key}")


def _slice(features, ids, selected_ids):
    index = {track_id: i for i, track_id in enumerate(ids)}
    rows = [index[track_id] for track_id in selected_ids]
    return {key: ([value[i] for i in rows] if isinstance(value, (list, tuple)) else np.asarray(value)[rows])
            for key, value in features.items()}


def _embed(model, batch):
    args = {name: batch[name] for name in ("audio", "mask", *model.active_modalities)}
    with torch.no_grad():
        return model(**args).detach().cpu().numpy()


def _evaluate_checkpoint(checkpoint, validation_features, validation_ids, positive_edges, expected_metrics):
    """Evaluate one frozen D checkpoint; parity is checked before any masked pass."""
    model = NeuroTuneV1Encoder("D", int(validation_features["genre"].shape[1]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = NeuroTuneV1Dataset(validation_ids, validation_features, modalities=("bpm", "genre", "lyrics"))
    batch = neurotune_v1_collate([dataset[i] for i in range(len(dataset))])
    original_batch = dict(batch)
    original_embeddings = _embed(model, original_batch)
    original_metrics = evaluate_neurotune_v1_retrieval(original_embeddings, validation_ids, positive_edges, k=10)
    parity_metrics = {**original_metrics, "eligible_positive_edges": original_metrics["unique_undirected_edge_count"]}
    _assert_original_parity(parity_metrics, expected_metrics)

    rows = []
    for condition in CONDITIONS:
        if condition == "original":
            result = measure_behavior_change(original_embeddings, original_embeddings, validation_ids, positive_edges)
        else:
            masked_batch = mask_one_modality(batch, MASKS[condition])
            masked_embeddings = _embed(model, masked_batch)
            result = measure_behavior_change(original_embeddings, masked_embeddings, validation_ids, positive_edges)
        metrics = result["retrieval_metrics"]
        endpoint_metrics = original_metrics if condition == "original" else metrics
        endpoint_edges = original_metrics["unique_undirected_edge_count"] if condition == "original" else metrics["unique_undirected_edge_count"]
        rows.append({"condition": condition, **{key: float(endpoint_metrics[key]) for key in METRICS},
                     "query_count": int(endpoint_metrics["query_count"]),
                     "eligible_positive_edges": int(endpoint_edges),
                     "mean_top_10_overlap": result["mean_top_k_overlap"],
                     "mean_embedding_cosine": result["mean_embedding_cosine"],
                     "mean_embedding_cosine_shift": result["mean_embedding_cosine_shift"]})
    if batch.keys() != original_batch.keys() or any(batch[key] is not original_batch[key] for key in batch if key not in MASKS):
        raise RuntimeError("original validation batch was mutated during masking")
    return rows


def _validate_p0_reference(reference, checkpoint, checkpoint_hash, fold, seed, schedule_hash, transform,
                           expected_checkpoint_path):
    provenance = checkpoint.get("provenance", {})
    if reference.get("checkpoint_path") != expected_checkpoint_path:
        raise ValueError("P0 D checkpoint path does not match the exact loaded checkpoint")
    if (reference.get("fold") != fold or reference.get("seed") != seed or reference.get("variant") != "D"
            or provenance.get("fold") != fold or provenance.get("seed") != seed or provenance.get("variant") != "D"
            or reference.get("checkpoint_sha256") != checkpoint_hash
            or reference.get("selected_epoch") != checkpoint.get("epoch")
            or reference.get("schedule_sha256") != schedule_hash
            or provenance.get("schedule_sha256") != schedule_hash
            or reference.get("transform_sha256") != _digest(transform)
            or provenance.get("transform_sha256") != _digest(transform)):
        raise ValueError("P0 D checkpoint fold/seed/variant/epoch/hash/transform/schedule mismatch")
    if checkpoint.get("variant") != "D" or checkpoint.get("fold", fold) != fold or checkpoint.get("seed") != seed:
        raise ValueError("D checkpoint identity mismatch")
    expected_metrics = reference.get("validation")
    if not isinstance(expected_metrics, dict):
        raise ValueError("P0 D reference is missing validation metrics")
    selection_metrics = checkpoint.get("selection_metrics")
    if not isinstance(selection_metrics, dict):
        raise ValueError("D checkpoint selection metrics are missing")
    required = (*METRICS, "query_count", "eligible_positive_edges")
    if any(key not in expected_metrics or key not in selection_metrics for key in required):
        raise ValueError("D checkpoint selection metrics or P0 validation metrics are incomplete")
    for key in required:
        selected, reference_value = selection_metrics[key], expected_metrics[key]
        if (not isinstance(selected, (int, float)) or isinstance(selected, bool)
                or not isinstance(reference_value, (int, float)) or isinstance(reference_value, bool)
                or not np.isfinite(selected) or not np.isfinite(reference_value)):
            raise ValueError(f"D checkpoint selection metric {key} is invalid")
        if key in ("query_count", "eligible_positive_edges"):
            equal = selected == reference_value and int(selected) == selected and int(reference_value) == reference_value
        else:
            equal = np.isclose(selected, reference_value, rtol=0, atol=1e-6)
        if not equal:
            raise ValueError(f"D checkpoint selection metric {key} differs from P0 validation reference")
    return expected_metrics


def _atomic_publish(stage, final):
    """Publish staged files with hard-link no-overwrite semantics and rollback."""
    made_dirs, made_files = [], []
    try:
        final.mkdir()
        st = final.stat(); made_dirs.append((final, st.st_dev, st.st_ino))
        for source in sorted(stage.rglob("*")):
            if not source.is_file():
                continue
            destination = final / source.relative_to(stage)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Record newly created parents for safe rollback.
            parent = destination.parent
            while parent != final and parent not in [item[0] for item in made_dirs]:
                st = parent.stat(); made_dirs.append((parent, st.st_dev, st.st_ino)); parent = parent.parent
            os.link(source, destination)
            st = destination.stat(); made_files.append((destination, st.st_dev, st.st_ino))
    except BaseException:
        for path, device, inode in reversed(made_files):
            try:
                st = path.stat()
                if (st.st_dev, st.st_ino) == (device, inode): path.unlink()
            except FileNotFoundError:
                pass
        for path, device, inode in reversed(made_dirs):
            try:
                st = path.stat()
                if (st.st_dev, st.st_ino) == (device, inode): path.rmdir()
            except FileNotFoundError:
                pass
        raise


def _run(data_root=ROOT, results_root=ROOT / "results", *, _test_only=False, _dependencies=None):
    started = time.monotonic()
    deps = _dependencies or {}
    if deps and not _test_only:
        raise ValueError("production dependencies and protocol are fixed")
    loader = deps.get("loader", load_abcd_cv_training)
    mxm_builder = deps.get("mxm_builder", build_abcd_cv_fold_mxm)
    feature_builder = deps.get("feature_builder", build_abcd_cv_development_features)
    root, result_root = Path(data_root), Path(results_root)
    final = result_root / OUTPUT_NAME
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    p0_root = result_root / P0_RELATIVE
    p0_report_path = p0_root / "experiment.json"
    inputs = {key: root / rel for key, rel in INPUTS.items()}
    checkpoint_paths = {f"{fold}/{seed}": p0_root / f"checkpoints/fold_{fold}_D_seed_{seed}.pt"
                        for fold in FOLDS for seed in SEEDS}
    schedule_paths = {f"{fold}/{seed}": p0_root / f"triplet_schedules/fold_{fold}_seed_{seed}.npz"
                      for fold in FOLDS for seed in SEEDS}
    protected_paths = {**inputs, **{f"code:{p}": ROOT / p for p in CODE_PATHS},
                       "p0_report": p0_report_path, **{f"checkpoint:{k}": v for k, v in checkpoint_paths.items()},
                       **{f"schedule:{k}": v for k, v in schedule_paths.items()}}
    before = {key: _hash_path(path) for key, path in protected_paths.items()}
    p0_report = json.loads(p0_report_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(inputs["cv_dir"] / "manifest.json")
    if p0_report.get("hashes", {}).get("split_manifest") != manifest_hash:
        raise ValueError("P0 split manifest hash differs from the current CV manifest")
    source_hashes = {key: before[key] for key in INPUTS
                     if key not in ("test_edges_path", "ab_test_edges_path", "mxm_path", "cv_dir")}
    if p0_report.get("hashes", {}).get("data_sources") != source_hashes:
        raise ValueError("P0 source hashes differ from current source snapshot")
    references = [row for row in p0_report.get("model_validation", []) if row.get("variant") == "D"]
    refs = {(row.get("fold"), row.get("seed")): row for row in references}
    if len(references) != 9 or set(refs) != {(f, s) for f in FOLDS for s in SEEDS}:
        raise ValueError("P0 must contain exactly nine D validation references")

    loader_inputs = {key: value for key, value in inputs.items() if key != "mxm_path"}
    loaded = loader(**loader_inputs)
    if [fold.get("fold") for fold in loaded.get("folds", [])] != list(FOLDS):
        raise ValueError("loaded folds must be fixed folds 0, 1, and 2")
    mxm_loaded = {key: loaded[key] for key in ("manifest", "catalog", "cache_path") if key in loaded}
    lyrics, mxm_provenance = mxm_builder(mxm_loaded, inputs["mxm_path"])
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.stage-", dir=final.parent))
    try:
        records, checkpoint_provenance = [], {}
        for fold in loaded["folds"]:
            f = fold["fold"]
            fold_input = {key: loaded[key] for key in ("manifest", "catalog", "cache_path") if key in loaded}
            development_fold = {key: fold[key] for key in ("fold", "train", "validation")}
            features, transform = feature_builder(fold_input, development_fold, lyrics)
            expected_ids = sorted(fold["train"] + fold["validation"])
            if set(features) != {"audio", "bpm", "genre", "lyrics"}:
                raise ValueError("development feature builder must return only D inputs")
            transform_hash = _digest(transform)
            validation_ids = sorted(fold["validation"])
            validation_features = _slice(features, expected_ids, validation_ids)
            if any("test" in key.lower() for key in validation_features):
                raise ValueError("test features are forbidden")
            positive_edges = list(fold["validation_edges"])
            if any(a not in set(validation_ids) or b not in set(validation_ids) for a, b in positive_edges):
                raise ValueError("validation relevance edges must have validation endpoints")
            for seed in SEEDS:
                key = f"{f}/{seed}"
                cp_bytes = checkpoint_paths[key].read_bytes()
                cp_hash = hashlib.sha256(cp_bytes).hexdigest()
                if cp_hash != before[f"checkpoint:{key}"]:
                    raise ValueError("D checkpoint changed after pre-run snapshot")
                checkpoint = torch.load(io.BytesIO(cp_bytes), map_location="cpu", weights_only=False)
                schedule_bytes = schedule_paths[key].read_bytes()
                schedule_hash = hashlib.sha256(schedule_bytes).hexdigest()
                with np.load(io.BytesIO(schedule_bytes), allow_pickle=False) as archive:
                    schedule_ids = list(archive["validation_ids"].astype(str))
                if schedule_ids != list(fold["validation"]):
                    raise ValueError("P0 schedule validation IDs differ from current fold")
                reference = refs[(f, seed)]
                expected_metrics = _validate_p0_reference(reference, checkpoint, cp_hash, f, seed,
                                                          schedule_hash, transform,
                                                          checkpoint_paths[key].relative_to(p0_root).as_posix())
                provenance = checkpoint.get("provenance", {})
                old_transform = checkpoint.get("transform_manifest")
                if not isinstance(old_transform, dict) or _digest(old_transform) != provenance.get("transform_sha256"):
                    raise ValueError("D checkpoint transform manifest hash mismatch")
                active_fields = {key: old_transform.get(key) for key in ACTIVE_TRANSFORM_FIELDS}
                current_fields = {key: transform.get(key) for key in ACTIVE_TRANSFORM_FIELDS}
                if active_fields != current_fields:
                    raise ValueError("D checkpoint active transform differs from current development transform")
                active_source_keys = ("audio_cache", "split_manifest")
                active_sources = {key: old_transform.get("source_sha256", {}).get(key) for key in active_source_keys}
                current_sources = {key: transform.get("source_sha256", {}).get(key) for key in active_source_keys}
                active_code_paths = ("src/features_neurotune_v1.py", "src/features.py")
                active_code = {key: old_transform.get("code_sha256", {}).get(key) for key in active_code_paths}
                current_code = {key: sha256_file(ROOT / key) for key in active_code_paths}
                expected_input_hashes = {key: before[key] for key in INPUTS
                                         if key not in ("mxm_path", "test_edges_path", "ab_test_edges_path", "cv_dir")}
                if (active_sources != current_sources or active_code != current_code
                        or current_sources.get("split_manifest") != _digest(loaded["manifest"])
                        or provenance.get("input_sha256") != expected_input_hashes
                        or provenance.get("split_manifest_sha256") != manifest_hash):
                    raise ValueError("D checkpoint source/split/code provenance mismatch")
                rows = _evaluate_checkpoint(checkpoint, validation_features, validation_ids, positive_edges,
                                            expected_metrics)
                for row in rows:
                    records.append({"fold": f, "variant": "D", "seed": seed,
                                    "selected_epoch": checkpoint["epoch"], "checkpoint_sha256": cp_hash,
                                    "schedule_sha256": schedule_hash, "transform_sha256": transform_hash,
                                    "split_manifest_sha256": provenance.get("split_manifest_sha256"), **row})
                checkpoint_provenance[key] = {"path": checkpoint_paths[key].relative_to(result_root).as_posix(),
                                              "sha256": cp_hash, "schedule_sha256": schedule_hash,
                                              "transform_sha256": transform_hash,
                                              "fold": f, "seed": seed, "variant": "D",
                                              "selected_epoch": checkpoint["epoch"]}
                del checkpoint, cp_bytes
            del features, validation_features
        if len(records) != 45:
            raise RuntimeError("expected nine checkpoints times five conditions")
        aggregates = {}
        for condition in CONDITIONS:
            group = [row for row in records if row["condition"] == condition]
            aggregates[condition] = {metric: {"mean": float(np.mean([r[metric] for r in group])),
                                               "population_sd": float(np.std([r[metric] for r in group]))}
                                     for metric in (*METRICS, "mean_top_10_overlap", "mean_embedding_cosine",
                                                    "mean_embedding_cosine_shift")}
        source_hashes = {key: before[key] for key in INPUTS if key not in ("test_edges_path", "ab_test_edges_path", "mxm_path", "cv_dir")}
        report = {"status": "complete", "protocol": {"folds": FOLDS, "seeds": SEEDS,
                   "conditions": CONDITIONS, "k": 10, "variant": "D", "observations": 9,
                   "artist_folds": 3, "observations_are_correlated": True,
                   "test_evaluation_deferred": True},
                  "hashes": {"data_sources": source_hashes,
                             "code": {path: before[f"code:{path}"] for path in CODE_PATHS},
                             "p0_experiment": before["p0_report"],
                             "split_manifest": sha256_file(inputs["cv_dir"] / "manifest.json"),
                             "mxm": mxm_provenance,
                             "d_checkpoints": {key: value["sha256"] for key, value in checkpoint_provenance.items()},
                             "d_schedules": {key: value["schedule_sha256"] for key, value in checkpoint_provenance.items()}},
                  "checkpoint_records": list(checkpoint_provenance.values()), "records": records,
                  "aggregate_descriptive": aggregates,
                  "interpretation": "Zero masking is distribution-shifted diagnostic evidence, not a causal modality contribution estimate.",
                  "counts": {"checkpoint_observations": 9, "conditions_per_checkpoint": 5, "condition_records": 45,
                             "artist_folds": 3},
                  "runtime": {"seconds": time.monotonic() - started, "python": platform.python_version(),
                              "numpy": np.__version__, "pytorch": torch.__version__}}
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (stage / "condition_records.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ("fold", "variant", "seed", "selected_epoch", "condition", *METRICS,
                      "query_count", "eligible_positive_edges", "mean_top_10_overlap",
                      "mean_embedding_cosine", "mean_embedding_cosine_shift", "checkpoint_sha256",
                      "schedule_sha256", "transform_sha256", "split_manifest_sha256")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        after = {key: _hash_path(path) for key, path in protected_paths.items()}
        if before != after:
            raise RuntimeError("protected inputs, code, P0 report, checkpoints, or schedules changed during run")
        _atomic_publish(stage, final)
        return report
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    _run(args.data_root, args.results_root)


if __name__ == "__main__":
    main()

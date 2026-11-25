"""Fresh paired D0/D4 modality masking on frozen validation checkpoints."""

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

from scripts.run_neurotune_v1_full_modal_ablation import FOLDS, INPUTS as D0_INPUTS, ROOT, SEEDS, _digest, _hash_path
from scripts.run_neurotune_v1_d4_multi_window_audio import (
    CODE_PATHS as D4_CODE_PATHS, INPUTS as D4_INPUTS, _take,
)
from src.data.neurotune_v1_abcd_cv_training import (
    build_abcd_cv_development_features, build_abcd_cv_d4_multi_window_features,
    build_abcd_cv_fold_mxm, load_abcd_cv_training,
)
from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
from src.evaluate_d_modality_sensitivity import measure_behavior_change
from src.model_v1 import NeuroTuneV1Encoder

CONDITIONS = ("unmasked", "audio_zero", "bpm_zero", "genre_zero", "lyrics_zero")
MASKS = {"audio_zero": "audio", "bpm_zero": "bpm", "genre_zero": "genre", "lyrics_zero": "lyrics"}
METRICS = ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10")
OUTPUT_NAME = "neurotune_v1_d4_modality_sensitivity"
PROTECTED_RESULT_NAMESPACES = (
    "neurotune_v1_full_modal_ablation", "neurotune_v1_d_modality_sensitivity",
    "neurotune_v1_training_improvements", "neurotune_v1_d2_modality_dropout",
    "neurotune_v1_d3_combined", "neurotune_v1_d4_multi_window_audio",
    "neurotune_v1_d4_multi_window_audio_pilot", "neurotune_v1_abcd_cv",
)
ACTIVE_TRANSFORM_FIELDS = ("fit_ids_hash", "audio_mean", "audio_std", "audio_fit_rows", "bpm_mean",
                           "bpm_std", "genre_vocabulary", "split_hash")
CODE_PATHS = tuple(dict.fromkeys((
    "src/evaluate_d4_modality_sensitivity.py", "scripts/evaluate_neurotune_v1_d4_modality_sensitivity.py",
    "src/evaluate_d_modality_sensitivity.py", "src/model_v1.py", "src/evaluate_neurotune_v1.py",
    "src/retrieve.py", "src/data/neurotune_v1_dataset.py", "src/data/neurotune_v1_abcd_cv_training.py",
    "scripts/run_neurotune_v1_full_modal_ablation.py", *D4_CODE_PATHS,
)))


def _assert_original_parity(measured, expected, *, atol=1e-6):
    required = (*METRICS, "query_count", "eligible_positive_edges")
    if not isinstance(measured, dict) or not isinstance(expected, dict) or any(
            key not in measured or key not in expected for key in required):
        raise ValueError("unmasked validation parity record is incomplete")
    for key in required:
        a, b = measured[key], expected[key]
        if key in ("query_count", "eligible_positive_edges"):
            if type(a) not in (int, np.int64, np.int32) or type(b) not in (int, np.int64, np.int32) or a != b:
                raise ValueError(f"unmasked validation parity mismatch for {key}")
        elif not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a, b, rtol=0, atol=atol):
            raise ValueError(f"unmasked validation parity mismatch for {key}")


def _mask_batch(batch, modality):
    key = ("audio_windows" if "audio_windows" in batch else "audio") if modality == "audio" else modality
    if key not in batch or not torch.is_tensor(batch[key]):
        raise ValueError(f"batch must contain tensor modality {key!r}")
    result = dict(batch)
    result[key] = torch.zeros_like(batch[key])
    return result


def _paired_rows(d0_rows, d4_rows):
    def keyed(rows):
        result = {}
        for row in rows:
            key = (row["fold"], row["seed"], row["condition"])
            if key in result:
                raise ValueError("duplicate fold/seed/condition in paired comparison")
            result[key] = row
        return result
    left, right = keyed(d0_rows), keyed(d4_rows)
    expected = {(fold, seed, condition) for fold in FOLDS for seed in SEEDS for condition in MASKS}
    if set(left) != expected or set(right) != expected:
        raise ValueError("paired comparison requires the exact 36 masked D0/D4 conditions")
    paired = []
    for key in sorted(expected):
        a, b = left[key], right[key]
        paired.append({"fold": key[0], "seed": key[1], "condition": key[2],
                       "D0_delta_ndcg": a["delta_ndcg"], "D4_delta_ndcg": b["delta_ndcg"],
                       "delta_ndcg_D4_minus_D0": b["delta_ndcg"] - a["delta_ndcg"],
                       "relative_delta_ndcg_change_percent": (
                           100 * (b["delta_ndcg"] - a["delta_ndcg"]) / abs(a["delta_ndcg"])
                           if a["delta_ndcg"] != 0 else None),
                       "D0_mean_top_10_overlap": a["mean_top_10_overlap"],
                       "D4_mean_top_10_overlap": b["mean_top_10_overlap"],
                       "D0_mean_embedding_cosine_shift": a["mean_embedding_cosine_shift"],
                       "D4_mean_embedding_cosine_shift": b["mean_embedding_cosine_shift"]})
    return paired


def _select_references(report, variant):
    rows = report.get("model_validation")
    if not isinstance(rows, list):
        raise ValueError(f"{variant} model validation references are missing")
    selected = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("variant") != "D":
            continue
        key = (row.get("fold"), row.get("seed"))
        if key in selected:
            raise ValueError(f"{variant} contains duplicate D fold/seed references")
        selected[key] = row
    expected = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if len(selected) != 9 or set(selected) != expected:
        raise ValueError(f"{variant} must contain exactly nine unique D references")
    return selected


def _validate_report_hashes(d0report, d4report, before):
    d0_sources = d0report.get("hashes", {}).get("data_sources")
    if not isinstance(d0_sources, dict) or not d0_sources:
        raise ValueError("D0 report data-source hashes are missing")
    for key, digest in d0_sources.items():
        current = before.get(f"input:{key}")
        if current is None or digest != current:
            raise ValueError(f"D0 report data-source hash mismatch for {key}")
    if (d4report.get("evaluation_scope") != "validation_only_development"
            or d4report.get("independent_test_artifacts_created") is not False):
        raise ValueError("D4 report must confirm validation-only scope and no independent-test artifacts")
    d4_protected = d4report.get("hashes", {}).get("protected_before")
    if not isinstance(d4_protected, dict):
        raise ValueError("D4 protected-before hashes are missing")
    required = {key for key in before if key.startswith("input:")}
    required.add("d0_report")
    required.update(f"checkpoint:{fold}/{seed}" for fold in FOLDS for seed in SEEDS)
    required.update(f"schedule:{fold}/{seed}" for fold in FOLDS for seed in SEEDS)
    if not required <= set(d4_protected):
        raise ValueError("D4 protected-before hashes are incomplete")
    for key in required:
        current_key = key
        if key.startswith("checkpoint:"):
            _, pair = key.split(":", 1)
            current_key = f"checkpoint:{pair}/D0"
        if d4_protected[key] != before.get(current_key):
            raise ValueError(f"D4 protected-before hash mismatch for {key}")
    copies = d4report.get("hashes", {}).get("schedule_copies")
    expected_copies = {f"{fold}/{seed}" for fold in FOLDS for seed in SEEDS}
    if not isinstance(copies, dict) or set(copies) != expected_copies:
        raise ValueError("D4 report schedule-copy hashes must cover all nine checkpoints")


def _result_namespace_hashes(results_root):
    hashes = {}
    for name in PROTECTED_RESULT_NAMESPACES:
        path = Path(results_root) / name
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(f"protected result namespace must be a directory: {path}")
        hashes[name] = _hash_path(path)
    return hashes


def _validate_schedule_copy(source_path, copy_path, validation_ids, expected_hash):
    source_bytes, copy_bytes = Path(source_path).read_bytes(), Path(copy_path).read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    copy_hash = hashlib.sha256(copy_bytes).hexdigest()
    if source_hash != expected_hash or copy_hash != expected_hash:
        raise ValueError("D4 copied schedule hash differs from the frozen D0 schedule")
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as source, np.load(
            io.BytesIO(copy_bytes), allow_pickle=False) as copied:
        fields = ("train_ids", "validation_ids", "train_schedule", "validation_schedule")
        if set(source.files) != set(fields) or set(copied.files) != set(fields):
            raise ValueError("D0/D4 schedules must contain the exact frozen schedule fields")
        for field in fields:
            left, right = source[field], copied[field]
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                raise ValueError(f"D4 copied schedule differs from D0 for {field}")
        if (list(source["validation_ids"].astype(str)) != list(validation_ids)
                or list(copied["validation_ids"].astype(str)) != list(validation_ids)
                or list(source["train_ids"].astype(str)) != list(copied["train_ids"].astype(str))):
            raise ValueError("D0/D4 copied schedule IDs/order differ from the current fold")
    return copy_hash


def _validate_d0_transform_manifest(current, frozen):
    """Allow only the D4-added feature module's expected D0 code-hash drift."""
    code_path = "src/features_neurotune_v1.py"
    if not isinstance(current, dict) or not isinstance(frozen, dict):
        raise ValueError("D0 current and frozen transforms must be mappings")
    current_code, frozen_code = current.get("code_sha256"), frozen.get("code_sha256")
    if (not isinstance(current_code, dict) or not isinstance(frozen_code, dict)
            or code_path not in current_code or code_path not in frozen_code
            or not isinstance(current_code[code_path], str) or not isinstance(frozen_code[code_path], str)):
        raise ValueError("D0 transforms must include the approved feature-code hash")
    current_without_code, frozen_without_code = dict(current), dict(frozen)
    current_without_code["code_sha256"] = dict(current_code)
    frozen_without_code["code_sha256"] = dict(frozen_code)
    current_without_code["code_sha256"].pop(code_path)
    frozen_without_code["code_sha256"].pop(code_path)
    if _digest(current_without_code) != _digest(frozen_without_code):
        raise ValueError("D0 current transform differs beyond the approved feature-code hash drift")
    return {"path": code_path, "checkpoint_sha256": frozen_code[code_path],
            "current_sha256": current_code[code_path], "drift_detected": current_code[code_path] != frozen_code[code_path],
            "checkpoint_transform_sha256": _digest(frozen), "current_transform_sha256": _digest(current)}


def _validate_checkpoint_transform_identity(checkpoint, variant, reference, current_transform):
    """Bind the frozen manifest for both models, honoring their provenance schemas."""
    expected = reference.get("transform_sha256")
    provenance = checkpoint.get("provenance", {})
    manifest = checkpoint.get("transform_manifest")
    if not isinstance(expected, str) or not isinstance(provenance, dict) or not isinstance(manifest, dict):
        raise ValueError("checkpoint transform identity is incomplete")
    provenance_hash = provenance.get("transform_sha256")
    if variant == "D0":
        if provenance_hash != expected:
            raise ValueError("D0 checkpoint provenance transform hash mismatch")
    elif variant == "D4":
        if "transform_sha256" in provenance and provenance_hash != expected:
            raise ValueError("D4 checkpoint provenance transform hash mismatch")
        if _digest(current_transform) != expected:
            raise ValueError("D4 current transform hash mismatch")
    else:
        raise ValueError("unknown checkpoint transform variant")
    if _digest(manifest) != expected:
        raise ValueError(f"{variant} checkpoint transform manifest hash mismatch")
    return True


def _slice(features, ids, selected_ids):
    index = {track_id: i for i, track_id in enumerate(ids)}
    rows = [index[track_id] for track_id in selected_ids]
    return {key: ([value[i] for i in rows] if isinstance(value, (list, tuple)) else np.asarray(value)[rows])
            for key, value in features.items()}


def _embed(model, batch, d4):
    names = ("audio_windows", "mask", "window_mask", *model.active_modalities) if d4 else (
        "audio", "mask", *model.active_modalities)
    with torch.no_grad():
        return model(**{name: batch[name] for name in names}).detach().cpu().numpy()


def _evaluate(model, features, ids, edges, d4, expected):
    modalities = ("bpm", "genre", "lyrics")
    ds = NeuroTuneV1Dataset(ids, features, modalities=modalities)
    batch = neurotune_v1_collate([ds[i] for i in range(len(ds))])
    original = _embed(model, batch, d4)
    baseline = measure_behavior_change(original, original, ids, edges)
    endpoint = {**baseline["retrieval_metrics"],
                "eligible_positive_edges": baseline["unique_undirected_edge_count"]}
    _assert_original_parity(endpoint, expected)
    results = []
    for condition in CONDITIONS:
        metric = baseline if condition == "unmasked" else measure_behavior_change(
            original, _embed(model, _mask_batch(batch, MASKS[condition]), d4), ids, edges)
        retrieval = metric["retrieval_metrics"]
        results.append({"condition": condition, **{key: float(retrieval[key]) for key in METRICS},
                        "query_count": int(retrieval["query_count"]),
                        "eligible_positive_edges": int(metric["unique_undirected_edge_count"]),
                        "mean_top_10_overlap": metric["mean_top_k_overlap"],
                        "mean_embedding_cosine": metric["mean_embedding_cosine"],
                        "mean_embedding_cosine_shift": metric["mean_embedding_cosine_shift"],
                        "delta_ndcg": float(endpoint["NDCG@10"] - retrieval["NDCG@10"])})
    return results


def _atomic_publish(stage, final):
    made_dirs, made_files = [], []
    try:
        final.mkdir()
        st = final.stat()
        made_dirs.append((final, st.st_dev, st.st_ino))
        for source in sorted(stage.rglob("*")):
            if source.is_file():
                dest = final / source.relative_to(stage)
                dest.parent.mkdir(parents=True, exist_ok=True)
                parent = dest.parent
                while parent != final and parent not in [entry[0] for entry in made_dirs]:
                    st = parent.stat()
                    made_dirs.append((parent, st.st_dev, st.st_ino))
                    parent = parent.parent
                os.link(source, dest)
                st = dest.stat()
                made_files.append((dest, st.st_dev, st.st_ino))
    except BaseException:
        for path, device, inode in reversed(made_files):
            try:
                st = path.stat()
                if (st.st_dev, st.st_ino) == (device, inode):
                    path.unlink()
            except FileNotFoundError:
                pass
        for path, device, inode in reversed(made_dirs):
            try:
                st = path.stat()
                if (st.st_dev, st.st_ino) == (device, inode):
                    path.rmdir()
            except FileNotFoundError:
                pass
        raise


def _run(data_root=ROOT, results_root=ROOT / "results", *, _test_only=False, _dependencies=None):
    started = time.monotonic()
    deps = _dependencies or {}
    if deps and not _test_only:
        raise ValueError("production dependencies and protocol are fixed")
    root, outroot = Path(data_root), Path(results_root)
    final = outroot / OUTPUT_NAME
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    d0root, d4root, schedroot = (outroot / "neurotune_v1_full_modal_ablation",
                                 outroot / "neurotune_v1_d4_multi_window_audio",
                                 outroot / "neurotune_v1_abcd_cv")
    d0report_path, d4report_path = d0root / "experiment.json", d4root / "experiment.json"
    inputs = {**{k: root / v for k, v in D0_INPUTS.items()},
              **{k: root / v for k, v in D4_INPUTS.items()}}
    cps = {(f, s, variant): (d0root if variant == "D0" else d4root) /
           f"checkpoints/fold_{f}_D_seed_{s}.pt" for f in FOLDS for s in SEEDS for variant in ("D0", "D4")}
    schedules = {(f, s): schedroot / f"triplet_schedules/fold_{f}_seed_{s}.npz" for f in FOLDS for s in SEEDS}
    d4_schedule_copies = {(f, s): d4root / f"triplet_schedules/fold_{f}_seed_{s}.npz"
                          for f in FOLDS for s in SEEDS}
    protected = {**{f"input:{k}": p for k, p in inputs.items()}, "d0_report": d0report_path,
                 "d4_report": d4report_path, **{f"checkpoint:{f}/{s}/{v}": p for (f, s, v), p in cps.items()},
                 **{f"schedule:{f}/{s}": p for (f, s), p in schedules.items()},
                 **{f"schedule_copy:{f}/{s}": p for (f, s), p in d4_schedule_copies.items()},
                 **{f"code:{p}": ROOT / p for p in CODE_PATHS}}
    before = {k: _hash_path(v) for k, v in protected.items()}
    namespaces_before = _result_namespace_hashes(outroot)
    d0report = json.loads(d0report_path.read_text(encoding="utf-8"))
    d4report = json.loads(d4report_path.read_text(encoding="utf-8"))
    d0refs = _select_references(d0report, "D0")
    d4refs = _select_references(d4report, "D4")
    pairs = {(f, s) for f in FOLDS for s in SEEDS}
    if d0report.get("status") != "complete" or d4report.get("status") != "complete":
        raise ValueError("D0/D4 references must be complete for all fixed fold/seed pairs")
    _validate_report_hashes(d0report, d4report, before)
    loader = deps.get("loader", load_abcd_cv_training)
    loaded = loader(**{k: p for k, p in inputs.items() if k in D0_INPUTS and k not in ("mxm_path",)})
    mxm_builder = deps.get("mxm_builder", build_abcd_cv_fold_mxm)
    lyrics, mxm_provenance = mxm_builder({k: loaded[k] for k in ("manifest", "catalog", "cache_path")}, inputs["mxm_path"])
    d0_builder = deps.get("d0_builder", build_abcd_cv_development_features)
    d4_builder = deps.get("d4_builder", build_abcd_cv_d4_multi_window_features)
    records, d0_transform_code_drift = [], {}
    stage = None
    outroot.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.stage-", dir=outroot))
    try:
        for fold in loaded["folds"]:
            f = fold["fold"]
            fold_input = {k: loaded[k] for k in ("manifest", "catalog", "cache_path")}
            development_fold = {k: fold[k] for k in ("fold", "train", "validation")}
            d0features, d0transform = d0_builder(fold_input, development_fold, lyrics)
            d4features, d4transform = d4_builder(fold_input, development_fold, lyrics, inputs["d4_cache_path"])
            if set(d0features) != {"audio", "bpm", "genre", "lyrics"}:
                raise ValueError("D0 development builder must return only D inputs")
            d4non = {k: v for k, v in d4transform.items() if k not in {"audio_mean", "audio_std", "audio_fit_rows"}}
            d0non = {k: v for k, v in d0transform.items() if k not in {"audio_mean", "audio_std", "audio_fit_rows"}}
            d4non["source_sha256"] = {k: v for k, v in d4non["source_sha256"].items() if k != "d4_audio_cache"}
            d0non["source_sha256"] = {k: v for k, v in d0non["source_sha256"].items() if k != "audio_cache"}
            if _digest(d4non) != _digest(d0non):
                raise ValueError("D0 and D4 non-audio train-fitted transforms differ")
            all_ids = sorted(fold["train"] + fold["validation"])
            val_ids = list(fold["validation"])
            if set(all_ids) & set(fold.get("test", [])):
                raise ValueError("test IDs may not enter feature builders")
            edges = list(fold["validation_edges"])
            if any(a not in set(val_ids) or b not in set(val_ids) for a, b in edges):
                raise ValueError("validation edge endpoints must be validation IDs")
            dv0, dv4 = _slice(d0features, all_ids, val_ids), _take(d4features, all_ids, val_ids)
            for seed in SEEDS:
                key = (f, seed)
                ref0, ref4 = d0refs[key], d4refs[key]
                cp0, cp4, sp = cps[(f, seed, "D0")], cps[(f, seed, "D4")], schedules[key]
                copied_schedule = d4_schedule_copies[key]
                b0, b4 = cp0.read_bytes(), cp4.read_bytes()
                h0, h4, hs = hashlib.sha256(b0).hexdigest(), hashlib.sha256(b4).hexdigest(), before[f"schedule:{f}/{seed}"]
                if (h0 != before[f"checkpoint:{f}/{seed}/D0"]
                        or h4 != before[f"checkpoint:{f}/{seed}/D4"]):
                    raise ValueError("checkpoint changed after protected input snapshot")
                copy_hash = _validate_schedule_copy(
                    sp, copied_schedule, val_ids, ref0.get("schedule_sha256"))
                if d4report["hashes"]["schedule_copies"].get(f"{f}/{seed}") != copy_hash:
                    raise ValueError("D4 report schedule-copy hash differs from its physical copy")
                for variant, ref, h, cp, transform in (
                        ("D0", ref0, h0, cp0, d0transform), ("D4", ref4, h4, cp4, d4transform)):
                    expected_path = cp.relative_to(d0root if variant == "D0" else d4root).as_posix()
                    if (ref.get("checkpoint_path") != expected_path or ref.get("checkpoint_sha256") != h
                            or ref.get("schedule_sha256") != hs
                            or (variant == "D4" and ref.get("transform_sha256") != _digest(transform))):
                        raise ValueError("reference checkpoint/schedule/transform identity mismatch")
                cp0obj, cp4obj = (torch.load(io.BytesIO(b), map_location="cpu", weights_only=False) for b in (b0, b4))
                for runner_variant, obj, ref, transform in (
                        ("D0", cp0obj, ref0, d0transform), ("D4", cp4obj, ref4, d4transform)):
                    provenance = obj.get("provenance", {})
                    if (obj.get("variant") != "D" or obj.get("fold", f) != f or obj.get("seed") != seed
                            or provenance.get("fold") != f or provenance.get("seed") != seed
                            or provenance.get("variant") != "D" or provenance.get("schedule_sha256") != hs
                            or obj.get("epoch") != ref.get("selected_epoch")):
                        raise ValueError("checkpoint fold/seed/epoch/schedule identity mismatch")
                    _validate_checkpoint_transform_identity(obj, runner_variant, ref, transform)
                    selected = obj.get("selection_metrics")
                    expected_metrics = ref.get("validation")
                    _assert_original_parity(selected, expected_metrics)
                    if runner_variant == "D0":
                        drift = _validate_d0_transform_manifest(transform, obj["transform_manifest"])
                        prior = d0_transform_code_drift.setdefault(str(f), drift)
                        if prior != drift:
                            raise ValueError("D0 checkpoint transform/code-hash drift differs across seeds")
                m0, m4 = NeuroTuneV1Encoder("D", dv0["genre"].shape[1]), NeuroTuneV1Encoder("D", dv4["genre"].shape[1])
                m0.load_state_dict(cp0obj["model_state_dict"]); m4.load_state_dict(cp4obj["model_state_dict"])
                m0.eval(); m4.eval()
                for variant, model, features, transform, reference, d4 in (
                        ("D0", m0, dv0, d0transform, ref0, False), ("D4", m4, dv4, d4transform, ref4, True)):
                    expected = reference["validation"]
                    if reference.get("fold") != f or reference.get("seed") != seed:
                        raise ValueError("reference fold/seed mismatch")
                    rows = _evaluate(model, features, val_ids, edges, d4, expected)
                    for row in rows:
                        values = [row[k] for k in (*METRICS, "mean_top_10_overlap", "mean_embedding_cosine_shift")]
                        if not np.isfinite(values).all():
                            raise ValueError("non-finite sensitivity metric")
                        records.append({"fold": f, "seed": seed, "variant": variant,
                                        "selected_epoch": reference["selected_epoch"], **row,
                                        "checkpoint_sha256": h0 if variant == "D0" else h4,
                                        "schedule_sha256": hs, "transform_sha256": _digest(transform),
                                        "d4_schedule_copy_sha256": copy_hash,
                                        "validation_ids_sha256": _digest(val_ids),
                                        "validation_edges_sha256": _digest(sorted(tuple(sorted(e)) for e in edges))})
        if len(records) != 90:
            raise RuntimeError("P0-A acceptance requires 45 records per model (90 total)")
        d0masked = [r for r in records if r["variant"] == "D0" and r["condition"] in MASKS]
        d4masked = [r for r in records if r["variant"] == "D4" and r["condition"] in MASKS]
        paired = _paired_rows(d0masked, d4masked)
        namespaces_after = _result_namespace_hashes(outroot)
        if namespaces_before != namespaces_after:
            raise RuntimeError("protected result namespaces changed during evaluation")
        report = {"status": "complete", "evaluation_scope": "validation_only_development",
                  "protocol": {"folds": FOLDS, "seeds": SEEDS, "conditions": CONDITIONS, "k": 10,
                               "test_features_embeddings_relevance_metrics": False,
                               "audio_zero_preserves_masks_lengths_ids_and_padding": True},
                  "counts": {"D0_condition_records": 45, "D4_condition_records": 45,
                             "condition_records": 90, "paired_modality_rows": 36,
                             "D4_45_record_acceptance": "passed"},
                  "hashes": {"protected_before": before, "sources": {k: v for k, v in before.items() if k.startswith("input:")},
                             "code": {k: v for k, v in before.items() if k.startswith("code:")},
                             "protected_result_namespaces": {
                                 name: {"before": namespaces_before[name], "after": namespaces_after[name]}
                                 for name in PROTECTED_RESULT_NAMESPACES},
                             "schedule": {f"{f}/{s}": before[f"schedule:{f}/{s}"] for f, s in sorted(pairs)},
                             "checkpoint": {f"{f}/{s}/{v}": before[f"checkpoint:{f}/{s}/{v}"]
                                            for f, s in sorted(pairs) for v in ("D0", "D4")}},
                  "mxm_provenance": mxm_provenance, "records": records, "paired_comparison": paired,
                  "d0_transform_code_hash_drift": d0_transform_code_drift,
                  "relative_delta_ndcg_change_percent": {
                      "formula": "100 * (D4_delta_ndcg - D0_delta_ndcg) / abs(D0_delta_ndcg)",
                      "undefined_when": "D0_delta_ndcg is zero; serialized as null in JSON and an empty CSV cell"},
                  "inherited_training_boundary": (
                      "The frozen D0/D4 models retain the inherited negative-candidate exclusion policy, under which "
                      "official/legacy test-source positives may be exclusion-only. This validation-only masking "
                      "analysis created no test targets and did not change that policy."),
                  "caveat": ("Validation-only development results; zero masking is a distribution-shifted diagnostic, not causal modality importance. "
                             "D0 current/frozen transform comparison permits only the recorded code_sha256.src/features_neurotune_v1.py drift; "
                             "all other transform fields and D4 transform hashes remain exact."),
                  "runtime": {"seconds": time.monotonic() - started, "python": platform.python_version(),
                              "numpy": np.__version__, "pytorch": torch.__version__}}
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        fields = ("fold", "seed", "variant", "selected_epoch", "condition", *METRICS,
                  "query_count", "eligible_positive_edges", "mean_top_10_overlap", "mean_embedding_cosine",
                  "mean_embedding_cosine_shift", "delta_ndcg", "checkpoint_sha256", "schedule_sha256",
                  "d4_schedule_copy_sha256",
                  "transform_sha256", "validation_ids_sha256", "validation_edges_sha256")
        with (stage / "condition_records.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
        with (stage / "paired_d0_d4_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
        (stage / "README.md").write_text("# D4 modality sensitivity\n\nFresh frozen-checkpoint D0/D4 validation-only inference. No training or test evaluation. Zero masking is a distribution-shifted diagnostic, not causal modality importance. D0 transform provenance permits only the recorded `code_sha256.src/features_neurotune_v1.py` drift; all other D0 transform fields and the D4 transform hash remain exact.\n\nP0-A acceptance: 45 D4 records and 45 D0 records; 36 paired masked modality rows. Relative NDCG-delta change is `100 * (D4_delta_ndcg - D0_delta_ndcg) / abs(D0_delta_ndcg)`; it is undefined at a zero D0 delta and is recorded as null/empty.\n\nInherited training boundary: the frozen D0/D4 models retain the inherited negative-candidate exclusion policy, under which official/legacy test-source positives may be exclusion-only. This validation-only masking analysis created no test targets and did not change that policy.\n", encoding="utf-8")
        after = {k: _hash_path(v) for k, v in protected.items()}
        if before != after:
            raise RuntimeError("protected source, report, code, checkpoint, or schedule hashes changed")
        _atomic_publish(stage, final)
        return report
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    _run(args.data_root, args.results_root)


if __name__ == "__main__":
    main()

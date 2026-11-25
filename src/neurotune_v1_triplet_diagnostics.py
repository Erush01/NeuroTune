"""Read-only, schedule-matched D0/D4 triplet and embedding diagnostics."""

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
    build_abcd_cv_development_features, build_abcd_cv_d4_multi_window_features,
    build_abcd_cv_fold_mxm, load_abcd_cv_training,
)
from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
from src.data.neurotune_v1_split import sha256_file
from src.model_v1 import NeuroTuneV1Encoder

ROOT = Path(__file__).resolve().parents[1]
FOLDS, SEEDS, EPOCHS, MARGIN = (0, 1, 2), (684, 685, 686), 50, 0.2
TRAIN_PER_EPOCH, PAIR_COUNT, BASE_SEED, BATCH_SIZE = 200, 10_000, 20261123, 64
POLICY_CAVEAT = (
    "Saved D0 schedules are reused as-is; official/legacy test-source positives may remain exclusion-only "
    "in those schedules under the inherited policy. This diagnostic does not rebuild schedules, create test "
    "targets, or change that policy. Schedule negatives are `sampled_negative`; unknown pairs are not confirmed "
    "negatives."
)
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
    "cv_dir": "data/msd_hf_abcd_pilot/cv", "d4_cache_path": "data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch.h5",
    "d4_metadata_path": "data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch_meta.json",
    "d4_rejection_path": "data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch_rejections.json",
    "mxm_path": "data/processed/mxm_counts",
}


def _sample_train_rows(schedule, fold, seed):
    schedule = np.asarray(schedule)
    if schedule.dtype != np.int32 or schedule.shape[0] != EPOCHS or schedule.ndim != 3 or schedule.shape[2] != 3:
        raise ValueError("train schedule must be int32 with shape (50, N, 3)")
    selected = []
    for epoch, rows in enumerate(schedule):
        if len(rows) < TRAIN_PER_EPOCH:
            raise ValueError(f"train schedule epoch {epoch} has fewer than {TRAIN_PER_EPOCH} rows")
        count = TRAIN_PER_EPOCH
        rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, int(fold), int(seed), epoch]))
        ix = np.sort(rng.choice(len(rows), size=count, replace=False))
        selected.extend((epoch, int(i)) for i in ix)
    indices = np.asarray(selected, dtype="<i8").reshape(-1, 2)
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    return {"selected_indices": indices, "selected_rows": len(indices),
            "total_rows": int(schedule.shape[0] * schedule.shape[1]), "indices_sha256": digest,
            "method": "uniform_without_replacement_per_epoch", "base_seed": BASE_SEED,
            "derived_seed": "SeedSequence(base_seed, fold, seed, epoch)"}


def _sample_track_pairs(track_ids, seed, count=PAIR_COUNT):
    ids = sorted(set(track_ids))
    total = len(ids) * (len(ids) - 1) // 2
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or total < count:
        raise ValueError(f"split has only {total} unique pairs, fewer than requested {count}")
    rng = np.random.default_rng(seed)
    chosen = set()
    while len(chosen) < count:
        # Rejection sampling keeps memory bounded for large catalogs.
        left = int(rng.integers(len(ids)))
        right = int(rng.integers(len(ids) - 1))
        if right >= left:
            right += 1
        chosen.add((min(left, right), max(left, right)))
    pairs = [[ids[a], ids[b]] for a, b in sorted(chosen)]
    payload = json.dumps(pairs, separators=(",", ":")).encode()
    return {"pairs": pairs, "count": len(pairs), "seed": int(seed),
            "method": "uniform_unique_unordered_track_pairs", "pairs_sha256": hashlib.sha256(payload).hexdigest()}


def _triplet_statistics(d_ap, d_an, margin=MARGIN):
    d_ap, d_an = np.asarray(d_ap, dtype=np.float64), np.asarray(d_an, dtype=np.float64)
    if (d_ap.shape != d_an.shape or d_ap.ndim != 1 or len(d_ap) == 0
            or not np.isfinite(d_ap).all() or not np.isfinite(d_an).all()):
        raise ValueError("triplet distances must be aligned finite one-dimensional arrays")
    gap = d_an - d_ap
    active = gap < margin
    easy, semi, hard = gap >= margin, (gap > 0) & (gap < margin), gap <= 0
    return {"triplet_count": len(gap), "active_count": int(active.sum()), "active_rate": float(active.mean()),
            "easy_count": int(easy.sum()), "easy_rate": float(easy.mean()),
            "semi_hard_count": int(semi.sum()), "semi_hard_rate": float(semi.mean()),
            "hard_count": int(hard.sum()), "hard_rate": float(hard.mean()),
            "d_ap_mean": float(d_ap.mean()), "d_an_mean": float(d_an.mean()), "gap_mean": float(gap.mean()),
            "margin": float(margin)}


def _hash_path(path):
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        content = [(p.relative_to(path).as_posix(), sha256_file(p)) for p in sorted(path.rglob("*")) if p.is_file()]
        return hashlib.sha256(json.dumps(content, separators=(",", ":")).encode()).hexdigest()
    raise FileNotFoundError(path)


def _embed(model, dataset):
    model.eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            batch = neurotune_v1_collate([dataset[i] for i in range(start, min(start + BATCH_SIZE, len(dataset)))])
            args = {key: batch[key] for key in ("audio", "audio_windows", "mask", "window_mask", *model.active_modalities)
                    if key in batch}
            blocks.append(model(**args).cpu().numpy())
    embeddings = np.concatenate(blocks)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
        raise ValueError("model embeddings must be finite and unit normalized")
    return embeddings


def _assert_matching_non_audio_transforms(d0, d4):
    fields = ("fit_ids_hash", "bpm_mean", "bpm_std", "genre_vocabulary", "idf", "split_hash")
    for name in fields:
        if name not in d0 or name not in d4 or d0[name] != d4[name]:
            raise ValueError(f"D0/D4 non-audio transform mismatch: {name}")
    d0_sources = d0.get("source_sha256")
    d4_sources = d4.get("source_sha256")
    if not isinstance(d0_sources, dict) or not isinstance(d4_sources, dict):
        raise ValueError("D0/D4 transform source hashes are missing")
    d0_sources = {k: v for k, v in d0_sources.items() if k not in {"audio_cache", "d4_audio_cache"}}
    d4_sources = {k: v for k, v in d4_sources.items() if k not in {"audio_cache", "d4_audio_cache"}}
    if d0_sources != d4_sources:
        raise ValueError("D0/D4 non-audio transform source hashes differ")


def _summary_checkpoint_metadata(model, references, pair):
    model_index = 0 if model == "D0" else 1
    row = references[model_index][pair]
    return {"checkpoint_path": row.get("checkpoint_path", row.get("path")),
            "checkpoint_epoch": row["selected_epoch"]}


def _validate_checkpoint_identity(checkpoint, reference, fold, seed, schedule_hash, *, model="D0"):
    provenance = checkpoint.get("provenance")
    manifest = checkpoint.get("transform_manifest")
    expected_transform = reference.get("transform_sha256")
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint transform manifest is missing or malformed")
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest_hash != expected_transform:
        raise ValueError("checkpoint transform manifest hash differs from selected reference")
    if not isinstance(provenance, dict):
        raise ValueError("checkpoint provenance transform hash differs from selected reference")
    if model not in {"D0", "D4"}:
        raise ValueError("unknown model identity")
    if (model == "D0" and provenance.get("transform_sha256") != expected_transform):
        raise ValueError("checkpoint provenance transform hash differs from selected reference")
    if (model == "D4" and "transform_sha256" in provenance
            and provenance["transform_sha256"] != expected_transform):
        raise ValueError("checkpoint provenance transform hash differs from selected reference")
    if (checkpoint.get("variant") != "D" or checkpoint.get("seed") != seed
            or checkpoint.get("epoch") != reference.get("selected_epoch")
            or not isinstance(checkpoint.get("model_state_dict"), dict)
            or checkpoint.get("config", {}).get("margin") != MARGIN
            or provenance.get("fold") != fold or provenance.get("seed") != seed
            or provenance.get("schedule_sha256") != schedule_hash
            or provenance.get("variant", checkpoint.get("variant")) != "D"):
        raise ValueError("checkpoint identity/provenance mismatch")


def _validate_current_transform(model, current_transform, checkpoint_manifest, reference_hash):
    current_hash = hashlib.sha256(json.dumps(
        current_transform, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_hash = hashlib.sha256(json.dumps(
        checkpoint_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest_hash != reference_hash:
        raise ValueError("checkpoint transform manifest hash differs from selected reference")
    if current_hash == reference_hash:
        return {"current_transform_sha256": current_hash, "checkpoint_transform_sha256": manifest_hash,
                "allowance_field": None, "old_code_sha256": None, "current_code_sha256": None,
                "allowance_reason": None}
    if model != "D0" or not isinstance(current_transform, dict) or not isinstance(checkpoint_manifest, dict):
        raise ValueError(f"{model} freshly built transform hash differs from selected reference")
    field = "src/features_neurotune_v1.py"
    current_codes = current_transform.get("code_sha256")
    checkpoint_codes = checkpoint_manifest.get("code_sha256")
    if not isinstance(current_codes, dict) or not isinstance(checkpoint_codes, dict):
        raise ValueError("transform code hashes are missing")
    current_code, old_code = current_codes.get(field), checkpoint_codes.get(field)
    def valid_sha(value):
        return (isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value))
    if not valid_sha(current_code) or not valid_sha(old_code) or current_code == old_code:
        raise ValueError("D0 transform code-hash difference is not the permitted feature-builder drift")
    current_comparable, checkpoint_comparable = json.loads(json.dumps(current_transform)), json.loads(json.dumps(checkpoint_manifest))
    current_comparable["code_sha256"].pop(field)
    checkpoint_comparable["code_sha256"].pop(field)
    if current_comparable != checkpoint_comparable:
        raise ValueError("D0 transform differs beyond the permitted feature-builder code hash")
    return {"current_transform_sha256": current_hash, "checkpoint_transform_sha256": manifest_hash,
            "allowance_field": "code_sha256." + field, "old_code_sha256": old_code,
            "current_code_sha256": current_code,
            "allowance_reason": "D0 feature-builder source code hash differs only because of the D4 extension; all other transform values and source hashes match."}


def _validate_reference_hashes(reports, before, inputs):
    d0_hashes = reports[0].get("hashes", {})
    data_sources = d0_hashes.get("data_sources")
    if not isinstance(data_sources, dict):
        raise ValueError("D0 data source hashes are missing")
    for key, digest in data_sources.items():
        snapshot_key = f"input:{key}"
        if snapshot_key not in before or digest != before[snapshot_key]:
            raise ValueError(f"D0 data source hash differs from current input: {key}")
    manifest = Path(inputs["cv_dir"]) / "manifest.json"
    if d0_hashes.get("split_manifest") != _hash_path(manifest):
        raise ValueError("D0 split manifest hash differs from current CV manifest")

    d4_before = reports[1].get("hashes", {}).get("protected_before")
    if not isinstance(d4_before, dict):
        raise ValueError("D4 protected-before reference hashes are missing")
    required = {key for key in before if key.startswith("input:") or key == "d0_report"
                or key.startswith("schedule:")}
    required.update(f"checkpoint:{f}/{s}" for f in FOLDS for s in SEEDS)
    missing = required - set(d4_before)
    if missing:
        raise ValueError(f"D4 protected-before hashes are missing required references: {sorted(missing)[:3]}")
    def snapshot_value(key):
        if key.startswith("checkpoint:"):
            return before[key.replace("checkpoint:", "checkpoint:D0:", 1)]
        return before[key]

    mismatched = [key for key in required if d4_before[key] != snapshot_value(key)]
    if mismatched:
        raise ValueError(f"D4 protected-before hashes differ from current snapshot: {sorted(mismatched)[:3]}")


def _validate_schedule_copy(source_path, copy_path, fold, seed):
    if _hash_path(source_path) != _hash_path(copy_path):
        raise ValueError(f"D4 schedule copy hash differs from D0 source for {fold}/{seed}")
    required = {"train_ids", "validation_ids", "train_schedule", "validation_schedule"}
    with np.load(source_path, allow_pickle=False) as source, np.load(copy_path, allow_pickle=False) as copied:
        if set(source.files) != required or set(copied.files) != required:
            raise ValueError("D0/D4 schedule archive fields differ from contract")
        for name in required:
            if not np.array_equal(source[name], copied[name]):
                raise ValueError(f"D4 schedule copy array {name} differs from D0 source for {fold}/{seed}")


def _distance_rows(values, *, model, fold, seed, split, source, sample_info):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    rows = []
    for q, value in zip(("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"), quantiles):
        rows.append({"model": model, "fold": fold, "seed": seed, "split": split, "distance": source,
                     "statistic": q, "lower": "", "upper": "", "count": len(values), "value": float(value), **sample_info})
    hist, edges = np.histogram(values, bins=20)
    rows.extend({"model": model, "fold": fold, "seed": seed, "split": split, "distance": source,
                 "statistic": "histogram", "lower": float(edges[i]), "upper": float(edges[i + 1]),
                 "count": int(hist[i]), "value": "", **sample_info} for i in range(len(hist)))
    return rows


def _run(data_root=ROOT, results_root=ROOT / "results", *, _test_only=False, _dependencies=None):
    started = time.monotonic()
    deps = _dependencies or {}
    if deps and not _test_only:
        raise ValueError("diagnostic dependencies are fixed outside tests")
    loader = deps.get("loader", load_abcd_cv_training)
    mxm_builder = deps.get("mxm_builder", build_abcd_cv_fold_mxm)
    d0_feature_builder = deps.get("d0_feature_builder", build_abcd_cv_development_features)
    d4_feature_builder = deps.get("d4_feature_builder", build_abcd_cv_d4_multi_window_features)
    embedder = deps.get("embedder", _embed)
    root, results = Path(data_root), Path(results_root)
    final = results / "neurotune_v1_d4_triplet_diagnostics"
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    d0root, d4root, schedule_root = (results / "neurotune_v1_full_modal_ablation",
                                     results / "neurotune_v1_d4_multi_window_audio",
                                     results / "neurotune_v1_abcd_cv" / "triplet_schedules")
    d0_report_path, d4_report_path = d0root / "experiment.json", d4root / "experiment.json"
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in (d0_report_path, d4_report_path)]
    if any(report.get("status") != "complete" for report in reports):
        raise ValueError("D0 and D4 reference experiments must be complete")
    if (reports[1].get("evaluation_scope") != "validation_only_development"
            or reports[1].get("independent_test_artifacts_created") is not False):
        raise ValueError("D4 reference must declare validation-only scope and no independent-test artifacts")
    inputs = {k: root / v for k, v in INPUTS.items()}
    references = {}
    for index, report in enumerate(reports):
        rows = report.get("model_validation")
        if not isinstance(rows, list):
            raise ValueError("reference model_validation rows are missing")
        references[index] = {}
        for row in rows:
            if row.get("variant") != "D":
                continue
            pair = (row.get("fold"), row.get("seed"))
            if pair in references[index] or pair not in {(f, s) for f in FOLDS for s in SEEDS}:
                raise ValueError("reference fold/seed rows are duplicated or unknown")
            references[index][pair] = row
        if set(references[index]) != {(f, s) for f in FOLDS for s in SEEDS}:
            raise ValueError("reference experiment must contain all nine D checkpoints")
    cp_paths = {}
    schedule_paths = {}
    schedule_copy_paths = {}
    for pair in references[0]:
        f, s = pair
        for model, report, base in (("D0", reports[0], d0root), ("D4", reports[1], d4root)):
            row = references[0 if model == "D0" else 1][pair]
            rel = row.get("checkpoint_path", row.get("path"))
            expected = f"checkpoints/fold_{f}_D_seed_{s}.pt"
            if rel != expected or type(row.get("selected_epoch")) is not int:
                raise ValueError(f"{model} checkpoint identity/epoch is malformed")
            cp_paths[(model, pair)] = base / rel
        schedule_paths[pair] = schedule_root / f"fold_{f}_seed_{s}.npz"
        schedule_copy_paths[pair] = d4root / "triplet_schedules" / f"fold_{f}_seed_{s}.npz"
    protected = {"d0_report": d0_report_path, "d4_report": d4_report_path}
    protected.update({f"checkpoint:{m}:{f}/{s}": p for (m, (f, s)), p in cp_paths.items()})
    protected.update({f"schedule:{f}/{s}": p for (f, s), p in schedule_paths.items()})
    protected.update({f"schedule_copy:{f}/{s}": p for (f, s), p in schedule_copy_paths.items()})
    protected.update({f"input:{k}": p for k, p in inputs.items()})
    code_paths = ["src/neurotune_v1_triplet_diagnostics.py", "scripts/run_neurotune_v1_d4_triplet_diagnostics.py",
                  "config/neurotune_v1_d4_multi_window_audio.json", "src/data/neurotune_v1_abcd_cv_training.py",
                  "src/data/neurotune_v1_abcd_cv.py", "src/data/neurotune_v1_dataset.py",
                  "src/data/neurotune_v1_split.py", "src/data/preprocess_msd_hf_abcd.py",
                  "scripts/preprocess_msd_hf_abcd.py", "scripts/preprocess_neurotune_v1_d4_audio_windows.py",
                  "src/data/preprocess_audio_window.py", "src/data/parse_mxm.py", "src/features_neurotune_v1.py",
                  "src/features.py", "src/model_v1.py", "src/train_neurotune_v1.py", "src/triplets.py"]
    protected.update({f"code:{p}": ROOT / p for p in code_paths})
    before = {k: _hash_path(p) for k, p in protected.items()}
    _validate_reference_hashes(reports, before, inputs)
    for model, index in (("D0", 0), ("D4", 1)):
        for pair, row in references[index].items():
            if before[f"checkpoint:{model}:{pair[0]}/{pair[1]}"] != row.get("checkpoint_sha256", row.get("sha256")):
                raise ValueError(f"{model} checkpoint hash differs from its experiment reference")
    for pair, row in references[0].items():
        if before[f"schedule:{pair[0]}/{pair[1]}"] != row.get("schedule_sha256"):
            raise ValueError("D0 schedule hash differs from its reference")
        d4_row = references[1][pair]
        if d4_row.get("schedule_sha256") != row.get("schedule_sha256"):
            raise ValueError("D4 schedule reference differs from the D0 schedule")
        copied = reports[1].get("hashes", {}).get("schedule_copies", {}).get(f"{pair[0]}/{pair[1]}")
        copy_hash = before[f"schedule_copy:{pair[0]}/{pair[1]}"]
        if copied != row.get("schedule_sha256") or copy_hash != row.get("schedule_sha256"):
            raise ValueError("D4 recorded schedule copy hash differs from D0 schedule")
        _validate_schedule_copy(schedule_paths[pair], schedule_copy_paths[pair], *pair)

    loaded = loader(**{k: v for k, v in inputs.items() if k not in
                                      {"d4_cache_path", "d4_metadata_path", "d4_rejection_path", "mxm_path"}})
    lyrics, mxm_provenance = mxm_builder({k: loaded[k] for k in ("manifest", "catalog", "cache_path")}, inputs["mxm_path"])
    results.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".neurotune_v1_d4_triplet_diagnostics.stage-", dir=results))
    try:
        summaries, distributions, sample_manifest = [], [], {}
        transform_provenance = []
        for fold_data in loaded["folds"]:
            fold = fold_data["fold"]
            dev_fold = {k: fold_data[k] for k in ("fold", "train", "validation")}
            ids = sorted(fold_data["train"] + fold_data["validation"])
            if set(ids) & set(fold_data.get("test", [])):
                raise ValueError("train/validation feature IDs overlap test IDs")
            d0_features, d0_transform = d0_feature_builder(
                {k: loaded[k] for k in ("manifest", "catalog", "cache_path")}, dev_fold, lyrics)
            d4_features, d4_transform = d4_feature_builder(
                {k: loaded[k] for k in ("manifest", "catalog", "cache_path")}, dev_fold, lyrics, inputs["d4_cache_path"])
            _assert_matching_non_audio_transforms(d0_transform, d4_transform)
            for seed in SEEDS:
                pair = (fold, seed)
                with np.load(schedule_paths[pair], allow_pickle=False) as archive:
                    if set(archive.files) != {"train_ids", "validation_ids", "train_schedule", "validation_schedule"}:
                        raise ValueError("schedule fields mismatch")
                    schedule = {key: archive[key].copy() for key in archive.files}
                if (list(schedule["train_ids"].astype(str)) != list(fold_data["train"])
                        or list(schedule["validation_ids"].astype(str)) != list(fold_data["validation"])):
                    raise ValueError("schedule ID order differs from fold descriptors")
                train_schedule, val_schedule = schedule["train_schedule"], schedule["validation_schedule"]
                for name, arr, n, epochs in (("train", train_schedule, len(fold_data["train"]), 50),
                                             ("validation", val_schedule, len(fold_data["validation"]), 1)):
                    if arr.dtype != np.int32 or arr.ndim != 3 or arr.shape[0] != epochs or arr.shape[2] != 3 or np.any(arr < 0) or np.any(arr >= n):
                        raise ValueError(f"invalid {name} schedule")
                train_sample = _sample_train_rows(train_schedule, fold, seed)
                val_indices = np.asarray([(0, i) for i in range(len(val_schedule[0]))], dtype="<i8")
                val_sample = {"selected_indices": val_indices, "selected_rows": len(val_indices),
                              "total_rows": len(val_indices), "indices_sha256": hashlib.sha256(val_indices.tobytes()).hexdigest(),
                              "method": "entire_validation_schedule", "base_seed": None, "derived_seed": None}
                sample_manifest[f"{fold}/{seed}"] = {"train": {k: v for k, v in train_sample.items() if k != "selected_indices"},
                                                      "validation": {k: v for k, v in val_sample.items() if k != "selected_indices"}}
                pair_samples = {}
                feature_map = {"D0": d0_features, "D4": d4_features}
                transform_map = {"D0": d0_transform, "D4": d4_transform}
                model_embeddings = {}
                transform_checks = {}
                for split, split_ids in (("train", fold_data["train"]), ("validation", fold_data["validation"])):
                    pair_seed = BASE_SEED + fold * 100_000 + seed * 10 + (0 if split == "train" else 1)
                    pair_samples[split] = _sample_track_pairs(split_ids, pair_seed)
                    sample_manifest[f"{fold}/{seed}"][split + "_pairs"] = {k: v for k, v in pair_samples[split].items() if k != "pairs"}
                for model, mi in (("D0", 0), ("D4", 1)):
                    row = references[mi][pair]
                    checkpoint_path = cp_paths[(model, pair)]
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    _validate_checkpoint_identity(checkpoint, row, fold, seed,
                                                  before[f"schedule:{fold}/{seed}"], model=model)
                    transform_check = _validate_current_transform(
                        model, transform_map[model], checkpoint["transform_manifest"], row["transform_sha256"])
                    transform_checks[model] = transform_check
                    transform_provenance.append({"model": model, "fold": fold, "seed": seed, **transform_check})
                    model = NeuroTuneV1Encoder("D", np.asarray(feature_map["D0"]["genre"]).shape[1]).cpu()
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model_embeddings["D0" if mi == 0 else "D4"] = embedder(model, NeuroTuneV1Dataset(ids, feature_map["D0" if mi == 0 else "D4"]))
                for model in ("D0", "D4"):
                    embs = model_embeddings[model]
                    pos = {track: i for i, track in enumerate(ids)}
                    for split in ("train", "validation"):
                        schedule_arr = train_schedule if split == "train" else val_schedule
                        selected = train_sample if split == "train" else val_sample
                        indices = selected["selected_indices"]
                        local_ids = fold_data[split]
                        base = 0 if split == "train" else len(fold_data["train"])
                        tri = schedule_arr[indices[:, 0], indices[:, 1], :]
                        if tri.size and (np.max(tri) >= len(local_ids) or np.min(tri) < 0):
                            raise ValueError("schedule contains an unresolved sample index")
                        tri_ids = np.asarray(local_ids, dtype=object)[tri]
                        ix = np.vectorize(pos.__getitem__, otypes=[np.int64])(tri_ids)
                        d_ap = 1.0 - np.sum(embs[ix[:, 0]] * embs[ix[:, 1]], axis=1)
                        d_an = 1.0 - np.sum(embs[ix[:, 0]] * embs[ix[:, 2]], axis=1)
                        stats = _triplet_statistics(d_ap, d_an)
                        summary = {"model": model, "fold": fold, "seed": seed, "split": split,
                                   **_summary_checkpoint_metadata(model, references, pair),
                                    "checkpoint_sha256": before[f"checkpoint:{model}:{fold}/{seed}"],
                                    "current_transform_sha256": transform_checks[model]["current_transform_sha256"],
                                    "checkpoint_transform_sha256": transform_checks[model]["checkpoint_transform_sha256"],
                                    "transform_allowance_field": transform_checks[model]["allowance_field"] or "",
                                    "transform_allowance_old_code_sha256": transform_checks[model]["old_code_sha256"] or "",
                                    "transform_allowance_current_code_sha256": transform_checks[model]["current_code_sha256"] or "",
                                    "schedule_sha256": before[f"schedule:{fold}/{seed}"],
                                    "schedule_source_sha256": before[f"schedule:{fold}/{seed}"],
                                    "schedule_copy_sha256": before[f"schedule_copy:{fold}/{seed}"],
                                   "selected_rows": selected["selected_rows"], "total_rows": selected["total_rows"],
                                   "sample_method": selected["method"], "sample_base_seed": selected["base_seed"],
                                   "sample_indices_sha256": selected["indices_sha256"],
                                   "embedding_norm_validated": True, "embedding_norm_max_abs_error": float(np.max(np.abs(np.linalg.norm(embs, axis=1) - 1))),
                                   "negative_type": "sampled_negative", **stats}
                        summaries.append(summary)
                        for key, values in (("d_ap", d_ap), ("d_an", d_an), ("gap", d_an - d_ap)):
                            distributions.extend(_distance_rows(values, model=model, fold=fold, seed=seed,
                                                                split=split, source=key, sample_info={"sample_indices_sha256": selected["indices_sha256"]}))
                        pair_rows = pair_samples[split]["pairs"]
                        pair_cosines = np.asarray([np.dot(embs[pos[a]], embs[pos[b]]) for a, b in pair_rows])
                        distributions.extend(_distance_rows(pair_cosines, model=model, fold=fold, seed=seed,
                                                            split=split, source="pairwise_cosine", sample_info={"sample_indices_sha256": pair_samples[split]["pairs_sha256"]}))
        _write_csv(stage / "triplet_summary_by_fold_seed.csv", summaries)
        _write_csv(stage / "distance_distributions.csv", distributions)
        protected_after = {k: _hash_path(p) for k, p in protected.items()}
        if protected_after != before:
            raise RuntimeError("protected inputs/references/code changed during diagnostics")
        report = {"status": "complete", "protocol": {"margin": MARGIN, "loss": "relu(d_ap-d_an+0.2)",
                  "distance": "1-dot(normalized embeddings)", "train_samples_per_epoch": TRAIN_PER_EPOCH,
                  "train_epochs": EPOCHS, "validation_schedule": "entire", "pairwise_cosine_pairs_per_split": PAIR_COUNT,
                   "base_seed": BASE_SEED, "negative_label": "sampled_negative",
                   "schedule_policy": "use saved schedule rows as-is; no regeneration or test-derived features",
                   "transform_hash_allowance": "D0 may differ only at code_sha256.src/features_neurotune_v1.py when every other transform field matches the checkpoint manifest; D4 remains exact",
                   "comparisons": "D0 and D4 use identical selected triplet rows and pair indices"},
                  "samples": sample_manifest, "transform_provenance": transform_provenance,
                  "hashes": {"protected_before": before, "protected_after": protected_after},
                  "runtime": {"seconds": time.monotonic() - started, "python": platform.python_version(),
                              "numpy": np.__version__, "pytorch": torch.__version__},
                  "mxm_provenance": mxm_provenance, "independent_test_artifacts_created": False,
                  "caveat": POLICY_CAVEAT + " Descriptive validation/train diagnostics only; no superiority or significance claim is made."}
        (stage / "analysis.md").write_text(_analysis(summaries), encoding="utf-8")
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if {k: _hash_path(p) for k, p in protected.items()} != before:
            raise RuntimeError("protected inputs/references/code changed during diagnostics")
        if final.exists() or final.is_symlink():
            raise FileExistsError(f"refusing to overwrite output namespace: {final}")
        final.mkdir()
        try:
            for source in sorted(p for p in stage.rglob("*") if p.is_file()):
                destination = final / source.relative_to(stage)
                os.link(source, destination)
        except BaseException:
            shutil.rmtree(final)
            raise
        shutil.rmtree(stage)
        return report
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _write_csv(path, rows):
    import csv
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _analysis(rows):
    allowed = [row for row in rows if row.get("transform_allowance_field")]
    allowance_report = ""
    if allowed:
        details = "; ".join(f"fold {row['fold']} seed {row['seed']}: old={row['transform_allowance_old_code_sha256']} current={row['transform_allowance_current_code_sha256']}"
                             for row in allowed)
        allowance_report = ("\n\n## Transform provenance allowance\n\n"
                            "D0 only allowed `code_sha256.src/features_neurotune_v1.py` to differ from the frozen checkpoint manifest; "
                            "the manifest hash still matches the selected reference, and all other transform fields/source hashes matched. "
                            + details + ". D4 transform hashes were required to match exactly.\n")
    return ("# D0/D4 triplet and embedding diagnostics\n\n"
            "This is a descriptive train/validation analysis of existing checkpoints and saved schedules. "
            + POLICY_CAVEAT + " "
            + "No independent test features, embeddings, relevance, or metrics were produced. No significance or superiority claim is made.\n\n"
            + f"Generated {len(rows)} model × fold × seed × split summary rows. Interpret D0/D4 rows as paired because the same schedule indices and pair samples were used.\n"
            + allowance_report)

"""Read-only P1-B diagnostic of the frozen D0/D4 pre-fusion audio branches."""

from collections.abc import Mapping
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.evaluate_neurotune_v1_p1a_intra_genre_retrieval import (
    build_validation_candidate_galleries, evaluate_intra_genre_retrieval,
)

OUTPUT_NAME = "neurotune_v1_p1b_audio_branch_diagnostic"


def _require_fresh_output(path):
    if Path(path).exists() or Path(path).is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {path}")


def extract_pre_fusion_audio(model, batch, variant):
    """Capture raw 64-D projection output; apply D4's valid-window mean."""
    import torch
    if variant not in {"D0", "D4"}:
        raise ValueError("variant must be D0 or D4")
    if model.training:
        raise ValueError("audio extraction requires an eval-mode frozen model")
    if not isinstance(batch, Mapping):
        raise ValueError("batch must be a mapping of model inputs")
    calls, captured = [], []

    def capture(_module, _inputs, output):
        calls.append(1)
        captured.append(output.detach())

    hook = model.audio_projection.register_forward_hook(capture)
    try:
        with torch.no_grad():
            model_inputs = {key: batch[key] for key in
                            ("audio", "audio_windows", "mask", "window_mask", "bpm", "genre", "lyrics")
                            if key in batch}
            model(**model_inputs)
    finally:
        hook.remove()
    if len(calls) != 1:
        raise RuntimeError(f"expected exactly one audio_projection call, got {len(calls)}")
    projected = captured[0]
    if projected.ndim != 2 or projected.shape[1] != 64:
        raise ValueError("audio_projection must produce one 64-D vector per input window")
    if variant == "D0":
        if "audio_windows" in batch or projected.shape[0] != batch["audio"].shape[0]:
            raise ValueError("D0 projection output must align with single-window tracks")
        result = projected
    else:
        if "audio_windows" not in batch:
            raise ValueError("D4 requires a multi-window batch")
        window_mask = batch.get("window_mask")
        if (not torch.is_tensor(window_mask) or window_mask.dtype != torch.bool
                or window_mask.ndim != 2 or window_mask.shape[1] != 3
                or not window_mask.any(dim=1).all() or int(window_mask.sum()) != projected.shape[0]):
            raise ValueError("D4 window_mask must align with the projected valid windows")
        result = projected.new_zeros((window_mask.shape[0], 3, 64))
        result[window_mask] = projected
        result = result.sum(dim=1) / window_mask.sum(dim=1, keepdim=True)
    if not torch.isfinite(result).all():
        raise ValueError("audio branch representations must be finite")
    normalized = torch.nn.functional.normalize(result, p=2, dim=1)
    return normalized.cpu().numpy()


def evaluate_paired_representations(records, test_ids=()):
    """Evaluate fold/seed paired D0/D4 matrices against exactly identical galleries."""
    results = {}
    for key, pair in records.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("records must be keyed by exact (fold, seed) pairs")
        forbidden_ids = set(test_ids.get(key, ())) if isinstance(test_ids, Mapping) else set(test_ids)
        if set(pair) != {"D0", "D4"}:
            raise ValueError("each fold/seed must contain exactly D0 and D4")
        common = None
        for variant in ("D0", "D4"):
            row = pair[variant]
            ids = list(row["validation_ids"])
            if forbidden_ids.intersection(ids):
                raise ValueError("test IDs may not enter audio diagnostic evaluation")
            if common is None:
                common = (ids, row["candidate_galleries"])
            elif ids != common[0] or row["candidate_galleries"] != common[1]:
                raise ValueError("D0/D4 must share identical ordered validation IDs and galleries")
        evaluations = {}
        signatures = {}
        for variant in ("D0", "D4"):
            row = pair[variant]
            evaluated = evaluate_intra_genre_retrieval(
                row["embeddings"], row["validation_ids"], row["candidate_galleries"])
            evaluations[variant] = evaluated
            signatures[variant] = (
                evaluated["candidate_hashes"], evaluated["label_hashes"])
        if signatures["D0"] != signatures["D4"]:
            raise ValueError("D0/D4 candidate and label hashes differ")
        results[key] = evaluations
    return results


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root)
    parser.add_argument("--results-root", type=Path, default=root / "results")
    args = parser.parse_args(argv)
    return _run(args.data_root, args.results_root)


def _run(data_root, results_root, *, _test_only=False, _dependencies=None):
    """Extract/evaluate frozen branches on development validation only."""
    import csv
    import io
    import shutil
    import tempfile
    import torch
    from scripts.run_neurotune_v1_full_modal_ablation import (
        FOLDS, INPUTS as D0_INPUTS, ROOT, SEEDS, _hash_path, _slice,
    )
    from scripts.run_neurotune_v1_d4_multi_window_audio import INPUTS as D4_INPUTS, _take
    from src.data.neurotune_v1_abcd_cv_training import (
        build_abcd_cv_development_features, build_abcd_cv_d4_multi_window_features,
        build_abcd_cv_fold_mxm, load_abcd_cv_training,
    )
    from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
    from src.evaluate_d4_modality_sensitivity import (
        _select_references, _validate_report_hashes, _validate_schedule_copy,
        _validate_checkpoint_transform_identity, _validate_d0_transform_manifest,
        CODE_PATHS, PROTECTED_RESULT_NAMESPACES,
    )
    from src.model_v1 import NeuroTuneV1Encoder

    deps = dict(_dependencies or {})
    if deps and not _test_only:
        raise ValueError("dependency injection is test-only")
    data_root, results_root = Path(data_root), Path(results_root)
    final = results_root / OUTPUT_NAME
    _require_fresh_output(final)
    results_root.mkdir(parents=True, exist_ok=True)
    d0root, d4root = results_root / "neurotune_v1_full_modal_ablation", results_root / "neurotune_v1_d4_multi_window_audio"
    schedroot = results_root / "neurotune_v1_abcd_cv"
    D0_INPUTS = deps.get("d0_inputs", D0_INPUTS)
    D4_INPUTS = deps.get("d4_inputs", D4_INPUTS)
    FOLDS, SEEDS = deps.get("folds", FOLDS), deps.get("seeds", SEEDS)
    inputs = {**{k: data_root / v for k, v in D0_INPUTS.items()},
              **{k: data_root / v for k, v in D4_INPUTS.items()}}
    checkpoints = {(f, s, v): (d0root if v == "D0" else d4root) / f"checkpoints/fold_{f}_D_seed_{s}.pt"
                   for f in FOLDS for s in SEEDS for v in ("D0", "D4")}
    schedules = {(f, s): schedroot / f"triplet_schedules/fold_{f}_seed_{s}.npz" for f in FOLDS for s in SEEDS}
    copies = {(f, s): d4root / f"triplet_schedules/fold_{f}_seed_{s}.npz" for f in FOLDS for s in SEEDS}
    d0report_path, d4report_path = d0root / "experiment.json", d4root / "experiment.json"
    protected = {**{f"input:{k}": p for k, p in inputs.items()}, "d0_report": d0report_path,
                 "d4_report": d4report_path,
                 **{f"checkpoint:{f}/{s}/{v}": p for (f, s, v), p in checkpoints.items()},
                 **{f"schedule:{f}/{s}": p for (f, s), p in schedules.items()},
                 **{f"schedule_copy:{f}/{s}": p for (f, s), p in copies.items()},
                 **{f"code:{name}": ROOT / name for name in CODE_PATHS},
                 "code:p1a": ROOT / "src/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py",
                 "code:p1b": Path(__file__)}
    hash_path = deps.get("hash_path", _hash_path)
    before = {key: hash_path(path) for key, path in protected.items()}
    reports = deps.get("reports")
    if reports is None:
        d0report, d4report = json.loads(d0report_path.read_text()), json.loads(d4report_path.read_text())
    else:
        d0report, d4report = reports["D0"], reports["D4"]
    refs0, refs4 = deps.get("references", {}).get("D0"), deps.get("references", {}).get("D4")
    refs0 = _select_references(d0report, "D0") if refs0 is None else refs0
    refs4 = _select_references(d4report, "D4") if refs4 is None else refs4
    if d0report.get("status") != "complete" or d4report.get("status") != "complete":
        raise ValueError("D0/D4 references must be complete")
    if (d4report.get("evaluation_scope") != "validation_only_development"
            or d4report.get("independent_test_artifacts_created") is not False):
        raise ValueError("D4 report must confirm validation-only scope and no independent-test artifacts")
    deps.get("report_validator", _validate_report_hashes)(d0report, d4report, before)
    namespaces = tuple(dict.fromkeys((*PROTECTED_RESULT_NAMESPACES,
                                    "neurotune_v1_p1a_intra_genre_retrieval",
                                    "neurotune_v1_d4_modality_sensitivity",
                                    "neurotune_v1_d4_triplet_diagnostics")))
    namespace_before = {name: hash_path(results_root / name) for name in namespaces}
    loaded = deps.get("loader", load_abcd_cv_training)(**{k: p for k, p in inputs.items() if k in D0_INPUTS and k != "mxm_path"})
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.stage-", dir=results_root))
    try:
        records, test_ids_by_pair, mxm_feature_source_provenance_by_fold = {}, {}, {}
        for fold in loaded["folds"]:
            f = fold["fold"]
            all_ids = sorted(fold["train"] + fold["validation"])
            val_ids = list(fold["validation"])
            test_ids = set(fold.get("test", []))
            if set(all_ids) & test_ids:
                raise ValueError("test IDs may not enter feature builders")
            fi = {"manifest": loaded["manifest"], "catalog": {i: loaded["catalog"][i] for i in all_ids},
                  "cache_path": loaded["cache_path"]}
            development = {k: fold[k] for k in ("fold", "train", "validation")}
            lyrics, mxm_feature_source_provenance = deps.get("mxm_builder", build_abcd_cv_fold_mxm)(
                fi, inputs["mxm_path"])
            mxm_feature_source_provenance_by_fold[str(f)] = mxm_feature_source_provenance
            features0, transform0 = deps.get("d0_builder", build_abcd_cv_development_features)(fi, development, lyrics)
            features4, transform4 = deps.get("d4_builder", build_abcd_cv_d4_multi_window_features)(fi, development, lyrics, inputs["d4_cache_path"])
            edges = list(fold["validation_edges"])
            if any(a not in val_ids or b not in val_ids for a, b in edges):
                raise ValueError("validation edges must connect validation IDs")
            galleries = build_validation_candidate_galleries(
                val_ids, {i: loaded["catalog"][i].get("genre") for i in val_ids}, edges)
            val_features = {"D0": deps.get("d0_slice", _slice)(features0, all_ids, val_ids),
                            "D4": deps.get("d4_slice", _take)(features4, all_ids, val_ids)}
            fold_record = {}
            for seed in SEEDS:
                for variant, transform, refs, rootdir in (
                        ("D0", transform0, refs0, d0root), ("D4", transform4, refs4, d4root)):
                    ref, path = refs[(f, seed)], checkpoints[(f, seed, variant)]
                    cp_bytes = deps["checkpoint_bytes"]((f, seed, variant)) if "checkpoint_bytes" in deps else path.read_bytes()
                    if hashlib.sha256(cp_bytes).hexdigest() != before[f"checkpoint:{f}/{seed}/{variant}"]:
                        raise ValueError("checkpoint changed after protected input snapshot")
                    schedule_hash = before[f"schedule:{f}/{seed}"]
                    copy_hash = deps.get("schedule_validator", _validate_schedule_copy)(
                        schedules[(f, seed)], copies[(f, seed)], val_ids, ref.get("schedule_sha256"))
                    if d4report.get("hashes", {}).get("schedule_copies", {}).get(f"{f}/{seed}") != copy_hash:
                        raise ValueError("D4 schedule-copy hash mismatch")
                    cp = deps["checkpoint_loader"](cp_bytes) if "checkpoint_loader" in deps else torch.load(io.BytesIO(cp_bytes), map_location="cpu", weights_only=False)
                    expected_path = path.relative_to(rootdir).as_posix()
                    if (ref.get("checkpoint_path") != expected_path
                            or ref.get("checkpoint_sha256") != before[f"checkpoint:{f}/{seed}/{variant}"]
                            or ref.get("schedule_sha256") != schedule_hash):
                        raise ValueError("reference checkpoint/schedule identity mismatch")
                    prov = cp.get("provenance", {})
                    if (cp.get("variant") != "D" or cp.get("fold", f) != f or cp.get("seed") != seed
                            or prov.get("fold") != f or prov.get("seed") != seed or prov.get("variant") != "D"
                            or prov.get("schedule_sha256") != schedule_hash or cp.get("epoch") != ref.get("selected_epoch")):
                        raise ValueError("checkpoint fold/seed/epoch/schedule identity mismatch")
                    if "checkpoint_validator" in deps:
                        deps["checkpoint_validator"](cp, variant, ref, transform, f, seed, schedule_hash)
                    else:
                        _validate_checkpoint_transform_identity(cp, variant, ref, transform)
                    if variant == "D0":
                        deps.get("d0_transform_validator", _validate_d0_transform_manifest)(transform, cp["transform_manifest"])
                    parity = deps.get("parity_validator")
                    if parity is None:
                        from src.evaluate_d4_modality_sensitivity import _assert_original_parity
                        parity = _assert_original_parity
                    parity(cp.get("selection_metrics"), ref.get("validation", {}))
                    if "model_loader" in deps:
                        model = deps["model_loader"](cp, variant, val_features[variant])
                    else:
                        model = NeuroTuneV1Encoder("D", val_features[variant]["genre"].shape[1])
                        model.load_state_dict(cp["model_state_dict"])
                    model.eval()
                    if "batch_builder" in deps:
                        batch = deps["batch_builder"](variant, val_ids, val_features[variant])
                    else:
                        ds = NeuroTuneV1Dataset(val_ids, val_features[variant], modalities=("bpm", "genre", "lyrics"))
                        batch = neurotune_v1_collate([ds[i] for i in range(len(ds))])
                    vectors = extract_pre_fusion_audio(model, batch, variant)
                    fold_record[(f, seed, variant)] = {"validation_ids": val_ids, "embeddings": vectors,
                                                       "candidate_galleries": galleries}
            for seed in SEEDS:
                records[(f, seed)] = {v: fold_record[(f, seed, v)] for v in ("D0", "D4")}
                test_ids_by_pair[(f, seed)] = set(fold.get("test", []))
        evaluated = evaluate_paired_representations(records, test_ids_by_pair)
        after = {key: hash_path(path) for key, path in protected.items()}
        if before != after or namespace_before != {n: hash_path(results_root / n) for n in namespaces}:
            raise RuntimeError("protected inputs or prior result namespaces changed during P1-B")
        report = {"status": "complete", "evaluation_scope": "validation_only_development",
                  "representation": "frozen D0/D4 raw pre-fusion audio_projection output; L2-normalized",
                  "caveat": "D0 and D4 audio branches are diagnostics extracted from frozen multimodal checkpoints, not independently trained audio-only models.",
                  "test_boundary": "MXM text source files are read, but the parser receives only train+validation catalog IDs. Held-out ABCD test IDs/labels are not materialized into diagnostic features or retrieval evaluation; no test features, embeddings, queries, galleries, relevance, or metrics are produced.",
                  "schedule_positive_exclusion_policy": "The inherited official/legacy test-positive exclusion policy is recorded only as frozen schedule/checkpoint provenance. P1-B does not resample triplets or train.",
                  "mxm_feature_source_provenance": mxm_feature_source_provenance_by_fold,
                  "hashes": {"protected_before": before, "protected_after": after}, "records": []}
        rows, per_query_rows, coverage_rows = [], [], []
        for (f, seed), pair in evaluated.items():
            for variant, result in pair.items():
                rows.append({"fold": f, "seed": seed, "system": f"{variant}-audio-branch",
                             **result["metrics"], "small_gallery_metrics": result["small_gallery_metrics"],
                             "counts": result["counts"], "candidate_hashes": result["candidate_hashes"],
                             "label_hashes": result["label_hashes"]})
                coverage_rows.append({"fold": f, "seed": seed, "system": f"{variant}-audio-branch",
                                      **result["counts"]})
                for query_id, query in result["per_query"].items():
                    per_query_rows.append({"fold": f, "seed": seed, "system": f"{variant}-audio-branch",
                                           "query_id": query_id, "genre": query["genre"],
                                           "positive_count": query["positive_count"],
                                           "Recall@10": query["Recall@10"],
                                           "MRR@10": query["MRR@10"],
                                           "NDCG@10": query["NDCG@10"],
                                           "candidate_hash": result["candidate_hashes"][query_id],
                                           "label_hash": result["label_hashes"][query_id]})
        report["records"] = rows
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (stage / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("fold", "seed", "system", "macro_query", "genre_macro",
                                                          "micro_recall@10", "small_gallery_metrics", "counts"))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(row[k], sort_keys=True) if isinstance(row[k], (dict, list)) else row[k]
                                 for k in writer.fieldnames})
        with (stage / "per_query_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("fold", "seed", "system", "query_id", "genre",
                                                        "positive_count",
                                                        "Recall@10", "MRR@10", "NDCG@10",
                                                        "candidate_hash", "label_hash"))
            writer.writeheader()
            for row in per_query_rows:
                writer.writerow({k: json.dumps(row[k], sort_keys=True) if isinstance(row.get(k), (dict, list)) else row.get(k)
                                 for k in writer.fieldnames})
        with (stage / "fold_query_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = list(dict.fromkeys(key for row in coverage_rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(coverage_rows)
        summary = ["# P1-B audio branch diagnostic", "",
                    "D0 and D4 are reported as frozen audio-branch diagnostics, not as independently trained audio-only models.",
                   "Same-genre retrieval uses validation-only galleries and known in-gallery validation positives. These are descriptive development results.",
                   "MXM text source files are read, but parsing is restricted to train+validation catalog IDs. Held-out ABCD test IDs/labels are not materialized into diagnostic features or retrieval evaluation.",
                   "The inherited official/legacy test-positive exclusion policy is retained only as frozen schedule/checkpoint provenance; P1-B does not resample triplets or train.",
                   "", "## Fold/seed aggregate metrics", "",
                   "| Fold | Seed | D0 R@10 / MRR@10 / NDCG@10 | D4 R@10 / MRR@10 / NDCG@10 | D4−D0 R@10 / MRR@10 / NDCG@10 |",
                   "|---:|---:|---:|---:|---:|"]
        by_pair = {}
        for row in rows:
            by_pair.setdefault((row["fold"], row["seed"]), {})[row["system"][:2]] = row
        for (f, seed), pair in sorted(by_pair.items()):
            d0, d4 = pair["D0"], pair["D4"]
            def triple(record):
                return "/".join(str(record["macro_query"][metric]) for metric in ("Recall@10", "MRR@10", "NDCG@10"))
            delta = "/".join(
                str(d4["macro_query"][metric] - d0["macro_query"][metric])
                if d4["macro_query"][metric] is not None and d0["macro_query"][metric] is not None
                else "None" for metric in ("Recall@10", "MRR@10", "NDCG@10"))
            summary.append(f"| {f} | {seed} | {triple(d0)} | {triple(d4)} | {delta} |")
        summary += ["", "## Fold/seed eligibility coverage", "",
                    "`fold_query_coverage.csv` records each system's eligible and ineligible validation-query counts, positive counts, and retrieval count provenance. Per-query rows contain compact metrics and hashes; gallery/relevance ID lists are not repeated.", "",
                    "| Fold | Seed | System | Eligible queries | Missing/unusable genre | No candidates | No in-gallery positive | Eligible positives |",
                    "|---:|---:|---|---:|---:|---:|---:|---:|"]
        for row in coverage_rows:
            summary.append(f"| {row['fold']} | {row['seed']} | {row['system']} | {row.get('eligible', 0)} | "
                           f"{row.get('missing_or_unusable_genre', 0)} | {row.get('no_candidates', 0)} | "
                           f"{row.get('no_in_gallery_positive', 0)} | {row.get('eligible_positive_count', 0)} |")
        (stage / "analysis.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        final.parent.mkdir(parents=True, exist_ok=True)
        from src.evaluate_neurotune_v1_p1a_intra_genre_retrieval import _atomic_publish
        _atomic_publish(stage, final)
        return report
    finally:
        shutil.rmtree(stage, ignore_errors=True)

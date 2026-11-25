"""Helpers for the MSD audio-only CV baseline evaluation."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from scripts.audit_msd_audio_cv import _source_sha256
from scripts.run_msd_audio_cv import _sha256, _validate_audit
from src.data.preprocess_audio import load_audio_sequence_cache
from src.evaluate import build_test_relevance, evaluate_rankings
from src.retrieve import rank_top_k

ROOT = Path(__file__).resolve().parents[1]
K = 10
SEEDS = [684, 685, 686]
METRICS = [f"{metric}@{K}" for metric in ("Recall", "HitRate", "MRR", "NDCG")]
CODE_FILES = (
    "scripts/evaluate_msd_audio_cv_baselines.py",
    "scripts/audit_msd_audio_cv.py",
    "scripts/run_msd_audio_cv.py",
    "src/data/preprocess_audio.py",
    "src/evaluate.py",
    "src/retrieve.py",
)


def _metric_summary(observations):
    """Summarize descriptive observations without implying independence."""
    return {
        "observation_count": len(observations),
        "metrics": {
            metric: {
                "mean": float(np.mean([row.get(metric, 0.0) for row in observations])),
                "sample_std": (float(np.std([row.get(metric, 0.0) for row in observations], ddof=1))
                               if len(observations) > 1 else 0.0),
            }
            for metric in METRICS
        },
    }


def _descriptive_summary(observations, observation_unit):
    return {"observation_unit": observation_unit, **_metric_summary(observations)}


def _validate_inputs(audit_path, cache_dir, lastfm_path, output_path, seeds=None):
    """Validate provenance and output safety before baseline evaluation starts."""
    audit_path, cache_dir, lastfm_path, output_path = map(
        Path, (audit_path, cache_dir, lastfm_path, output_path))
    output_path = output_path.resolve()
    inputs = [audit_path.resolve(), lastfm_path.resolve()]
    completed_report = audit_path.parent / "msd_audio_only_cv_experiment.json"
    inputs.append(completed_report.resolve())
    checkpoint_dir = audit_path.parent / "checkpoints"
    inputs.extend(path.resolve() for path in checkpoint_dir.glob("msd_audio_only_cv_fold_*.pt"))
    cache_paths = {}
    metadata_paths = {}
    for fold in range(3):
        cache_paths[fold] = cache_dir / f"msd_audio_only_cv_fold_{fold}.h5"
        metadata_paths[fold] = cache_dir / f"msd_audio_only_cv_fold_{fold}_meta.json"
        inputs.extend((cache_paths[fold].resolve(), metadata_paths[fold].resolve()))
    if output_path in inputs:
        raise ValueError("output path must not collide with an input path")
    protected_directories = [path.resolve() for path in (lastfm_path, cache_dir)
                             if path.is_dir()]
    if any(output_path == directory or output_path.is_relative_to(directory)
           for directory in protected_directories):
        raise ValueError("output path must not be inside a read-only input directory")
    if output_path.exists():
        raise FileExistsError(f"output target already exists: {output_path}")

    with audit_path.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    ids_by_fold, _, all_ids = _validate_audit(audit)
    audit_sha256 = _sha256(audit_path)
    expected_lastfm = audit.get("source_sha256", {}).get("lastfm")
    if not isinstance(expected_lastfm, str):
        raise ValueError("Last.fm source provenance is missing from the CV audit")
    if expected_lastfm != _source_sha256(lastfm_path):
        raise ValueError("Last.fm source provenance does not match the current Last.fm tree")

    model_report_path = completed_report
    with model_report_path.open(encoding="utf-8") as stream:
        model_report = json.load(stream)
    model_seeds = model_report.get("config", {}).get("seeds")
    expected_seeds = list(SEEDS if seeds is None else seeds)
    if (model_report.get("fold_count") != 3 or not isinstance(model_seeds, list) or
            set(model_seeds) != set(expected_seeds)):
        raise ValueError("model report fold_count or seed set does not match baseline configuration")
    model_sources = model_report.get("source_sha256", {})
    expected_sources = {"audit": audit_sha256, "lastfm_similar": _source_sha256(lastfm_path)}
    for fold in ids_by_fold:
        expected_sources[f"fold_{fold}_cache"] = _sha256(cache_paths[fold])
        expected_sources[f"fold_{fold}_metadata"] = _sha256(metadata_paths[fold])
    if any(model_sources.get(key) != value for key, value in expected_sources.items()):
        raise ValueError("model report source provenance does not match current audit, Last.fm, or caches")

    for fold, ids in ids_by_fold.items():
        expected_train = [track_id for other in sorted(ids_by_fold) if other != fold
                          for track_id in ids_by_fold[other]]
        with metadata_paths[fold].open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("audit_sha256") != audit_sha256:
            raise ValueError(f"fold {fold} metadata audit_sha256 does not match current CV audit")
        split_ids = metadata.get("split_ids", {})
        if (split_ids.get("train") != expected_train or split_ids.get("validation") != [] or
                split_ids.get("test") != ids):
            raise ValueError(f"fold {fold} metadata split_ids do not match audit")
        normalization = metadata.get("normalization")
        if not isinstance(normalization, dict) or not isinstance(normalization.get("train_ids"), list):
            raise ValueError(f"fold {fold} normalization metadata must contain train_ids")
        if any(not isinstance(track_id, str) for track_id in normalization["train_ids"]):
            raise ValueError(f"fold {fold} normalization train_ids must be strings")
        if normalization["train_ids"] != expected_train:
            raise ValueError(f"fold {fold} normalization train_ids do not match expected train IDs")
        model_fold = model_report.get("per_fold", {}).get(str(fold), {})
        model_test = model_fold.get("test", {})
        heldout_relevance = build_test_relevance(
            [{"source": source, "target": target}
             for source, target in audit["folds"][str(fold)]["positive_edges"]], ids)
        expected_query_count = len(heldout_relevance)
        expected_edge_count = sum(map(len, heldout_relevance.values())) // 2
        if (model_fold.get("train_ids") != expected_train or model_fold.get("test_ids") != ids or
                model_test.get("gallery_count") != len(ids) or
                model_test.get("query_count") != expected_query_count or
                model_test.get("unique_edge_count") != expected_edge_count):
            raise ValueError(f"model report fold {fold} IDs or test count do not match audit")
        if set(model_fold.get("per_seed", {})) != {str(seed) for seed in model_seeds}:
            raise ValueError(f"model report fold {fold} seed set does not match configuration")
        load_audio_sequence_cache(cache_paths[fold], metadata_paths[fold], expected_ids=all_ids)
    return audit, ids_by_fold, all_ids, model_report, expected_sources


def _random_rankings(gallery_ids, fold, seed):
    """Sample up to K unique, non-query IDs without replacement per query."""
    gallery_ids = list(gallery_ids)
    rng = np.random.default_rng(np.random.SeedSequence([int(fold), int(seed)]))
    rankings = {}
    for query_id in gallery_ids:
        candidates = np.asarray([track_id for track_id in gallery_ids if track_id != query_id])
        rankings[query_id] = [
            {"song_id": track_id} for track_id in rng.choice(
                candidates, size=min(K, len(candidates)), replace=False)
        ]
    return rankings


def _mean_pool_audio(audio):
    """Mean-pool continuous audio channels and L2-normalize one vector per track."""
    pooled = np.asarray(audio)[:, :, :25].mean(axis=1)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return np.divide(pooled, norms, out=np.zeros_like(pooled), where=norms != 0)


def _run_impl(audit_path, cache_dir, lastfm_path, output_path, seeds):
    audit_path, cache_dir, lastfm_path, output_path = map(
        Path, (audit_path, cache_dir, lastfm_path, output_path))
    audit, ids_by_fold, all_ids, model_report, source_sha256 = _validate_inputs(
        audit_path, cache_dir, lastfm_path, output_path, seeds)
    model_report_path = audit_path.parent / "msd_audio_only_cv_experiment.json"
    output_folds = {}
    for fold, test_ids in sorted(ids_by_fold.items()):
        cache_path = cache_dir / f"msd_audio_only_cv_fold_{fold}.h5"
        metadata_path = cache_dir / f"msd_audio_only_cv_fold_{fold}_meta.json"
        cache_ids, audio, _ = load_audio_sequence_cache(
            cache_path, metadata_path, expected_ids=all_ids)
        indices = {track_id: index for index, track_id in enumerate(cache_ids)}
        test_audio = audio[[indices[track_id] for track_id in test_ids]]
        del audio
        test_edges = [{"source": source, "target": target}
                      for source, target in audit["folds"][str(fold)]["positive_edges"]]
        relevance = build_test_relevance(test_edges, test_ids)
        embeddings = _mean_pool_audio(test_audio)
        del test_audio
        output_folds[str(fold)] = {
            "test_ids": test_ids,
            "test": {"gallery_count": len(test_ids), "query_count": len(relevance),
                     "unique_edge_count": sum(map(len, relevance.values())) // 2},
            "audio_metrics": None,
            "per_seed": {},
        }
        audio_rankings = {track_id: rank_top_k(embeddings, test_ids, track_id, k=K)
                          for track_id in test_ids}
        output_folds[str(fold)]["audio_metrics"] = evaluate_rankings(
            audio_rankings, relevance, k=K)
        del audio_rankings
        for seed in seeds:
            random_rankings = _random_rankings(test_ids, fold, seed)
            reference = model_report["per_fold"][str(fold)]["per_seed"][str(seed)]["metrics"]
            output_folds[str(fold)]["per_seed"][str(seed)] = {
                "random_metrics": evaluate_rankings(random_rankings, relevance, k=K),
                "model_reference_metrics": reference,
            }
        output_folds[str(fold)]["summary"] = {
            "random": _descriptive_summary([
                record["random_metrics"] for record in output_folds[str(fold)]["per_seed"].values()],
                "seed observations within this fold"),
            "model_reference": _descriptive_summary([
                record["model_reference_metrics"]
                for record in output_folds[str(fold)]["per_seed"].values()],
                "seed observations within this fold"),
        }
        del embeddings, relevance, test_edges

    audio_observations = [record["audio_metrics"] for record in output_folds.values()]
    random_observations = [seed_record["random_metrics"]
                           for record in output_folds.values()
                           for seed_record in record["per_seed"].values()]
    model_observations = [seed_record["model_reference_metrics"]
                          for record in output_folds.values()
                          for seed_record in record["per_seed"].values()]
    code_sha256 = {relative: _sha256(ROOT / relative) for relative in CODE_FILES}
    report = {
        "experiment": "MSD_audio_only_CV_post_hoc_baselines",
        "exploratory": True,
        "caveat": "Post-hoc exploratory baselines; not model tuning and not directly comparable outside these held-out folds.",
        "random_sampling": "Random rankings are sampled independently per query: for each fold and seed, choose 10 distinct non-query gallery tracks without replacement.",
        "fold_count": 3,
        "seeds": list(seeds),
        "per_fold": output_folds,
        "summary": {
            "audio": _descriptive_summary(audio_observations, "held-out fold observations"),
            "random": _descriptive_summary(random_observations, "fold-seed observations"),
            "model_reference": _descriptive_summary(model_observations, "fold-seed observations"),
        },
        "source_sha256": {**source_sha256, "model_report": _sha256(model_report_path)},
        "code_sha256": code_sha256,
        "runtime_versions": {"python": platform.python_version(), "numpy": str(np.__version__)},
        "paths": {"audit": str(audit_path), "cache_dir": str(cache_dir),
                  "lastfm_similar": str(lastfm_path), "model_report": str(model_report_path),
                  "output": str(output_path)},
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def run(audit_path, cache_dir, lastfm_path, output_path, seeds=None):
    """Run fold-local baselines and remove a newly-created partial report on failure."""
    output_path = Path(output_path)
    existed_before = output_path.exists()
    try:
        return _run_impl(audit_path, cache_dir, lastfm_path, output_path,
                         list(SEEDS if seeds is None else seeds))
    except BaseException:
        if not existed_before and output_path.is_file():
            try:
                output_path.unlink()
            except BaseException:
                pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "results/msd_audio_only_cv_audit.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--lastfm", type=Path, default=ROOT / "data/processed/lastfm_similar")
    parser.add_argument("--output", type=Path, default=ROOT / "results/msd_audio_only_cv_baselines.json")
    args = parser.parse_args(argv)
    run(args.audit, args.cache_dir, args.lastfm, args.output)


if __name__ == "__main__":
    main()

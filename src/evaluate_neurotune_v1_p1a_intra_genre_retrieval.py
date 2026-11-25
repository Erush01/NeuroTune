"""Run validation-only same-genre retrieval for frozen D0/D4 systems."""

from collections.abc import Mapping
import argparse
import hashlib
import csv
import ctypes
import errno
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

from src.evaluate import evaluate_rankings
from src.retrieve import rank_top_k_many

OUTPUT_NAME = "neurotune_v1_p1a_intra_genre_retrieval"
_K = 10


def main(argv=None):
    """Run the fixed P1-A intra-genre retrieval protocol."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root,
                        help=f"data root (default: {root})")
    parser.add_argument("--results-root", type=Path, default=root / "results",
                        help=f"results root (default: {root / 'results'})")
    args = parser.parse_args(argv)
    return _run(args.data_root, args.results_root)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def _hash_path(path):
    from scripts.run_neurotune_v1_full_modal_ablation import _hash_path as canonical_hash_path

    return canonical_hash_path(path)


def _require_renameat2():
    if platform.system() != "Linux":
        raise RuntimeError("atomic no-overwrite publication requires Linux renameat2")
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-overwrite directory publication requires renameat2")
    return renameat2


def _atomic_publish(stage, final):
    renameat2 = _require_renameat2()
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(stage), -100, os.fsencode(final), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), str(final))
        raise OSError(error, os.strerror(error), str(final))


def evaluate_query_metrics(ranking, relevant_ids, k=10):
    """Compute Recall@K, MRR@K, and binary NDCG@K for one eligible query."""
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        raise ValueError("relevant_ids must contain at least one ID")
    relevant_ranks = [
        rank for rank, item in enumerate(ranking[:k], start=1)
        if item["song_id"] in relevant_ids
    ]
    recall = len(relevant_ranks) / len(relevant_ids)
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        f"Recall@{k}": recall,
        f"MRR@{k}": mrr,
        f"NDCG@{k}": dcg / idcg if idcg else 0.0,
    }


def build_validation_candidate_galleries(validation_ids, genres, validation_edges):
    """Build deterministic same-genre galleries and in-gallery positives.

    ``genres`` maps validation track IDs to their raw catalog genre strings;
    absent or blank values are treated as unusable labels. ``validation_edges``
    contains known undirected positive edges and may reference only validation IDs.
    """
    if isinstance(validation_ids, (str, bytes)):
        raise ValueError("validation_ids must be an iterable of track IDs")
    ids = list(validation_ids)
    if any(not isinstance(track_id, str) or not track_id.strip() for track_id in ids):
        raise ValueError("validation IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("validation IDs must be unique")
    id_set = set(ids)

    if not isinstance(genres, Mapping):
        raise ValueError("genres must map track IDs to raw genre labels")
    if any(track_id not in id_set for track_id in genres):
        raise ValueError("genre labels may only be supplied for validation IDs")
    for genre in genres.values():
        if genre is not None and not isinstance(genre, str):
            raise ValueError("genre labels must be strings or None")

    if isinstance(validation_edges, (str, bytes)):
        raise ValueError("validation_edges must be an iterable of ID pairs")
    positives = set()
    for edge in validation_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("each validation edge must contain exactly two IDs")
        left, right = edge
        if (not isinstance(left, str) or not isinstance(right, str)
                or left not in id_set or right not in id_set or left == right):
            raise ValueError("validation edges must connect distinct validation IDs")
        pair = frozenset((left, right))
        if pair in positives:
            raise ValueError("duplicate undirected validation edge")
        positives.add(pair)

    usable_genres = {
        track_id: genre for track_id, genre in genres.items()
        if isinstance(genre, str) and genre.strip()
    }
    queries = {}
    counts = {
        "missing_or_unusable_genre": 0,
        "no_candidates": 0,
        "no_in_gallery_positive": 0,
        "eligible": 0,
        "eligible_small_gallery": 0,
    }
    for query_id in sorted(ids):
        genre = usable_genres.get(query_id)
        if genre is None:
            reason = "missing_or_unusable_genre"
            gallery_ids = []
        else:
            gallery_ids = sorted(
                track_id for track_id, candidate_genre in usable_genres.items()
                if track_id != query_id and candidate_genre == genre
            )
            if not gallery_ids:
                reason = "no_candidates"
            else:
                reason = "no_in_gallery_positive"

        relevant_ids = []
        if gallery_ids:
            gallery_set = set(gallery_ids)
            relevant_ids = sorted(
                next(iter(pair - {query_id})) for pair in positives
                if query_id in pair and (pair - {query_id}) <= gallery_set
            )
            if relevant_ids:
                reason = "eligible"

        eligible = reason == "eligible"
        small_gallery = eligible and len(gallery_ids) < 10
        queries[query_id] = {
            "genre": genre,
            "reason": reason,
            "eligible": eligible,
            "gallery_ids": gallery_ids,
            "gallery_size": len(gallery_ids),
            "relevant_ids": relevant_ids,
            "positive_count": len(relevant_ids),
            "small_gallery": small_gallery,
        }
        counts[reason] += 1
        if small_gallery:
            counts["eligible_small_gallery"] += 1

    return {"queries": queries, "counts": counts}


def evaluate_intra_genre_retrieval(embeddings, song_ids, candidate_galleries):
    """Rank and evaluate only each eligible query's same-genre gallery."""
    k = 10
    try:
        vectors = np.asarray(embeddings, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("embeddings must be a finite 2D array") from error
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("embeddings must be a finite 2D array")
    if isinstance(song_ids, (str, bytes)):
        raise ValueError("song_ids must be an iterable of unique non-empty IDs")
    song_ids = list(song_ids)
    if (len(song_ids) != vectors.shape[0]
            or any(not isinstance(song_id, str) or not song_id.strip() for song_id in song_ids)
            or len(set(song_ids)) != len(song_ids)):
        raise ValueError("song_ids must be unique, non-empty, and row-aligned with embeddings")

    if (not isinstance(candidate_galleries, Mapping)
            or not isinstance(candidate_galleries.get("queries"), Mapping)
            or not isinstance(candidate_galleries.get("counts"), Mapping)):
        raise ValueError("candidate_galleries must contain query and count mappings")
    queries = candidate_galleries["queries"]
    if set(queries) != set(song_ids):
        raise ValueError("candidate query IDs must exactly match song_ids")

    status_keys = {
        "missing_or_unusable_genre", "no_candidates", "no_in_gallery_positive", "eligible",
    }
    count_keys = status_keys | {"eligible_small_gallery"}
    candidate_counts = candidate_galleries["counts"]
    if set(candidate_counts) != count_keys:
        raise ValueError("candidate status counts must contain exactly the builder count fields")

    expected_counts = {key: 0 for key in count_keys}
    eligible = {}
    genre_group_ids = {}
    required_fields = {
        "genre", "reason", "eligible", "gallery_ids", "gallery_size", "relevant_ids",
        "positive_count", "small_gallery",
    }
    song_id_set = set(song_ids)
    for query_id, data in queries.items():
        if not isinstance(data, Mapping) or not required_fields <= set(data):
            raise ValueError("each candidate query must contain a complete record")
        genre = data["genre"]
        if genre is not None and (not isinstance(genre, str) or not genre.strip()):
            raise ValueError("query genres must be non-empty strings or None")
        if not isinstance(data["eligible"], bool):
            raise ValueError("query eligible flags must be booleans")
        gallery_ids = data["gallery_ids"]
        relevant_ids = data["relevant_ids"]
        if (isinstance(gallery_ids, (str, bytes)) or not isinstance(gallery_ids, (list, tuple))
                or any(not isinstance(item, str) or not item.strip() for item in gallery_ids)
                or len(gallery_ids) != len(set(gallery_ids))
                or query_id in gallery_ids or not set(gallery_ids) <= song_id_set):
            raise ValueError("gallery IDs must be unique known IDs and exclude their query")
        if (isinstance(relevant_ids, (str, bytes)) or not isinstance(relevant_ids, (list, tuple))
                or any(not isinstance(item, str) or not item.strip() for item in relevant_ids)
                or len(relevant_ids) != len(set(relevant_ids))
                or not set(relevant_ids) <= set(gallery_ids)):
            raise ValueError("relevant IDs must be unique members of the query gallery")
        if (type(data["gallery_size"]) is not int or data["gallery_size"] != len(gallery_ids)
                or type(data["positive_count"]) is not int
                or data["positive_count"] != len(relevant_ids)
                or not isinstance(data["small_gallery"], bool)):
            raise ValueError("candidate gallery size, positive count, or small-gallery flag is inconsistent")

        if genre is None:
            expected_reason = "missing_or_unusable_genre"
            if gallery_ids or relevant_ids:
                raise ValueError("queries without a usable genre cannot have galleries or positives")
        elif not gallery_ids:
            expected_reason = "no_candidates"
            if relevant_ids:
                raise ValueError("queries without candidates cannot have in-gallery positives")
        elif relevant_ids:
            expected_reason = "eligible"
        else:
            expected_reason = "no_in_gallery_positive"
        if (data["reason"] != expected_reason
                or data["eligible"] is not (expected_reason == "eligible")
                or data["small_gallery"] is not (expected_reason == "eligible" and len(gallery_ids) < 10)):
            raise ValueError("candidate reason and query status fields are inconsistent")
        expected_counts[expected_reason] += 1
        if expected_reason == "eligible" and len(gallery_ids) < 10:
            expected_counts["eligible_small_gallery"] += 1

        if data["eligible"]:
            if any(
                not isinstance(queries[gallery_id], Mapping)
                or queries[gallery_id].get("genre") != genre
                for gallery_id in gallery_ids
            ):
                raise ValueError("eligible query galleries must contain only tracks of the same usable genre")
            group_ids = tuple(sorted([query_id, *gallery_ids]))
            previous_ids = genre_group_ids.setdefault(genre, group_ids)
            if previous_ids != group_ids:
                raise ValueError("eligible queries in one genre must share the exact candidate group")
            eligible[query_id] = data

    if any(type(candidate_counts[key]) is not int or candidate_counts[key] != value
           for key, value in expected_counts.items()):
        raise ValueError("candidate status counts disagree with query records")

    id_to_index = {song_id: index for index, song_id in enumerate(song_ids)}

    def digest(value):
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    candidate_hashes = {
        query_id: digest({"query_id": query_id, "candidate_ids": data["gallery_ids"]})
        for query_id, data in eligible.items()
    }
    label_hashes = {
        query_id: digest({"query_id": query_id, "relevant_ids": data["relevant_ids"]})
        for query_id, data in eligible.items()
    }

    grouped_queries = {}
    for query_id, data in eligible.items():
        grouped_queries.setdefault(data["genre"], []).append(query_id)

    rankings = {}
    relevance = {}
    for genre, query_ids in sorted(grouped_queries.items()):
        group_ids = genre_group_ids[genre]
        group_embeddings = [vectors[id_to_index[song_id]] for song_id in group_ids]
        group_rankings = rank_top_k_many(group_embeddings, group_ids, query_ids, k=k)
        rankings.update(group_rankings)
        relevance.update({query_id: set(eligible[query_id]["relevant_ids"]) for query_id in query_ids})

    per_query = {}
    for query_id, data in eligible.items():
        evaluated = evaluate_query_metrics(rankings[query_id], relevance[query_id], k=k)
        per_query[query_id] = {
            "genre": data["genre"],
            "gallery_ids": list(data["gallery_ids"]),
            "relevant_ids": list(data["relevant_ids"]),
            "positive_count": data["positive_count"],
            **evaluated,
        }

    def aggregate(query_ids):
        query_ids = list(query_ids)
        if not query_ids:
            return {f"Recall@{k}": None, f"MRR@{k}": None, f"NDCG@{k}": None,
                    "query_count": 0, "positive_count": 0}
        evaluated = evaluate_rankings(
            {query_id: rankings[query_id] for query_id in query_ids},
            {query_id: relevance[query_id] for query_id in query_ids},
            k=k,
        )
        return {
            f"Recall@{k}": evaluated[f"Recall@{k}"],
            f"MRR@{k}": evaluated[f"MRR@{k}"],
            f"NDCG@{k}": evaluated[f"NDCG@{k}"],
            "query_count": len(query_ids),
            "positive_count": sum(len(relevance[query_id]) for query_id in query_ids),
        }

    by_group = {genre: aggregate(query_ids) for genre, query_ids in grouped_queries.items()}
    all_ids = list(eligible)
    macro_query = aggregate(all_ids)
    group_metrics = list(by_group.values())
    genre_macro = {
        f"{metric}@{k}": (
            sum(item[f"{metric}@{k}"] for item in group_metrics) / len(group_metrics)
            if group_metrics else None
        )
        for metric in ("Recall", "MRR", "NDCG")
    }
    genre_macro["genre_count"] = len(group_metrics)
    hits = sum(
        1 for query_id in all_ids
        for item in rankings[query_id][:k]
        if item["song_id"] in relevance[query_id]
    )
    positive_count = sum(len(relevance[query_id]) for query_id in all_ids)
    metrics = {
        "macro_query": macro_query,
        "genre_macro": genre_macro,
        "micro_recall@10": hits / positive_count if positive_count else None,
    }
    small_ids = [query_id for query_id, data in eligible.items() if data["small_gallery"]]
    small_gallery_metrics = aggregate(small_ids)
    small_positive_count = sum(len(relevance[query_id]) for query_id in small_ids)
    small_hits = sum(
        1 for query_id in small_ids
        for item in rankings[query_id][:k]
        if item["song_id"] in relevance[query_id]
    )
    small_gallery_metrics["micro_recall@10"] = (
        small_hits / small_positive_count if small_positive_count else None
    )
    return {
        "rankings": rankings,
        "per_query": per_query,
        "per_genre": by_group,
        "metrics": metrics,
        "small_gallery_metrics": small_gallery_metrics,
        "counts": {
            **candidate_galleries["counts"],
            "eligible_positive_count": positive_count,
            "relevant_hits_at_10": hits,
        },
        "candidate_hashes": candidate_hashes,
        "label_hashes": label_hashes,
    }


def _run(data_root=None, results_root=None, *, _test_only=False, _dependencies=None):
    """Evaluate the frozen D0/D4 models and matched cosine baselines on validation only."""
    _require_renameat2()
    started = time.monotonic()
    import torch
    from scripts.run_neurotune_v1_full_modal_ablation import FOLDS, INPUTS as D0_INPUTS, ROOT, SEEDS
    from scripts.run_neurotune_v1_d4_multi_window_audio import INPUTS as D4_INPUTS, _take
    from src.data.neurotune_v1_abcd_cv_training import (
        build_abcd_cv_development_features, build_abcd_cv_d4_multi_window_features,
        build_abcd_cv_fold_mxm, load_abcd_cv_training,
    )
    from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
    from src.evaluate_d4_modality_sensitivity import (
        _assert_original_parity, _select_references, _validate_report_hashes,
        CODE_PATHS, PROTECTED_RESULT_NAMESPACES, _validate_checkpoint_transform_identity,
        _validate_d0_transform_manifest, _validate_schedule_copy, _slice, _embed,
    )
    from src.evaluate_neurotune_v1 import (
        build_neurotune_v1_cosine_baseline, build_d4_neurotune_v1_cosine_baseline,
        evaluate_neurotune_v1_retrieval,
    )
    from src.model_v1 import NeuroTuneV1Encoder

    deps = dict(_dependencies or {})
    if deps and not _test_only:
        raise ValueError("dependency injection is test-only; production protocol is fixed")
    root = Path(data_root if data_root is not None else ROOT)
    outroot = Path(results_root if results_root is not None else ROOT / "results")
    final = outroot / OUTPUT_NAME
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    d0root, d4root = outroot / "neurotune_v1_full_modal_ablation", outroot / "neurotune_v1_d4_multi_window_audio"
    schedroot = outroot / "neurotune_v1_abcd_cv"
    inputs = {**{k: root / v for k, v in D0_INPUTS.items()}, **{k: root / v for k, v in D4_INPUTS.items()}}
    checkpoint_paths = {(f, s, v): (d0root if v == "D0" else d4root) /
                        f"checkpoints/fold_{f}_D_seed_{s}.pt"
                        for f in FOLDS for s in SEEDS for v in ("D0", "D4")}
    schedule_paths = {(f, s): schedroot / f"triplet_schedules/fold_{f}_seed_{s}.npz"
                      for f in FOLDS for s in SEEDS}
    copies = {(f, s): d4root / f"triplet_schedules/fold_{f}_seed_{s}.npz"
              for f in FOLDS for s in SEEDS}
    d0report_path, d4report_path = d0root / "experiment.json", d4root / "experiment.json"
    protected = {**{f"input:{k}": p for k, p in inputs.items()}, "d0_report": d0report_path,
                 "d4_report": d4report_path,
                 **{f"checkpoint:{f}/{s}/{v}": p for (f, s, v), p in checkpoint_paths.items()},
                 **{f"schedule:{f}/{s}": p for (f, s), p in schedule_paths.items()},
                 **{f"schedule_copy:{f}/{s}": p for (f, s), p in copies.items()},
                  **{f"code:{name}": ROOT / name for name in CODE_PATHS},
                  "code:src/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py": Path(__file__),
                  "code:scripts/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py":
                      ROOT / "scripts/evaluate_neurotune_v1_p1a_intra_genre_retrieval.py"}
    hash_path = deps.get("hash_path", _hash_path)
    before = {key: hash_path(path) for key, path in protected.items()}
    reports = deps.get("reports")
    if reports is None:
        d0report = json.loads(d0report_path.read_text(encoding="utf-8"))
        d4report = json.loads(d4report_path.read_text(encoding="utf-8"))
    else:
        d0report, d4report = reports["D0"], reports["D4"]
    d0refs, d4refs = _select_references(d0report, "D0"), _select_references(d4report, "D4")
    if d0report.get("status") != "complete" or d4report.get("status") != "complete":
        raise ValueError("D0/D4 references must be complete for all fixed fold/seed pairs")
    if (d4report.get("evaluation_scope") != "validation_only_development"
            or d4report.get("independent_test_artifacts_created") is not False):
        raise ValueError("D4 report must confirm validation-only scope and no independent-test artifacts")
    _validate_report_hashes(d0report, d4report, before)
    namespace_names = tuple(dict.fromkeys((*PROTECTED_RESULT_NAMESPACES,
                                         "neurotune_v1_d4_modality_sensitivity", "neurotune_v1_d4_triplet_diagnostics")))
    namespace_before = {name: hash_path(outroot / name) for name in namespace_names}

    loader = deps.get("loader", load_abcd_cv_training)
    loaded = loader(**{k: p for k, p in inputs.items() if k in D0_INPUTS and k != "mxm_path"})
    mxm_builder = deps.get("mxm_builder", build_abcd_cv_fold_mxm)
    # MXM is parsed fold-locally below so test partition tracks never reach feature parsing.
    d0_builder, d4_builder = deps.get("d0_builder", build_abcd_cv_development_features), deps.get("d4_builder", build_abcd_cv_d4_multi_window_features)
    records, fold_genre_rows, query_rows = [], [], []
    outroot.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.stage-", dir=outroot))
    try:
        if {fold["fold"] for fold in loaded["folds"]} != set(FOLDS) or len(loaded["folds"]) != len(FOLDS):
            raise ValueError("validated loader must return exactly the three fixed folds")
        mxm_provenance = {}
        for fold in loaded["folds"]:
            f = fold["fold"]
            fold_input = {k: loaded[k] for k in ("manifest", "catalog", "cache_path")}
            development = {k: fold[k] for k in ("fold", "train", "validation")}
            all_ids, val_ids = sorted(fold["train"] + fold["validation"]), list(fold["validation"])
            test_ids = set(fold.get("test", []))
            if set(all_ids) & test_ids or set(val_ids) & test_ids:
                raise ValueError("test IDs may not enter feature builders or validation evaluation")
            fold_input["catalog"] = {track_id: loaded["catalog"][track_id] for track_id in all_ids}
            lyrics, mxm_provenance[str(f)] = mxm_builder(fold_input, inputs["mxm_path"])
            d0features, d0transform = d0_builder(fold_input, development, lyrics)
            d4features, d4transform = d4_builder(fold_input, development, lyrics, inputs["d4_cache_path"])
            edges = list(fold["validation_edges"])
            if any(a not in set(val_ids) or b not in set(val_ids) for a, b in edges):
                raise ValueError("validation edges must connect validation IDs")
            d0val, d4val = _slice(d0features, all_ids, val_ids), _take(d4features, all_ids, val_ids)
            genres = {track_id: loaded["catalog"][track_id].get("genre") for track_id in val_ids}
            galleries = build_validation_candidate_galleries(val_ids, genres, edges)
            baseline0 = deps.get("d0_baseline_builder", build_neurotune_v1_cosine_baseline)(d0val, val_ids)
            baseline4 = deps.get("d4_baseline_builder", build_d4_neurotune_v1_cosine_baseline)(d4val, val_ids)
            systems = [("D0-cosine", baseline0, None), ("D4-cosine", baseline4, None)]
            for seed in SEEDS:
                for variant, feats, transform, refs, rootdir in (
                        ("D0", d0val, d0transform, d0refs, d0root), ("D4", d4val, d4transform, d4refs, d4root)):
                    ref = refs[(f, seed)]
                    path = checkpoint_paths[(f, seed, variant)]
                    blob = deps["checkpoint_bytes"]((f, seed, variant)) if "checkpoint_bytes" in deps else path.read_bytes()
                    if hashlib.sha256(blob).hexdigest() != before[f"checkpoint:{f}/{seed}/{variant}"]:
                        raise ValueError("checkpoint changed after protected input snapshot")
                    schedule_hash = before[f"schedule:{f}/{seed}"]
                    copy_hash = _validate_schedule_copy(
                        schedule_paths[(f, seed)], copies[(f, seed)], val_ids,
                        ref.get("schedule_sha256"))
                    if d4report.get("hashes", {}).get("schedule_copies", {}).get(f"{f}/{seed}") != copy_hash:
                        raise ValueError("D4 report schedule-copy hash differs from its physical copy")
                    cp_obj = deps["checkpoint_loader"](blob) if "checkpoint_loader" in deps else torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
                    expected_path = path.relative_to(rootdir).as_posix()
                    if (ref.get("checkpoint_path") != expected_path or ref.get("checkpoint_sha256") != before[f"checkpoint:{f}/{seed}/{variant}"]
                            or ref.get("schedule_sha256") != schedule_hash):
                        raise ValueError("reference checkpoint/schedule identity mismatch")
                    provenance = cp_obj.get("provenance", {})
                    if (cp_obj.get("variant") != "D" or cp_obj.get("fold", f) != f or cp_obj.get("seed") != seed
                            or provenance.get("fold") != f or provenance.get("seed") != seed
                            or provenance.get("variant") != "D"
                            or provenance.get("schedule_sha256") != schedule_hash or cp_obj.get("epoch") != ref.get("selected_epoch")):
                        raise ValueError("checkpoint fold/seed/epoch/schedule identity mismatch")
                    _validate_checkpoint_transform_identity(cp_obj, variant, ref, transform)
                    if variant == "D0":
                        _validate_d0_transform_manifest(transform, cp_obj["transform_manifest"])
                    expected = ref.get("validation", {})
                    _assert_original_parity(cp_obj.get("selection_metrics"), expected)
                    model = deps["model_loader"](cp_obj, variant, feats) if "model_loader" in deps else None
                    if model is None:
                        model = NeuroTuneV1Encoder("D", feats["genre"].shape[1])
                        model.load_state_dict(cp_obj["model_state_dict"]); model.eval()
                    ds = NeuroTuneV1Dataset(val_ids, feats, modalities=("bpm", "genre", "lyrics"))
                    batch = neurotune_v1_collate([ds[i] for i in range(len(ds))])
                    embedder = deps.get("embedding_builder")
                    embeddings = embedder(model, batch, variant, f, seed) if embedder else _embed(model, batch, variant == "D4")
                    full = evaluate_neurotune_v1_retrieval(embeddings, val_ids, edges)
                    full_parity = {key: full[key] for key in ("Recall@10", "HitRate@10", "MRR@10", "NDCG@10", "query_count")}
                    full_parity["eligible_positive_edges"] = full["unique_undirected_edge_count"]
                    _assert_original_parity(full_parity, expected)
                    systems.append((f"{variant}-neural", embeddings, seed))
            for system, embedding, seed in systems:
                result = evaluate_intra_genre_retrieval(embedding, val_ids, galleries)
                for query_id, gallery in galleries["queries"].items():
                    data = result["per_query"].get(query_id, {})
                    query_rows.append({"fold": f, "seed": seed, "system": system, "query_id": query_id,
                                       "genre": gallery["genre"], "reason": gallery["reason"],
                                       "eligible": gallery["eligible"], "gallery_size": gallery["gallery_size"],
                                       "in_gallery_positive_count": gallery["positive_count"],
                                       "small_gallery": gallery["small_gallery"],
                                       "candidate_hash": result["candidate_hashes"].get(query_id),
                                       "label_hash": result["label_hashes"].get(query_id),
                                       **{k: data.get(k) for k in ("Recall@10", "MRR@10", "NDCG@10")}})
                unique_positive_edges = {frozenset((query_id, positive_id))
                                         for query_id, query in galleries["queries"].items()
                                         for positive_id in query["relevant_ids"]}
                directed_relevance_count = result["counts"]["eligible_positive_count"]
                records.append({"fold": f, "seed": seed, "system": system,
                                **result["metrics"], "small_gallery_metrics": result["small_gallery_metrics"],
                                "directed_query_relevance_count": directed_relevance_count,
                                "unique_undirected_in_genre_positive_edge_count": len(unique_positive_edges),
                                "counts": result["counts"], "per_genre": result["per_genre"]})
                fold_genre_rows.append({"fold": f, "seed": seed, "system": system, "genre": "__macro_query__",
                                        **result["metrics"]["macro_query"],
                                        "genre_macro_recall@10": result["metrics"]["genre_macro"]["Recall@10"],
                                        "genre_macro_mrr@10": result["metrics"]["genre_macro"]["MRR@10"],
                                        "genre_macro_ndcg@10": result["metrics"]["genre_macro"]["NDCG@10"],
                                        "micro_recall@10": result["metrics"]["micro_recall@10"],
                                        "directed_query_relevance_count": directed_relevance_count,
                                        "unique_undirected_in_genre_positive_edge_count": len(unique_positive_edges),
                                        "small_gallery_query_count": result["small_gallery_metrics"]["query_count"],
                                        **{key: result["counts"][key] for key in (
                                            "missing_or_unusable_genre", "no_candidates", "no_in_gallery_positive")}})
                for genre, metrics in result["per_genre"].items():
                    fold_genre_rows.append({"fold": f, "seed": seed, "system": system, "genre": genre, **metrics})
        # Publish only after all immutable-input and prior-result checks pass.
        after = {key: hash_path(path) for key, path in protected.items()}
        if before != after:
            raise RuntimeError("protected inputs changed during evaluation")
        namespace_after = {name: hash_path(outroot / name) for name in namespace_names}
        if namespace_before != namespace_after:
            raise RuntimeError("protected result namespaces changed during evaluation")
        if len({r["system"] for r in records}) != 4:
            raise RuntimeError("expected exactly four retrieval systems")
        report = {"status": "complete", "evaluation_scope": "validation_only_development",
                  "protocol": {"folds": list(FOLDS), "seeds": list(SEEDS), "k": 10,
                               "systems": ["D0-neural", "D4-neural", "D0-cosine", "D4-cosine"],
                               "cosine_seed": None, "test_source_integrity_only": True,
                               "publication_strategy": "linux_renameat2_rename_noreplace"},
                  "hashes": {"protected_before": before, "protected_after": after,
                             "protected_result_namespaces": {
                                 name: {"before": namespace_before[name], "after": namespace_after[name]}
                                 for name in namespace_names}}, "records": records,
                  "mxm_provenance": mxm_provenance,
                  "caveat": "Same-genre Last.fm positive links may be incomplete. This is descriptive validation-only analysis; no causal, superiority, significance, or generalization claim is made.",
                  "test_boundary": "Test-source files and partition IDs are integrity-only and never enter feature inputs, queries, galleries, relevance, metrics, or training.",
                  "runtime": {"seconds": 0.0, "python": platform.python_version(), "numpy": np.__version__, "pytorch": torch.__version__}}
        def write_csv(name, rows):
            if not rows:
                (stage / name).write_text("\n", encoding="utf-8"); return
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (stage / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        write_csv("per_query_metrics.csv", query_rows)
        write_csv("per_fold_seed_genre_metrics.csv", fold_genre_rows)
        write_csv("gallery_coverage.csv", query_rows)
        report["runtime"]["seconds"] = time.monotonic() - started
        summary_lines = ["# P1-A intra-genre retrieval", "", report["caveat"], "", report["test_boundary"], "",
                         "## Fold/seed metrics", "", "| Fold | Seed | System | Macro-query R@10 / MRR@10 / NDCG@10 | Genre-macro R@10 / MRR@10 / NDCG@10 | Micro R@10 | <10 queries | Directed relevance / unique undirected edges | Ineligible (missing/no candidates/no positives) |",
                         "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in records:
            macro, genre = row["macro_query"], row["genre_macro"]
            counts = row["counts"]
            summary_lines.append(
                f"| {row['fold']} | {row['seed'] if row['seed'] is not None else 'shared'} | {row['system']} | "
                f"{macro['Recall@10']} / {macro['MRR@10']} / {macro['NDCG@10']} | "
                f"{genre['Recall@10']} / {genre['MRR@10']} / {genre['NDCG@10']} | "
                f"{row['micro_recall@10']} | {row['small_gallery_metrics']['query_count']} | "
                f"{row['directed_query_relevance_count']} / {row['unique_undirected_in_genre_positive_edge_count']} | "
                f"{counts['missing_or_unusable_genre']}/{counts['no_candidates']}/{counts['no_in_gallery_positive']} |")
        summary_lines.append("")
        summary_lines.append("Ineligible query reasons are counted per fold/system as missing_or_unusable_genre, no_candidates, and no_in_gallery_positive; eligible_small_gallery separately counts eligible queries with fewer than 10 candidates.")
        summary_lines.append("")
        summary_lines.append("`directed_query_relevance_count` counts query-positive directions; `unique_undirected_in_genre_positive_edge_count` counts each in-gallery positive pair once (the directed count is normally twice the undirected count). Both values are recorded in experiment.json and the macro-query row of per_fold_seed_genre_metrics.csv.")
        summary_lines += ["", "## Per-genre metrics", "", "| Fold | Seed | System | Genre | Queries | Directed positives | Recall@10 | MRR@10 | NDCG@10 |", "|---|---:|---|---|---:|---:|---:|---:|---:|"]
        for item in fold_genre_rows:
            if item["genre"] != "__macro_query__":
                summary_lines.append(f"| {item['fold']} | {item['seed'] if item['seed'] is not None else 'shared'} | {item['system']} | {item['genre']} | {item['query_count']} | {item['positive_count']} | {item['Recall@10']} | {item['MRR@10']} | {item['NDCG@10']} |")
        summary_lines += ["", "Same-genre Last.fm positive links may be incomplete. No causal, superiority, significance, or generalization claim is made.", ""]
        (stage / "analysis.md").write_text("\n".join(summary_lines), encoding="utf-8")
        (stage / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _atomic_publish(stage, final)
        return report
    finally:
        shutil.rmtree(stage, ignore_errors=True)

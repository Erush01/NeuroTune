"""Pure masking and behavior-change helpers for D-model sensitivity checks."""

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import torch

from .evaluate_neurotune_v1 import evaluate_neurotune_v1_retrieval
from .retrieve import rank_top_k_many


_MODALITIES = ("audio", "bpm", "genre", "lyrics")


def mask_one_modality(batch, modality):
    """Return a shallow batch copy with one model input zeroed for every row."""
    if not isinstance(batch, Mapping):
        raise ValueError("batch must be a mapping")
    if modality not in _MODALITIES:
        raise ValueError(f"modality must be one of {_MODALITIES}")
    if modality not in batch or not torch.is_tensor(batch[modality]):
        raise ValueError(f"batch must contain tensor modality {modality!r}")

    masked = dict(batch)
    masked[modality] = torch.zeros_like(batch[modality])
    return masked


def _normalized_embeddings(embeddings, name, row_count):
    vectors = np.asarray(embeddings)
    if np.iscomplexobj(vectors):
        raise ValueError(f"{name} embeddings must be real-valued")
    vectors = vectors.astype(np.float64, copy=False)
    if vectors.ndim != 2 or vectors.shape[0] != row_count or vectors.shape[1] == 0:
        raise ValueError(f"{name} embeddings must be a non-empty-width row-aligned matrix")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} embeddings must contain only finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


def measure_behavior_change(original_embeddings, masked_embeddings, song_ids, positive_edges, k=10):
    """Measure masked retrieval quality, per-query top-k overlap and embedding drift."""
    song_ids = list(song_ids)
    if not song_ids or len(set(song_ids)) != len(song_ids):
        raise ValueError("song_ids must be non-empty and unique")
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    original = _normalized_embeddings(original_embeddings, "original", len(song_ids))
    masked = _normalized_embeddings(masked_embeddings, "masked", len(song_ids))
    if masked.shape != original.shape:
        raise ValueError("original and masked embeddings must have matching shapes")

    # The evaluator validates the edge protocol and restricts queries to eligible endpoints.
    positive_edges = list(positive_edges)
    retrieval_metrics = evaluate_neurotune_v1_retrieval(masked, song_ids, positive_edges, k=k)
    relevance_query_ids = sorted({endpoint for edge in positive_edges for endpoint in edge})
    if not relevance_query_ids:
        raise ValueError("positive_edges must contain at least one eligible query")
    original_rankings = rank_top_k_many(original, song_ids, relevance_query_ids, k=k)
    masked_rankings = rank_top_k_many(masked, song_ids, relevance_query_ids, k=k)
    denominator = min(k, len(song_ids) - 1)
    overlaps = [
        len({row["song_id"] for row in original_rankings[query_id]}
            & {row["song_id"] for row in masked_rankings[query_id]}) / denominator
        for query_id in relevance_query_ids
    ]

    cosine = np.sum(original * masked, axis=1)
    both_zero = (np.linalg.norm(original, axis=1) == 0) & (np.linalg.norm(masked, axis=1) == 0)
    cosine[both_zero] = 1.0
    mean_cosine = float(np.mean(cosine))
    return {
        "retrieval_metrics": retrieval_metrics,
        "eligible_query_count": retrieval_metrics["query_count"],
        "unique_undirected_edge_count": retrieval_metrics["unique_undirected_edge_count"],
        "mean_top_k_overlap": float(np.mean(overlaps)),
        "mean_embedding_cosine": mean_cosine,
        "mean_embedding_cosine_shift": float(1.0 - mean_cosine),
    }

"""Full-gallery cosine retrieval helpers."""

from numbers import Integral

import numpy as np


def rank_top_k(embeddings, song_ids, query_id, k=10):
    """Rank all songs except ``query_id`` by cosine similarity.

    Embedding rows must align with ``song_ids``. Equal scores are ordered by
    song ID for deterministic output. Results contain ``song_id`` and ``score``.
    """
    vectors = np.asarray(embeddings, dtype=float)
    song_ids = list(song_ids)
    if vectors.ndim != 2 or vectors.shape[0] != len(song_ids):
        raise ValueError("embeddings rows must align with song_ids")
    if len(set(song_ids)) != len(song_ids):
        raise ValueError("song_ids must be unique")
    try:
        query_index = song_ids.index(query_id)
    except ValueError as error:
        raise ValueError(f"query_id not found: {query_id}") from error
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
        raise ValueError("k must be a non-negative integer")
    k = int(k)

    norms = np.linalg.norm(vectors, axis=1)
    normalized = np.divide(
        vectors,
        norms[:, np.newaxis],
        out=np.zeros_like(vectors),
        where=norms[:, np.newaxis] != 0,
    )
    scores = normalized @ normalized[query_index]
    candidates = [
        {"song_id": song_id, "score": float(scores[index])}
        for index, song_id in enumerate(song_ids)
        if index != query_index
    ]
    candidates.sort(key=lambda item: (-item["score"], item["song_id"]))
    return candidates[: min(k, len(candidates))]


def rank_top_k_many(embeddings, song_ids, query_ids, k=10):
    """Rank several queries against one gallery using a single score matrix."""
    vectors = np.asarray(embeddings, dtype=float)
    song_ids = list(song_ids)
    query_ids = list(query_ids)
    if vectors.ndim != 2 or vectors.shape[0] != len(song_ids):
        raise ValueError("embeddings rows must align with song_ids")
    if len(set(song_ids)) != len(song_ids):
        raise ValueError("song_ids must be unique")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids must be unique")
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must contain only finite values")

    song_index = {song_id: index for index, song_id in enumerate(song_ids)}
    try:
        query_indices = [song_index[query_id] for query_id in query_ids]
    except (KeyError, TypeError) as error:
        raise ValueError(f"query_id not found: {error.args[0]}") from error

    norms = np.linalg.norm(vectors, axis=1)
    normalized = np.divide(
        vectors,
        norms[:, np.newaxis],
        out=np.zeros_like(vectors),
        where=norms[:, np.newaxis] != 0,
    )
    scores = normalized[query_indices] @ normalized.T
    rankings = {}
    for query_row, (query_id, query_index) in enumerate(zip(query_ids, query_indices)):
        candidates = [
            {"song_id": song_id, "score": float(scores[query_row, index])}
            for index, song_id in enumerate(song_ids)
            if index != query_index
        ]
        candidates.sort(key=lambda item: (-item["score"], item["song_id"]))
        rankings[query_id] = candidates[: min(int(k), len(candidates))]
    return rankings

"""Fixed equal-modality cosine baseline and retrieval evaluation for NeuroTune v1."""

from collections.abc import Mapping

import numpy as np

from .evaluate import evaluate_rankings
from .retrieve import rank_top_k_many


_MODALITIES = ("audio", "bpm", "lyrics", "genre")


def build_neurotune_v1_cosine_baseline(features, song_ids, modalities=_MODALITIES):
    """Build fixed equal-weight cosine embeddings from transformed v1 features."""
    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping")
    song_ids = list(song_ids)
    if not song_ids or len(set(song_ids)) != len(song_ids):
        raise ValueError("song_ids must be non-empty and unique")
    if (not isinstance(modalities, (list, tuple)) or not modalities
            or any(not isinstance(name, str) or name not in _MODALITIES for name in modalities)
            or len(set(modalities)) != len(modalities)
            ):
        raise ValueError("modalities must be a non-empty unique selection of v1 modalities")
    if any(key not in features for key in modalities):
        raise ValueError("features must contain every selected modality")

    matrices = []
    for name in modalities:
        if name == "audio":
            audio = features[name]
            if not isinstance(audio, (list, tuple)) or len(audio) != len(song_ids):
                raise ValueError("audio sequences must align with song_ids")
            summaries = []
            for sequence in audio:
                sequence = np.asarray(sequence)
                if (sequence.dtype != np.float32 or sequence.ndim != 2
                        or sequence.shape[0] == 0 or sequence.shape[1] != 26):
                    raise ValueError("each audio sequence must be non-empty float32 with shape (T, 26)")
                if not np.isfinite(sequence).all():
                    raise ValueError("audio sequences must contain only finite values")
                summaries.append(sequence.mean(axis=0, dtype=np.float32))
            matrices.append(np.stack(summaries, axis=0))
            continue

        matrix = np.asarray(features[name])
        if (matrix.dtype != np.float32 or matrix.ndim != 2
                or matrix.shape[0] != len(song_ids)):
            raise ValueError(f"{name} must be a row-aligned float32 matrix")
        if name == "bpm" and matrix.shape[1] != 1:
            raise ValueError("bpm must have shape (N, 1)")
        if name == "lyrics" and matrix.shape[1] != 5000:
            raise ValueError("lyrics must have shape (N, 5000)")
        if name == "genre" and matrix.shape[1] == 0:
            raise ValueError("genre must have positive width")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} must contain only finite values")
        matrices.append(matrix)

    weighted = []
    for matrix in matrices:
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
        normalized = np.divide(
            matrix.astype(np.float64), norms,
            out=np.zeros(matrix.shape, dtype=np.float64), where=norms != 0,
        )
        weighted.append((normalized * 0.5).astype(np.float32))
    fused = np.concatenate(weighted, axis=1)
    final_norms = np.linalg.norm(fused.astype(np.float64), axis=1, keepdims=True)
    return np.divide(
        fused.astype(np.float64), final_norms,
        out=np.zeros(fused.shape, dtype=np.float64), where=final_norms != 0,
    ).astype(np.float32)


def build_d4_neurotune_v1_cosine_baseline(features, song_ids, modalities=_MODALITIES):
    """Build the equal-weight cosine baseline from D4 valid-window audio summaries."""
    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping")
    song_ids = list(song_ids)
    audio_windows = features.get("audio_windows")
    if not isinstance(audio_windows, Mapping):
        raise ValueError("audio_windows must be a track_id-to-windows mapping")
    window_valid = np.asarray(features.get("window_valid"))
    if window_valid.dtype != np.bool_ or window_valid.shape != (len(song_ids), 3):
        raise ValueError("window_valid must be a bool matrix with shape (N, 3)")

    summaries = []
    for row, track_id in enumerate(song_ids):
        if track_id not in audio_windows:
            raise ValueError("audio_windows must contain every requested track")
        windows = audio_windows[track_id]
        if not isinstance(windows, (list, tuple)) or len(windows) != 3:
            raise ValueError("each track must have exactly three audio windows")
        valid_summaries = []
        for index, window in enumerate(windows):
            window = np.asarray(window)
            if (window.dtype != np.float32 or window.ndim != 2
                    or window.shape[1] != 26):
                raise ValueError("each audio window must be float32 with shape (T, 26)")
            if not np.isfinite(window).all():
                raise ValueError("audio windows must contain only finite values")
            is_nonempty = window.shape[0] > 0
            if bool(window_valid[row, index]) != is_nonempty:
                raise ValueError("window_valid must agree with non-empty audio windows")
            if is_nonempty:
                valid_summaries.append(window.mean(axis=0, dtype=np.float32))
        if not valid_summaries:
            raise ValueError("each track must have at least one valid audio window")
        summaries.append(np.mean(np.stack(valid_summaries), axis=0, dtype=np.float32)[None, :])

    baseline_features = dict(features)
    baseline_features["audio"] = summaries
    return build_neurotune_v1_cosine_baseline(baseline_features, song_ids, modalities)


def evaluate_neurotune_v1_retrieval(embeddings, song_ids, positive_edges, k=10):
    """Evaluate undirected positives against each query's full non-self gallery."""
    song_ids = list(song_ids)
    vectors = np.asarray(embeddings)
    if (vectors.ndim != 2 or vectors.shape[0] != len(song_ids)
            or len(set(song_ids)) != len(song_ids)):
        raise ValueError("embeddings rows must align with unique song_ids")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must contain only finite values")
    song_set = set(song_ids)
    relevance = {}
    undirected_edges = set()
    try:
        edge_iterator = iter(positive_edges)
    except TypeError as error:
        raise ValueError("positive_edges must be iterable") from error
    for edge in edge_iterator:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("each positive edge must be a pair of song IDs")
        source, target = edge
        if source not in song_set or target not in song_set:
            raise ValueError("positive edge endpoint is not in song_ids")
        if source == target:
            raise ValueError("positive edges cannot be self-edges")
        canonical = frozenset((source, target))
        if canonical in undirected_edges:
            raise ValueError("positive edges must be unique and undirected")
        undirected_edges.add(canonical)
        relevance.setdefault(source, set()).add(target)
        relevance.setdefault(target, set()).add(source)

    rankings = rank_top_k_many(vectors, song_ids, list(relevance), k=k)
    metrics = evaluate_rankings(rankings, relevance, k=k)
    metrics["unique_undirected_edge_count"] = len(undirected_edges)
    return metrics

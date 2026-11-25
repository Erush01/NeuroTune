"""Source-separated cross-validation helpers for NeuroTune v1."""

from .evaluate_neurotune_v1 import (
    build_neurotune_v1_cosine_baseline,
    evaluate_neurotune_v1_retrieval,
)


_C_MODALITIES = ("audio", "bpm", "genre")


def _canonical_edge_keys(edges):
    result = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("each positive edge must be a pair of song IDs")
        try:
            canonical = frozenset(edge)
        except TypeError as error:
            raise ValueError("positive edge endpoints must be hashable") from error
        if len(canonical) != 2:
            raise ValueError("positive edges cannot be self-edges")
        result.add(canonical)
    return result


def build_neurotune_v1_c_cosine_baseline(features, song_ids):
    """Build the fixed C-matched cosine baseline using audio, BPM, and genre."""
    return build_neurotune_v1_cosine_baseline(
        features, song_ids, modalities=_C_MODALITIES
    )


def evaluate_neurotune_v1_source_separated_retrieval(
    embeddings, song_ids, held_out_train_source_edges, official_test_source_edges,
):
    """Return separate retrieval metric records for the two edge sources."""
    try:
        held_out_edges = list(held_out_train_source_edges)
        official_edges = list(official_test_source_edges)
    except TypeError as error:
        raise ValueError("edge sources must be iterable") from error
    song_ids = list(song_ids)

    held_out_canonical = _canonical_edge_keys(held_out_edges)
    official_canonical = _canonical_edge_keys(official_edges)
    if held_out_canonical & official_canonical:
        raise ValueError("edge sources must not overlap")

    # Delegate full ID, uniqueness, embedding, and metric validation to the
    # existing evaluator after cheaply rejecting cross-source overlap.
    held_out_metrics = evaluate_neurotune_v1_retrieval(
        embeddings, song_ids, held_out_edges, k=10
    )
    official_metrics = evaluate_neurotune_v1_retrieval(
        embeddings, song_ids, official_edges, k=10
    )

    return {
        "held_out_train_source": held_out_metrics,
        "official_test_source": official_metrics,
    }

"""Partition-local triplet sampling and cosine-margin loss utilities."""

import math
import numbers
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MixedTripletSample:
    """One schedule; pool count -1 means its semi-hard pool was not evaluated."""

    triplets: np.ndarray
    sampling_kinds: tuple[str, ...]
    eligible_uniform_counts: np.ndarray
    eligible_semi_hard_counts: np.ndarray
    semi_hard_evaluated: np.ndarray
    attempted_semi_hard_count: int
    fallback_count: int
    realized_uniform_fraction: float
    realized_semi_hard_fraction: float


def _validated_track_ids(track_ids):
    try:
        ids = list(track_ids)
    except TypeError as error:
        raise ValueError("track_ids must be iterable") from error
    if not ids or any(not isinstance(track_id, str) or not track_id for track_id in ids):
        raise ValueError("track_ids must contain nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("track_ids must be unique")
    return ids


def _prepare_triplet_edges(ids, positive_edges, known_positive_edges):
    id_to_index = {track_id: index for index, track_id in enumerate(ids)}

    def normalize_edges(edges, label):
        try:
            raw_edges = list(edges)
        except TypeError as error:
            raise ValueError(f"{label} must be iterable") from error
        normalized = []
        for edge in raw_edges:
            try:
                left, right = edge
            except (TypeError, ValueError) as error:
                raise ValueError("each positive edge must have two endpoints") from error
            if not isinstance(left, str) or not isinstance(right, str):
                raise ValueError("positive edge endpoints must be strings")
            if left == right:
                raise ValueError("positive edges cannot be self-edges")
            if left not in id_to_index or right not in id_to_index:
                raise ValueError("positive edge endpoint is not in track_ids")
            normalized.append(tuple(sorted((left, right))))
        if len(set(normalized)) != len(normalized):
            raise ValueError("positive edges must be unique undirected edges")
        return sorted(normalized)

    canonical_edges = normalize_edges(positive_edges, "positive_edges")
    if known_positive_edges is None:
        known_edges = canonical_edges
    else:
        known_edges = normalize_edges(known_positive_edges, "known_positive_edges")
        if not set(canonical_edges).issubset(known_edges):
            raise ValueError("positive edges must be a subset of known positive edges")
    adjacency = {track_id: set() for track_id in ids}
    for left, right in known_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    directed_edges = [direction for left, right in canonical_edges
                      for direction in ((left, right), (right, left))]
    return id_to_index, adjacency, directed_edges


class MixedSemiHardTripletSampler:
    """Reusable exact semi-hard index for one fixed train embedding snapshot.

    ``sample(seed)`` makes a fresh deterministic epoch schedule. This sampler
    stores one bounded float32 pairwise similarity matrix for the fixed embedding
    snapshot and computes semi-hard masks and pools transiently per attempted row.
    The embedding array must not be mutated during this sampler's lifetime.
    """

    def __init__(self, track_ids, positive_edges, embeddings, margin=0.2,
                 semi_hard_fraction=0.25, known_positive_edges=None):
        if (isinstance(margin, bool) or not isinstance(margin, numbers.Real)
                or not math.isfinite(margin) or margin < 0):
            raise ValueError("margin must be a finite nonnegative real number")
        if (isinstance(semi_hard_fraction, bool)
                or not isinstance(semi_hard_fraction, numbers.Real)
                or not math.isfinite(semi_hard_fraction)
                or not 0 <= semi_hard_fraction <= 1):
            raise ValueError("semi_hard_fraction must be a finite real number in [0, 1]")
        self.track_ids = _validated_track_ids(track_ids)
        self.id_to_index, adjacency, directed_edges = _prepare_triplet_edges(
            self.track_ids, positive_edges, known_positive_edges,
        )
        if not directed_edges:
            raise ValueError("positive_edges must contain at least one positive edge")
        try:
            vectors = np.asarray(embeddings)
        except (TypeError, ValueError) as error:
            raise ValueError("embeddings must be a finite real array with one row per track") from error
        if (vectors.ndim != 2 or vectors.shape[0] != len(self.track_ids) or vectors.shape[1] == 0
                or not np.issubdtype(vectors.dtype, np.number)
                or np.issubdtype(vectors.dtype, np.complexfloating)
                or not np.isfinite(vectors).all()):
            raise ValueError("embeddings must be a finite real array with one row per track")
        vectors = vectors.astype(np.float64, copy=True)
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("embeddings must have nonzero row norms")
        normalized_vectors = (vectors / norms[:, None]).astype(np.float32)
        self._similarity_matrix = normalized_vectors @ normalized_vectors.T
        self.margin = margin
        self.semi_hard_fraction = semi_hard_fraction
        self.directed_edges = tuple(directed_edges)
        self._forbidden_by_anchor = {}
        for anchor_id, _ in self.directed_edges:
            anchor_index = self.id_to_index[anchor_id]
            if anchor_index not in self._forbidden_by_anchor:
                forbidden = {self.id_to_index[track_id] for track_id in adjacency[anchor_id]}
                forbidden.add(anchor_index)
                self._forbidden_by_anchor[anchor_index] = frozenset(forbidden)
            if len(self._forbidden_by_anchor[anchor_index]) == len(self.track_ids):
                raise ValueError(f"no eligible negative for anchor {anchor_id!r}")

    def _semi_hard_candidates(self, row_index):
        anchor_id, positive_id = self.directed_edges[row_index]
        anchor_index = self.id_to_index[anchor_id]
        positive_index = self.id_to_index[positive_id]
        similarities = self._similarity_matrix[anchor_index]
        positive_distance = 1.0 - similarities[positive_index]
        negative_distances = 1.0 - similarities
        eligible_mask = (
            (positive_distance < negative_distances)
            & (negative_distances < positive_distance + self.margin)
        )
        forbidden = self._forbidden_by_anchor[anchor_index]
        eligible_mask[np.fromiter(forbidden, dtype=np.intp, count=len(forbidden))] = False
        return np.flatnonzero(eligible_mask).astype(np.int32, copy=False)

    def _sample_uniform_negative(self, rng, anchor_index):
        forbidden = self._forbidden_by_anchor[anchor_index]
        while True:
            candidate = rng.randrange(len(self.track_ids))
            if candidate not in forbidden:
                return candidate

    def sample(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        rng = random.Random(seed)
        row_count = len(self.directed_edges)
        attempted_count = int(math.floor(row_count * self.semi_hard_fraction + 0.5))
        attempted_rows = set(rng.sample(range(row_count), attempted_count))
        triplets = np.empty((row_count, 3), dtype=np.int32)
        kinds = []
        uniform_counts = np.empty(row_count, dtype=np.int32)
        semi_hard_counts = np.full(row_count, -1, dtype=np.int32)
        evaluated = np.zeros(row_count, dtype=np.bool_)
        for row_index, (anchor_id, positive_id) in enumerate(self.directed_edges):
            anchor_index = self.id_to_index[anchor_id]
            positive_index = self.id_to_index[positive_id]
            forbidden = self._forbidden_by_anchor[anchor_index]
            uniform_counts[row_index] = len(self.track_ids) - len(forbidden)
            triplets[row_index, :2] = (anchor_index, positive_index)
            if row_index in attempted_rows:
                semi_hard = self._semi_hard_candidates(row_index)
                evaluated[row_index] = True
                semi_hard_counts[row_index] = len(semi_hard)
                if len(semi_hard):
                    negative_index = int(rng.choice(semi_hard))
                    kinds.append("semi_hard")
                else:
                    negative_index = self._sample_uniform_negative(rng, anchor_index)
                    kinds.append("fallback_uniform")
            else:
                negative_index = self._sample_uniform_negative(rng, anchor_index)
                kinds.append("uniform")
            triplets[row_index, 2] = negative_index
        semi_hard_realized = kinds.count("semi_hard") / row_count
        return MixedTripletSample(
            triplets=triplets,
            sampling_kinds=tuple(kinds),
            eligible_uniform_counts=uniform_counts,
            eligible_semi_hard_counts=semi_hard_counts,
            semi_hard_evaluated=evaluated,
            attempted_semi_hard_count=attempted_count,
            fallback_count=kinds.count("fallback_uniform"),
            realized_uniform_fraction=1.0 - semi_hard_realized,
            realized_semi_hard_fraction=semi_hard_realized,
        )


def sample_mixed_semi_hard_triplets(
    track_ids, positive_edges, embeddings, seed, margin=0.2,
    semi_hard_fraction=0.25, known_positive_edges=None,
):
    """Sample one directed-positive schedule; see reusable sampler for cache API."""
    sampler = MixedSemiHardTripletSampler(
        track_ids, positive_edges, embeddings, margin=margin,
        semi_hard_fraction=semi_hard_fraction,
        known_positive_edges=known_positive_edges,
    )
    return sampler.sample(seed)


def sample_triplet_schedule(track_ids, positive_edges, epochs, seed, known_positive_edges=None):
    """Return an ``(epochs, 2 * edges, 3)`` index schedule for one partition."""
    try:
        ids = list(track_ids)
    except TypeError as error:
        raise ValueError("track_ids must be iterable") from error
    if not ids or any(not isinstance(track_id, str) or not track_id for track_id in ids):
        raise ValueError("track_ids must contain nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("track_ids must be unique")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    id_to_index, adjacency, directed_edges = _prepare_triplet_edges(
        ids, positive_edges, known_positive_edges,
    )
    for anchor, positive in directed_edges:
        forbidden = adjacency[anchor] | {anchor, positive}
        if len(forbidden) == len(ids):
            raise ValueError(f"no eligible negative for anchor {anchor!r}")

    schedule = np.empty((epochs, len(directed_edges), 3), dtype=np.int32)
    rng = random.Random(seed)
    for epoch in range(epochs):
        for row, (anchor, positive) in enumerate(directed_edges):
            forbidden = adjacency[anchor] | {anchor, positive}
            while True:
                negative = ids[rng.randrange(len(ids))]
                if negative not in forbidden:
                    break
            schedule[epoch, row] = (
                id_to_index[anchor], id_to_index[positive], id_to_index[negative]
            )
    return schedule


def triplet_margin_cosine_loss(anchor, positive, negative, margin=0.2):
    """Mean cosine-distance margin loss for L2-normalized embedding triplets.

    Inputs are expected to be L2-normalized because cosine distance is computed
    as ``1 - dot``; this function does not normalize embeddings internally.
    """
    if isinstance(margin, bool) or not isinstance(margin, numbers.Real) or not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be a finite nonnegative real number")
    tensors = (anchor, positive, negative)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("anchor, positive, and negative must be tensors")
    if any(tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0 for tensor in tensors):
        raise ValueError("embeddings must have nonempty (B, D) shapes")
    if not (anchor.shape == positive.shape == negative.shape):
        raise ValueError("anchor, positive, and negative shapes must match")
    if any(not tensor.is_floating_point() or tensor.is_complex() for tensor in tensors):
        raise ValueError("embeddings must have real floating-point dtypes")
    if any(tensor.device != anchor.device or tensor.dtype != anchor.dtype for tensor in tensors[1:]):
        raise ValueError("embeddings must share dtype and device")
    if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError("embeddings must be finite")

    positive_distance = 1.0 - torch.sum(anchor * positive, dim=1)
    negative_distance = 1.0 - torch.sum(anchor * negative, dim=1)
    return F.relu(positive_distance - negative_distance + margin).mean()

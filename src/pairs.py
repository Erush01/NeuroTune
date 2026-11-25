import math
import random


def _canonical_pair(left_id, right_id):
    return tuple(sorted((left_id, right_id)))


def build_pair_pools(edges, partition_ids, genres):
    """Build compact positive and candidate-count pools within one partition."""
    ordered_ids = sorted(set(partition_ids))
    partition = set(ordered_ids)
    positives = set()
    for edge in edges:
        left_id = edge.get("source")
        right_id = edge.get("target")
        if left_id == right_id or left_id not in partition or right_id not in partition:
            continue
        positives.add(_canonical_pair(left_id, right_id))

    genre_groups = {}
    for track_id in ordered_ids:
        genre = genres.get(track_id)
        if genre is not None and genre != "":
            genre_groups.setdefault(genre, []).append(track_id)
    genre_groups = {genre: genre_groups[genre] for genre in sorted(genre_groups)}
    id_genres = {
        track_id: genre
        for genre, track_ids in genre_groups.items()
        for track_id in track_ids
    }
    positive_same_genre_counts = {genre: 0 for genre in genre_groups}
    for left_id, right_id in positives:
        genre = id_genres.get(left_id)
        if genre is not None and genre == id_genres.get(right_id):
            positive_same_genre_counts[genre] += 1

    random_pair_count = math.comb(len(ordered_ids), 2) - len(positives)
    same_genre_pair_count = sum(
        math.comb(len(track_ids), 2) - positive_same_genre_counts[genre]
        for genre, track_ids in genre_groups.items()
    )
    return {
        "positives": sorted(positives),
        "partition_ids": ordered_ids,
        "genre_groups": genre_groups,
        "random_pair_count": random_pair_count,
        "same_genre_pair_count": same_genre_pair_count,
    }


def _weighted_choice(rng, weighted_items):
    total_weight = sum(weight for _, weight in weighted_items)
    draw = rng.randrange(total_weight)
    for item, weight in weighted_items:
        if draw < weight:
            return item
        draw -= weight
    raise AssertionError("weighted choice requires a positive total weight")


def sample_pairs(pools, seed, negatives_per_positive=1, same_genre_fraction=0.5):
    """Return positive rows and unique seeded negatives sampled by rejection."""
    positives = list(pools["positives"])
    if not isinstance(negatives_per_positive, int) or negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must be a non-negative integer")
    if not 0 <= same_genre_fraction <= 1:
        raise ValueError("same_genre_fraction must be between 0 and 1")

    total_negatives = len(positives) * negatives_per_positive
    same_genre_count = total_negatives * same_genre_fraction
    if same_genre_fraction == 0.5:
        same_genre_count = total_negatives // 2
    else:
        if not math.isclose(same_genre_count, round(same_genre_count), abs_tol=1e-12):
            raise ValueError("same_genre_fraction does not yield an integer sample count")
        same_genre_count = int(round(same_genre_count))
    random_count = total_negatives - same_genre_count

    if pools["same_genre_pair_count"] < same_genre_count:
        raise ValueError(
            f"same-genre candidate pool has {pools['same_genre_pair_count']} pairs; "
            f"{same_genre_count} required"
        )
    if pools["random_pair_count"] < total_negatives:
        raise ValueError(
            f"random candidate pool has {pools['random_pair_count']} pairs; "
            f"{total_negatives} unique negatives required"
        )

    rng = random.Random(seed)
    positive_set = set(positives)
    selected = set()
    sampled_same_genre = []
    genre_groups = pools["genre_groups"]
    id_genres = {
        track_id: genre
        for genre, track_ids in genre_groups.items()
        for track_id in track_ids
    }
    remaining_by_genre = {
        genre: math.comb(len(track_ids), 2)
        for genre, track_ids in genre_groups.items()
    }
    for left_id, right_id in positive_set:
        genre = id_genres.get(left_id)
        if genre is not None and genre == id_genres.get(right_id):
            remaining_by_genre[genre] -= 1

    for _ in range(same_genre_count):
        weighted_groups = [
            (genre, count) for genre, count in remaining_by_genre.items() if count
        ]
        genre = _weighted_choice(rng, weighted_groups)
        track_ids = genre_groups[genre]
        while True:
            left_id, right_id = rng.sample(track_ids, 2)
            pair = _canonical_pair(left_id, right_id)
            if pair not in positive_set and pair not in selected:
                selected.add(pair)
                sampled_same_genre.append(pair)
                remaining_by_genre[genre] -= 1
                break

    sampled_random = []
    partition_ids = pools["partition_ids"]
    while len(sampled_random) < random_count:
        left_id, right_id = rng.sample(partition_ids, 2)
        pair = _canonical_pair(left_id, right_id)
        if pair not in positive_set and pair not in selected:
            selected.add(pair)
            sampled_random.append(pair)

    return (
        [(left_id, right_id, 1.0) for left_id, right_id in positives]
        + [(left_id, right_id, 0.0) for left_id, right_id in sampled_random]
        + [(left_id, right_id, 0.0) for left_id, right_id in sampled_same_genre]
    )

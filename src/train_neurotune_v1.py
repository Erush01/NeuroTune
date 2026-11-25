"""Train one NeuroTune v1 variant and select its validation checkpoint."""

import copy
import math
import numbers
from pathlib import Path
import time

import numpy as np
import torch

from src.data.neurotune_v1_dataset import neurotune_v1_collate
from src.evaluate import build_test_relevance, evaluate_rankings
from src.model_v1 import NeuroTuneV1Encoder
from src.retrieve import rank_top_k
from src.triplets import triplet_margin_cosine_loss


def _validate_schedule(schedule, name, expected_epochs, dataset_size):
    if not isinstance(schedule, np.ndarray) or schedule.dtype != np.int32:
        raise ValueError(f"{name} must be an int32 NumPy array")
    if schedule.ndim != 3 or schedule.shape[0] != expected_epochs or schedule.shape[2] != 3:
        raise ValueError(f"{name} must have shape ({expected_epochs}, N, 3)")
    if schedule.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one triplet row")
    if np.any(schedule < 0) or np.any(schedule >= dataset_size):
        raise ValueError(f"{name} contains an index outside its dataset")


def _collate_indices(dataset, indices):
    return neurotune_v1_collate([dataset[int(index)] for index in indices])


def _make_keep_masks(batch_size, probabilities, generator):
    return {
        modality: torch.rand(batch_size, generator=generator) >= probability
        for modality, probability in probabilities.items()
        if probability > 0
    }


def _batch_loss(model, dataset, triplets, batch_size, margin):
    total = 0.0
    count = 0
    for start in range(0, len(triplets), batch_size):
        rows = triplets[start:start + batch_size]
        anchor = _collate_indices(dataset, rows[:, 0])
        positive = _collate_indices(dataset, rows[:, 1])
        negative = _collate_indices(dataset, rows[:, 2])
        embeddings = model.forward_triplets(anchor, positive, negative)
        loss = triplet_margin_cosine_loss(*embeddings, margin=margin)
        row_count = len(rows)
        total += float(loss.item()) * row_count
        count += row_count
    return total / count


def _validation_rankings(model, dataset, batch_size):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            inputs = neurotune_v1_collate(rows)
            active = model.active_modalities
            audio_keys = ("audio_windows", "mask", "window_mask") if "audio_windows" in inputs else ("audio", "mask")
            args = {key: inputs[key] for key in (*audio_keys, *active)}
            embeddings.append(model(**args).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def train_neurotune_v1(
    train_dataset,
    validation_dataset,
    train_schedule,
    validation_schedule,
    validation_positive_edges,
    variant,
    seed,
    checkpoint_path,
    transform_manifest,
    provenance,
    *,
    epochs=50,
    batch_size=64,
    train_schedule_provider=None,
    record_epoch_timing=False,
    bpm_dropout_probability=0.0,
    genre_dropout_probability=0.0,
    lyrics_dropout_probability=0.0,
    dropout_seed=None,
):
    """Train using partition-local schedules and select on validation NDCG@10 only."""
    if variant not in {"A", "B", "C", "D"}:
        raise ValueError("variant must be one of A, B, C, or D")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(epochs, bool) or not isinstance(epochs, numbers.Integral) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    dropout_probabilities = {
        "bpm": bpm_dropout_probability,
        "genre": genre_dropout_probability,
        "lyrics": lyrics_dropout_probability,
    }
    for modality, probability in dropout_probabilities.items():
        if (isinstance(probability, bool) or not isinstance(probability, numbers.Real)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError(f"{modality}_dropout_probability must be a finite number in [0, 1]")
        dropout_probabilities[modality] = float(probability)
    dropout_enabled = any(probability > 0 for probability in dropout_probabilities.values())
    if dropout_seed is not None and (
        isinstance(dropout_seed, bool) or not isinstance(dropout_seed, numbers.Integral)
        or dropout_seed < 0
    ):
        raise ValueError("dropout_seed must be a nonnegative integer")
    if dropout_enabled and dropout_seed is None:
        raise ValueError("dropout_seed is required when modality dropout is enabled")
    if (dropout_seed is not None
            and int(dropout_seed) > (2**63 - 1) - (int(epochs) - 1)):
        raise ValueError("dropout_seed plus epochs - 1 must not exceed 2^63 - 1")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_path}")
    if set(train_dataset.track_ids) & set(validation_dataset.track_ids):
        raise ValueError("train and validation dataset IDs must be disjoint")
    active_modalities = NeuroTuneV1Encoder._MODALITIES[variant]
    inactive_dropout = [
        name for name, probability in dropout_probabilities.items()
        if probability > 0 and name not in active_modalities
    ]
    if inactive_dropout:
        raise ValueError(
            f"dropout probabilities cannot be enabled for inactive modalities: {inactive_dropout}"
        )
    for name, dataset in (("train", train_dataset), ("validation", validation_dataset)):
        missing = set(active_modalities) - set(dataset.modalities)
        if missing:
            raise ValueError(f"{name} dataset is missing active modalities: {sorted(missing)}")
    num_genres = 1
    if variant in {"C", "D"}:
        train_genre_width = train_dataset.features["genre"].shape[1]
        validation_genre_width = validation_dataset.features["genre"].shape[1]
        if train_genre_width != validation_genre_width:
            raise ValueError("train and validation genre feature widths must match")
        num_genres = train_genre_width
    if train_schedule_provider is None:
        _validate_schedule(train_schedule, "train_schedule", int(epochs), len(train_dataset))
    elif train_schedule is not None:
        raise ValueError("train_schedule must be None when train_schedule_provider is provided")
    _validate_schedule(validation_schedule, "validation_schedule", 1, len(validation_dataset))

    validation_positive_edges = list(validation_positive_edges)
    validation_ids = set(validation_dataset.track_ids)
    for edge in validation_positive_edges:
        if not isinstance(edge, dict) or edge.get("source") not in validation_ids or edge.get("target") not in validation_ids:
            raise ValueError("validation edge endpoints must belong to the validation dataset")
    relevance = build_test_relevance(validation_positive_edges, validation_dataset.track_ids)

    torch.manual_seed(int(seed))
    model = NeuroTuneV1Encoder(variant, num_genres).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_epoch = None
    best_metrics = None
    best_loss = None
    best_state = None

    for epoch in range(int(epochs)):
        epoch_started = time.perf_counter() if record_epoch_timing else None
        sample = None
        if train_schedule_provider is not None:
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    sample = train_schedule_provider(model, train_dataset, epoch + 1)
            finally:
                model.train(was_training)
            epoch_schedule = sample.triplets
            if (not isinstance(epoch_schedule, np.ndarray) or epoch_schedule.dtype != np.int32
                    or epoch_schedule.ndim != 2 or epoch_schedule.shape[1] != 3
                    or epoch_schedule.shape[0] == 0):
                raise ValueError("provider triplets must be a nonempty int32 array with shape (N, 3)")
            if np.any(epoch_schedule < 0) or np.any(epoch_schedule >= len(train_dataset)):
                raise ValueError("provider triplets contain an index outside the train dataset")
        else:
            epoch_schedule = train_schedule[epoch]
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + epoch)
        order = torch.randperm(len(epoch_schedule), generator=generator).numpy()
        epoch_dropout_generator = None
        if dropout_enabled:
            epoch_dropout_generator = torch.Generator(device="cpu").manual_seed(
                int(dropout_seed) + epoch
            )
        epoch_rows = epoch_schedule[order]
        train_total = 0.0
        train_count = 0
        dropout_attempted = {name: 0 for name in active_modalities}
        dropout_dropped = {name: 0 for name in active_modalities}
        if sample is not None:
            positive_distances = []
            negative_distances = []
            active_count = 0
        for start in range(0, len(epoch_rows), int(batch_size)):
            rows = epoch_rows[start:start + int(batch_size)]
            anchor = _collate_indices(train_dataset, rows[:, 0])
            positive = _collate_indices(train_dataset, rows[:, 1])
            negative = _collate_indices(train_dataset, rows[:, 2])
            optimizer.zero_grad()
            forward_kwargs = {}
            if dropout_enabled:
                dropout_generator = epoch_dropout_generator
                for role, keep_masks in zip(
                    ("anchor_keep_masks", "positive_keep_masks", "negative_keep_masks"),
                    (_make_keep_masks(
                        len(rows),
                        {name: probability for name, probability in dropout_probabilities.items()
                         if name in active_modalities},
                        dropout_generator,
                    )
                     for _ in range(3)),
                ):
                    forward_kwargs[role] = keep_masks
                    for modality in active_modalities:
                        dropout_attempted[modality] += len(rows)
                        if modality in keep_masks:
                            keep_mask = keep_masks[modality]
                            dropout_dropped[modality] += int((~keep_mask).sum().item())
            embeddings = model.forward_triplets(anchor, positive, negative, **forward_kwargs)
            loss = triplet_margin_cosine_loss(*embeddings, margin=0.2)
            if sample is not None:
                anchor_embeddings, positive_embeddings, negative_embeddings = embeddings
                positive_distance = 1.0 - torch.sum(anchor_embeddings * positive_embeddings, dim=1)
                negative_distance = 1.0 - torch.sum(anchor_embeddings * negative_embeddings, dim=1)
                positive_distances.extend(positive_distance.detach().cpu().tolist())
                negative_distances.extend(negative_distance.detach().cpu().tolist())
                active_count += int((positive_distance - negative_distance + 0.2 > 0).sum().item())
            loss.backward()
            optimizer.step()
            row_count = len(rows)
            train_total += float(loss.item()) * row_count
            train_count += row_count
        train_loss = train_total / train_count

        model.eval()
        with torch.no_grad():
            validation_loss = _batch_loss(
                model, validation_dataset, validation_schedule[0], int(batch_size), 0.2
            )
            gallery_embeddings = _validation_rankings(model, validation_dataset, int(batch_size))
            rankings = {
                query_id: rank_top_k(gallery_embeddings, validation_dataset.track_ids, query_id, k=10)
                for query_id in relevance
            }
            metrics = evaluate_rankings(rankings, relevance, k=10)
        record = {
            "epoch": epoch + 1,
            "train_triplet_loss": train_loss,
            "validation_triplet_loss": validation_loss,
            "Recall@10": metrics["Recall@10"],
            "NDCG@10": metrics["NDCG@10"],
            "MRR@10": metrics["MRR@10"],
            "query_count": metrics["query_count"],
            "eligible_positive_edges": metrics["eligible_positive_edges"],
        }
        if dropout_enabled:
            for modality in active_modalities:
                record[f"{modality}_dropout_attempted_count"] = dropout_attempted[modality]
                record[f"{modality}_dropout_dropped_count"] = dropout_dropped[modality]
                record[f"{modality}_dropout_realized_rate"] = (
                    dropout_dropped[modality] / dropout_attempted[modality]
                    if dropout_attempted[modality] else 0.0
                )
        if record_epoch_timing:
            epoch_seconds = time.perf_counter() - epoch_started
            if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
                raise RuntimeError("epoch timing must be finite and positive")
            record["epoch_seconds"] = epoch_seconds
        if sample is not None:
            semi_hard_counts = np.asarray(sample.eligible_semi_hard_counts)
            evaluated_counts = semi_hard_counts[semi_hard_counts >= 0]
            record.update({
                "active_triplet_fraction": active_count / train_count,
                "mean_positive_cosine_distance": float(np.mean(positive_distances)),
                "mean_negative_cosine_distance": float(np.mean(negative_distances)),
                "margin_violation_fraction": active_count / train_count,
                "sampler_attempted_semi_hard_count": int(sample.attempted_semi_hard_count),
                "sampler_requested_semi_hard_fraction": sample.attempted_semi_hard_count / train_count,
                "sampler_requested_uniform_fraction": 1.0 - sample.attempted_semi_hard_count / train_count,
                "sampler_realized_uniform_fraction": float(sample.realized_uniform_fraction),
                "sampler_realized_semi_hard_fraction": float(sample.realized_semi_hard_fraction),
                "sampler_fallback_count": int(sample.fallback_count),
                "sampler_mean_eligible_uniform_pool": float(np.mean(sample.eligible_uniform_counts)),
                "sampler_mean_eligible_semi_hard_pool": (
                    float(np.mean(evaluated_counts)) if evaluated_counts.size else float("nan")
                ),
            })
        history.append(record)
        ndcg = metrics["NDCG@10"]
        if (best_epoch is None or ndcg > best_metrics["NDCG@10"]
                or (ndcg == best_metrics["NDCG@10"] and validation_loss < best_loss)):
            best_epoch = epoch + 1
            best_loss = validation_loss
            best_metrics = dict(metrics)
            best_state = copy.deepcopy(model.state_dict())

    checkpoint = {
        "model_state_dict": best_state,
        "variant": variant,
        "seed": int(seed),
        "epoch": best_epoch,
        "config": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "margin": 0.2,
            "device": "cpu",
        },
        "transform_manifest": copy.deepcopy(transform_manifest),
        "provenance": copy.deepcopy(provenance),
        "selection_metrics": dict(best_metrics),
    }
    torch.save(checkpoint, checkpoint_path)
    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "checkpoint_path": str(checkpoint_path),
    }

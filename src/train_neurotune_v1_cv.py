"""Train and validation-select NeuroTune v1 C for edge-split cross-validation."""

import copy
import hashlib
import json
import numbers
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from src.data.neurotune_v1_dataset import neurotune_v1_collate
from src.evaluate_neurotune_v1 import evaluate_neurotune_v1_retrieval
from src.model_v1 import NeuroTuneV1Encoder
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


def _canonical_edges(edges, ids, name):
    id_set = set(ids)
    try:
        values = list(edges)
    except TypeError as error:
        raise ValueError(f"{name} must be iterable") from error
    result = set()
    for edge in values:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError(f"each {name} item must be an endpoint pair")
        left, right = edge
        if left not in id_set or right not in id_set or left == right:
            raise ValueError(f"{name} endpoints must be distinct dataset IDs")
        canonical = tuple(sorted((left, right)))
        if canonical in result:
            raise ValueError(f"{name} must contain unique undirected edges")
        result.add(canonical)
    return result


def _verify_schedule(schedule, expected_edges, known_edges, ids, name):
    id_to_index = {track_id: index for index, track_id in enumerate(ids)}
    expected_directed = {
        (id_to_index[left], id_to_index[right])
        for edge in expected_edges
        for left, right in (edge, edge[::-1])
    }
    known_neighbors = {track_id: set() for track_id in ids}
    for left, right in known_edges:
        known_neighbors[left].add(right)
        known_neighbors[right].add(left)
    for epoch_rows in schedule:
        observed = {(int(row[0]), int(row[1])) for row in epoch_rows}
        if len(observed) != len(epoch_rows) or observed != expected_directed:
            raise ValueError(f"{name} positive pairs do not exactly match its edge list")
        for anchor, _, negative in epoch_rows:
            anchor_id, negative_id = ids[int(anchor)], ids[int(negative)]
            if negative_id == anchor_id or negative_id in known_neighbors[anchor_id]:
                raise ValueError(f"{name} contains a known-positive or self negative")


def _collate_indices(dataset, indices):
    return neurotune_v1_collate([dataset[int(index)] for index in indices])


def _validation_loss(model, dataset, schedule, batch_size):
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(schedule), batch_size):
            rows = schedule[start:start + batch_size]
            inputs = [_collate_indices(dataset, rows[:, column]) for column in range(3)]
            embeddings = model.forward_triplets(*inputs)
            loss = triplet_margin_cosine_loss(*embeddings, margin=0.2)
            total += float(loss.item()) * len(rows)
            count += len(rows)
    return total / count


def _embed_gallery(model, dataset, batch_size):
    embedded = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            inputs = neurotune_v1_collate(rows)
            embedded.append(model(audio=inputs["audio"], mask=inputs["mask"],
                                  bpm=inputs["bpm"], genre=inputs["genre"]).cpu().numpy())
    return np.concatenate(embedded, axis=0)


def _digest(value):
    if isinstance(value, np.ndarray):
        payload = value.tobytes()
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_provenance(provenance):
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    digest_pattern = re.compile(r"[0-9a-fA-F]{64}\Z")
    for name in ("split_sha256", "cache_sha256"):
        value = provenance.get(name)
        if not isinstance(value, str) or digest_pattern.fullmatch(value) is None:
            raise ValueError(f"provenance.{name} must be a 64-character hexadecimal SHA-256")
    for name in ("source_sha256", "code_sha256"):
        value = provenance.get(name)
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"provenance.{name} must be a nonempty mapping")
        for key, digest in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"provenance.{name} keys must be nonempty strings")
            if not isinstance(digest, str) or digest_pattern.fullmatch(digest) is None:
                raise ValueError(f"provenance.{name}[{key!r}] must be a 64-character hexadecimal SHA-256")


def train_neurotune_v1_cv(
    dataset,
    train_positive_edges,
    validation_positive_edges,
    known_positive_edges,
    train_schedule,
    validation_schedule,
    seed,
    checkpoint_path,
    transform_manifest,
    provenance,
    settings,
):
    """Train C on edge-split labels, using only the shared non-test pool dataset."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _validate_provenance(provenance)
    if not isinstance(settings, dict):
        raise ValueError("settings must be a mapping")
    epochs = settings.get("epochs", 50)
    batch_size = settings.get("batch_size", 64)
    if isinstance(epochs, bool) or not isinstance(epochs, numbers.Integral) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_path}")

    ids = list(dataset.track_ids)
    train_edges = _canonical_edges(train_positive_edges, ids, "train_positive_edges")
    validation_edges = _canonical_edges(validation_positive_edges, ids, "validation_positive_edges")
    known_edges = _canonical_edges(known_positive_edges, ids, "known_positive_edges")
    if train_edges & validation_edges:
        raise ValueError("train and validation positive edges must be disjoint")
    if train_edges | validation_edges != known_edges:
        raise ValueError("train and validation edges must partition known positive edges")
    _validate_schedule(train_schedule, "train_schedule", int(epochs), len(dataset))
    _validate_schedule(validation_schedule, "validation_schedule", 1, len(dataset))
    _verify_schedule(train_schedule, train_edges, known_edges, ids, "train_schedule")
    _verify_schedule(validation_schedule, validation_edges, known_edges, ids, "validation_schedule")

    torch.manual_seed(int(seed))
    model = NeuroTuneV1Encoder("C", dataset.features["genre"].shape[1]).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_epoch = None
    best_metrics = None
    best_loss = None
    best_state = None

    for epoch in range(int(epochs)):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + epoch)
        order = torch.randperm(len(train_schedule[epoch]), generator=generator).numpy()
        rows = train_schedule[epoch, order]
        total, count = 0.0, 0
        for start in range(0, len(rows), int(batch_size)):
            batch = rows[start:start + int(batch_size)]
            inputs = [_collate_indices(dataset, batch[:, column]) for column in range(3)]
            optimizer.zero_grad()
            embeddings = model.forward_triplets(*inputs)
            loss = triplet_margin_cosine_loss(*embeddings, margin=0.2)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
            count += len(batch)
        train_loss = total / count

        model.eval()
        validation_loss = _validation_loss(model, dataset, validation_schedule[0], int(batch_size))
        gallery = _embed_gallery(model, dataset, int(batch_size))
        metrics = evaluate_neurotune_v1_retrieval(gallery, ids, sorted(validation_edges), k=10)
        record = {"epoch": epoch + 1, "train_triplet_loss": train_loss,
                  "validation_triplet_loss": validation_loss, **metrics}
        history.append(record)
        ndcg = metrics["NDCG@10"]
        if (best_epoch is None or ndcg > best_metrics["NDCG@10"]
                or (ndcg == best_metrics["NDCG@10"] and validation_loss < best_loss)):
            best_epoch, best_loss = epoch + 1, validation_loss
            best_metrics = dict(metrics)
            best_state = copy.deepcopy(model.state_dict())

    provenance_copy = copy.deepcopy(provenance)
    checkpoint = {
        "model_state_dict": best_state,
        "variant": "C",
        "fold": settings.get("fold"),
        "seed": int(seed),
        "epoch": best_epoch,
        "config": {"optimizer": "Adam", "learning_rate": 0.001,
                   "batch_size": int(batch_size), "epochs": int(epochs),
                   "margin": 0.2, "device": "cpu", "variant": "C"},
        "transform_manifest": copy.deepcopy(transform_manifest),
        "provenance": provenance_copy,
        "fit_id_hash": _digest(sorted(ids)),
        "artifact_hashes": {
            "split_sha256": provenance_copy["split_sha256"],
            "source_sha256": copy.deepcopy(provenance_copy["source_sha256"]),
            "code_sha256": copy.deepcopy(provenance_copy["code_sha256"]),
            "cache_sha256": provenance_copy["cache_sha256"],
            "provenance": _digest(provenance_copy),
            "train_schedule": _digest(train_schedule),
            "validation_schedule": _digest(validation_schedule),
            "known_edges": _digest(sorted(known_edges)),
        },
        "selection_metrics": dict(best_metrics),
    }
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=checkpoint_path.parent
    )
    os.close(descriptor)
    staged_path = Path(staged_name)
    try:
        torch.save(checkpoint, staged_path)
        os.link(staged_path, checkpoint_path)
    finally:
        staged_path.unlink(missing_ok=True)
    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_epoch": best_epoch,
            "best_metrics": best_metrics, "checkpoint_path": str(checkpoint_path)}

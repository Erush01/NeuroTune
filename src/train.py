from numbers import Integral

import numpy as np
import torch

from src.model import SiameseEncoder, contrastive_loss
from src.pairs import sample_pairs


def train_seed(
    modalities,
    song_ids,
    train_pools,
    validation_pools,
    seed,
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    margin=1.0,
):
    """Train one CPU Siamese encoder using partition-local sampled pairs."""
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    required_modalities = ("bpm", "lyrics", "genre")
    for name in required_modalities:
        if name not in modalities:
            raise ValueError(f"missing required modality: {name}")

    arrays = {}
    for name in required_modalities:
        array = np.asarray(modalities[name])
        if array.ndim != 2:
            raise ValueError(f"modality {name} must be a 2D array")
        if array.shape[0] != len(song_ids):
            raise ValueError(f"modality {name} row count must match song_ids")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise ValueError(f"modality {name} must contain real numeric values")
        if not np.isfinite(array).all():
            raise ValueError(f"modality {name} must contain only finite values")
        arrays[name] = array
    if arrays["bpm"].shape[1] != 1:
        raise ValueError("modality bpm must have width 1")
    if arrays["lyrics"].shape[1] != 5000:
        raise ValueError("modality lyrics must have width 5000")
    if arrays["genre"].shape[1] == 0:
        raise ValueError("modality genre must have width greater than 0")

    if len(set(song_ids)) != len(song_ids):
        raise ValueError("song_ids must be unique")
    if set(train_pools["partition_ids"]) & set(validation_pools["partition_ids"]):
        raise ValueError("train and validation partitions must not overlap")
    for name, pools in (("train", train_pools), ("validation", validation_pools)):
        if not pools.get("positives"):
            raise ValueError(f"{name} pool must contain at least one positive pair")

    torch.manual_seed(seed)
    song_index = {song_id: index for index, song_id in enumerate(song_ids)}
    tensors = {
        name: torch.as_tensor(arrays[name], dtype=torch.float32, device="cpu")
        for name in required_modalities
    }

    model = SiameseEncoder(num_genres=tensors["genre"].shape[1]).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation_rows = sample_pairs(
        validation_pools, seed + 1_000_000, 1, 0.5
    )

    def pair_tensors(rows):
        left = torch.tensor([song_index[row[0]] for row in rows], dtype=torch.long)
        right = torch.tensor([song_index[row[1]] for row in rows], dtype=torch.long)
        labels = torch.tensor([row[2] for row in rows], dtype=torch.float32)
        return left, right, labels

    validation_left, validation_right, validation_labels = pair_tensors(validation_rows)

    def loss_for_pairs(left_indices, right_indices, labels):
        left_embedding = model(
            tensors["bpm"][left_indices],
            tensors["lyrics"][left_indices],
            tensors["genre"][left_indices],
        )
        right_embedding = model(
            tensors["bpm"][right_indices],
            tensors["lyrics"][right_indices],
            tensors["genre"][right_indices],
        )
        return contrastive_loss(left_embedding, right_embedding, labels, margin=margin)

    history = []
    train_rows_by_epoch = []
    for epoch in range(epochs):
        model.train()
        rows = sample_pairs(train_pools, seed + epoch, 1, 0.5)
        train_rows_by_epoch.append(rows)
        left_indices, right_indices, labels = pair_tensors(rows)
        shuffle_generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        permutation = torch.randperm(len(rows), generator=shuffle_generator)
        batch_losses = []

        for start in range(0, len(rows), batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_for_pairs(left_indices[batch], right_indices[batch], labels[batch])
            loss.backward()
            optimizer.step()
            batch_losses.append((loss.detach().item(), len(batch)))

        train_loss = sum(loss * count for loss, count in batch_losses) / len(rows)
        model.eval()
        with torch.no_grad():
            validation_loss = loss_for_pairs(
                validation_left, validation_right, validation_labels
            ).item()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

    pair_trace = {
        "train_by_epoch": train_rows_by_epoch,
        "validation": validation_rows,
    }
    return model, history, pair_trace

from numbers import Integral

import numpy as np
import torch

from src.model import AudioOnlySiameseEncoder, contrastive_loss
from src.pairs import sample_pairs


def train_audio_only_seed(
    audio_sequences,
    song_ids,
    train_pools,
    validation_pools,
    seed,
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    margin=1.0,
):
    """Train an audio-only Siamese encoder; song_ids must be canonical strings."""
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    song_ids = list(song_ids)
    if any(not isinstance(song_id, str) for song_id in song_ids):
        raise ValueError("song_ids must contain only strings")
    try:
        song_id_set = set(song_ids)
    except TypeError as exc:
        raise ValueError("song_ids must be hashable") from exc
    if len(song_id_set) != len(song_ids):
        raise ValueError("song_ids must be unique")

    audio = np.asarray(audio_sequences)
    if audio.shape != (len(song_ids), 1024, 28):
        raise ValueError("audio_sequences must have shape (N, 1024, 28) aligned with song_ids")
    if not np.issubdtype(audio.dtype, np.number) or np.issubdtype(
        audio.dtype, np.complexfloating
    ):
        raise ValueError("audio_sequences must contain real numeric values")
    if not np.isfinite(audio).all():
        raise ValueError("audio_sequences must contain only finite values")

    pool_id_sets = {}
    pools_to_validate = [("train", train_pools)]
    if validation_pools is not None:
        pools_to_validate.append(("validation", validation_pools))
    for name, pools in pools_to_validate:
        try:
            partition_ids = list(pools["partition_ids"])
            positives = list(pools["positives"])
            partition_id_set = set(partition_ids)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed {name} pool") from exc
        if not positives:
            raise ValueError(f"{name} pool must contain at least one positive pair")
        if len(partition_id_set) != len(partition_ids):
            raise ValueError(f"{name} partition_ids must be unique")
        if not partition_id_set <= song_id_set:
            raise ValueError(f"{name} pool contains IDs not present in song_ids")
        try:
            referenced_ids = {track_id for pair in positives for track_id in pair}
        except TypeError as exc:
            raise ValueError(f"malformed {name} positive pairs") from exc
        if not referenced_ids <= partition_id_set:
            raise ValueError(f"{name} pool references IDs outside its partition_ids")
        if any(len(pair) != 2 for pair in positives):
            raise ValueError(f"malformed {name} positive pairs")
        pool_id_sets[name] = partition_id_set
    if validation_pools is not None and pool_id_sets["train"] & pool_id_sets["validation"]:
        raise ValueError("train and validation partitions must not overlap")

    torch.manual_seed(seed)
    song_index = {song_id: index for index, song_id in enumerate(song_ids)}
    audio_tensor = torch.as_tensor(audio, dtype=torch.float32, device="cpu")
    if not torch.isfinite(audio_tensor).all():
        raise ValueError("input values must remain finite after float32 conversion")

    model = AudioOnlySiameseEncoder().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation_rows = (
        sample_pairs(validation_pools, seed + 1_000_000, 1, 0.0)
        if validation_pools is not None
        else []
    )

    def pair_tensors(rows):
        left = torch.tensor([song_index[row[0]] for row in rows], dtype=torch.long)
        right = torch.tensor([song_index[row[1]] for row in rows], dtype=torch.long)
        labels = torch.tensor([row[2] for row in rows], dtype=torch.float32)
        return left, right, labels

    if validation_pools is not None:
        validation_left, validation_right, validation_labels = pair_tensors(validation_rows)

    def loss_for_pairs(left_indices, right_indices, labels):
        left_embedding = model(audio_tensor[left_indices])
        right_embedding = model(audio_tensor[right_indices])
        return contrastive_loss(left_embedding, right_embedding, labels, margin=margin)

    history = []
    train_rows_by_epoch = []
    for epoch in range(epochs):
        model.train()
        rows = sample_pairs(train_pools, seed + epoch, 1, 0.0)
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
        validation_loss = None
        if validation_pools is not None:
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

    pair_trace = {"train_by_epoch": train_rows_by_epoch, "validation": validation_rows}
    return model, history, pair_trace

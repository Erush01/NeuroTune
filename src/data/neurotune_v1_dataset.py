"""Index transformed NeuroTune v1 features and collate ragged audio batches."""

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class NeuroTuneV1Dataset(Dataset):
    """Align transformed feature arrays to the requested order.

    Input arrays are shared; callers must treat them and returned NumPy rows as
    read-only. Collated tensors are newly allocated.
    """

    def __init__(self, track_ids, features, modalities=None):
        self.track_ids = list(track_ids)
        if not self.track_ids or any(not isinstance(track_id, str) or not track_id.strip()
                                     for track_id in self.track_ids):
            raise ValueError("track_ids must be non-empty strings")
        if len(set(self.track_ids)) != len(self.track_ids):
            raise ValueError("track_ids must be unique")
        if not isinstance(features, dict):
            raise ValueError("features must be a mapping")

        count = len(self.track_ids)
        if "audio" in features and "audio_windows" in features:
            raise ValueError("features cannot contain both audio and audio_windows")
        self.multi_window = "audio_windows" in features
        if self.multi_window:
            audio_windows = features["audio_windows"]
            window_valid = features.get("window_valid")
            if not isinstance(audio_windows, dict) or set(audio_windows) != set(self.track_ids):
                raise ValueError("audio_windows must map exactly the requested track IDs")
            if (not isinstance(window_valid, np.ndarray) or window_valid.dtype != np.bool_
                    or window_valid.shape != (count, 3)):
                raise ValueError(f"window_valid must be a bool array with shape ({count}, 3)")
            for index, track_id in enumerate(self.track_ids):
                windows = audio_windows[track_id]
                if not isinstance(windows, (list, tuple)) or len(windows) != 3:
                    raise ValueError(f"audio_windows for {track_id} must contain three sequences")
                if not window_valid[index].any():
                    raise ValueError(f"track {track_id} must have at least one valid window")
                for window_index, sequence in enumerate(windows):
                    is_valid = window_valid[index, window_index]
                    if (not isinstance(sequence, np.ndarray) or sequence.dtype != np.float32
                            or sequence.ndim != 2 or sequence.shape[1] != 26
                            or (is_valid and sequence.shape[0] == 0)
                            or (not is_valid and sequence.shape[0] != 0)
                            or not np.isfinite(sequence).all()):
                        raise ValueError(
                            f"audio window {window_index} for {track_id} must be finite float32 "
                            f"with shape (T, 26) and validity-consistent length")
            self.features = {"audio_windows": audio_windows, "window_valid": window_valid}
        else:
            audio = features.get("audio")
            if not isinstance(audio, (list, tuple)) or len(audio) != count:
                raise ValueError("audio must contain one sequence per track")
            for index, sequence in enumerate(audio):
                if (not isinstance(sequence, np.ndarray) or sequence.dtype != np.float32
                        or sequence.ndim != 2 or sequence.shape[1] != 26
                        or sequence.shape[0] == 0 or not np.isfinite(sequence).all()):
                    raise ValueError(f"audio sequence {index} must be non-empty finite float32 with shape (T, 26)")
            self.features = {"audio": audio}

        if modalities is None:
            modalities = ("bpm", "lyrics", "genre")
        if (not isinstance(modalities, (list, tuple))
                or any(not isinstance(name, str) or name not in {"bpm", "lyrics", "genre"} for name in modalities)
                or len(modalities) != len(set(modalities))):
            raise ValueError("modalities must be a unique list of bpm, lyrics, and genre")

        self.modalities = tuple(modalities)
        expected_shapes = {"bpm": (count, 1), "lyrics": (count, 5000)}
        for name, shape in expected_shapes.items():
            if name not in self.modalities:
                continue
            matrix = features.get(name)
            if (not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32
                    or matrix.shape != shape or not np.isfinite(matrix).all()):
                raise ValueError(f"{name} must be a finite float32 matrix with shape {shape}")
            self.features[name] = matrix
        if "genre" in self.modalities:
            genre = features.get("genre")
            if (not isinstance(genre, np.ndarray) or genre.dtype != np.float32
                    or genre.ndim != 2 or genre.shape[0] != count or genre.shape[1] == 0
                    or not np.isfinite(genre).all()):
                raise ValueError(f"genre must be a finite float32 matrix with {count} rows")
            self.features["genre"] = genre
        self.track_to_index = {track_id: index for index, track_id in enumerate(self.track_ids)}

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, index):
        row = {"track_id": self.track_ids[index]}
        if self.multi_window:
            track_id = self.track_ids[index]
            row["audio_windows"] = self.features["audio_windows"][track_id]
            row["window_valid"] = self.features["window_valid"][index]
        else:
            row["audio"] = self.features["audio"][index]
        row.update({name: self.features[name][index] for name in self.modalities})
        return row


def neurotune_v1_collate(rows):
    """Right-pad audio to this batch's longest sequence and build its valid mask."""
    if not rows:
        raise ValueError("cannot collate an empty batch")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("all rows must be mappings")
    if any(("audio" in row) == ("audio_windows" in row) for row in rows):
        raise ValueError("each row must contain exactly one audio mode")
    multi_window = "audio_windows" in rows[0]
    if any(("audio_windows" in row) != multi_window for row in rows):
        raise ValueError("cannot mix single- and multi-window rows in one batch")
    audio_keys = {"track_id", "audio_windows", "window_valid"} if multi_window else {"track_id", "audio"}
    modality_keys = set(rows[0]) - audio_keys
    if any(set(row) - audio_keys != modality_keys for row in rows):
        raise ValueError("all rows in a batch must have consistent modality keys")
    if multi_window:
        lengths = []
        valid = []
        for row in rows:
            if "window_valid" not in row:
                raise ValueError("each D4 row must include window_valid")
            windows = row["audio_windows"]
            window_valid = row["window_valid"]
            if not isinstance(windows, (list, tuple)) or len(windows) != 3:
                raise ValueError("each D4 row must contain exactly three audio windows")
            if (not isinstance(window_valid, np.ndarray) or window_valid.dtype != np.bool_
                    or window_valid.shape != (3,)):
                raise ValueError("each D4 row window_valid must be a bool array with shape (3,)")
            if not window_valid.any():
                raise ValueError("each D4 row must have at least one valid window")
            row_lengths = []
            for index, (window, is_valid) in enumerate(zip(windows, window_valid)):
                if (not isinstance(window, np.ndarray) or window.dtype != np.float32
                        or window.ndim != 2 or window.shape[1] != 26
                        or (is_valid and window.shape[0] == 0)
                        or (not is_valid and window.shape[0] != 0)
                        or not np.isfinite(window).all()):
                    raise ValueError(
                        f"D4 row window {index} must be finite float32 with shape (T, 26) "
                        "and validity-consistent length")
                row_lengths.append(len(window))
            lengths.append(row_lengths)
            valid.append(window_valid)
        max_length = max(length for row_lengths, row_valid in zip(lengths, valid)
                         for length, is_valid in zip(row_lengths, row_valid) if is_valid)
        audio = torch.zeros((len(rows), 3, max_length, 26), dtype=torch.float32)
        mask = torch.zeros((len(rows), 3, max_length), dtype=torch.bool)
        for row_index, (row, row_lengths, row_valid) in enumerate(zip(rows, lengths, valid)):
            for window_index, (window, length, is_valid) in enumerate(
                    zip(row["audio_windows"], row_lengths, row_valid)):
                if is_valid:
                    audio[row_index, window_index, :length] = torch.as_tensor(window, dtype=torch.float32)
                    mask[row_index, window_index, :length] = True
        batch = {
            "audio_windows": audio,
            "mask": mask,
            "window_mask": torch.as_tensor(np.stack(valid), dtype=torch.bool),
            "lengths": torch.as_tensor(lengths, dtype=torch.long),
            "track_ids": [row["track_id"] for row in rows],
        }
    else:
        lengths = [len(row["audio"]) for row in rows]
        batch_size, max_length = len(rows), max(lengths)
        audio = torch.zeros((batch_size, max_length, 26), dtype=torch.float32)
        mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        for index, (row, length) in enumerate(zip(rows, lengths)):
            audio[index, :length] = torch.as_tensor(row["audio"], dtype=torch.float32)
            mask[index, :length] = True
        batch = {
            "audio": audio,
            "mask": mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "track_ids": [row["track_id"] for row in rows],
        }
    for name in ("bpm", "lyrics", "genre"):
        if name in modality_keys:
            batch[name] = torch.stack([torch.as_tensor(row[name], dtype=torch.float32) for row in rows])
    return batch

import torch
from torch import nn
from torch.nn import functional as F


class NeuroTuneV1Encoder(nn.Module):
    _MODALITIES = {
        "A": (),
        "B": ("bpm",),
        "C": ("bpm", "genre"),
        "D": ("bpm", "genre", "lyrics"),
    }

    def __init__(self, variant, num_genres):
        super().__init__()
        if variant not in self._MODALITIES:
            raise ValueError("variant must be one of A, B, C, or D")
        if not isinstance(num_genres, int) or num_genres <= 0:
            raise ValueError("num_genres must be a positive integer")
        self.variant = variant
        self.num_genres = num_genres

        self.audio_branch = nn.ModuleList(
            [
                nn.Conv1d(26, 32, kernel_size=5, stride=1, padding=2),
                nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.audio_projection = nn.Linear(64, 64)
        active = self._MODALITIES[variant]
        if "bpm" in active:
            self.bpm_branch = nn.Sequential(nn.Linear(1, 8), nn.ReLU())
        if "genre" in active:
            self.genre_branch = nn.Sequential(nn.Linear(num_genres, 16), nn.ReLU())
        if "lyrics" in active:
            self.lyrics_branch = nn.Sequential(nn.Linear(5000, 32), nn.ReLU())
        if active:
            dimensions = 64 + sum({"bpm": 8, "genre": 16, "lyrics": 32}[name] for name in active)
            self.fusion = nn.Sequential(
                nn.Linear(dimensions, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )

    @property
    def active_modalities(self):
        """Return the auxiliary modalities used by this encoder variant."""
        return self._MODALITIES[self.variant]

    @staticmethod
    def _validate_tensor(name, value, shape):
        if not torch.is_tensor(value) or value.ndim != len(shape):
            raise ValueError(f"{name} must be a tensor with shape {shape}")
        for actual, expected in zip(value.shape, shape):
            if expected is not None and actual != expected:
                raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")

    def _validate_inputs(self, audio, mask, bpm, lyrics, genre):
        self._validate_tensor("audio", audio, (None, None, 26))
        if audio.dtype != torch.float32:
            raise ValueError("audio must have dtype float32")
        batch, steps, _ = audio.shape
        if batch < 1 or steps < 1:
            raise ValueError("audio batch and time dimensions must be non-empty")
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or tuple(mask.shape) != (batch, steps):
            raise ValueError("mask must be a boolean tensor with shape (B, T)")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each sequence must contain at least one valid timestep")

        values = {"bpm": bpm, "lyrics": lyrics, "genre": genre}
        expected = set(self._MODALITIES[self.variant])
        for name, value in values.items():
            if (name in expected) != (value is not None):
                state = "required" if name in expected else "not accepted"
                raise ValueError(f"{name} is {state} for variant {self.variant}")
            if value is not None:
                width = {"bpm": 1, "lyrics": 5000, "genre": self.num_genres}[name]
                self._validate_tensor(name, value, (batch, width))
                if value.dtype != audio.dtype:
                    raise ValueError(f"{name} must have the same dtype as audio")
                if value.device != audio.device:
                    raise ValueError(f"{name} must be on the same device as audio")

    def _validate_keep_masks(self, keep_masks, batch, device):
        if keep_masks is None:
            return
        if not isinstance(keep_masks, dict):
            raise ValueError("keep_masks must be a mapping from auxiliary modality names to tensors")
        for name, keep_mask in keep_masks.items():
            if name not in {"bpm", "genre", "lyrics"}:
                raise ValueError(f"unknown keep-mask modality: {name}")
            if name not in self.active_modalities:
                raise ValueError(f"{name} cannot be masked for variant {self.variant}")
            if not torch.is_tensor(keep_mask) or keep_mask.dtype != torch.bool:
                raise ValueError(f"{name} keep mask must be a boolean tensor")
            if tuple(keep_mask.shape) != (batch,):
                raise ValueError(f"{name} keep mask must have shape ({batch},)")
            if keep_mask.device != device:
                raise ValueError(f"{name} keep mask must be on the same device as audio")

    def forward(
        self, audio=None, mask=None, bpm=None, lyrics=None, genre=None, keep_masks=None,
        audio_windows=None, window_mask=None,
    ):
        if audio_windows is None:
            if window_mask is not None:
                raise ValueError("window_mask is only accepted with audio_windows")
            self._validate_inputs(audio, mask, bpm, lyrics, genre)
            self._validate_keep_masks(keep_masks, audio.shape[0], audio.device)
            mask = mask.to(device=audio.device)
            # Ignore arbitrary values in padded input cells, matching zero-padding semantics.
            features = audio.masked_fill(~mask.unsqueeze(-1), 0).transpose(1, 2)
            for conv in self.audio_branch:
                features = F.relu(conv(features))
                features = features.masked_fill(~mask.unsqueeze(1), 0)
            pooled = features.sum(dim=2) / mask.sum(dim=1, keepdim=True).to(features.dtype)
            audio_embedding = self.audio_projection(pooled)
        else:
            if audio is not None:
                raise ValueError("provide audio or audio_windows, not both")
            if self.variant != "D":
                raise ValueError("audio_windows are only accepted for variant D")
            self._validate_tensor("audio_windows", audio_windows, (None, 3, None, 26))
            if audio_windows.dtype != torch.float32:
                raise ValueError("audio_windows must have dtype float32")
            batch, windows, steps, _ = audio_windows.shape
            if batch < 1 or steps < 1:
                raise ValueError("audio_windows batch and time dimensions must be non-empty")
            if (not torch.is_tensor(mask) or mask.dtype != torch.bool
                    or tuple(mask.shape) != (batch, windows, steps)):
                raise ValueError("mask must be a boolean tensor with shape (B, 3, T)")
            if (not torch.is_tensor(window_mask) or window_mask.dtype != torch.bool
                    or tuple(window_mask.shape) != (batch, windows)):
                raise ValueError("window_mask must be a boolean tensor with shape (B, 3)")
            if mask.device != audio_windows.device or window_mask.device != audio_windows.device:
                raise ValueError("audio_windows, mask, and window_mask must be on the same device")
            timestep_valid = mask.any(dim=2)
            if not torch.equal(timestep_valid, window_mask):
                raise ValueError("window_mask must match timestep-mask validity")
            if not torch.all(window_mask.any(dim=1)):
                raise ValueError("each track must contain at least one valid window")
            values = {"bpm": bpm, "lyrics": lyrics, "genre": genre}
            for name, value in values.items():
                if value is None:
                    raise ValueError(f"{name} is required for variant D")
                width = {"bpm": 1, "lyrics": 5000, "genre": self.num_genres}[name]
                self._validate_tensor(name, value, (batch, width))
                if value.dtype != audio_windows.dtype or value.device != audio_windows.device:
                    raise ValueError(f"{name} must have the same dtype and device as audio_windows")
            self._validate_keep_masks(keep_masks, batch, audio_windows.device)
            valid_audio = audio_windows[window_mask]
            valid_mask = mask[window_mask]
            features = valid_audio.masked_fill(~valid_mask.unsqueeze(-1), 0).transpose(1, 2)
            for conv in self.audio_branch:
                features = F.relu(conv(features))
                features = features.masked_fill(~valid_mask.unsqueeze(1), 0)
            pooled = features.sum(dim=2) / valid_mask.sum(dim=1, keepdim=True).to(features.dtype)
            projected = self.audio_projection(pooled)
            window_embeddings = audio_windows.new_zeros((batch, windows, 64))
            window_embeddings[window_mask] = projected
            audio_embedding = window_embeddings.sum(dim=1) / window_mask.sum(dim=1, keepdim=True)
        if self.variant == "A":
            return F.normalize(audio_embedding, p=2, dim=1)

        branches = [audio_embedding]
        if self.variant in {"B", "C", "D"}:
            branch = self.bpm_branch(bpm)
            if keep_masks is not None and "bpm" in keep_masks:
                branch = branch * keep_masks["bpm"].unsqueeze(1)
            branches.append(branch)
        if self.variant in {"C", "D"}:
            branch = self.genre_branch(genre)
            if keep_masks is not None and "genre" in keep_masks:
                branch = branch * keep_masks["genre"].unsqueeze(1)
            branches.append(branch)
        if self.variant == "D":
            branch = self.lyrics_branch(lyrics)
            if keep_masks is not None and "lyrics" in keep_masks:
                branch = branch * keep_masks["lyrics"].unsqueeze(1)
            branches.append(branch)
        return F.normalize(self.fusion(torch.cat(branches, dim=1)), p=2, dim=1)

    def forward_triplets(
        self,
        anchor_inputs,
        positive_inputs,
        negative_inputs,
        anchor_keep_masks=None,
        positive_keep_masks=None,
        negative_keep_masks=None,
    ):
        """Embed three collator-style input dictionaries; return (anchor, positive, negative)."""
        def encode(inputs, keep_masks):
            if not isinstance(inputs, dict):
                raise ValueError("triplet inputs must be dictionaries")
            accepted = {"audio", "audio_windows", "window_mask", "mask", "bpm", "lyrics", "genre", "lengths", "track_ids"}
            unknown = set(inputs) - accepted
            if unknown:
                raise ValueError(f"unexpected triplet input keys: {sorted(unknown)}")
            active = self._MODALITIES[self.variant]
            args = {key: inputs[key] for key in ("audio", "audio_windows", "mask", "window_mask") if key in inputs}
            args.update({key: inputs[key] for key in active if key in inputs})
            if keep_masks is not None:
                args["keep_masks"] = keep_masks
            if ("mask" not in args or not ("audio" in args or "audio_windows" in args)):
                raise ValueError("triplet inputs must include audio (or audio_windows) and mask")
            return self(**args)

        return (
            encode(anchor_inputs, anchor_keep_masks),
            encode(positive_inputs, positive_keep_masks),
            encode(negative_inputs, negative_keep_masks),
        )

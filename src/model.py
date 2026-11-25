import torch
from torch import nn
from torch.nn import functional as F


class SiameseEncoder(nn.Module):
    def __init__(self, num_genres):
        super().__init__()
        self.bpm_branch = nn.Sequential(nn.Linear(1, 8), nn.ReLU())
        self.lyrics_branch = nn.Sequential(nn.Linear(5000, 32), nn.ReLU())
        self.genre_branch = nn.Sequential(nn.Linear(num_genres, 16), nn.ReLU())
        self.projection = nn.Linear(56, 64)

    def forward(self, bpm, lyrics, genre):
        features = torch.cat(
            (
                self.bpm_branch(bpm),
                self.lyrics_branch(lyrics),
                self.genre_branch(genre),
            ),
            dim=1,
        )
        return F.normalize(self.projection(features), p=2, dim=1)


class AudioFirstSiameseEncoder(nn.Module):
    def __init__(self, num_genres):
        super().__init__()
        self.audio_branch = nn.Sequential(
            nn.Conv1d(28, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(1024, 32),
            nn.ReLU(),
        )
        self.bpm_branch = nn.Sequential(nn.Linear(1, 8), nn.ReLU())
        self.lyrics_branch = nn.Sequential(nn.Linear(5000, 32), nn.ReLU())
        self.genre_branch = nn.Sequential(nn.Linear(num_genres, 16), nn.ReLU())
        self.projection = nn.Linear(88, 64)

    def forward(self, audio, bpm, lyrics, genre):
        features = torch.cat(
            (
                self.audio_branch(audio.transpose(1, 2)),
                self.bpm_branch(bpm),
                self.lyrics_branch(lyrics),
                self.genre_branch(genre),
            ),
            dim=1,
        )
        return F.normalize(self.projection(features), p=2, dim=1)


class AudioOnlySiameseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_branch = nn.Sequential(
            nn.Conv1d(28, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(1024, 32),
            nn.ReLU(),
        )

    def forward(self, audio):
        features = self.audio_branch(audio.transpose(1, 2))
        return F.normalize(features, p=2, dim=1)


def contrastive_loss(left_embeddings, right_embeddings, labels, margin=1.0):
    distances = torch.linalg.vector_norm(left_embeddings - right_embeddings, dim=1)
    if torch.is_complex(labels):
        raise ValueError("labels must be real-valued")
    labels = labels.to(dtype=distances.dtype, device=distances.device).reshape(-1)
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must contain only 0 or 1")
    losses = labels * distances.square() + (1 - labels) * F.relu(margin - distances).square()
    return losses.mean()

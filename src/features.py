"""Train-fitted B0 feature construction."""

from decimal import Decimal, InvalidOperation
import math

import numpy as np


VOCABULARY_SIZE = 5000


def build_b0_vectors(tracks, lyrics, genres, train_ids, song_ids):
    """Fit feature statistics on ``train_ids`` and transform ``song_ids``.

    Feature columns are standardized BPM, 5000 official MXM word indices, and
    one column per sorted genre observed in training.
    """
    track_by_id = {}
    for track in tracks:
        if not isinstance(track, dict) or "track_id" not in track:
            raise ValueError("each track must contain a track_id")
        track_by_id[track["track_id"]] = track

    train_ids = list(train_ids)
    song_ids = list(song_ids)
    if not train_ids:
        raise ValueError("train_ids must not be empty")
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("train_ids must be unique")

    def bpm_for(track_id):
        try:
            bpm = float(track_by_id[track_id]["tempo"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing or invalid tempo for track {track_id!r}") from exc
        if not math.isfinite(bpm):
            raise ValueError(f"missing or invalid tempo for track {track_id!r}")
        return bpm

    def lyric_count(count, track_id):
        try:
            value = Decimal(str(count))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"invalid lyric entry for track {track_id!r}") from exc
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            raise ValueError(f"invalid lyric entry for track {track_id!r}")
        return int(value)

    train_bpm = np.asarray([bpm_for(track_id) for track_id in train_ids], dtype=float)
    bpm_mean = float(train_bpm.mean())
    bpm_std = float(train_bpm.std())
    bpm_divisor = bpm_std if bpm_std > 0 else 1.0

    document_frequency = np.zeros(VOCABULARY_SIZE, dtype=np.int64)
    for track_id in train_ids:
        document = lyrics.get(track_id, {})
        if not isinstance(document, dict):
            raise ValueError(f"invalid lyric document for track {track_id!r}")
        seen = set()
        for word_id, count in document.items():
            try:
                index = int(word_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid lyric entry for track {track_id!r}") from exc
            value = lyric_count(count, track_id)
            if str(index) != str(word_id) or not 1 <= index <= VOCABULARY_SIZE:
                raise ValueError(f"invalid lyric entry for track {track_id!r}")
            if value > 0:
                seen.add(index - 1)
        if seen:
            document_frequency[list(seen)] += 1
    idf = np.log((1 + len(train_ids)) / (1 + document_frequency)) + 1

    train_genres = []
    for track_id in train_ids:
        genre = genres.get(track_id)
        if not isinstance(genre, str) or not genre:
            raise ValueError(f"missing or invalid genre for track {track_id!r}")
        train_genres.append(genre)
    genre_vocabulary = sorted(set(train_genres))
    genre_index = {genre: index for index, genre in enumerate(genre_vocabulary)}

    scale = 1 / math.sqrt(3)
    vectors = np.zeros((len(song_ids), 1 + VOCABULARY_SIZE + len(genre_vocabulary)), dtype=float)
    for row, track_id in enumerate(song_ids):
        vectors[row, 0] = ((bpm_for(track_id) - bpm_mean) / bpm_divisor) * scale

        document = lyrics.get(track_id, {})
        if not isinstance(document, dict):
            raise ValueError(f"invalid lyric document for track {track_id!r}")
        for word_id, count in document.items():
            try:
                index = int(word_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid lyric entry for track {track_id!r}") from exc
            value = lyric_count(count, track_id)
            if str(index) != str(word_id) or not 1 <= index <= VOCABULARY_SIZE:
                raise ValueError(f"invalid lyric entry for track {track_id!r}")
            vectors[row, 1 + index - 1] = value * idf[index - 1]
        lyric_row = vectors[row, 1 : 1 + VOCABULARY_SIZE]
        lyric_norm = np.linalg.norm(lyric_row)
        if lyric_norm:
            lyric_row /= lyric_norm
        lyric_row *= scale

        genre = genres.get(track_id)
        if isinstance(genre, str) and genre in genre_index:
            vectors[row, 1 + VOCABULARY_SIZE + genre_index[genre]] = scale

        row_norm = np.linalg.norm(vectors[row])
        if row_norm:
            vectors[row] /= row_norm

    metadata = {
        "bpm_mean": bpm_mean,
        "bpm_std": bpm_std,
        "idf": idf,
        "genre_vocabulary": genre_vocabulary,
    }
    return vectors, metadata


def build_train_fitted_modalities(tracks, lyrics, genres, train_ids, song_ids):
    """Return separately transformed modalities fitted on ``train_ids``.

    The fitting and input validation are shared with B0 so its established
    train-only statistics remain the source of truth.
    """
    _, metadata = build_b0_vectors(tracks, lyrics, genres, train_ids, song_ids)
    track_by_id = {track["track_id"]: track for track in tracks}
    song_ids = list(song_ids)
    bpm_divisor = metadata["bpm_std"] if metadata["bpm_std"] > 0 else 1.0

    bpm = np.empty((len(song_ids), 1), dtype=float)
    lyric_vectors = np.zeros((len(song_ids), VOCABULARY_SIZE), dtype=float)
    genre_vocabulary = metadata["genre_vocabulary"]
    genre_index = {genre: index for index, genre in enumerate(genre_vocabulary)}
    genre_vectors = np.zeros((len(song_ids), len(genre_vocabulary)), dtype=float)

    for row, track_id in enumerate(song_ids):
        bpm[row, 0] = (
            float(track_by_id[track_id]["tempo"]) - metadata["bpm_mean"]
        ) / bpm_divisor

        for word_id, count in lyrics.get(track_id, {}).items():
            lyric_vectors[row, int(word_id) - 1] = int(Decimal(str(count))) * metadata["idf"][
                int(word_id) - 1
            ]
        norm = np.linalg.norm(lyric_vectors[row])
        if norm:
            lyric_vectors[row] /= norm

        genre = genres.get(track_id)
        if isinstance(genre, str) and genre in genre_index:
            genre_vectors[row, genre_index[genre]] = 1.0

    return {"bpm": bpm, "lyrics": lyric_vectors, "genre": genre_vectors}, metadata

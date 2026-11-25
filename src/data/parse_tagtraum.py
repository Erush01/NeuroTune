from .table import normalize_track_id, read_rows


def parse_tagtraum(path):
    """Parse a track_id,genre extract; conflicting labels are rejected."""
    genres = {}
    for row in read_rows(path):
        track_id = normalize_track_id(row.get("track_id"))
        genre = row.get("genre", "").strip()
        if not genre:
            continue
        if track_id in genres and genres[track_id] != genre:
            raise ValueError(f"Conflicting primary genres for {track_id}")
        genres[track_id] = genre
    return genres

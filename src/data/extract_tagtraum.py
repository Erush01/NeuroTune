"""Convert the official tagtraum .cls annotations into tagtraum_genres.csv."""
import csv
from pathlib import Path

from .table import normalize_track_id


def convert_cls(cls_path, output_path):
    """Parse a majority-vote cls file; emit track_id,genre (primary genre only).

    Repeated track IDs with differing labels are rejected; identical repeats collapse.
    """
    cls_path = Path(cls_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genres = {}
    with cls_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(f"{cls_path}:{line_number}: unexpected row {line!r}")
            track_id = normalize_track_id(fields[0])
            genre = fields[1].strip()
            if not genre:
                continue
            if track_id in genres and genres[track_id] != genre:
                raise ValueError(
                    f"Conflicting primary genres for {track_id}: "
                    f"{genres[track_id]!r} vs {genre!r}"
                )
            genres[track_id] = genre
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["track_id", "genre"])
        for track_id in sorted(genres):
            writer.writerow([track_id, genres[track_id]])
    return genres

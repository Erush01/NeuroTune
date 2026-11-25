import math

from .table import normalize_track_id, read_rows


def parse_msd(path):
    """Parse the documented extracted MSD track table; duplicate IDs collapse."""
    tracks = {}
    for row in read_rows(path):
        track_id = normalize_track_id(row.get("track_id"))
        try:
            tempo = float(row.get("tempo", ""))
            if not math.isfinite(tempo):
                tempo = None
        except (TypeError, ValueError):
            tempo = None
        track = {
            "track_id": track_id,
            "tempo": tempo,
            "artist_id": row.get("artist_id", ""),
            "song_id": row.get("song_id", ""),
            "artist_name": row.get("artist_name", ""),
            "title": row.get("title", row.get("song_name", "")),
            "duplicate_rows": 0,
        }
        if track_id in tracks:
            tracks[track_id]["duplicate_rows"] += 1
        else:
            tracks[track_id] = track
    return list(tracks.values())

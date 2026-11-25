"""Extract MSD track metadata and tempo from official MillionSongSubset HDF5 files."""
import csv
import math
from pathlib import Path


def extract_track(h5_path):
    """Read one MSD HDF5 file into the normalized track dict."""
    import h5py

    track = {
        "track_id": "",
        "tempo": None,
        "artist_id": "",
        "song_id": "",
        "artist_name": "",
        "title": "",
    }
    with h5py.File(h5_path, "r") as f:
        analysis = f["analysis/songs"][0]
        track_id = analysis["track_id"]
        if isinstance(track_id, bytes):
            track_id = track_id.decode("utf-8")
        track["track_id"] = track_id.strip().upper()
        tempo = float(analysis["tempo"])
        track["tempo"] = tempo if math.isfinite(tempo) and tempo > 0 else None
        meta = f["metadata/songs"][0]
        for field in ("artist_id", "song_id", "artist_name", "title"):
            value = meta[field]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            track[field] = value.strip() if isinstance(value, str) else value
    return track


def extract_dataset(root):
    """Walk an extracted subset directory and return sorted track dicts."""
    root = Path(root)
    tracks = {}
    h5_paths = sorted(root.rglob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files under {root}")
    for h5_path in h5_paths:
        track = extract_track(h5_path)
        track_id = track["track_id"]
        if not track_id.startswith("TR") or not track_id[2:].isalnum():
            raise ValueError(f"Invalid MSD track ID from {h5_path}: {track_id!r}")
        if track_id in tracks:
            continue
        tracks[track_id] = track
    return [tracks[track_id] for track_id in sorted(tracks)]


def write_tracks_csv(tracks, output_path):
    """Write the msd_tracks.csv table consumed by parse_msd."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["track_id", "tempo", "artist_id", "song_id", "artist_name", "title"]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for track in tracks:
            row = {field: track.get(field, "") for field in fields}
            row["tempo"] = "" if track.get("tempo") is None else track.get("tempo")
            writer.writerow(row)

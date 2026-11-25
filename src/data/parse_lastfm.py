import json

from .table import normalize_track_id, read_rows


def parse_lastfm(path):
    """Parse an edge table or the official JSON-per-track similarity files."""
    from pathlib import Path

    path = Path(path)
    if path.is_dir():
        return _parse_json_directory(path)
    edges = []
    for row in read_rows(path):
        source = normalize_track_id(row.get("track_id", row.get("source_track_id")))
        target = normalize_track_id(row.get("similar_track_id", row.get("target_track_id")))
        try:
            score = float(row.get("score", ""))
        except (TypeError, ValueError):
            score = None
        edges.append({"source": source, "target": target, "score": score})
    return edges


def _parse_json_directory(path):
    edges = []
    for file_path in sorted(path.glob("*.json")):
        with file_path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        source_value = record.get("track_id") or file_path.stem
        source = normalize_track_id(source_value)
        for similar in record.get("similars", []):
            if not isinstance(similar, list) or len(similar) < 2:
                raise ValueError(f"Invalid similar-track entry in {file_path}")
            target = normalize_track_id(similar[0])
            try:
                score = float(similar[1])
            except (TypeError, ValueError):
                score = None
            edges.append({"source": source, "target": target, "score": score})
    return edges

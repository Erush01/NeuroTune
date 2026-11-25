import csv
from pathlib import Path


def read_rows(path):
    """Read a documented CSV/TSV extract with a header row."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"Missing header row: {path}")
        reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]
        return [
            {str(key).strip().lower(): (value.strip() if isinstance(value, str) else value)
             for key, value in row.items() if key is not None}
            for row in reader
        ]


def normalize_track_id(value):
    track_id = (value or "").strip().upper()
    if not track_id.startswith("TR") or not track_id[2:].isalnum():
        raise ValueError(f"Invalid MSD track ID: {value!r}")
    return track_id

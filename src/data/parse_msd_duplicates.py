from pathlib import Path

from .table import normalize_track_id


def parse_msd_duplicates(path):
    """Parse official MSD duplicate groups represented by '%' headings."""
    groups = []
    current = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("%"):
                if current:
                    groups.append(current)
                    current = []
                continue
            current.append(normalize_track_id(line))
    if current:
        groups.append(current)
    return groups

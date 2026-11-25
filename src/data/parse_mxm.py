import csv
from pathlib import Path

from .table import normalize_track_id


def parse_mxm(path, track_ids=None):
    """Parse official sparse text files or a long-form track_id/word_id/count extract."""
    path = Path(path)
    requested = None if track_ids is None else {normalize_track_id(track_id) for track_id in track_ids}
    if path.is_dir():
        songs = {}
        for text_path in sorted(path.glob("*.txt")):
            _merge_counts(songs, _parse_official_text(text_path, requested))
    elif path.suffix.lower() == ".txt":
        songs = _parse_official_text(path, requested)
    else:
        songs = {}
        for row in _iter_rows(path):
            track_id = normalize_track_id(row.get("track_id"))
            word_id = row.get("word_id", "")
            count = int(row.get("count", ""))
            if not word_id or count < 0:
                raise ValueError(f"Invalid sparse lyric count for {track_id}")
            if requested is not None and track_id not in requested:
                continue
            counts = songs.setdefault(track_id, {})
            counts[word_id] = counts.get(word_id, 0) + count
    return songs


def _merge_counts(destination, source):
    for track_id, counts in source.items():
        target = destination.setdefault(track_id, {})
        for word_id, count in counts.items():
            target[word_id] = target.get(word_id, 0) + count


def _parse_official_text(path, requested=None):
    songs = {}
    vocabulary_size = None
    header_seen = False
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("%"):
                words = next(csv.reader([line[1:]]))
                if not words or any(not word for word in words):
                    raise ValueError(f"Invalid musiXmatch vocabulary header at {path}:{line_number}")
                vocabulary_size = len(words)
                header_seen = True
                continue
            if not header_seen:
                raise ValueError(f"Missing musiXmatch vocabulary header at {path}:{line_number}")
            fields = next(csv.reader([line]))
            if len(fields) < 2:
                raise ValueError(f"Invalid musiXmatch row at {path}:{line_number}")
            track_id = normalize_track_id(fields[0])
            try:
                mxm_record_id = int(fields[1])
            except ValueError as error:
                raise ValueError(f"Invalid MXM record ID at {path}:{line_number}") from error
            if mxm_record_id <= 0:
                raise ValueError(f"Invalid MXM record ID at {path}:{line_number}")
            keep = requested is None or track_id in requested
            counts = songs.setdefault(track_id, {}) if keep else None
            for item in fields[2:]:
                try:
                    word_id, count = item.split(":", 1)
                    word_index = int(word_id)
                    word_count = int(count)
                except ValueError as error:
                    raise ValueError(f"Invalid sparse count at {path}:{line_number}") from error
                if word_index < 1 or word_count < 0:
                    raise ValueError(f"Invalid sparse count at {path}:{line_number}")
                if vocabulary_size is not None and word_index > vocabulary_size:
                    raise ValueError(f"Word index exceeds vocabulary at {path}:{line_number}")
                if keep:
                    key = str(word_index)
                    counts[key] = counts.get(key, 0) + word_count
        if not header_seen:
            raise ValueError(f"Missing musiXmatch vocabulary header: {path}")
    return songs


def _iter_rows(path):
    """Stream the same header-normalized CSV/TSV rows as table.read_rows."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
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
        for row in reader:
            yield {str(key).strip().lower(): (value.strip() if isinstance(value, str) else value)
                   for key, value in row.items() if key is not None}

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.audio_only_audit import build_audio_only_audit
from src.data.parse_lastfm import parse_lastfm
from src.data.parse_msd import parse_msd
from src.data.parse_msd_duplicates import parse_msd_duplicates


def _source_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _write_json(output_path, audit):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output_path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(audit, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run(tracks_path, lastfm_path, duplicates_path, output_path, seed=684):
    paths = {
        "msd": Path(tracks_path),
        "lastfm": Path(lastfm_path),
        "msd_duplicates": Path(duplicates_path),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        audit = {
            "status": "blocked_missing_sources",
            "seed": seed,
            "source_counts": {"msd": None, "lastfm_edges_raw": None, "lastfm_tracks": None},
            "missing_sources": [
                {"source": name, "expected_path": str(paths[name])} for name in missing
            ],
            "source_sha256": {
                name: _source_sha256(path) for name, path in paths.items() if path.exists()
            },
        }
    else:
        tracks = parse_msd(paths["msd"])
        edges = parse_lastfm(paths["lastfm"])
        duplicate_groups = parse_msd_duplicates(paths["msd_duplicates"])
        audit = build_audio_only_audit(
            tracks, edges, duplicate_groups=duplicate_groups, seed=seed
        )
        audit["source_sha256"] = {
            name: _source_sha256(path) for name, path in paths.items()
        }
    _write_json(output_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description="Audit the MSD audio-only training split.")
    parser.add_argument("--tracks", type=Path, default=REPO_ROOT / "data/processed/msd_tracks.csv")
    parser.add_argument("--lastfm", type=Path, default=REPO_ROOT / "data/processed/lastfm_similar")
    parser.add_argument("--duplicates", type=Path, default=REPO_ROOT / "downloads/msd_duplicates.txt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/msd_audio_only_audit.json")
    parser.add_argument("--seed", type=int, default=684)
    args = parser.parse_args()
    audit = run(args.tracks, args.lastfm, args.duplicates, args.output, args.seed)
    print(f"Audio-only audit status: {audit['status']}")
    print(f"Audit written to {args.output}")


if __name__ == "__main__":
    main()

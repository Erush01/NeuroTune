"""Build a deterministic audit of the MSD audio-only cross-validation folds."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from src.data.audio_only_cv import build_audio_only_cv_audit
from src.data.parse_lastfm import parse_lastfm
from src.data.parse_msd import parse_msd
from src.data.parse_msd_duplicates import parse_msd_duplicates


CHUNK_SIZE = 1024 * 1024


def _update_file_hash(digest, path):
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)


def _source_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_dir():
        for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = file_path.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            _update_file_hash(digest, file_path)
    else:
        _update_file_hash(digest, path)
    return digest.hexdigest()


def _write_json_atomic(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parser():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=repo_root / "data/processed/msd_tracks.csv")
    parser.add_argument("--lastfm", type=Path, default=repo_root / "data/processed/lastfm_similar")
    parser.add_argument("--duplicates", type=Path, default=repo_root / "downloads/msd_duplicates.txt")
    parser.add_argument("--output", type=Path, default=repo_root / "results/msd_audio_only_cv_audit.json")
    parser.add_argument("--seed", type=int, default=684)
    parser.add_argument("--fold-count", type=int, default=3)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    sources = {"tracks": args.tracks, "lastfm": args.lastfm, "duplicates": args.duplicates}
    missing = [name for name, path in sources.items() if not path.exists()]
    source_hashes = {
        name: _source_sha256(path) for name, path in sources.items() if path.exists()
    }
    if missing:
        audit = {
            "status": "blocked",
            "seed": args.seed,
            "fold_count": args.fold_count,
            "source_sha256": source_hashes,
            "missing_sources": missing,
            "source_counts": None,
            "duplicate_counts": None,
            "catalog_counts": None,
            "folds": None,
            "cross_fold_edges": None,
        }
    else:
        tracks = parse_msd(args.tracks)
        edges = parse_lastfm(args.lastfm)
        duplicate_groups = parse_msd_duplicates(args.duplicates)
        audit = build_audio_only_cv_audit(
            tracks, edges, duplicate_groups=duplicate_groups,
            fold_count=args.fold_count, seed=args.seed,
        )
        audit["source_sha256"] = source_hashes

    _write_json_atomic(args.output, audit)
    print(f"{audit['status']}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

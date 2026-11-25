"""Validate official 10K catalog inputs and atomically publish its CV edge audit."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from scripts.audit_msd_hf_ab import _lastfm_edges, _mxm_ids
from src.data.neurotune_v1_official10k_cv import build_cv_audit
from src.data.parse_tagtraum import parse_tagtraum

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results/neurotune_v1_official10k_cv_audit.json"


def _sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for child in paths:
        if path.is_dir():
            relative = child.relative_to(path).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _eligible_ids(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        split_ids = data["splits"]["artist_disjoint"]["track_ids"]
        ids = set().union(*(set(split_ids[name]) for name in ("train", "validation", "test")))
    except (KeyError, TypeError) as exc:
        raise ValueError("data audit lacks artist-disjoint split track IDs") from exc
    return ids


def _atomic_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    published = False
    try:
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            staged = Path(directory) / path.name
            staged.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.link(staged, path)
            published = True
    except BaseException:
        if published:
            path.unlink(missing_ok=True)
        raise


def run(data_audit, tracks_csv, mxm_dir, genres_csv, lastfm_train_zip, output=DEFAULT_OUTPUT):
    paths = {
        "data_audit": Path(data_audit), "tracks": Path(tracks_csv),
        "mxm": Path(mxm_dir), "genres": Path(genres_csv),
        "lastfm_train": Path(lastfm_train_zip),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing sources (coverage counts unknown): " + ", ".join(missing))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    eligible = _eligible_ids(paths["data_audit"])
    records = {}
    with paths["tracks"].open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["track_id"] in eligible:
                records[row["track_id"]] = row
    lyrics_ids = _mxm_ids(paths["mxm"])
    genres = parse_tagtraum(paths["genres"])
    coverage = {
        "tracks_csv": len(eligible & records.keys()),
        "mxm_counts": len(eligible & lyrics_ids),
        "cd2c_genres": len(eligible & genres.keys()),
    }
    incomplete = {name: len(eligible) - count for name, count in coverage.items() if count != len(eligible)}
    if incomplete:
        raise ValueError(f"selected catalog coverage incomplete; unknown usable edge counts: {incomplete}")
    if len(eligible) != 939:
        raise ValueError(f"expected 939 official common-catalog IDs, found {len(eligible)}")

    metadata = {track_id: records[track_id] for track_id in eligible}
    edges, sources_found = _lastfm_edges(paths["lastfm_train"], "train", eligible)
    if len(edges) != 600:
        raise ValueError(f"expected 600 common-catalog Last.fm train edges, found {len(edges)}")
    audit = build_cv_audit(metadata, edges)
    expected = {
        "track_counts": [323, 310, 306], "test_edges": [63, 78, 79],
        "pool_edges": [289, 262, 269], "train_edges": [232, 210, 216],
        "validation_edges": [57, 52, 53],
    }
    actual = {
        "track_counts": [fold["counts"]["test_tracks"] for fold in audit["folds"]],
        "test_edges": [fold["counts"]["test_edges"] for fold in audit["folds"]],
        "pool_edges": [fold["counts"]["non_test_pool_edges"] for fold in audit["folds"]],
        "train_edges": [fold["counts"]["train_edges"] for fold in audit["folds"]],
        "validation_edges": [fold["counts"]["validation_edges"] for fold in audit["folds"]],
    }
    if actual != expected or audit["counts"]["cross_fold_edges"] != 380:
        raise ValueError(f"split counts differ from approved audit expectations: {actual}")

    audit.update({
        "catalog": "official MSD 10K common catalog",
        "coverage": coverage,
        "source_edges_found": len(edges),
        "lastfm_sources_found": len(sources_found),
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
        "code_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in (
                Path(__file__), REPO_ROOT / "src/data/neurotune_v1_official10k_cv.py",
                REPO_ROOT / "scripts/audit_msd_hf_ab.py",
                REPO_ROOT / "src/data/parse_tagtraum.py",
            )
        },
        "expected_counts": expected,
    })
    _atomic_json(output, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-audit", type=Path, default=REPO_ROOT / "results/data_audit.json")
    parser.add_argument("--tracks", type=Path, default=REPO_ROOT / "data/processed/msd_tracks.csv")
    parser.add_argument("--mxm-dir", type=Path, default=REPO_ROOT / "data/processed/mxm_counts")
    parser.add_argument("--genres", type=Path, default=REPO_ROOT / "data/processed/tagtraum_genres.csv")
    parser.add_argument("--lastfm-train", type=Path, default=REPO_ROOT / "lastfm_train.zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = run(args.data_audit, args.tracks, args.mxm_dir, args.genres,
                args.lastfm_train, args.output)
    print(json.dumps({"counts": audit["counts"], "folds": [fold["counts"] for fold in audit["folds"]]}, sort_keys=True))


if __name__ == "__main__":
    main()

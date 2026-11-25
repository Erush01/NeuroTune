import argparse
import hashlib
import json
from pathlib import Path

from src.data.join_audit import build_audit
from src.data.parse_lastfm import parse_lastfm
from src.data.parse_msd import parse_msd
from src.data.parse_msd_duplicates import parse_msd_duplicates
from src.data.parse_mxm import parse_mxm
from src.data.parse_tagtraum import parse_tagtraum


SOURCE_FILES = {
    "msd": "msd_tracks.csv",
    "mxm": "mxm_counts.csv",
    "tagtraum_cd2c": "tagtraum_genres.csv",
    "lastfm": "lastfm_similar",
}


def _source_sha256(path):
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(child.read_bytes())
    return digest.hexdigest()


def run(data_dir, output_path, seed=684):
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    paths = {name: data_dir / filename for name, filename in SOURCE_FILES.items()}
    duplicates_path = data_dir / "msd_duplicates.txt"
    if not paths["mxm"].exists() and (data_dir / "mxm_counts").is_dir():
        paths["mxm"] = data_dir / "mxm_counts"
    if not paths["lastfm"].exists() and (data_dir / "lastfm_similar.tsv").exists():
        paths["lastfm"] = data_dir / "lastfm_similar.tsv"
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        audit = {
            "status": "blocked_missing_sources",
            "seed": seed,
            "source_counts": {"msd": None, "mxm": None, "tagtraum_cd2c": None, "lastfm_edges_raw": None},
            "missing_sources": [
                {"source": name, "expected_path": str(paths[name])} for name in missing
            ],
            "dataset_access_notes": "See data/README.md. No source counts are inferred from missing files.",
        }
    else:
        parsed = {
            "msd": parse_msd(paths["msd"]),
            "mxm": parse_mxm(paths["mxm"]),
            "genres": parse_tagtraum(paths["tagtraum_cd2c"]),
            "edges": parse_lastfm(paths["lastfm"]),
        }
        duplicate_groups = parse_msd_duplicates(duplicates_path) if duplicates_path.exists() else []
        audit = build_audit(
            parsed["msd"], parsed["mxm"], parsed["genres"], parsed["edges"], seed,
            duplicate_groups=duplicate_groups,
        )
        audit["source_sha256"] = {
            name: _source_sha256(path) for name, path in paths.items()
        }
        if duplicates_path.exists():
            audit["source_sha256"]["msd_duplicates"] = _source_sha256(duplicates_path)
        audit["dataset_access_notes"] = "Dataset access/licensing is documented in data/README.md; source files are not bundled."
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description="Audit NeuroTune's M0 source overlap and observed edges.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/data_audit.json"))
    parser.add_argument("--seed", type=int, default=684)
    args = parser.parse_args()
    audit = run(args.data_dir, args.output, args.seed)
    print(f"M0 status: {audit['status']}")
    print(f"Audit written to {args.output}")


if __name__ == "__main__":
    main()

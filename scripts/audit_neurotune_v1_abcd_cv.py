"""Validate A-D cache provenance and publish separate artist-grouped CV labels."""

import argparse
import json
from pathlib import Path

from src.data.neurotune_v1_abcd_cv import build_abcd_cv


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = REPO_ROOT / "data/msd_hf_abcd_pilot"
DEFAULTS = {
    "audit_path": PILOT_DIR / "abcd_catalog_audit.json",
    "catalog_path": PILOT_DIR / "abcd_catalog_tracks.csv",
    "train_edges_path": PILOT_DIR / "abcd_lastfm_train_edges.csv",
    "test_edges_path": PILOT_DIR / "abcd_lastfm_test_edges.csv",
    "cache_path": PILOT_DIR / "abcd_audio_30s_26ch.h5",
    "cache_metadata_path": PILOT_DIR / "abcd_audio_30s_26ch_meta.json",
    "rejection_path": PILOT_DIR / "abcd_audio_30s_26ch_rejections.json",
    "ab_catalog_path": REPO_ROOT / "data/msd_hf_pilot/ab_catalog_tracks.csv",
    "ab_split_path": REPO_ROOT / "data/msd_hf_pilot/neurotune_v1_split.json",
    "ab_test_edges_path": REPO_ROOT / "data/msd_hf_pilot/ab_lastfm_test_edges.csv",
    "output_dir": PILOT_DIR / "cv",
}


def _summary(manifest):
    result = []
    for fold in manifest["folds"]:
        counts = fold["counts"]
        result.append({
            "fold": fold["fold"],
            "valid_tracks": {name: len(fold[f"{name}_track_ids"]) for name in ("train", "validation", "test")},
            "valid_artists": {name: len(fold[f"{name}_artist_keys"]) for name in ("train", "validation", "test")},
            "edges": {
                "train_source": {name: counts[name] for name in ("train", "validation", "test_train_source")},
                "official_test_source": counts["test_official_source"],
            },
        })
    exclusions = manifest["exclusions"]
    return {
        "folds": result,
        "exclusions": {
            "invalid_endpoint": {source: len(edges) for source, edges in exclusions["invalid_endpoint"].items()},
            "by_fold": [{
                reason: {source: len(edges) for source, edges in groups.items()}
                for reason, groups in fold.items()
            } for fold in exclusions["by_fold"]],
            "ab_test_source_policy": len(manifest["ab_test_source_policy_exclusions"]),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    options = (
        ("--audit", "audit_path"), ("--catalog", "catalog_path"),
        ("--train-edges", "train_edges_path"), ("--test-edges", "test_edges_path"),
        ("--cache", "cache_path"), ("--cache-metadata", "cache_metadata_path"),
        ("--rejections", "rejection_path"), ("--ab-catalog", "ab_catalog_path"),
        ("--ab-split", "ab_split_path"), ("--ab-test-edges", "ab_test_edges_path"),
        ("--output-dir", "output_dir"),
    )
    for option, destination in options:
        parser.add_argument(option, dest=destination, type=Path, default=DEFAULTS[destination])
    parser.add_argument("--seed", dest="seed", type=int, default=684)
    args = vars(parser.parse_args(argv))
    result = build_abcd_cv(**args)
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    print(json.dumps(_summary(manifest), sort_keys=True))


if __name__ == "__main__":
    main()

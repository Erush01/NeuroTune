"""Build the source-aware NeuroTune v1 artist-disjoint split."""

import argparse
from pathlib import Path

from src.data.neurotune_v1_split import build_split, publish_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "msd_hf_pilot"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--test-zip", type=Path, default=REPO_ROOT / "lastfm_test.zip")
    args = parser.parse_args(argv)
    data_dir = args.data_dir
    manifest, edges = build_split(
        data_dir / "ab_catalog_audit.json",
        data_dir / "ab_catalog_tracks.csv",
        data_dir / "ab_lastfm_train_edges.csv",
        data_dir / "ab_lastfm_test_edges.csv",
        args.test_zip,
        data_dir,
    )
    publish_split(manifest, edges, data_dir)
    print("NeuroTune v1 split created")
    print(f"tracks: {manifest['counts']['tracks']}")
    print(f"edges: {manifest['counts']['edges']}")
    print(f"test queries: {manifest['counts']['test_queries']}")
    print(f"cross-partition exclusions: {manifest['counts']['excluded_cross_partition_edges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the combined MSD A/B/C/D audit and compare it with the A/B pilot."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from scripts.audit_msd_hf_ab import REPO_ROOT, _run_named_shards


DEFAULTS = {
    "a_tar": REPO_ROOT / "data/msd_hf_pilot/A.tar.gz",
    "b_tar": REPO_ROOT / "data/msd_hf_pilot/B.tar.gz",
    "c_tar": REPO_ROOT / "data/msd_hf_pilot/C.tar.gz",
    "d_tar": REPO_ROOT / "data/msd_hf_pilot/D.tar.gz",
    "lyrics_dir": REPO_ROOT / "data/processed/mxm_counts",
    "genres_csv": REPO_ROOT / "data/processed/tagtraum_genres.csv",
    "duplicates_path": REPO_ROOT / "data/processed/msd_duplicates.txt",
    "lastfm_train_zip": REPO_ROOT / "lastfm_train.zip",
    "lastfm_test_zip": REPO_ROOT / "lastfm_test.zip",
    "reference_catalog": REPO_ROOT / "data/msd_hf_pilot/ab_catalog_tracks.csv",
    "reference_train_edges": REPO_ROOT / "data/msd_hf_pilot/ab_lastfm_train_edges.csv",
    "reference_test_edges": REPO_ROOT / "data/msd_hf_pilot/ab_lastfm_test_edges.csv",
    "output_dir": REPO_ROOT / "data/msd_hf_abcd_pilot",
}
LEGACY_PILOT_DIR = (REPO_ROOT / "data/msd_hf_pilot").resolve()
AUTHORIZATION = "The user confirmed project-use permission."


def run_abcd(a_tar=DEFAULTS["a_tar"], b_tar=DEFAULTS["b_tar"],
             c_tar=DEFAULTS["c_tar"], d_tar=DEFAULTS["d_tar"],
             lyrics_dir=DEFAULTS["lyrics_dir"], genres_csv=DEFAULTS["genres_csv"],
             duplicates_path=DEFAULTS["duplicates_path"],
             lastfm_train_zip=DEFAULTS["lastfm_train_zip"],
             lastfm_test_zip=DEFAULTS["lastfm_test_zip"],
             reference_catalog=DEFAULTS["reference_catalog"],
             reference_train_edges=DEFAULTS["reference_train_edges"],
             reference_test_edges=DEFAULTS["reference_test_edges"],
             output_dir=DEFAULTS["output_dir"]):
    output_path = Path(output_dir).resolve()
    if output_path == LEGACY_PILOT_DIR or LEGACY_PILOT_DIR in output_path.parents:
        raise ValueError(f"A-D outputs cannot be written inside {LEGACY_PILOT_DIR}")
    return _run_named_shards(
        OrderedDict((label, path) for label, path in (
            ("A", a_tar), ("B", b_tar), ("C", c_tar), ("D", d_tar),
        )),
        "abcd", lyrics_dir, genres_csv, duplicates_path,
        lastfm_train_zip, lastfm_test_zip, output_path,
        comparison_paths={
            "catalog": reference_catalog,
            "train_edges": reference_train_edges,
            "test_edges": reference_test_edges,
        },
        third_party_caveat=None,
        project_use_authorization=AUTHORIZATION,
        additional_code_paths=(Path(__file__),),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option, key in (
        ("--a-tar", "a_tar"), ("--b-tar", "b_tar"),
        ("--c-tar", "c_tar"), ("--d-tar", "d_tar"),
        ("--lyrics-dir", "lyrics_dir"), ("--genres", "genres_csv"),
        ("--duplicates", "duplicates_path"), ("--lastfm-train", "lastfm_train_zip"),
        ("--lastfm-test", "lastfm_test_zip"),
        ("--ab-catalog", "reference_catalog"),
        ("--ab-train-edges", "reference_train_edges"),
        ("--ab-test-edges", "reference_test_edges"), ("--output-dir", "output_dir"),
    ):
        parser.add_argument(option, dest=key, type=Path, default=DEFAULTS[key])
    args = vars(parser.parse_args())
    audit = run_abcd(**args)
    print(json.dumps(audit["counts"], sort_keys=True))


if __name__ == "__main__":
    main()

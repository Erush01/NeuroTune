"""Build the audited raw MSD A-D audio cache."""

import argparse
import json
from pathlib import Path

from src.data.preprocess_msd_hf_abcd import PILOT_DIR, REPO_ROOT, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option, dest, default in (
        ("--a-tar", "a_tar", REPO_ROOT / "data/msd_hf_pilot/A.tar.gz"),
        ("--b-tar", "b_tar", REPO_ROOT / "data/msd_hf_pilot/B.tar.gz"),
        ("--c-tar", "c_tar", REPO_ROOT / "data/msd_hf_pilot/C.tar.gz"),
        ("--d-tar", "d_tar", REPO_ROOT / "data/msd_hf_pilot/D.tar.gz"),
        ("--audit", "audit_path", PILOT_DIR / "abcd_catalog_audit.json"),
        ("--catalog", "catalog_path", PILOT_DIR / "abcd_catalog_tracks.csv"),
        ("--output-h5", "output_h5", PILOT_DIR / "abcd_audio_30s_26ch.h5"),
        ("--output-meta", "output_meta", PILOT_DIR / "abcd_audio_30s_26ch_meta.json"),
        ("--output-rejections", "output_rejections", PILOT_DIR / "abcd_audio_30s_26ch_rejections.json"),
    ):
        parser.add_argument(option, dest=dest, type=Path, default=default)
    args = vars(parser.parse_args(argv))
    result = run(**args)
    print(json.dumps({key: result[key] for key in ("track_count", "rejected_track_count", "segment_count")}, sort_keys=True))


if __name__ == "__main__":
    main()

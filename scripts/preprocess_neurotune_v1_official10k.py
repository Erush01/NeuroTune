"""Build the isolated official-10K raw audio cache."""

import argparse

from src.data.preprocess_neurotune_v1_official10k import build_official10k_audio_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="results/neurotune_v1_official10k_cv_audit.json")
    parser.add_argument("--subset-root", default="data/msd_subset_raw/MillionSongSubset")
    parser.add_argument("--output", default="data/processed/neurotune_v1_official10k_audio_30s_26ch.h5")
    parser.add_argument("--metadata", default="data/processed/neurotune_v1_official10k_audio_30s_26ch_meta.json")
    args = parser.parse_args(argv)
    result = build_official10k_audio_cache(args.audit, args.subset_root, args.output, args.metadata)
    print(f"Built {result['track_count']} tracks / {result['segment_count']} segments: {result['output_path']}")


if __name__ == "__main__":
    main()

"""Build the processed audio-sequence cache from the split audit."""

import argparse
import json
from pathlib import Path

from src.data.preprocess_audio import build_audio_sequence_cache


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "data/msd_subset_raw/MillionSongSubset"
DEFAULT_AUDIT = ROOT / "results/data_audit.json"
DEFAULT_OUTPUT = ROOT / "data/processed/audio_sequences.h5"
DEFAULT_METADATA_OUTPUT = ROOT / "data/processed/audio_sequences_meta.json"


def run(input_root=DEFAULT_INPUT_ROOT, audit=DEFAULT_AUDIT, output=DEFAULT_OUTPUT,
        metadata_output=DEFAULT_METADATA_OUTPUT, grid_size=1024):
    """Read split IDs from the audit and build the audio sequence cache."""
    audit_path = Path(audit)
    with audit_path.open(encoding="utf-8") as source:
        audit_data = json.load(source)
    split_ids = audit_data["splits"]["artist_disjoint"]["track_ids"]
    return build_audio_sequence_cache(
        input_root=input_root,
        split_ids=split_ids,
        audit_json_path=audit_path,
        hdf5_output_path=output,
        metadata_json_path=metadata_output,
        grid_size=grid_size,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--grid-size", type=int, default=1024)
    args = parser.parse_args(argv)
    return run(
        input_root=args.input_root,
        audit=args.audit,
        output=args.output,
        metadata_output=args.metadata_output,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()

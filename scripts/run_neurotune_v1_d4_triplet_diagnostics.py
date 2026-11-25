"""Run read-only, schedule-matched D0/D4 triplet diagnostics."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.neurotune_v1_triplet_diagnostics import ROOT, _run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    _run(args.data_root, args.results_root)


if __name__ == "__main__":
    main()

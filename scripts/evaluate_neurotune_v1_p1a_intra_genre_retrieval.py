"""Run the fixed P1-A intra-genre retrieval evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_neurotune_v1_p1a_intra_genre_retrieval import main


if __name__ == "__main__":
    main()

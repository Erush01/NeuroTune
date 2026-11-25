"""Run fresh paired D0/D4 modality sensitivity inference on validation only."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_d4_modality_sensitivity import main


if __name__ == "__main__":
    main()

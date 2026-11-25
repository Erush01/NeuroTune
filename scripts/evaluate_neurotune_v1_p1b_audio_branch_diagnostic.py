"""Run the fixed read-only P1-B audio branch diagnostic."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_neurotune_v1_p1b_audio_branch_diagnostic import main


if __name__ == "__main__":
    main()

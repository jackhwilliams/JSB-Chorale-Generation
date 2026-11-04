"""Edit SETTINGS, then run this file. CLI options override these defaults.
Paths are relative to the project root, regardless of IDE working directory.
"""

# ---- SETTINGS: edit these values ----
CHECKPOINT = "GRU/checkpoints/tonality/best.pt"
DATA_DIRECTORY = "GRU/artifacts/preprocessing"
SPLIT = "validation"  # Test is already observed; use validation for development.
OUTPUT_DIRECTORY = None  # Automatically choose a new GRU/evaluations/run_<id> folder.

# ---- Implementation ----
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from GRU._bootstrap import configure
configure(__file__, run_as_script=__name__ == '__main__')
from GRU.launchers import evaluate_model

if __name__ == "__main__":
    evaluate_model("gru", CHECKPOINT, DATA_DIRECTORY, SPLIT, OUTPUT_DIRECTORY)

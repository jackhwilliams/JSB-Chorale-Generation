"""Edit SETTINGS, then run this file. CLI options override these defaults.
Paths are relative to the project root, regardless of IDE working directory.
"""

# ---- SETTINGS: edit these values ----
CHECKPOINT = "MLP/checkpoints/baseline/best.pt"
DATA_DIRECTORY = "MLP/artifacts/preprocessing"
SPLIT = "validation"  # Test is already observed; use validation for development.
OUTPUT_DIRECTORY = None  # Automatically choose a new MLP/evaluations/run_<id> folder.

# ---- Implementation ----
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MLP._bootstrap import configure
configure(__file__, run_as_script=__name__ == '__main__')
from MLP.launchers import evaluate_model

if __name__ == "__main__":
    evaluate_model("mlp", CHECKPOINT, DATA_DIRECTORY, SPLIT, OUTPUT_DIRECTORY)

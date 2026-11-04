"""Edit SETTINGS, then run this file. CLI options override these defaults.
Paths are relative to the project root, regardless of IDE working directory.
"""

# ---- SETTINGS: edit these values ----
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DURATION_HIDDEN_SIZES = [64]
CHORD_EMBEDDING_DIM = 32
DURATION_EMBEDDING_DIM = 8
INPUT_DROPOUT = 0.2
RECURRENT_DROPOUT = 0.0
TONALITY_CONDITIONING = "one_hot_direct"
TRAINING = {
    "seed": 8675309,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 16,
    "max_epochs": 100,
    "patience": 12,
    "gradient_clip": 1.0,
    "device": "auto",
    "num_threads": 1,
    "overfit_updates": 500,
}
DATA_DIRECTORY = "GRU/artifacts/preprocessing"
OUTPUT_DIRECTORY = None  # Automatically create a new GRU/runs/run_<id> folder.
MODE = "train"  # Use "overfit" for a two-chorale learning check.

# ---- Implementation (no edits needed below) ----
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from GRU._bootstrap import configure
configure(__file__, run_as_script=__name__ == '__main__')
from GRU.launchers import train

if __name__ == "__main__":
    train("gru", dict(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, duration_hidden_sizes=DURATION_HIDDEN_SIZES, chord_embedding_dim=CHORD_EMBEDDING_DIM, duration_embedding_dim=DURATION_EMBEDDING_DIM, input_dropout=INPUT_DROPOUT, recurrent_dropout=RECURRENT_DROPOUT, tonality_conditioning=TONALITY_CONDITIONING), TRAINING, DATA_DIRECTORY, OUTPUT_DIRECTORY, MODE)

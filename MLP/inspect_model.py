"""Edit SETTINGS, then run this file. CLI options override these defaults.
Paths are relative to the project root, regardless of IDE working directory.
"""

# ---- SETTINGS: edit these values ----
WINDOW_SIZE = 16
HIDDEN_SIZES = [128, 64]
DURATION_HIDDEN_SIZES = [64]
CHORD_EMBEDDING_DIM = 32
DURATION_EMBEDDING_DIM = 8
DROPOUT = 0.2
DATA_DIRECTORY = "MLP/artifacts/preprocessing"

# ---- Implementation ----
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MLP._bootstrap import configure
configure(__file__, run_as_script=__name__ == '__main__')
from MLP.launchers import inspect_model

if __name__ == "__main__":
    inspect_model("mlp", dict(window_size=WINDOW_SIZE, hidden_sizes=HIDDEN_SIZES, duration_hidden_sizes=DURATION_HIDDEN_SIZES, chord_embedding_dim=CHORD_EMBEDDING_DIM, duration_embedding_dim=DURATION_EMBEDDING_DIM, dropout=DROPOUT), DATA_DIRECTORY)

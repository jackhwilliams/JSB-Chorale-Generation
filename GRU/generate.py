"""Edit SETTINGS, then run this file. CLI options override these defaults.
Paths are relative to the project root, regardless of IDE working directory.
"""

# ---- SETTINGS: edit these values ----
CHECKPOINT = "GRU/checkpoints/tonality/best.pt"
TONALITY = "major"
SEED = 8675309
CHORD_TEMPERATURE = 1.0
DURATION_TEMPERATURE = 1.0
MAX_EVENTS = 512
OUTPUT_FILE = None  # Automatically choose a new GRU/generated/run_<id>.csv file.

# ---- Implementation ----
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from GRU._bootstrap import configure
configure(__file__, run_as_script=__name__ == '__main__')
from GRU.launchers import generate

if __name__ == "__main__":
    generate("gru", CHECKPOINT, TONALITY, dict(seed=SEED, chord_temperature=CHORD_TEMPERATURE, duration_temperature=DURATION_TEMPERATURE, max_events=MAX_EVENTS), OUTPUT_FILE)

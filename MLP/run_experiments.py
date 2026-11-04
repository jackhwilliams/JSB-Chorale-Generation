"""Run the explicit experiment list (training). Edit SETTINGS, then run this file."""

# ---- SETTINGS ----
EXPERIMENTS_FILE = "MLP/configs/experiments.json"
OUTPUT_DIRECTORY = None  # Automatically choose a fresh artifact directory.
DATA_DIRECTORY = "MLP/artifacts/preprocessing"

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MLP._bootstrap import configure
configure(__file__, run_as_script=__name__ == "__main__")
from MLP.launchers import project_path, show
from MLP.experiments.runner import run_experiments
import argparse
import uuid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default=EXPERIMENTS_FILE)
    parser.add_argument("--output", default=OUTPUT_DIRECTORY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data", default=DATA_DIRECTORY)
    args = parser.parse_args()
    output = project_path(args.output or ("MLP/artifacts/experiments/run_" + uuid.uuid4().hex[:10]))
    show(dict(configuration=project_path(args.experiments), output=output))
    if not args.dry_run:
        show(run_experiments(project_path(args.experiments), project_path(args.data), output))

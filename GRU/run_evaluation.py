"""Run the full frozen-checkpoint report, including TEST evaluation. Edit SETTINGS, then run this file."""

# ---- SETTINGS ----
EVALUATION_FILE = "GRU/configs/evaluation.json"
OUTPUT_DIRECTORY = None  # Automatically choose a fresh artifact directory.

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from GRU._bootstrap import configure
configure(__file__, run_as_script=__name__ == "__main__")
from GRU.launchers import project_path, show
from GRU.evaluation.runner import evaluate
import argparse
import uuid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=EVALUATION_FILE)
    parser.add_argument("--output", default=OUTPUT_DIRECTORY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = project_path(args.output or ("GRU/artifacts/evaluation/run_" + uuid.uuid4().hex[:10]))
    show(dict(configuration=project_path(args.config), output=output))
    if not args.dry_run:
        show(evaluate(project_path(args.config), output))

"""Run directly to prepare a new derived copy. Never modifies data/."""

# ---- SETTINGS ----
SOURCE_DIRECTORY = "data/chorale chords"
OUTPUT_DIRECTORY = None  # New artifacts/preprocessing_runs/run_<id> folder.

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MLP._bootstrap import configure
configure(__file__, run_as_script=__name__ == "__main__")
from MLP.launchers import project_path, show
from MLP.data.preprocessing import preprocess
import argparse
import uuid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_DIRECTORY)
    parser.add_argument("--output", default=OUTPUT_DIRECTORY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = project_path(args.output or ("MLP/artifacts/preprocessing_runs/run_" + uuid.uuid4().hex[:10]))
    source = project_path(args.source)
    show(dict(source=source, output=output))
    if not args.dry_run:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new or empty output directory")
        metadata, chorales = preprocess(source, output)
        show(dict(chorales=len(chorales), progress_scale=metadata["progress"]["scale"]))

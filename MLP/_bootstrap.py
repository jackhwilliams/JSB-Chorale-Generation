"""Locate shared code and select the existing project environment for direct runs."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure(script, run_as_script=False):
    for path in [ROOT]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    interpreter = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if run_as_script and interpreter.exists() and Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        # subprocess quotes Windows paths with spaces correctly; os.execv on
        # Windows can split script arguments when a project path contains spaces.
        flags = ['-B'] if sys.dont_write_bytecode else []
        raise SystemExit(subprocess.call([str(interpreter), *flags, str(Path(script).resolve()), *sys.argv[1:]]))


# MLP workflow

The MLP pipeline is independent: its datasets, prediction utilities, trainer,
sampler, evaluator, and experiment runner live in this folder. Shared utilities
and frozen preprocessing are copied into each model folder deliberately.

See the [project README](../README.md) for the project overview, environment setup,
data representation, Git policy, and team development practices.

## Run

With your environment activated, run these commands from the repository root:

```sh
python -B MLP/inspect_model.py
python -B MLP/train.py --dry-run
python -B MLP/generate.py --tonality major --seed 123
python -B MLP/evaluate.py --dry-run
python -B MLP/run_experiments.py --dry-run
python -B -m unittest discover -s MLP/tests -v
```

Editable entry scripts resolve paths from the repository root even when invoked
from another working directory. They select the repository's `.venv` if present;
otherwise they use the interpreter that launched them. Package imports begin
with `MLP`, for example `from MLP.training.runner import load_checkpoint`.

Edit the settings at the top of a script, or use CLI options to override them.
Settings are local to the script you edit. `train.py` defaults to full training;
`--dry-run` previews settings without starting work. `evaluate.py` defaults to
validation. Fresh output names protect existing runs and samples.

## Files

| Path | Purpose |
|---|---|
| `model.py` | Architecture and configuration |
| `train.py`, `generate.py` | Editable training and generation entry points |
| `inspect_model.py`, `evaluate.py` | Inspect dimensions or score a checkpoint |
| `prepare_data.py` | Read source CSVs and write a fresh derived corpus |
| `run_experiments.py` | Train the configurations explicitly listed in the local manifest |
| `run_evaluation.py` | Full checkpoint report, including the already observed test split |
| `data/`, `models/` | Local dataset, feature, prediction-head, and loss utilities |
| `training/`, `generation/`, `evaluation/`, `experiments/` | Local workflow implementations |
| `configs/` | Model and workflow configs; frozen chord labels |
| `checkpoints/` | Included selected weights and training summaries |
| `artifacts/preprocessing/` | Included frozen encoded corpus and metadata |
| `tests/fixtures/` | Included regression samples and selected-asset checksums |
| `requirements-cpu.txt` | Direct dependency versions, matching the other model folder |

New `runs/`, `generated/`, `evaluations/`, and other `artifacts/` outputs are ignored
by Git. Historical `documentation/` and `reorganization/` folders may exist in a
local working copy; they are excluded and are not required by teammates or tests.

Source data lives in `../data/` and should remain unchanged. Current two-column
chord CSVs use the frozen `configs/chord_labels.json` mapping for labels; generated
three-column CSVs retain their existing format. Do not change the vocabulary or
split underneath an existing checkpoint.

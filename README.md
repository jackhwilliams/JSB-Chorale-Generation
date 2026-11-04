# Bach chorale progression generator

This project learns to generate harmonic progressions from Bach chorales using
PyTorch. Each generated event contains a Roman-numeral chord identity and its
duration. The models generate symbolic harmony, rather than four-part SATB
voicings, sheet music, or audio.

The project provides two independently editable pipelines:

| Model | How it uses history | Default configuration |
|---|---|---|
| [MLP](MLP/README.md) | A fixed window of preceding chord-duration events | 16-event window; dense layers of 128 and 64 units |
| [GRU](GRU/README.md) | Recurrent memory carried through the progression | 64 hidden units; one layer; one-hot tonality at the input and both prediction heads |

Both predict the next chord, then predict its duration conditioned on that chord.
They also receive major/minor tonality and elapsed musical time. Generation
samples from the learned distributions until an end-of-sequence token or a
configurable event limit. It applies no forced tonic ending, minimum length,
diatonic mask, or other musical correction. Requesting major or minor does not
guarantee consistent mode adherence or musical quality.

## Data and experiment design

- The harmonic corpus contains **380 chorales and 24,227 events**.
- The vocabulary contains 76 observed chord identities and 24 observed durations.
  Durations are measured in sixteenth-note units; extra tokens mark sequence
  boundaries and padding.
- A frozen, whole-chorale split contains **300 training, 40 validation, and 40 test
  chorales**. Windows never cross chorale boundaries.
- Training minimizes chord cross-entropy plus conditional-duration cross-entropy.
  The duration branch receives the true next chord during training and evaluation,
  and the sampled chord during generation.
- Validation selects checkpoints. The test set has already been evaluated for the
  original baselines; use validation for further development and avoid using the
  observed test results for tuning.

Source CSVs and original analysis tools live in `data/`. Each model keeps an
identical copy of the authoritative encoded corpus and metadata in
`artifacts/preprocessing/`. These preserve the vocabulary, split, and training-only
progress scale. The current source chord CSVs have two columns; each pipeline
recovers their existing labels from `configs/chord_labels.json` without modifying
the source files.

Selected trained checkpoints are included so teammates can generate immediately:
`MLP/checkpoints/baseline/best.pt`, `GRU/checkpoints/baseline/best.pt`, and
`GRU/checkpoints/tonality/best.pt`. GRU generation defaults to the tonality variant;
the scalar-conditioning baseline remains available. A checkpoint saves its own
architecture and vocabulary, so generation reconstructs the trained configuration.

## Set up a local environment

Use a separate virtual environment for each clone. **Python 3.11 is recommended**;
the existing local verification used Python 3.9 and PyTorch 2.8 on Windows CPU.
Requirements pin the direct dependencies, while pip resolves platform-specific
transitive dependencies. The same dependency list is kept inside each model folder.

From the repository root, create and activate an environment:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```sh
python3 -m venv .venv
source .venv/bin/activate
```

For **Windows or Linux CPU**, install the matching PyTorch wheel, then the project
requirements:

```sh
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r GRU/requirements-cpu.txt
```

For **macOS with a compatible PyTorch 2.8 wheel**, install from the standard index:

```sh
python -m pip install --upgrade pip
python -m pip install -r GRU/requirements-cpu.txt
```

PyTorch wheel availability depends on the OS, processor, and Python version;
consult the [official PyTorch 2.8 installation instructions](https://pytorch.org/get-started/previous-versions/#v280).
Use the CPU environment for regression checks. CUDA support exists in training,
but GPU behavior and macOS have not been verified locally. The included GitHub
Actions workflow is configured to test Windows and Linux with Python 3.11.

If environment activation is unavailable, replace `python` in the commands below
with `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux/macOS.
No user-specific installation path or `PYTHONPATH` setting is needed. Direct
scripts locate the repository from their own location and select its `.venv`
when present; otherwise they use the interpreter that launched them.

## Inspect, generate, train, and evaluate

Run these examples from the repository root with the environment activated.

```sh
# Inspect model dimensions without fitting.
python -B MLP/inspect_model.py
python -B GRU/inspect_model.py

# Generate from included trained checkpoints.
python -B MLP/generate.py --tonality major --seed 123
python -B GRU/generate.py --tonality minor --seed 123

# Preview training settings without writing outputs.
python -B GRU/train.py --dry-run --hidden-size 128 --num-layers 2

# Run a small, two-training-chorale diagnostic with an explicit update budget.
python -B MLP/train.py --mode overfit --overfit-updates 500

# Start a full training run only when intended.
python -B GRU/train.py --mode train

# Score the selected checkpoint on validation; this is the default subset.
python -B GRU/evaluate.py
```

Open an entry script and edit its top-level settings, or pass CLI options to
override them. Settings belong to the script you edit: changing
`inspect_model.py` does not change `train.py`. Use `--help` to see options.
**The editable `train.py` scripts default to full training**; use `--dry-run` for
inspection. New runs and generated CSV/JSON pairs receive unique output names.
Existing outputs are not overwritten.

Each model also has:

- `prepare_data.py`: read the source CSVs and create a new derived corpus outside
  `data/`; leave the frozen preprocessing copies intact.
- `run_experiments.py`: execute the explicit, model-specific experiment manifest.
  It launches training unless `--dry-run` is supplied.
- `run_evaluation.py`: reproduce a full report, including test inference and
  generated samples. This is an explicit reporting action, not a tuning step;
  use `--dry-run` to inspect its configuration.

On Windows, `GRU/Train GRU and Generate 6.cmd` launches the accompanying PowerShell
workflow: train a GRU, then generate three major and three minor samples. Its
project root is derived from the script location. Preview all seven commands with:

```powershell
& './GRU/Train GRU and Generate 6.ps1' -Preview
```

## Repository layout and collaboration

```text
GRU/                         Recurrent-model pipeline
MLP/                         Fixed-window-model pipeline
  model.py                   Architecture and model config
  train.py, generate.py      Editable entry scripts
  data/, models/             Local dataset and prediction/loss utilities
  training/, generation/     Local training and sampling implementations
  evaluation/, experiments/  Metrics, reports, and explicit experiment runner
  configs/                   Model and workflow configuration files
  checkpoints/               Selected trained weights and run summaries
  artifacts/preprocessing/   Frozen encoded inputs and metadata
  tests/fixtures/            Portable regression examples and asset checksums
data/                        Original CSVs and analysis/export tools
.github/workflows/            Automated tests on pushes and pull requests
```

Work in a branch, keep changes scoped to the intended model, and run its tests
before opening a pull request. Shared utility code is deliberately copied into
both folders so each pipeline is independent. When changing shared behavior,
review whether the equivalent change belongs in the other pipeline too.

Do not edit source CSVs or overwrite selected checkpoints as part of routine
development. Save new experiments under fresh output paths. Keep the vocabulary,
split, checkpoint architecture, and preprocessing metadata compatible. Changes to
representation, objective, or experimental budget should be discussed with the team.

Run all tests, or one model's suite:

```sh
python -B -m unittest discover -s . -p "test_*.py" -v
python -B -m unittest discover -s GRU/tests -v
python -B -m unittest discover -s MLP/tests -v
```

Tests cover data alignment, masks, gradients, checkpoint loading, deterministic
sampling, small training updates, CLI behavior, and report generation. Report
tests substitute small training-only probes for held-out inference. Regression
fixtures omit host/version metadata and allow small floating-point differences
across CPU implementations while preserving generated events and stopping behavior.

## What belongs in Git

The ignore rules include source, configs, tests, source CSVs, frozen preprocessing,
and the three selected checkpoints. Small selected checkpoints make a fresh clone
usable without retraining. Their recorded training environment is provenance,
rather than a required local path.

They exclude virtual environments, IDE settings, caches, credentials, new runs,
generated outputs, duplicate historical artifacts, migration audits, personal chat
records, and obsolete local documentation. These files remain on their owner's
machine. Publish an intentionally selected result separately when the team needs it.

The optional piano SoundFont (`data/*.sf2`) is excluded. It is not needed for model
training, generation, or tests. The original playback script additionally needs
`tinysoundfont`, an audio device, and that SoundFont; run it with `data/` as the
working directory. It is an optional legacy utility rather than part of the tested
model environment.

The standard-library MIDI exporter can be used without writing into `data/`:

```sh
mkdir exports
python -B data/chorale_to_midi.py "data/data original chorales/chorale_001.csv" -o exports/example.mid
```

Use this command in place of the older `code/` paths in the unchanged data guide.
MIDI export consumes note CSVs, not the model's Roman-numeral progression output.

Before making a public release, the team should choose a project license and
document the source and redistribution terms of the corpus. No license has been
selected on the team's behalf. Keep raw chat transcripts out of the repository;
record lasting design decisions in reviewed documentation instead.

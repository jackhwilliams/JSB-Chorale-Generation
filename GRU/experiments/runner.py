"""Resolve an explicit experiment list, then execute it sequentially."""

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from ..model import GRUConfig

from ..training.runner import TrainingConfig, fit


def resolve_experiments(path):
    """Resolve relative files and overrides into self-contained configurations."""
    path = Path(path)
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('schema_version') != 1:
        raise ValueError('Experiment list requires schema_version=1')
    base = json.loads((path.parent / document['training_config']).read_text(encoding='utf-8'))
    entries = document['experiments']
    if not isinstance(entries, list) or not entries:
        raise ValueError('Provide a nonempty explicit experiment list')
    resolved, names = [], set()
    for entry in entries:
        name, kind = entry['name'], entry['model']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in names:
            raise ValueError('Experiment names must be unique safe directory names')
        names.add(name)
        if kind != 'gru':
            raise ValueError('Experiment model must be gru')
        values = json.loads((path.parent / entry['model_config']).read_text(encoding='utf-8'))
        values.update(entry.get('model_overrides', {}))
        cls = GRUConfig
        model_config = cls(**values).to_dict()
        training = dict(base)
        training.update(entry.get('training_overrides', {}))
        training = asdict(TrainingConfig(**training))
        resolved.append({'name': name, 'model': kind, 'model_config': model_config,
                         'training_config': training})
    weights = {(e['training_config']['chord_weight'], e['training_config']['duration_weight']) for e in resolved}
    if len(weights) != 1 or next(iter(weights)) != (1.0, 1.0):
        raise ValueError('This comparison uses the approved equally weighted validation objective')
    return resolved


def comparison_row(name, folder):
    folder = Path(folder)
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    if run['mode'] != 'train' or summary['monitor_kind'] != 'validation':
        raise ValueError('Only validation-selected full training runs belong in this table')
    metrics = summary['best']
    if not all(math.isfinite(metrics[k]) for k in ['loss', 'chord_loss', 'duration_loss']):
        raise ValueError('Cannot rank nonfinite validation losses')
    return {'name': name, 'model': run['model_type'], 'seed': summary['seed'],
            'parameters': run['parameters'], 'epochs': summary['epochs'],
            'updates': summary['updates'], 'best_epoch': summary['best_epoch'],
            'elapsed_seconds': summary['elapsed_seconds'], 'stop_reason': summary['stop_reason'],
            'validation_loss': metrics['loss'], 'validation_chord_loss': metrics['chord_loss'],
            'validation_duration_loss': metrics['duration_loss'],
            'validation_chord_accuracy': metrics['chord_accuracy'],
            'validation_duration_accuracy': metrics['duration_accuracy'],
            'checkpoint': str((folder / 'best.pt').resolve())}


def save_comparison(rows, output):
    ordered = sorted(rows, key=lambda row: (row['validation_loss'], row['name']))
    ranked = [dict(rank=i, **row) for i, row in enumerate(ordered, 1)]
    output = Path(output)
    (output / 'comparison.json').write_text(json.dumps(ranked, indent=2) + '\n', encoding='utf-8')
    if ranked:
        with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ranked[0]))
            writer.writeheader()
            writer.writerows(ranked)
    lines = ['# Validation comparison', '',
             'Ranked by the sum of equally weighted chord and duration validation cross-entropies.', '',
             '| Rank | Experiment | Parameters | Best epoch | Val chord loss | Val duration loss | Val total | Chord accuracy | Duration accuracy | Seconds |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in ranked:
        lines.append(f'| {row["rank"]} | {row["name"]} | {row["parameters"]:,} | '
                     f'{row["best_epoch"]} | {row["validation_chord_loss"]:.4f} | '
                     f'{row["validation_duration_loss"]:.4f} | {row["validation_loss"]:.4f} | '
                     f'{row["validation_chord_accuracy"]:.1%} | {row["validation_duration_accuracy"]:.1%} | '
                     f'{row["elapsed_seconds"]:.1f} |')
    lines.extend(['', 'Duration accuracy is conditioned on the correct next chord. '
                  'Chord accuracy includes EOS. These are validation results, not test or generation results.',
                  'Single-seed differences do not establish an architecture-wide advantage. '
                  'Runtime includes data/model setup, training, validation, checkpoint writing and reload checks.', ''])
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return ranked


def run_experiments(manifest, data, output):
    """Preflight the list; preserve each run and stop on failure, without retries.

    All runs consume the same frozen preprocessing directory and split. No test
    evaluation, automatic search, or additional seed expansion occurs here.
    """
    resolved = resolve_experiments(manifest)
    output, data = Path(output), Path(data)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty experiment directory')
    snapshot = {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                for name in ['metadata.json', 'chorales.json']}
    output.mkdir(parents=True, exist_ok=True)
    plan = {'schema_version': 1, 'ranking': 'ascending combined validation loss',
            'data_sha256': snapshot, 'experiments': resolved}
    (output / 'resolved_experiments.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    rows = []
    for experiment in resolved:
        name = experiment['name']
        current = {filename: hashlib.sha256((data / filename).read_bytes()).hexdigest() for filename in snapshot}
        if current != snapshot:
            raise ValueError('Preprocessed data changed during the experiment suite')
        print(f'Starting experiment: {name}', flush=True)
        folder = output / name
        try:
            fit(experiment['model'], data, folder, experiment['model_config'],
                TrainingConfig(**experiment['training_config']), mode='train')
            rows.append(comparison_row(name, folder))
            save_comparison(rows, output)
        except Exception as error:
            (output / 'failure.json').write_text(json.dumps({'experiment': name,
                'error_type': type(error).__name__, 'message': str(error)}, indent=2) + '\n', encoding='utf-8')
            raise
    return save_comparison(rows, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, default=Path('GRU/configs/experiments.json'))
    parser.add_argument('--data', type=Path, default=Path('GRU/artifacts/preprocessing'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_experiments(args.experiments, args.data, args.output), indent=2))


if __name__ == '__main__':
    main()

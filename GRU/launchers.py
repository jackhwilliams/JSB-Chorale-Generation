"""Implementation behind the editable GRU/ entry scripts.

Paths in these entry points are relative to the repository, regardless of the
terminal/IDE working directory. Importing an entry script does not run a task.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import torch

from .data.datasets import load_encoded_split
from .evaluation.metrics import predictive_metrics
from .generation.runner import GenerationConfig, export_sample, generate_from_checkpoint
from .model import GRUConfig

from .training.runner import TrainingConfig, fit, load_checkpoint, make_loader, make_model

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def fresh_output(kind, action, output=None, suffix=''):
    return project_path(output) if output else ROOT / kind.upper() / action / ('run_' + uuid.uuid4().hex[:10] + suffix)


def read_json(path):
    return json.loads(project_path(path).read_text(encoding='utf-8-sig'))


def show(value):
    print(json.dumps(value, indent=2, default=str))


def _model_arguments(parser, kind):
    parser.add_argument('--model-config', '--config', type=str)
    for name in (['hidden-size', 'num-layers']):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--input-dropout', type=float)
    parser.add_argument('--recurrent-dropout', type=float)
    parser.add_argument('--tonality-conditioning', choices=['scalar', 'one_hot_direct'])
    parser.add_argument('--duration-hidden-sizes', nargs='*', type=int)
    parser.add_argument('--chord-embedding-dim', type=int)
    parser.add_argument('--duration-embedding-dim', type=int)


def _model_config(kind, defaults, args):
    cls = GRUConfig
    values = cls(**read_json(args.model_config)).to_dict() if args.model_config else dict(defaults)
    for name in cls.__dataclass_fields__:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    return cls(**values).to_dict()


def train(kind, model_parameters, training_parameters, data_directory, output_directory, mode):
    parser = argparse.ArgumentParser(description='Train using the editable settings at the top of this file.')
    _model_arguments(parser, kind)
    parser.add_argument('--training-config')
    parser.add_argument('--mode', choices=['train', 'overfit'], default=mode)
    parser.add_argument('--data', default=data_directory)
    parser.add_argument('--output', default=output_directory)
    parser.add_argument('--dry-run', action='store_true', help='Print settings without training or writing files')
    for name, value_type in [('seed', int), ('batch-size', int), ('max-epochs', int), ('patience', int),
                             ('learning-rate', float), ('weight-decay', float), ('gradient-clip', float),
                             ('overfit-updates', int), ('num-threads', int)]:
        parser.add_argument('--' + name, type=value_type)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    model_config = _model_config(kind, model_parameters, args)
    values = dict(training_parameters)
    if args.training_config:
        values.update(read_json(args.training_config))
    for name in TrainingConfig.__dataclass_fields__:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    config = TrainingConfig(**values)
    output = fresh_output(kind, 'runs', args.output)
    settings = dict(model=kind, mode=args.mode, model_config=model_config,
                    training_config=asdict(config), data=project_path(args.data), output=output)
    show(settings)
    if not args.dry_run:
        result = fit(kind, project_path(args.data), output, model_config, config, args.mode)
        show(result)
        if args.mode == 'overfit' and not result['diagnostic_passed']:
            raise SystemExit('Overfit diagnostic did not meet the reported learning criteria')


def inspect_model(kind, model_parameters, data_directory):
    parser = argparse.ArgumentParser(description='Check editable model settings without training.')
    _model_arguments(parser, kind)
    parser.add_argument('--data', default=data_directory)
    args = parser.parse_args()
    config = _model_config(kind, model_parameters, args)
    torch.set_num_threads(1)
    torch.manual_seed(8675309)
    chorales, metadata = load_encoded_split(project_path(args.data), 'train')
    loader = make_loader(kind, chorales[:2], metadata, config, 2)
    model = make_model(kind, metadata, config).eval()
    with torch.inference_mode():
        outputs = model(next(iter(loader)))
    show(dict(model=kind, config=config, input_features=model.input_dim,
              parameters=sum(p.numel() for p in model.parameters()),
              output_shapes={k: list(v.shape) for k, v in outputs.items() if isinstance(v, torch.Tensor)}))


def generate(kind, checkpoint, tonality, generation_parameters, output_file):
    parser = argparse.ArgumentParser(description='Generate using editable settings at the top of this file.')
    parser.add_argument('--checkpoint', default=checkpoint)
    parser.add_argument('--tonality', choices=['major', 'minor'], default=tonality)
    parser.add_argument('--config')
    parser.add_argument('--output', default=output_file)
    parser.add_argument('--dry-run', action='store_true')
    for name, value_type in [('seed', int), ('chord-temperature', float), ('duration-temperature', float), ('max-events', int)]:
        parser.add_argument('--' + name, type=value_type)
    args = parser.parse_args()
    values = dict(generation_parameters)
    if args.config:
        values.update(read_json(args.config))
    for name in GenerationConfig.__dataclass_fields__:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    config = GenerationConfig(**values)
    output = fresh_output(kind, 'generated', args.output, '.csv')
    show(dict(checkpoint=project_path(args.checkpoint), tonality=args.tonality,
              generation_config=asdict(config), output=output))
    if args.dry_run:
        return
    torch.set_num_threads(1)
    result = generate_from_checkpoint(project_path(args.checkpoint), args.tonality, config)
    if result['model_type'] != kind:
        raise ValueError(f'This script expects a {kind.upper()} checkpoint')
    export_sample(result, output)
    show({k: result[k] for k in ['event_count', 'total_duration', 'termination_reason', 'truncated']})


def evaluate_model(kind, checkpoint, data_directory, split, output_directory):
    parser = argparse.ArgumentParser(description='Score a checkpoint; validation is the default subset.')
    parser.add_argument('--checkpoint', default=checkpoint)
    parser.add_argument('--data', default=data_directory)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default=split)
    parser.add_argument('--output', default=output_directory)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = fresh_output(kind, 'evaluations', args.output)
    show(dict(checkpoint=project_path(args.checkpoint), data=project_path(args.data), split=args.split, output=output))
    if args.dry_run:
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory')
    torch.set_num_threads(1)
    model, saved = load_checkpoint(project_path(args.checkpoint))
    if saved['model_type'] != kind:
        raise ValueError(f'This script expects a {kind.upper()} checkpoint')
    chorales, metadata = load_encoded_split(project_path(args.data), args.split)
    if metadata != saved['metadata']:
        raise ValueError('Checkpoint and preprocessing metadata do not match')
    loader = make_loader(kind, chorales, metadata, model.config.to_dict(), 16)
    metrics = predictive_metrics(model, loader, metadata)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(dict(split=args.split, checkpoint=str(project_path(args.checkpoint)),
                                                      metrics=metrics), indent=2) + '\n', encoding='utf-8')
    show(metrics)

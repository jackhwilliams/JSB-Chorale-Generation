"""MLP trainer. CLI defaults to a two-training-chorale diagnostic."""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.datasets import MLPDataset, load_encoded_split

from ..model import MLPConfig, MLPModel
from ..models.shared import EventCrossEntropy


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 8675309
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = None  # None selects 64.
    max_epochs: int = 100
    patience: int = 12
    gradient_clip: float = 1.0
    chord_weight: float = 1.0
    duration_weight: float = 1.0
    device: str = 'auto'
    num_threads: int = 1
    overfit_updates: int = 500

    def __post_init__(self):
        for name in ['max_epochs', 'patience', 'num_threads', 'overfit_updates']:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.batch_size is not None and (isinstance(self.batch_size, bool)
                or not isinstance(self.batch_size, int) or self.batch_size <= 0):
            raise ValueError('batch_size must be a positive integer or null')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        for name in ['learning_rate', 'gradient_clip', 'weight_decay', 'chord_weight', 'duration_weight']:
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError('learning_rate and gradient_clip must be positive')
        if self.device not in ['auto', 'cpu', 'cuda']:
            raise ValueError('device must be auto, cpu, or cuda')


def seed_everything(seed, num_threads=1):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def make_model(kind, metadata, model_config):
    if kind != 'mlp':
        raise ValueError('This pipeline accepts only MLP checkpoints/configurations')
    return MLPModel(metadata, MLPConfig(**model_config))


def make_loader(kind, chorales, metadata, model_config, batch_size, shuffle=False, generator=None):
    if kind != 'mlp':
        raise ValueError('This pipeline accepts only MLP datasets')
    dataset = MLPDataset(chorales, metadata, model_config['window_size'])
    collator = None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collator, generator=generator, num_workers=0)


def move_batch(batch, device):
    # Packed sequence lengths stay on CPU; all other tensors follow the model.
    return {k: v.to(device) if isinstance(v, torch.Tensor) and k != 'lengths' else v
            for k, v in batch.items()}


def run_epoch(model, loader, criterion, device, optimizer=None, gradient_clip=1.0, max_updates=None):
    """Token-count aggregation, with the same masks in fitting and validation.

    Accuracy is teacher-forced next-event accuracy; duration uses the true chord.
    Training metrics describe predictions before each batch's parameter update.
    """
    training = optimizer is not None
    model.train(training)
    totals = {k: 0 for k in ['chord_sum', 'duration_sum', 'chord_count', 'duration_count',
                             'chord_correct', 'duration_correct']}
    updates = 0
    with torch.set_grad_enabled(training):
        for raw in loader:
            batch = move_batch(raw, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = criterion(outputs, batch)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Nonfinite event loss')
            if training:
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            totals['chord_sum'] += losses['chord_loss_sum'].detach().item()
            totals['duration_sum'] += losses['duration_loss_sum'].detach().item()
            for name in ['chord', 'duration']:
                totals[name + '_count'] += losses[name + '_count'].item()
                targets = getattr(criterion.vocabulary, name + '_targets')(batch['target_' + name + 's'])
                mask = batch[name + '_loss_mask']
                totals[name + '_correct'] += ((outputs[name + '_logits'].argmax(-1) == targets) & mask).sum().item()
            if training and max_updates is not None and updates >= max_updates:
                break
    if totals['chord_count'] == 0:
        raise ValueError('An epoch needs at least one valid chord target')
    metrics = dict(totals)
    for name in ['chord', 'duration']:
        count = totals[name + '_count']
        metrics[name + '_loss'] = totals[name + '_sum'] / max(count, 1)
        metrics[name + '_accuracy'] = totals[name + '_correct'] / max(count, 1)
    metrics['loss'] = (criterion.chord_weight * metrics['chord_loss']
                       + criterion.duration_weight * metrics['duration_loss'])
    metrics['updates'] = updates
    return metrics


def load_checkpoint(path, device='cpu'):
    """Rebuild a model from trusted local metadata/config and its saved weights."""
    saved = torch.load(path, map_location=device, weights_only=True)
    model = make_model(saved['model_type'], saved['metadata'], saved['model_config']).to(device)
    model.load_state_dict(saved['model_state'])
    model.eval()
    return model, saved


def fit(kind, data_directory, output_directory, model_config, config=None, mode='overfit'):
    """Fit one approved configuration; mode=train is an explicit full-corpus action.

    The diagnostic selects the first two saved TRAIN IDs, uses those same examples
    for monitoring, and disables dropout/weight decay. It is not validation/test
    evaluation. Never loads the test split into a model or uses it for selection.
    """
    config = config if config is not None else TrainingConfig()
    started = time.perf_counter()
    if mode not in ['overfit', 'train']:
        raise ValueError('mode must be overfit or train')
    if kind != 'mlp':
        raise ValueError('model must be mlp')
    model_config = (MLPConfig(**model_config)).to_dict()
    training, metadata = load_encoded_split(data_directory, 'train')
    if mode == 'overfit':
        training = training[:2]
        if len(training) != 2:
            raise ValueError('Overfit diagnostic requires two training chorales')
        validation = training
        model_config = dict(model_config)
        for key in ['dropout', 'input_dropout', 'recurrent_dropout']:
            if key in model_config:
                model_config[key] = 0.0
        config = replace(config, weight_decay=0.0)
    else:
        validation, _ = load_encoded_split(data_directory, 'validation')
    seed_everything(config.seed, config.num_threads)
    device = torch.device('cuda' if config.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if config.device == 'auto' else config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    model = make_model(kind, metadata, model_config).to(device)
    model_config = model.config.to_dict()
    batch_size = config.batch_size or (64)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = make_loader(kind, training, metadata, model_config, batch_size, True, generator)
    monitor_loader = make_loader(kind, validation, metadata, model_config, batch_size)
    criterion = EventCrossEntropy(metadata, config.chord_weight, config.duration_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty run directory; existing results will not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    training_settings = asdict(config)
    training_settings['batch_size'] = batch_size
    run = {'model_type': kind, 'model_config': model_config, 'training_config': training_settings,
           'mode': mode, 'training_ids': [c['chorale_id'] for c in training],
           'monitor_ids': [c['chorale_id'] for c in validation],
           'monitor_kind': 'same_training_examples' if mode == 'overfit' else 'validation',
           'metadata': metadata, 'device': str(device),
           'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'numpy': str(np.__version__), 'platform': platform.platform()},
           'parameters': sum(p.numel() for p in model.parameters()),
           'encoded_corpus_sha256': hashlib.sha256((Path(data_directory) / 'chorales.json').read_bytes()).hexdigest()}
    (output / 'run.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')
    initial = run_epoch(model, monitor_loader, criterion, device)
    best, stale, steps, history = float('inf'), 0, 0, []
    epoch_budget = config.overfit_updates if mode == 'overfit' else config.max_epochs

    def checkpoint(epoch, monitor):
        # Lists/native scalars keep the checkpoint compatible with weights_only.
        np_state = np.random.get_state()
        return {**run, 'epoch': epoch, 'updates': steps, 'monitor_metrics': monitor,
                'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'torch_rng_state': torch.get_rng_state(), 'python_rng_state': random.getstate(),
                'numpy_rng_state': [np_state[0], np_state[1].tolist(), int(np_state[2]),
                                    int(np_state[3]), float(np_state[4])],
                'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'loader_rng_state': generator.get_state(), 'best_loss': best, 'stale_epochs': stale}

    for epoch in range(1, epoch_budget + 1):
        epoch_started = time.perf_counter()
        remaining = config.overfit_updates - steps if mode == 'overfit' else None
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer,
                                  config.gradient_clip, remaining)
        steps += train_metrics['updates']
        monitor = run_epoch(model, monitor_loader, criterion, device)
        improved = monitor['loss'] < best
        if improved:
            best, stale = monitor['loss'], 0
            torch.save(checkpoint(epoch, monitor), output / 'best.pt')
        else:
            stale += 1
        record = {'epoch': epoch, 'updates': steps, 'training': train_metrics,
                  'monitor': monitor, 'improved': improved,
                  'elapsed_seconds': time.perf_counter() - epoch_started}
        history.append(record)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        if mode == 'train' or epoch == 1 or epoch % 25 == 0:
            print(f'{kind} epoch={epoch} updates={steps} monitor_loss={monitor["loss"]:.4f} '
                  f'chord_acc={monitor["chord_accuracy"]:.3f} duration_acc={monitor["duration_accuracy"]:.3f}', flush=True)
        if mode == 'overfit' and steps >= config.overfit_updates:
            break
        if mode == 'train' and stale >= config.patience:
            break
    torch.save(checkpoint(epoch, monitor), output / 'last.pt')
    best_model, saved = load_checkpoint(output / 'best.pt', device)
    final = run_epoch(best_model, monitor_loader, criterion, device)
    # Verify the selected checkpoint reproduces its saved monitoring loss.
    if not math.isclose(final['loss'], saved['monitor_metrics']['loss'], rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError('Reloaded checkpoint does not reproduce monitoring loss')
    summary = {'model_type': kind, 'mode': mode, 'seed': config.seed,
               'training_ids': run['training_ids'], 'monitor_kind': run['monitor_kind'],
               'epochs': epoch, 'updates': steps, 'best_epoch': saved['epoch'],
               'stop_reason': 'update_budget' if mode == 'overfit' else
                              'early_stopping' if stale >= config.patience else 'epoch_budget',
               'initial': initial, 'best': final,
               'loss_reduction_fraction': 1 - final['loss'] / initial['loss'],
               'checkpoint_reload_verified': True,
               'elapsed_seconds': time.perf_counter() - started}
    if mode == 'overfit':
        summary['diagnostic_passed'] = (summary['loss_reduction_fraction'] >= .9
                                         and final['chord_accuracy'] >= .95
                                         and final['duration_accuracy'] >= .95)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['mlp'], default='mlp')
    parser.add_argument('--model-config', type=Path)
    parser.add_argument('--training-config', type=Path)
    parser.add_argument('--mode', choices=['overfit', 'train'], default='overfit')
    parser.add_argument('--data', type=Path, default=Path('MLP/artifacts/preprocessing'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    cls = MLPConfig
    model_config = (cls.from_json(args.model_config) if args.model_config else cls()).to_dict()
    settings = json.loads(args.training_config.read_text(encoding='utf-8')) if args.training_config else {}
    if args.seed is not None:
        settings['seed'] = args.seed
    result = fit(args.model, args.data, args.output, model_config, TrainingConfig(**settings), args.mode)
    print(json.dumps(result, indent=2))
    if args.mode == 'overfit' and not result['diagnostic_passed']:
        raise SystemExit('Overfit diagnostic did not meet the reported learning criteria')


if __name__ == '__main__':
    main()

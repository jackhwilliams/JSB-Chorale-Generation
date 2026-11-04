"""Generate from BOS using a learned EOS, without musical restrictions."""

import argparse
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import torch

from ..data.preprocessing import decode_event
from ..model import MLPModel

from ..training.runner import load_checkpoint


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 8675309
    chord_temperature: float = 1.0
    duration_temperature: float = 1.0
    max_events: int = 512

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if isinstance(self.max_events, bool) or not isinstance(self.max_events, int) or self.max_events <= 0:
            raise ValueError('max_events must be a positive integer')
        for name in ['chord_temperature', 'duration_temperature']:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')


def sample_class(logits, temperature, generator):
    """Sample all output categories, using a dedicated CPU generator.

    Shift before temperature scaling for stability, even at very small positive
    temperatures. Return the sampled class and its temperature-adjusted chance.
    No top-k/p, frequency masking, or musical restrictions are applied.
    """
    scores = logits.detach().to(device='cpu', dtype=torch.float64)
    if scores.ndim != 1 or scores.numel() == 0 or not torch.isfinite(scores).all():
        raise ValueError('Sampling requires a finite nonempty logit vector')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    probabilities = torch.softmax((scores - scores.max()) / temperature, dim=-1)
    selected = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return selected, float(probabilities[selected].item())


def generate(model, metadata, tonality, config=None):
    """Generate one in-memory sample, preserving early/empty EOS results.

    Progress is the sum of durations already generated divided by the saved
    training scale. The window contains BOS once and rolls over prior events.
    The model runs in eval/inference mode; its previous train/eval mode is restored.
    """
    config = config if config is not None else GenerationConfig()
    if tonality not in metadata['tonality_by_token']:
        raise ValueError('Tonality must be major or minor')
    if not isinstance(model, MLPModel):
        raise ValueError('Expected a MLP model')
    kind = 'mlp'
    device = next(model.parameters()).device
    generator = torch.Generator(device='cpu').manual_seed(config.seed)
    special = metadata['chord_special_tokens']
    placeholder = metadata['duration_placeholder_token']
    scale = metadata['progress']['scale']
    if scale <= 0:
        raise ValueError('Progress scale must be positive')
    chords, durations = [special['BOS']], [placeholder]
    events, trace, hidden, elapsed = [], [], None, 0
    mode = torch.tensor([metadata['tonality_by_token'].index(tonality)], dtype=torch.long, device=device)
    termination = 'max_events'
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(config.max_events):
                progress = torch.tensor([elapsed / scale], dtype=torch.float32, device=device)
                window = model.config.window_size
                valid = min(len(chords), window)
                batch = {
                    'input_chords': torch.tensor([[special['PAD']] * (window - valid) + chords[-valid:]],
                                                 dtype=torch.long, device=device),
                    'input_durations': torch.tensor([[placeholder] * (window - valid) + durations[-valid:]],
                                                    dtype=torch.long, device=device),
                    'context_mask': torch.tensor([[False] * (window - valid) + [True] * valid], device=device),
                    'tonality': mode, 'progress': progress}
                context = model.encode_context(batch)
                head_conditioning = {'tonality': mode} if model.components.direct_tonality else {}
                chord_logits = model.components.predict_chords(context, **head_conditioning)[0]
                chord_class, chord_probability = sample_class(chord_logits, config.chord_temperature, generator)
                chord_token = int(model.components.vocabulary.chord_tokens(
                    torch.tensor(chord_class, device=device)).item())
                step = {'prediction_index': len(events) + 1, 'elapsed_duration': elapsed,
                        'progress': elapsed / scale, 'chord_class': chord_class,
                        'chord_token': chord_token, 'chord_probability': chord_probability}
                if chord_token == special['EOS']:
                    step['eos'] = True
                    trace.append(step)
                    termination = 'eos'
                    break
                next_chord = torch.tensor([chord_token], dtype=torch.long, device=device)
                duration_logits, valid_duration = model.components.predict_durations(context, next_chord, **head_conditioning)
                if not valid_duration.item():
                    raise ValueError('Sampled a nonmusical chord other than EOS')
                duration_class, duration_probability = sample_class(duration_logits[0], config.duration_temperature, generator)
                duration_token = int(model.components.vocabulary.duration_tokens(
                    torch.tensor(duration_class, device=device)).item())
                event = decode_event(chord_token, duration_token, metadata)
                events.append(asdict(event))
                chords.append(chord_token)
                durations.append(duration_token)
                step.update({'eos': False, 'duration_class': duration_class,
                             'duration_token': duration_token, 'duration_probability': duration_probability})
                trace.append(step)
                elapsed += event.duration
    finally:
        model.train(was_training)
    return {'schema_version': 1, 'model_type': kind, 'model_config': model.config.to_dict(),
            'tonality': tonality, 'generation_config': asdict(config),
            'termination_reason': termination, 'truncated': termination != 'eos',
            'event_count': len(events), 'total_duration': elapsed, 'events': events,
            'trace': trace, 'progress_settings': metadata['progress'],
            'versions': {'python': platform.python_version(), 'torch': str(torch.__version__)},
            'device': str(device)}


def export_sample(result, csv_path):
    """Write compatible musical-event CSV and an auditable JSON sidecar.

    Empty EOS samples deliberately contain metadata/header only. All provenance
    and truncation information lives in the sidecar, never as fake musical rows.
    """
    csv_path = Path(csv_path)
    if csv_path.suffix.lower() != '.csv':
        raise ValueError('Output must have a .csv extension')
    json_path = csv_path.with_suffix('.json')
    if csv_path.exists() or json_path.exists():
        raise ValueError('Refusing to overwrite an existing sample or sidecar')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([result['tonality']])
        writer.writerow(['chord_index', 'duration', 'chord_label'])
        for event in result['events']:
            writer.writerow([event['chord_index'], event['duration'], event['chord_label']])
    with json_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return csv_path, json_path


def generate_from_checkpoint(checkpoint, tonality, config=None):
    checkpoint = Path(checkpoint)
    # CPU is the verified reproducible inference environment for this phase.
    model, saved = load_checkpoint(checkpoint, device='cpu')
    result = generate(model, saved['metadata'], tonality, config)
    result['checkpoint'] = {'path': str(checkpoint.resolve()),
                            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            'epoch': saved['epoch'], 'training_seed': saved['training_config']['seed'],
                            'monitor_kind': saved['monitor_kind']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tonality', choices=['major', 'minor'], required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--chord-temperature', type=float)
    parser.add_argument('--duration-temperature', type=float)
    parser.add_argument('--max-events', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    for name in ['seed', 'chord_temperature', 'duration_temperature', 'max_events']:
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    torch.set_num_threads(1)
    result = generate_from_checkpoint(args.checkpoint, args.tonality, GenerationConfig(**values))
    export_sample(result, args.output)
    print(json.dumps({k: result[k] for k in ['model_type', 'tonality', 'event_count',
                                            'total_duration', 'termination_reason', 'truncated']}, indent=2))


if __name__ == '__main__':
    main()

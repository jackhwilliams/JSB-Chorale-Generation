"""Configurable fixed-window MLP. Running this module inspects a batch only."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
from MLP._bootstrap import configure as _configure
_configure(__file__, run_as_script=__name__ == '__main__')

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn

from MLP.models.shared import FeatureConfig, SharedPredictionComponents


@dataclass(frozen=True)
class MLPConfig:
    """The three main controls accept an int and two lists/tuples of ints.

    hidden_sizes defines the shared backbone on the chord prediction path.
    duration_hidden_sizes defines additional layers in the duration branch.
    Internally lists are copied to tuples so later edits cannot change a model's
    recorded configuration. to_dict() converts them back to JSON-friendly lists.
    """

    window_size: int = 16
    hidden_sizes: tuple = (128, 64)
    duration_hidden_sizes: tuple = (64,)
    chord_embedding_dim: int = 32
    duration_embedding_dim: int = 8
    dropout: float = 0.2

    def __post_init__(self):
        if isinstance(self.window_size, bool) or not isinstance(self.window_size, int) or self.window_size <= 0:
            raise ValueError('window_size must be a positive integer')
        for name in ['hidden_sizes', 'duration_hidden_sizes']:
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)):
                raise ValueError(f'{name} must be a list or tuple of positive integers')
            if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
                raise ValueError(f'{name} must contain positive integers')
            object.__setattr__(self, name, tuple(values))
        if not self.hidden_sizes:
            raise ValueError('hidden_sizes must contain at least one hidden layer')
        FeatureConfig(self.chord_embedding_dim, self.duration_embedding_dim, self.duration_hidden_sizes)
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError('dropout must be a finite number in [0, 1)')

    def to_dict(self):
        result = asdict(self)
        result['hidden_sizes'] = list(self.hidden_sizes)
        result['duration_hidden_sizes'] = list(self.duration_hidden_sizes)
        return result

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


class MLPModel(nn.Module):
    """[B,W] events -> [B,H] context -> 77 chord / 24 duration logits.

    Single examples [W] are supported too. Prediction conditioning is supplied
    once per window. Training forward(batch) uses the batch's true target chord
    for duration prediction; it never supplies that target to the backbone or
    chord head. For future generation, encode_context(batch), predict a chord
    through components, then condition duration on the selected chord token.
    """

    def __init__(self, metadata, config=None):
        super().__init__()
        self.config = config if config is not None else MLPConfig()
        feature_config = FeatureConfig(self.config.chord_embedding_dim,
                                       self.config.duration_embedding_dim,
                                       self.config.duration_hidden_sizes)
        self.components = SharedPredictionComponents(self.config.hidden_sizes[-1], metadata, feature_config)
        encoder = self.components.encoder
        self.input_dim = self.config.window_size * encoder.output_dim + encoder.conditioning_dim
        layers = []
        previous = self.input_dim
        for width in self.config.hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(self.config.dropout)])
            previous = width
        self.backbone = nn.Sequential(*layers)

    def encode_context(self, batch):
        chords, durations = batch['input_chords'], batch['input_durations']
        if chords.ndim not in (1, 2) or chords.shape[-1] != self.config.window_size:
            raise ValueError(f'Expected [W] or [B,W] with W={self.config.window_size}')
        if chords.shape != durations.shape:
            raise ValueError('Chord and duration input shapes must match')
        mask = batch['context_mask']
        if mask.shape != chords.shape or mask.dtype != torch.bool:
            raise ValueError('context_mask must be boolean with the input shape')
        encoder = self.components.encoder
        events = encoder(chords, durations) * mask.unsqueeze(-1)
        conditioning = encoder.conditioning(batch['tonality'], batch['progress'])
        if conditioning.shape[:-1] != chords.shape[:-1]:
            raise ValueError('Tonality and progress require one value per window')
        flattened = events.flatten(start_dim=-2)
        return self.backbone(torch.cat((flattened, conditioning), dim=-1))

    def forward(self, batch, next_chord_tokens=None):
        context = self.encode_context(batch)
        if next_chord_tokens is None:
            if 'target_chords' not in batch:
                raise ValueError('Supply next_chord_tokens or a training batch with target_chords')
            next_chord_tokens = batch['target_chords']
        return self.components(context, next_chord_tokens)


def main():
    """Display configuration/shapes for one training batch; no fitting or writes."""
    from torch.utils.data import DataLoader
    from MLP.data.datasets import MLPDataset, load_encoded_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON configuration file')
    parser.add_argument('--window-size', type=int)
    parser.add_argument('--hidden-sizes', type=int, nargs='+')
    parser.add_argument('--duration-hidden-sizes', type=int, nargs='*')
    parser.add_argument('--data', type=Path, default=_BootstrapPath(__file__).resolve().parents[1] / 'MLP/artifacts/preprocessing')
    args = parser.parse_args()
    values = (MLPConfig.from_json(args.config) if args.config else MLPConfig()).to_dict()
    for name in ['window_size', 'hidden_sizes', 'duration_hidden_sizes']:
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    config = MLPConfig(**values)
    chorales, metadata = load_encoded_split(args.data, 'train')
    dataset = MLPDataset(chorales, metadata, config.window_size)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = MLPModel(metadata, config).eval()
    with torch.no_grad():
        outputs = model(batch)
    print(json.dumps({'config': config.to_dict(), 'input_features': model.input_dim,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'input_shape': list(batch['input_chords'].shape),
                      'chord_output_shape': list(outputs['chord_logits'].shape),
                      'duration_output_shape': list(outputs['duration_logits'].shape)}, indent=2))


if __name__ == '__main__':
    main()

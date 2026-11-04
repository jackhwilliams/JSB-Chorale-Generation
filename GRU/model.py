"""Packed, full-chorale GRU with the shared chord/conditional-duration heads."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
from GRU._bootstrap import configure as _configure
_configure(__file__, run_as_script=__name__ == '__main__')

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from GRU.models.shared import FeatureConfig, SharedPredictionComponents


@dataclass(frozen=True)
class GRUConfig:
    hidden_size: int = 64
    num_layers: int = 1
    duration_hidden_sizes: tuple = (64,)
    chord_embedding_dim: int = 32
    duration_embedding_dim: int = 8
    input_dropout: float = 0.2
    recurrent_dropout: float = 0.0
    tonality_conditioning: str = 'scalar'  # Legacy checkpoint default.

    def __post_init__(self):
        if self.tonality_conditioning not in ['scalar', 'one_hot_direct']:
            raise ValueError('tonality_conditioning must be scalar or one_hot_direct')
        for name in ['hidden_size', 'num_layers']:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(self.duration_hidden_sizes, (list, tuple)):
            raise ValueError('duration_hidden_sizes must be a list or tuple of positive integers')
        object.__setattr__(self, 'duration_hidden_sizes', tuple(self.duration_hidden_sizes))
        FeatureConfig(self.chord_embedding_dim, self.duration_embedding_dim, self.duration_hidden_sizes)
        for name in ['input_dropout', 'recurrent_dropout']:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value < 1):
                raise ValueError(f'{name} must be finite and in [0, 1)')

    def to_dict(self):
        result = asdict(self)
        result['duration_hidden_sizes'] = list(self.duration_hidden_sizes)
        return result

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


class GRUModel(nn.Module):
    """[B,T] tokens -> [B,T,H] recurrent context -> shared prediction outputs.

    forward(batch) and encode_context(batch) always start from zero state unless
    explicitly supplied a state. They never store state on the model. Training
    uses complete independent chorales; optional state is for sequential inference.
    Hidden state has shape [num_layers,B,H] in the original batch order.
    tonality_conditioning='one_hot_direct' uses [major,minor] one-hot features
    at every recurrent input and directly at both prediction heads. 'scalar'
    retains the baseline architecture and is the default for old checkpoints.
    """

    def __init__(self, metadata, config=None):
        super().__init__()
        self.config = config if config is not None else GRUConfig()
        features = FeatureConfig(self.config.chord_embedding_dim,
                                 self.config.duration_embedding_dim,
                                 self.config.duration_hidden_sizes)
        stronger_mode = self.config.tonality_conditioning == 'one_hot_direct'
        self.components = SharedPredictionComponents(
            self.config.hidden_size, metadata, features,
            tonality_encoding='one_hot' if stronger_mode else 'scalar',
            direct_tonality=stronger_mode)
        self.input_dropout = nn.Dropout(self.config.input_dropout)
        self.input_dim = self.components.encoder.output_dim + self.components.encoder.conditioning_dim
        # nn.GRU's dropout is inter-layer only, so disable it for one layer.
        self.effective_recurrent_dropout = self.config.recurrent_dropout if self.config.num_layers > 1 else 0.0
        self.recurrent = nn.GRU(self.input_dim, self.config.hidden_size,
                                num_layers=self.config.num_layers, batch_first=True,
                                dropout=self.effective_recurrent_dropout)

    def _features(self, chords, durations, tonality, progress):
        encoder = self.components.encoder
        events = self.input_dropout(encoder(chords, durations))
        conditioning = encoder.conditioning(tonality, progress)
        if conditioning.shape[:-1] != events.shape[:-1]:
            raise ValueError('Conditioning must have one value per input event')
        return torch.cat((events, conditioning), dim=-1)

    def _validate_hidden(self, hidden, batch_size):
        expected = (self.config.num_layers, batch_size, self.config.hidden_size)
        if hidden is not None and hidden.shape != expected:
            raise ValueError(f'Hidden state must have shape {expected}')

    def encode_context(self, batch, hidden=None):
        """Return padded contexts [B,T,H] and final state [layers,B,H].

        Packing excludes right padding from recurrence and final state. Unpacking
        restores original batch order and zero contexts in padded positions.
        Later heads may have nonzero biases there: losses must still use masks.
        """
        chords, durations = batch['input_chords'], batch['input_durations']
        if chords.ndim != 2 or chords.shape != durations.shape or min(chords.shape) == 0:
            raise ValueError('Expected nonempty matching [B,T] chord/duration tensors')
        lengths = batch['lengths']
        if (lengths.dtype != torch.long or lengths.shape != (chords.shape[0],)
                or torch.any(lengths <= 0) or torch.any(lengths > chords.shape[1])):
            raise ValueError('lengths must be int64 [B], between 1 and T')
        self._validate_hidden(hidden, chords.shape[0])
        features = self._features(chords, durations, batch['tonality'], batch['progress'])
        packed = pack_padded_sequence(features, lengths.detach().cpu(), batch_first=True,
                                       enforce_sorted=False)
        packed_context, final_hidden = self.recurrent(packed, hidden)
        context, _ = pad_packed_sequence(packed_context, batch_first=True,
                                         total_length=chords.shape[1])
        return context, final_hidden

    def encode_step(self, chords, durations, tonality, progress, hidden=None):
        """One observed event per sequence: [B] inputs -> [B,H] context.

        Call in eval mode for inference. Caller supplies the next prediction
        boundary's progress and carries the returned state to the next step.
        This is a recurrent primitive, not a sampler or stopping policy.
        """
        if chords.ndim != 1 or chords.numel() == 0 or chords.shape != durations.shape:
            raise ValueError('Step inputs must be nonempty matching [B] tensors')
        self._validate_hidden(hidden, chords.shape[0])
        features = self._features(chords, durations, tonality, progress)
        context, final_hidden = self.recurrent(features.unsqueeze(1), hidden)
        return context[:, 0], final_hidden

    def forward(self, batch, next_chord_tokens=None, hidden=None):
        context, final_hidden = self.encode_context(batch, hidden)
        if next_chord_tokens is None:
            if 'target_chords' not in batch:
                raise ValueError('Supply next_chord_tokens or a training batch with target_chords')
            next_chord_tokens = batch['target_chords']
        if self.components.direct_tonality:
            outputs = self.components(context, next_chord_tokens, tonality=batch['tonality'])
        else:
            outputs = self.components(context, next_chord_tokens)
        outputs['hidden_state'] = final_hidden
        return outputs


def main():
    """Inspect one training batch with no fitting or file writes."""
    from torch.utils.data import DataLoader
    from GRU.data.datasets import GRUDataset, GRUCollator, load_encoded_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--num-layers', type=int)
    parser.add_argument('--duration-hidden-sizes', type=int, nargs='*')
    parser.add_argument('--input-dropout', type=float)
    parser.add_argument('--recurrent-dropout', type=float)
    parser.add_argument('--tonality-conditioning', choices=['scalar', 'one_hot_direct'])
    parser.add_argument('--data', type=Path, default=_BootstrapPath(__file__).resolve().parents[1] / 'GRU/artifacts/preprocessing')
    args = parser.parse_args()
    values = (GRUConfig.from_json(args.config) if args.config else GRUConfig()).to_dict()
    for name in ['hidden_size', 'num_layers', 'duration_hidden_sizes', 'input_dropout',
                 'recurrent_dropout', 'tonality_conditioning']:
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    config = GRUConfig(**values)
    chorales, metadata = load_encoded_split(args.data, 'train')
    batch = next(iter(DataLoader(GRUDataset(chorales, metadata), batch_size=2,
                                collate_fn=GRUCollator(metadata))))
    model = GRUModel(metadata, config).eval()
    with torch.no_grad():
        outputs = model(batch)
    print(json.dumps({'config': config.to_dict(), 'input_features': model.input_dim,
                      'effective_recurrent_dropout': model.effective_recurrent_dropout,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'lengths': batch['lengths'].tolist(),
                      'chord_output_shape': list(outputs['chord_logits'].shape),
                      'duration_output_shape': list(outputs['duration_logits'].shape),
                      'hidden_state_shape': list(outputs['hidden_state'].shape)}, indent=2))


if __name__ == '__main__':
    main()

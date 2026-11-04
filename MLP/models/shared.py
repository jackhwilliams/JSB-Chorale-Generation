"""Shared event embeddings and P(chord|context) P(duration|chord,context).

No architecture backbone is defined here. Context is [..., H], where ... can be a
batch [B] or sequence batch [B,T]. Heads return logits, never softmax values.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FeatureConfig:
    chord_embedding_dim: int = 32
    duration_embedding_dim: int = 8
    duration_hidden_sizes: tuple = (64,)

    def __post_init__(self):
        for value in (self.chord_embedding_dim, self.duration_embedding_dim,
                      *self.duration_hidden_sizes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Embedding and hidden dimensions must be positive integers')


class OutputVocabulary(nn.Module):
    """Explicit output-class/token mappings, saved as device-aware buffers.

    Input vocabularies contain BOS/PAD and a duration placeholder. Output
    vocabularies contain only musical chords plus EOS, and musical durations.
    Invalid target tokens map to -1 and must be masked before cross entropy.
    """

    def __init__(self, metadata):
        super().__init__()
        self.num_musical_chords = len(metadata['chord_index_by_token'])
        special = metadata['chord_special_tokens']
        self.eos_token = special['EOS']
        self.pad_token = special['PAD']
        self.placeholder = metadata['duration_placeholder_token']
        chords = list(range(self.num_musical_chords)) + [self.eos_token]
        durations = [i for i, duration in enumerate(metadata['duration_by_token'])
                     if duration is not None]
        self.register_buffer('chord_token_by_class', torch.tensor(chords, dtype=torch.long))
        self.register_buffer('duration_token_by_class', torch.tensor(durations, dtype=torch.long))
        chord_lookup = torch.full((max(special.values()) + 1,), -1, dtype=torch.long)
        chord_lookup[self.chord_token_by_class] = torch.arange(len(chords))
        duration_lookup = torch.full((len(metadata['duration_by_token']),), -1, dtype=torch.long)
        duration_lookup[self.duration_token_by_class] = torch.arange(len(durations))
        self.register_buffer('chord_class_by_token', chord_lookup)
        self.register_buffer('duration_class_by_token', duration_lookup)

    @staticmethod
    def lookup(tokens, table):
        if tokens.dtype != torch.long:
            raise ValueError('Token/class tensors must be int64')
        if torch.any((tokens < 0) | (tokens >= len(table))):
            raise ValueError('Token/class outside vocabulary')
        return table[tokens]

    def chord_targets(self, tokens):
        return self.lookup(tokens, self.chord_class_by_token)

    def duration_targets(self, tokens):
        return self.lookup(tokens, self.duration_class_by_token)

    def chord_tokens(self, classes):
        return self.lookup(classes, self.chord_token_by_class)

    def duration_tokens(self, classes):
        return self.lookup(classes, self.duration_token_by_class)


class EventEncoder(nn.Module):
    """Encode chord/duration inputs [...]->[..., Ec+Ed].

    Chord PAD and duration placeholder rows stay zero with no embedding gradient.
    BOS has its own learned chord vector. Conditioning is separate:
    it is appended once per window.
    """

    def __init__(self, metadata, config, tonality_encoding='scalar'):
        super().__init__()
        if tonality_encoding not in ['scalar', 'one_hot']:
            raise ValueError('tonality_encoding must be scalar or one_hot')
        self.tonality_encoding = tonality_encoding
        self.chord_embedding = nn.Embedding(
            max(metadata['chord_special_tokens'].values()) + 1,
            config.chord_embedding_dim, padding_idx=metadata['chord_special_tokens']['PAD'])
        self.duration_embedding = nn.Embedding(
            len(metadata['duration_by_token']), config.duration_embedding_dim,
            padding_idx=metadata['duration_placeholder_token'])
        self.output_dim = config.chord_embedding_dim + config.duration_embedding_dim
        self.tonality_dim = 2 if tonality_encoding == 'one_hot' else 1
        self.conditioning_dim = self.tonality_dim + 1

    def forward(self, chords, durations):
        if chords.shape != durations.shape:
            raise ValueError('Chord and duration input shapes must match')
        return torch.cat((self.chord_embedding(chords), self.duration_embedding(durations)), dim=-1)

    def conditioning(self, tonality, progress):
        """Return scalar/one-hot mode followed by unchanged elapsed progress."""
        if tonality.shape != progress.shape:
            raise ValueError('Conditioning requires matching shapes')
        if not torch.all(torch.isfinite(progress)) or torch.any(progress < 0):
            raise ValueError('Progress must be finite and nonnegative')
        return torch.cat((self.encode_tonality(tonality), progress.to(self.chord_embedding.weight.dtype).unsqueeze(-1)), dim=-1)

    def encode_tonality(self, tonality):
        if torch.any((tonality != 0) & (tonality != 1)):
            raise ValueError('Tonality must be binary: major=0, minor=1')
        dtype = self.chord_embedding.weight.dtype
        if self.tonality_encoding == 'one_hot':
            return F.one_hot(tonality.long(), num_classes=2).to(dtype)
        return tonality.to(dtype).unsqueeze(-1)


class SharedPredictionComponents(nn.Module):
    """Reusable encoders and conditional heads for any context backbone.

    Training: forward(context, target_chords) conditions duration on ground truth.
    Generation: predict_chords(context), select/decode a chord class, then call
    predict_durations(context, selected_chord_tokens). EOS/PAD never enter the
    duration network: their returned rows are zeros and explicitly masked out.
    Input and next-chord conditioning share the same learned chord embedding.
    With direct_tonality=True, both heads additionally require matching mode
    tokens, encoded by the encoder as scalar or one-hot features. The default
    scalar/no-direct path preserves the original checkpoint layout.
    """

    def __init__(self, context_dim, metadata, config=None, *,
                 tonality_encoding='scalar', direct_tonality=False):
        super().__init__()
        if isinstance(context_dim, bool) or not isinstance(context_dim, int) or context_dim <= 0:
            raise ValueError('context_dim must be a positive integer')
        self.config = config if config is not None else FeatureConfig()
        self.context_dim = context_dim
        self.encoder = EventEncoder(metadata, self.config, tonality_encoding)
        self.direct_tonality = direct_tonality
        self.head_context_dim = context_dim + (self.encoder.tonality_dim if direct_tonality else 0)
        self.vocabulary = OutputVocabulary(metadata)
        self.chord_head = nn.Linear(self.head_context_dim, len(self.vocabulary.chord_token_by_class))
        layers = []
        previous = self.head_context_dim + self.config.chord_embedding_dim
        for width in self.config.duration_hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, len(self.vocabulary.duration_token_by_class)))
        self.duration_head = nn.Sequential(*layers)

    def _head_context(self, context, tonality):
        if context.shape[-1] != self.context_dim:
            raise ValueError('Unexpected context feature dimension')
        if not self.direct_tonality:
            return context
        if tonality is None or tonality.shape != context.shape[:-1]:
            raise ValueError('Direct tonality requires one mode token per context')
        return torch.cat((context, self.encoder.encode_tonality(tonality)), dim=-1)

    def predict_chords(self, context, tonality=None):
        return self.chord_head(self._head_context(context, tonality))

    def predict_durations(self, context, next_chord_tokens, tonality=None):
        if context.shape[:-1] != next_chord_tokens.shape or context.shape[-1] != self.context_dim:
            raise ValueError('Context and next-chord shapes must align')
        classes = self.vocabulary.chord_targets(next_chord_tokens)
        allowed = (classes >= 0) | (next_chord_tokens == self.vocabulary.pad_token)
        if not torch.all(allowed):
            raise ValueError('BOS cannot be a next-chord target')
        valid = (next_chord_tokens >= 0) & (next_chord_tokens < self.vocabulary.num_musical_chords)
        # Flatten only leading batch/time dimensions; preserve arbitrary layouts.
        flat_context = self._head_context(context, tonality).reshape(-1, self.head_context_dim)
        flat_tokens = next_chord_tokens.reshape(-1)
        indices = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        logits = context.new_zeros((flat_context.shape[0], len(self.vocabulary.duration_token_by_class)))
        if indices.numel():
            features = torch.cat((flat_context[indices], self.encoder.chord_embedding(flat_tokens[indices])), dim=-1)
            logits = logits.index_copy(0, indices, self.duration_head(features))
        else:
            # Differentiable empty result, without inventing an EOS duration.
            logits = logits + flat_context.sum() * 0
        return logits.reshape(*next_chord_tokens.shape, -1), valid

    def forward(self, context, next_chord_tokens, tonality=None):
        duration_logits, valid = self.predict_durations(context, next_chord_tokens, tonality)
        return {'chord_logits': self.predict_chords(context, tonality),
                'duration_logits': duration_logits, 'duration_valid_mask': valid,
                'duration_conditioning_tokens': next_chord_tokens}


class EventCrossEntropy(nn.Module):
    """Equal-weight masked means by default, independent of batch/time layout.

    Each term is averaged over its own valid targets. This is the approved
    balanced objective, not a single per-event mean of joint log likelihood.
    Return sums/counts too, so later epoch reporting can aggregate correctly.
    """

    def __init__(self, metadata, chord_weight=1.0, duration_weight=1.0):
        super().__init__()
        if any(not math.isfinite(w) or w < 0 for w in [chord_weight, duration_weight]):
            raise ValueError('Loss weights must be finite and nonnegative')
        self.vocabulary = OutputVocabulary(metadata)
        self.chord_weight = chord_weight
        self.duration_weight = duration_weight

    @staticmethod
    def _masked(logits, targets, mask):
        if logits.shape[:-1] != targets.shape or targets.shape != mask.shape or mask.dtype != torch.bool:
            raise ValueError('Logits, targets and boolean masks must align')
        count = mask.sum()
        if torch.any(mask):
            total = F.cross_entropy(logits[mask], targets[mask], reduction='sum')
        else:
            total = logits[mask].sum()  # zero with gradient, even if excluded rows contain NaN
        return total / count.clamp_min(1), total, count

    def forward(self, outputs, batch):
        chord_targets = self.vocabulary.chord_targets(batch['target_chords'])
        duration_targets = self.vocabulary.duration_targets(batch['target_durations'])
        chord_mask, duration_mask = batch['chord_loss_mask'], batch['duration_loss_mask']
        musical = (batch['target_chords'] >= 0) & (batch['target_chords'] < self.vocabulary.num_musical_chords)
        if chord_mask.dtype != torch.bool or duration_mask.dtype != torch.bool:
            raise ValueError('Loss masks must be boolean')
        if (torch.any(chord_mask & (chord_targets < 0))
                or not torch.equal(duration_mask, chord_mask & musical)
                or torch.any(duration_mask & (duration_targets < 0))):
            raise ValueError('Invalid EOS/padding masks or musical duration targets')
        if (not torch.equal(outputs['duration_valid_mask'], musical)
                or not torch.equal(outputs['duration_conditioning_tokens'], batch['target_chords'])):
            raise ValueError('Duration head must be conditioned on these target chords')
        if (outputs['chord_logits'].shape[-1] != len(self.vocabulary.chord_token_by_class)
                or outputs['duration_logits'].shape[-1] != len(self.vocabulary.duration_token_by_class)):
            raise ValueError('Incorrect output class count')
        chord, chord_sum, chord_count = self._masked(outputs['chord_logits'], chord_targets, chord_mask)
        duration, duration_sum, duration_count = self._masked(outputs['duration_logits'], duration_targets, duration_mask)
        return {'loss': self.chord_weight * chord + self.duration_weight * duration,
                'chord_loss': chord, 'duration_loss': duration,
                'chord_loss_sum': chord_sum, 'duration_loss_sum': duration_sum,
                'chord_count': chord_count, 'duration_count': duration_count}

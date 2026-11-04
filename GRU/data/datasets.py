"""PyTorch next-event datasets over the shared, saved Phase 1 representation.

All conditioning refers to the next prediction boundary. No hidden states or
trainable encoders live here. Each item uses this model folder's independent dataset implementation.
"""

from bisect import bisect_right
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .preprocessing import decode_event


def load_encoded_split(directory, split):
    """Select saved IDs without resplitting; return (encoded chorales, metadata)."""
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    chorales = json.loads((directory / 'chorales.json').read_text(encoding='utf-8'))
    by_id = {c['chorale_id']: c for c in chorales}
    ids = [i for subset in metadata['split'].values() for i in subset]
    if (len(by_id) != len(chorales) or len(ids) != len(set(ids))
            or set(ids) != set(by_id)):
        raise ValueError('Saved split and encoded chorales must partition the same unique IDs')
    if split not in metadata['split']:
        raise ValueError(f'Unknown split: {split}')
    return tuple(by_id[i] for i in metadata['split'][split]), metadata


def _sequence(chorale, metadata):
    """Validate boundaries and derive causal progress from input durations only."""
    chords = list(chorale['chord_tokens'])
    durations = list(chorale['duration_tokens'])
    special = metadata['chord_special_tokens']
    placeholder = metadata['duration_placeholder_token']
    if (len(chords) < 3 or len(chords) != len(durations)
            or chords[0] != special['BOS'] or chords[-1] != special['EOS']
            or durations[0] != placeholder or durations[-1] != placeholder):
        raise ValueError('Expected BOS, musical events, EOS with matching durations')
    elapsed = [0]
    clock = 0
    for chord, duration in zip(chords[1:-1], durations[1:-1]):
        event = decode_event(chord, duration, metadata)
        if event is None:
            raise ValueError('Special tokens cannot occur inside a musical sequence')
        clock += event.duration
        elapsed.append(clock)
    # elapsed[t] is the time after input[t], before target[t]. This is known
    # from the past, including at EOS, without reading the target duration.
    scale = metadata['progress']['scale']
    if scale <= 0:
        raise ValueError('Progress scale must be positive')
    tonality = chorale['tonality_token']
    if not 0 <= tonality < len(metadata['tonality_by_token']):
        raise ValueError('Invalid tonality token')
    length = len(chords) - 1
    return {
        'chorale_id': chorale['chorale_id'],
        'input_chords': torch.tensor(chords[:-1], dtype=torch.long),
        'input_durations': torch.tensor(durations[:-1], dtype=torch.long),
        'tonality': torch.full((length,), tonality, dtype=torch.long),
        'progress': torch.tensor([v / scale for v in elapsed], dtype=torch.float32),
        'target_chords': torch.tensor(chords[1:], dtype=torch.long),
        'target_durations': torch.tensor(durations[1:], dtype=torch.long),
        'chord_loss_mask': torch.ones(length, dtype=torch.bool),
        'duration_loss_mask': torch.tensor([True] * (length - 1) + [False]),
    }





class GRUDataset(Dataset):
    """One full chorale per item, with [L] shifted input and target tensors.

    L is the musical-event count plus one (the EOS prediction). Use
    GRUCollator(metadata) with DataLoader for right-padded [B,T] batches.
    """

    def __init__(self, chorales, metadata):
        self.sequences = tuple(_sequence(c, metadata) for c in chorales)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        # Return fresh tensors so callers cannot change future samples in place.
        return {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in self.sequences[index].items()}


class GRUCollator:
    """Pickleable DataLoader collator; preserves order and true lengths.

    Token fields are int64, progress float32, masks bool. lengths is [B]; all
    other tensor fields are [B,T]. sequence_mask includes EOS prediction steps.
    Padded conditioning uses zero, which is ignored using sequence_mask.
    """

    def __init__(self, metadata):
        self.pad_chord = metadata['chord_special_tokens']['PAD']
        self.pad_duration = metadata['duration_placeholder_token']

    def __call__(self, items):
        if not items:
            raise ValueError('Cannot collate an empty batch')
        lengths = torch.tensor([len(item['input_chords']) for item in items], dtype=torch.long)
        batch = {'chorale_id': [item['chorale_id'] for item in items], 'lengths': lengths}
        padding = {'input_chords': self.pad_chord, 'target_chords': self.pad_chord,
                   'input_durations': self.pad_duration, 'target_durations': self.pad_duration,
                   'tonality': 0, 'progress': 0.0,
                   'chord_loss_mask': False, 'duration_loss_mask': False}
        for key, value in padding.items():
            batch[key] = pad_sequence([item[key] for item in items], batch_first=True,
                                      padding_value=value)
        batch['sequence_mask'] = torch.arange(batch['input_chords'].shape[1])[None, :] < lengths[:, None]
        return batch

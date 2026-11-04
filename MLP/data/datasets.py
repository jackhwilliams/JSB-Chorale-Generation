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


class MLPDataset(Dataset):
    """One example per musical target plus EOS, independent of window size.

    An item has input token vectors [W], context_mask [W], scalar conditioning,
    scalar targets/masks, and a chorale ID plus one-based target_index for audit.
    Default DataLoader collation produces input tensors [B,W] and targets [B].
    Only tokens strictly preceding the target enter the window. PAD fills left
    history, and the single BOS is retained only while it is within the window.
    """

    def __init__(self, chorales, metadata, window_size=16):
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError('window_size must be a positive integer')
        self.window_size = window_size
        self.pad_chord = metadata['chord_special_tokens']['PAD']
        self.pad_duration = metadata['duration_placeholder_token']
        self.sequences = tuple(_sequence(c, metadata) for c in chorales)
        self.ends = []
        total = 0
        for sequence in self.sequences:
            total += len(sequence['target_chords'])
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        chorale_index = bisect_right(self.ends, index)
        offset = index - (self.ends[chorale_index - 1] if chorale_index else 0)
        sequence = self.sequences[chorale_index]
        start = max(0, offset + 1 - self.window_size)
        count = offset + 1 - start
        chords = torch.full((self.window_size,), self.pad_chord, dtype=torch.long)
        durations = torch.full((self.window_size,), self.pad_duration, dtype=torch.long)
        context_mask = torch.zeros(self.window_size, dtype=torch.bool)
        chords[-count:] = sequence['input_chords'][start:offset + 1]
        durations[-count:] = sequence['input_durations'][start:offset + 1]
        context_mask[-count:] = True
        item = {'chorale_id': sequence['chorale_id'], 'target_index': offset + 1,
                'input_chords': chords, 'input_durations': durations,
                'context_mask': context_mask}
        for key in ['tonality', 'progress', 'target_chords', 'target_durations',
                    'chord_loss_mask', 'duration_loss_mask']:
            item[key] = sequence[key][offset].clone()
        return item







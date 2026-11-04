"""Read-only source loading and reproducible Phase 1 preprocessing.

Durations are integer sixteenth-note units. No events are merged or relabeled.
Only the explicit save function writes files; importing this module is safe.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import statistics


@dataclass(frozen=True)
class Event:
    chord_index: int
    duration: int
    chord_label: str


@dataclass(frozen=True)
class Chorale:
    chorale_id: str
    tonality: str
    events: tuple

    @property
    def total_duration(self):
        return sum(event.duration for event in self.events)


def load_chorales(source):
    """Validate the metadata/header schema and global index/label consistency."""
    chorales = []
    index_labels = {}
    for path in sorted(Path(source).glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        if len(rows) < 3 or rows[0] not in [['major'], ['minor']]:
            raise ValueError(f'{path}: invalid tonality or empty sequence')
        unlabeled = rows[1] == ['chord_index', 'duration']
        if not unlabeled and rows[1] != ['chord_index', 'duration', 'chord_label']:
            raise ValueError(f'{path}: invalid header')
        # The source copy uses two columns. Recover the existing catalog labels
        # from the frozen model vocabulary, without rewriting source CSVs.
        labels = (json.loads((Path(__file__).resolve().parents[1] / 'configs/chord_labels.json').read_text(encoding='utf-8'))
                  if unlabeled else {})
        events = []
        for line, row in enumerate(rows[2:], 3):
            if len(row) != (2 if unlabeled else 3) or any(not value or value != value.strip() for value in row):
                raise ValueError(f'{path}:{line}: malformed event')
            chord, duration = int(row[0]), int(row[1])
            if unlabeled:
                if str(chord) not in labels:
                    raise ValueError(f'{path}:{line}: chord absent from frozen vocabulary')
                row = [*row, labels[str(chord)]]
            if chord < 0 or duration <= 0:
                raise ValueError(f'{path}:{line}: invalid chord or duration')
            if chord in index_labels and index_labels[chord] != row[2]:
                raise ValueError(f'{path}:{line}: inconsistent chord label')
            index_labels[chord] = row[2]
            events.append(Event(chord, duration, row[2]))
        chorales.append(Chorale(path.stem, rows[0][0], tuple(events)))
    if not chorales:
        raise ValueError('No chorale CSVs found')
    return tuple(chorales)


def approved_split(chorales):
    """Reproduce the approved 380-file split, including the two approved swaps."""
    by_id = {c.chorale_id: c for c in chorales}
    if len(by_id) != 380 or len(chorales) != 380:
        raise ValueError('Approved split requires the inspected 380-chorale corpus')
    rng = random.Random(42)
    split = {'train': [], 'validation': [], 'test': []}
    for mode, holdout in [('major', 21), ('minor', 19)]:
        ids = sorted(c.chorale_id for c in chorales if c.tonality == mode)
        rng.shuffle(ids)
        split['validation'].extend(ids[:holdout])
        split['test'].extend(ids[holdout:2 * holdout])
        split['train'].extend(ids[2 * holdout:])
    for incoming, outgoing in [('147', '000'), ('335', '004')]:
        split['test'].remove(incoming)
        split['train'].remove(outgoing)
        split['train'].append(incoming)
        split['test'].append(outgoing)
    for ids in split.values():
        ids.sort()
    validate_split(chorales, split)
    return split


def validate_split(chorales, split):
    """Reject leakage, missing sequences, and categories absent from training."""
    by_id = {c.chorale_id: c for c in chorales}
    ids = [i for subset in split.values() for i in subset]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError('Split must partition whole chorales exactly once')
    for field in ['chord_index', 'duration']:
        observed = {getattr(e, field) for c in chorales for e in c.events}
        training = {getattr(e, field) for i in split['train'] for e in by_id[i].events}
        if observed != training:
            raise ValueError(f'Training lacks {field} categories: {sorted(observed - training)}')


def build_metadata(chorales, split):
    validate_split(chorales, split)
    labels = {e.chord_index: e.chord_label for c in chorales for e in c.events}
    original_indices = sorted(labels)
    durations = sorted({e.duration for c in chorales for e in c.events})
    training_ids = set(split['train'])
    scale = statistics.median(c.total_duration for c in chorales if c.chorale_id in training_ids)
    return {
        'schema_version': 1,
        'chord_index_by_token': original_indices,
        'chord_label_by_token': [labels[i] for i in original_indices],
        'chord_special_tokens': {name: len(original_indices) + i
                                 for i, name in enumerate(['BOS', 'EOS', 'PAD'])},
        'duration_by_token': [None] + durations,
        'duration_placeholder_token': 0,
        'tonality_by_token': ['major', 'minor'],
        'progress': {'kind': 'absolute_elapsed_duration', 'unit': 'sixteenth_note',
                     'scale': scale, 'scale_method': 'median_training_total_duration',
                     'clip': False},
        'split': split,
        'split_seed': 42,
        'split_method': 'sorted IDs; shuffle major then minor; validation, test, train',
        'approved_swaps': [{'test_to_train': '147', 'train_to_test': '000'},
                           {'test_to_train': '335', 'train_to_test': '004'}],
        'related_arrangements': 'No provenance grouping available; relationship leakage remains possible',
    }


def encode_chorale(chorale, metadata):
    """Return BOS/events/EOS and elapsed time *before* each represented event.

For shifted prediction later, use the target boundary's elapsed time: after
observing a musical event, advance the clock by its duration before predicting.
Special events have null musical durations, with a placeholder token and mask.
"""
    chord_tokens = {v: i for i, v in enumerate(metadata['chord_index_by_token'])}
    duration_tokens = {v: i for i, v in enumerate(metadata['duration_by_token']) if v is not None}
    special = metadata['chord_special_tokens']
    placeholder = metadata['duration_placeholder_token']
    chords, durations, elapsed, duration_mask = [special['BOS']], [placeholder], [0], [False]
    clock = 0
    for event in chorale.events:
        chords.append(chord_tokens[event.chord_index])
        durations.append(duration_tokens[event.duration])
        elapsed.append(clock)
        duration_mask.append(True)
        clock += event.duration
    chords.append(special['EOS'])
    durations.append(placeholder)
    elapsed.append(clock)
    duration_mask.append(False)
    return {'chorale_id': chorale.chorale_id,
            'tonality_token': metadata['tonality_by_token'].index(chorale.tonality),
            'chord_tokens': chords, 'duration_tokens': durations,
            'elapsed_duration': elapsed,
            'progress': [v / metadata['progress']['scale'] for v in elapsed],
            'duration_loss_mask': duration_mask,
            'chord_loss_mask': [False] + [True] * (len(chords) - 1),
            'total_duration': clock}


def decode_event(chord_token, duration_token, metadata):
    """Decode a musical event; boundary/padding tokens have no musical event."""
    if chord_token in metadata['chord_special_tokens'].values():
        if duration_token != metadata['duration_placeholder_token']:
            raise ValueError('Special chord tokens require placeholder duration')
        return None
    if not 0 <= chord_token < len(metadata['chord_index_by_token']):
        raise ValueError('Invalid chord token')
    if not 0 < duration_token < len(metadata['duration_by_token']):
        raise ValueError('Musical chords require a musical duration token')
    return Event(metadata['chord_index_by_token'][chord_token],
                 metadata['duration_by_token'][duration_token],
                 metadata['chord_label_by_token'][chord_token])


def padding_event(metadata):
    """Padding template; both losses must ignore padding in later datasets."""
    return {'chord_token': metadata['chord_special_tokens']['PAD'],
            'duration_token': metadata['duration_placeholder_token'],
            'chord_loss_mask': False, 'duration_loss_mask': False}


def preprocess(source, output):
    """Save reconstructible metadata and encoded data outside the source folder."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Output and source directories must be separate')
    chorales = load_chorales(source)
    metadata = build_metadata(chorales, approved_split(chorales))
    metadata['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(source.glob('*.csv'))}
    encoded = [encode_chorale(c, metadata) for c in chorales]
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('metadata.json', metadata), ('chorales.json', encoded)]:
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return metadata, encoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta, encoded = preprocess(args.source, args.output)
    print(f'Saved {len(encoded)} chorales; progress scale={meta["progress"]["scale"]}')

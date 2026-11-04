"""Metric definitions: probabilistic prediction, distributions, and exact reuse."""

from collections import Counter
from difflib import SequenceMatcher
import math
import statistics

import torch
from torch.nn import functional as F

from ..data.preprocessing import decode_event
from ..models.shared import EventCrossEntropy
from ..training.runner import move_batch


def predictive_metrics(model, loader, metadata):
    """One forward pass; aggregate overall and by mode using valid token counts.

    Joint NLL is total chord+duration NLL divided by musical-event-plus-EOS
    count. EOS has chord likelihood only. All logs are natural logarithms.
    """
    device = next(model.parameters()).device
    criterion = EventCrossEntropy(metadata).to(device)
    groups = {name: {'chord_nll_sum': 0., 'duration_nll_sum': 0., 'chord_count': 0,
                     'duration_count': 0, 'chord_correct': 0, 'duration_correct': 0}
              for name in ['overall', 'major', 'minor']}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for raw in loader:
                batch = move_batch(raw, device)
                outputs = model(batch)
                criterion(outputs, batch)  # Validate teacher forcing, shapes, masks.
                for name in ['chord', 'duration']:
                    targets = getattr(criterion.vocabulary, name + '_targets')(batch['target_' + name + 's'])
                    mask = batch[name + '_loss_mask']
                    costs = torch.zeros_like(batch['progress'], dtype=torch.float64)
                    costs[mask] = F.cross_entropy(outputs[name + '_logits'][mask], targets[mask], reduction='none').double()
                    correct = outputs[name + '_logits'].argmax(-1) == targets
                    for group, totals in groups.items():
                        subset = mask if group == 'overall' else mask & (batch['tonality'] == metadata['tonality_by_token'].index(group))
                        totals[name + '_nll_sum'] += costs[subset].sum().item()
                        totals[name + '_count'] += subset.sum().item()
                        totals[name + '_correct'] += (correct & subset).sum().item()
    finally:
        model.train(was_training)
    for totals in groups.values():
        for name in ['chord', 'duration']:
            n = totals[name + '_count']
            totals[name + '_cross_entropy'] = totals[name + '_nll_sum'] / n if n else None
            totals[name + '_accuracy'] = totals[name + '_correct'] / n if n else None
        n = totals['chord_count']
        totals['chord_perplexity'] = math.exp(totals['chord_cross_entropy']) if n else None
        totals['balanced_loss'] = (totals['chord_cross_entropy'] + totals['duration_cross_entropy']) if n and totals['duration_count'] else None
        totals['joint_nll_per_event_including_eos'] = (totals['chord_nll_sum'] + totals['duration_nll_sum']) / n if n else None
        totals['chorale_count'] = totals['chord_count'] - totals['duration_count']
        totals['mean_sequence_nll'] = ((totals['chord_nll_sum'] + totals['duration_nll_sum']) / totals['chorale_count']
                                       if totals['chorale_count'] else None)
    return groups


def decode_reference(chorales, metadata):
    result = []
    for c in chorales:
        events = [decode_event(a, b, metadata) for a, b in zip(c['chord_tokens'][1:-1], c['duration_tokens'][1:-1])]
        result.append({'chorale_id': c['chorale_id'],
                       'tonality': metadata['tonality_by_token'][c['tonality_token']],
                       'events': [dict(chord_index=e.chord_index, duration=e.duration, chord_label=e.chord_label) for e in events],
                       'event_count': len(events), 'total_duration': c['total_duration'],
                       'termination_reason': 'source', 'truncated': False})
    return result


def distribution_distance(left, right):
    """Total variation and base-2 Jensen-Shannon divergence of pooled counts.

    Both range 0..1 for nonempty distributions. Empty sides are undefined, not
    silently counted as a perfect match. This is not a significance test.
    """
    nl, nr = sum(left.values()), sum(right.values())
    if not nl or not nr:
        return {'total_variation': None, 'js_divergence_bits': None}
    variation, divergence = 0., 0.
    for key in set(left) | set(right):
        p, q = left.get(key, 0) / nl, right.get(key, 0) / nr
        midpoint = (p + q) / 2
        variation += abs(p - q)
        if p:
            divergence += .5 * p * math.log2(p / midpoint)
        if q:
            divergence += .5 * q * math.log2(q / midpoint)
    return {'total_variation': .5 * variation, 'js_divergence_bits': max(0., divergence)}


def pooled_counts(samples):
    counts = {name: Counter() for name in ['chords', 'durations', 'transitions', 'chord_durations']}
    for sample in samples:
        pairs = [(e['chord_index'], e['duration']) for e in sample['events']]
        counts['chords'].update(c for c, d in pairs)
        counts['durations'].update(d for c, d in pairs)
        counts['transitions'].update((a[0], b[0]) for a, b in zip(pairs, pairs[1:]))
        counts['chord_durations'].update(pairs)
    return counts


def longest_training_match(events, references):
    """Longest contiguous exact chord-duration match, with offsets for inspection.

    All training chorales are searched; mode of the match is reported rather
    than filtering references by mode. Common patterns are not proof of copying.
    """
    query = tuple((e['chord_index'], e['duration']) for e in events)
    best = {'length': 0, 'training_chorale_id': None, 'training_tonality': None,
            'generated_start': None, 'training_start': None}
    for reference in references:
        target = tuple((e['chord_index'], e['duration']) for e in reference['events'])
        match = SequenceMatcher(None, query, target, autojunk=False).find_longest_match(0, len(query), 0, len(target))
        if match.size > best['length']:
            best = {'length': match.size, 'training_chorale_id': reference['chorale_id'],
                    'training_tonality': reference['tonality'],
                    'generated_start': match.a, 'training_start': match.b}
    best['fraction_of_generated_events'] = best['length'] / len(query) if query else None
    return best


def sequence_statistics(sample):
    events = sample['events']
    chords = [e['chord_index'] for e in events]
    pairs = [(e['chord_index'], e['duration']) for e in events]
    bigrams = list(zip(chords, chords[1:]))
    duration = sum(e['duration'] for e in events)
    tonic = 0 if sample['tonality'] == 'major' else 1
    return {'event_count': len(events), 'total_duration': duration,
            'mean_event_duration': duration / len(events) if events else None,
            'unique_chords': len(set(chords)),
            'adjacent_chord_repetition': sum(a == b for a, b in bigrams) / len(bigrams) if bigrams else None,
            'repeated_chord_bigram_fraction': 1 - len(set(bigrams)) / len(bigrams) if bigrams else None,
            'repeated_event_fraction': 1 - len(set(pairs)) / len(pairs) if pairs else None,
            'final_mode_tonic': chords[-1] == tonic if chords else None,
            'final_either_tonic': chords[-1] in [0, 1] if chords else None,
            'final_dominant_to_mode_tonic': chords[-2] in [56, 59] and chords[-1] == tonic if len(chords) >= 2 else None,
            'diminished_event_fraction': sum('°' in e['chord_label'] for e in events) / len(events) if events else None,
            'seventh_event_fraction': sum('7' in e['chord_label'] for e in events) / len(events) if events else None}


def numeric_summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'min': None, 'max': None}
    return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)}


def describe_samples(samples):
    stats = [sequence_statistics(s) for s in samples]
    result = {'sample_count': len(samples), 'empty_count': sum(not s['events'] for s in samples),
              'eos_count': sum(s['termination_reason'] == 'eos' for s in samples),
              'truncated_count': sum(s['truncated'] for s in samples)}
    for name in stats[0] if stats else []:
        result[name] = numeric_summary([float(s[name]) if isinstance(s[name], bool) else s[name] for s in stats])
    sequences = [tuple((e['chord_index'], e['duration']) for e in s['events']) for s in samples]
    result['unique_complete_sequences'] = len(set(sequences))
    return result

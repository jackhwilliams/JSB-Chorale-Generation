"""Approved Phase 9 evaluation: frozen checkpoints, test split, 40 samples."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import torch

from ..data.datasets import load_encoded_split
from ..generation.runner import GenerationConfig, generate, export_sample
from ..training.runner import load_checkpoint, make_loader
from .metrics import (decode_reference, describe_samples, distribution_distance,
                      longest_training_match, pooled_counts, predictive_metrics,
                      sequence_statistics, numeric_summary)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def create_plots(output, predictive, generated, references, metadata):
    os.environ['MPLCONFIGDIR'] = str((output / 'plot_cache').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {'Training': '#59636e', 'GRU': '#227f9f'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = list(predictive)
    for index, kind in enumerate(names):
        metrics = predictive[kind]['overall']
        axes[0].bar(np.arange(2) + index * .35, [metrics['chord_cross_entropy'], metrics['duration_cross_entropy']],
                    .35, label=kind.upper(), color=colors[kind.upper()])
        axes[1].bar(np.arange(2) + index * .35, [metrics['chord_accuracy'], metrics['duration_accuracy']],
                    .35, label=kind.upper(), color=colors[kind.upper()])
    for ax in axes:
        ax.set_xticks(np.arange(2) + .175, ['Chord (includes EOS)', 'Duration (true chord)'])
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(ylabel='Cross-entropy (nats)', title='Held-out test loss — lower is better')
    axes[1].set(ylabel='Accuracy', ylim=(0, 1), title='Held-out test accuracy')
    fig.savefig(output / 'test_prediction.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row, mode in enumerate(['major', 'minor']):
        for col, (key, label) in enumerate([('event_count', 'Musical events'), ('total_duration', 'Duration (sixteenth-note units)')]):
            ax = axes[row, col]
            groups = {'Training': [s for s in references if s['tonality'] == mode]}
            groups.update({k.upper(): [s for s in v if s['tonality'] == mode] for k, v in generated.items()})
            for name, samples in groups.items():
                x = sorted(s[key] for s in samples)
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', label=name, color=colors[name])
            ax.set(xlabel=label, ylabel='Cumulative sample fraction', title=mode.title(), ylim=(0, 1.03))
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False)
    fig.savefig(output / 'generated_lengths.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    labels = dict(zip(metadata['chord_index_by_token'], metadata['chord_label_by_token']))
    for row, mode in enumerate(['major', 'minor']):
        train = pooled_counts([s for s in references if s['tonality'] == mode])
        for col, (key, title) in enumerate([('chords', '12 most frequent training chords'), ('durations', '8 most frequent training durations')]):
            top = [k for k, n in train[key].most_common(12 if key == 'chords' else 8)]
            groups = {'Training': train}
            groups.update({k.upper(): pooled_counts([s for s in v if s['tonality'] == mode]) for k, v in generated.items()})
            ax = axes[row, col]
            for i, (name, counts) in enumerate(groups.items()):
                total = sum(counts[key].values())
                ax.bar(np.arange(len(top)) + i * .25, [counts[key][k] / total if total else 0 for k in top],
                       width=.25, label=name, color=colors[name])
            ax.set_xticks(np.arange(len(top)) + .25,
                          [labels[k] if key == 'chords' else str(k) for k in top], rotation=45, ha='right')
            ax.set(ylabel='Pooled event fraction', title=mode.title() + ': ' + title)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False)
    fig.savefig(output / 'generated_frequencies.png', dpi=160)
    plt.close(fig)


def write_report(output, predictive, descriptive, distances, copying, generated):
    lines = ['# Phase 9 evaluation', '', 'Frozen best-validation checkpoints, seed 8675309 for training. '
             'Test evaluation is final reporting, not checkpoint selection or tuning.', '',
             '## Held-out prediction', '',
             '| Model | Subset | Chorales | Chord CE | Chord perplexity | Chord accuracy | Duration CE | Duration accuracy | Balanced loss | Joint NLL/event incl. EOS |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind, groups in predictive.items():
        for name, m in groups.items():
            lines.append(f'| {kind.upper()} | {name} | {m["chorale_count"]} | {m["chord_cross_entropy"]:.4f} | '
                         f'{m["chord_perplexity"]:.2f} | {m["chord_accuracy"]:.1%} | {m["duration_cross_entropy"]:.4f} | '
                         f'{m["duration_accuracy"]:.1%} | {m["balanced_loss"]:.4f} | {m["joint_nll_per_event_including_eos"]:.4f} |')
    lines += ['', '![Held-out prediction](test_prediction.png)', '', '## Generated behavior', '',
              '20 major and 20 minor samples per model. Major seeds: 8675309–8675328; minor seeds: 8675329–8675348. '
              'The configured seed list is preserved. Both temperatures are 1.0; the cap is 512 musical events. '
              'Every sample, including empty or truncated output, is retained.', '',
              '| Source | Tonality | Samples | Empty | EOS | Truncated | Mean events | Median duration | Final mode tonic* | Final V/V7 → mode tonic* |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for source, groups in descriptive.items():
        for mode, d in groups.items():
            fmt = lambda v: 'N/A' if v is None else f'{v:.1%}'
            lines.append(f'| {source} | {mode} | {d["sample_count"]} | {d["empty_count"]} | {d["eos_count"]} | '
                         f'{d["truncated_count"]} | {d["event_count"]["mean"]:.1f} | {d["total_duration"]["median"]:.0f} | '
                         f'{fmt(d["final_mode_tonic"]["mean"])} | {fmt(d["final_dominant_to_mode_tonic"]["mean"])} |')
    lines += ['', '*Ending statistics apply only to source sequences or nontruncated EOS samples; empty sequences '
              'are undefined and excluded from ending denominators. Two-event proxies also exclude sequences '
              'shorter than two events. A final V/V7-to-tonic identity pair is a '
              'cadence-like proxy, not a voice-leading or phrase analysis. Mode tonic is I in major and i in minor; '
              'the separate `final_either_tonic` statistic includes major-tonic endings in minor chorales.', '']
    lines += ['| Source | Tonality | Either-tonic ending | Mean adjacent chord repetition | Mean repeated bigram fraction | Unique complete sequences |',
              '|---|---|---:|---:|---:|---:|']
    for source, groups in descriptive.items():
        for mode, d in groups.items():
            fmt = lambda v: 'N/A' if v is None else f'{v:.1%}'
            lines.append(f'| {source} | {mode} | {fmt(d["final_either_tonic"]["mean"])} | '
                         f'{fmt(d["adjacent_chord_repetition"]["mean"])} | '
                         f'{fmt(d["repeated_chord_bigram_fraction"]["mean"])} | {d["unique_complete_sequences"]} |')
    lines += ['', 'Repetition fractions are computed per sequence, then averaged over defined sequences. '
              'Bigram reuse depends on sequence length; a lower fraction is not automatically better music.', '',
              '![Length distributions](generated_lengths.png)', '', '![Frequency distributions](generated_frequencies.png)', '',
              '## Distribution differences from the training corpus', '',
              '| Model | Tonality | Statistic | Total variation | JS divergence (bits) |',
              '|---|---|---|---:|---:|']
    for kind, modes in distances.items():
        for mode, categories in modes.items():
            for category, values in categories.items():
                fmt = lambda v: 'N/A' if v is None else f'{v:.4f}'
                lines.append(f'| {kind.upper()} | {mode} | {category} | {fmt(values["total_variation"])} | {fmt(values["js_divergence_bits"])} |')
    lines += ['', 'Both distances are 0 for identical nonempty distributions and at most 1. Counts are pooled '
              'over events/pairs, so longer samples contribute more. Transition pairs never cross chorale boundaries. '
              'The frequency plot displays common training categories only; distance calculations include every category.', '',
              '## Exact reuse and repetition', '',
              '| Model | Tonality | Median longest training match (events) | Maximum match | Median matched fraction |',
              '|---|---|---:|---:|---:|']
    for kind, modes in copying.items():
        for mode, c in modes.items():
            fraction = c['matched_fraction']['median']
            fraction_text = 'N/A' if fraction is None else f'{fraction:.1%}'
            lines.append(f'| {kind.upper()} | {mode} | {c["longest_match"]["median"]:.0f} | '
                         f'{c["longest_match"]["max"]} | {fraction_text} |')
    lines += ['', 'Matches are contiguous exact **chord-duration** subsequences searched across all 300 training '
              'chorales, including both modes. `sample_statistics.json` records source ID, mode, zero-based offsets, '
              'match length, and fraction for inspection. Shared harmonic patterns alone do not establish copying. '
              'Repetition metrics separately record adjacent repeated chords, repeated chord bigrams, repeated '
              'event categories, and unique complete sequences; the latter counts empty sequences as a sequence.', '',
              '## Examples for musical review', '',
              'Fixed selection rule: show the first seed for each model and tonality, without quality-based selection. '
              'CSV files contain the complete sequences; opening excerpts below show 16 events.', '']
    for kind, samples in generated.items():
        for mode in ['major', 'minor']:
            sample = next(s for s in samples if s['tonality'] == mode)
            stem = sample['sample_name']
            text = ' → '.join(e['chord_label'] + ' (' + str(e['duration']) + ')' for e in sample['events'][:16]) or '(empty EOS sample)'
            lines += [f'### {kind.upper()} {mode}', '', f'[{stem}.csv](samples/{stem}.csv)', '', text, '']
    lines += ['## Definitions and practical limits', '',
              '- Cross-entropies/NLL use natural logarithms; chord perplexity is exp(chord CE). Chord targets '
              'include EOS; duration targets exclude EOS and padding and receive the true next chord.',
              '- Balanced loss is mean chord CE + mean duration CE. Joint NLL/event is (sum chord NLL + sum '
              'musical duration NLL) / (musical events + EOS). Mean sequence NLL is also saved in JSON.',
              '- Generated durations, chords, transitions, and chord-duration frequencies are compared with '
              'training data, separately by major/minor. Generated lengths/repetition include all samples; '
              'truncations are not silently treated as musical endings.',
              '- Symbolic diminished/seventh fractions come from existing labels. No inferred functions, '
              'diatonic substitutions, extra harmonic relabeling, or composite Bach-quality score is introduced.',
              '- This is one training seed, one architecture, and 20 generated samples per mode. '
              'Generation seeds are fixed; numerical differences are descriptive, not significance tests.',
              '- The test set has now been evaluated. Do not use these results to select new configurations; '
              'any further development should use validation and treat this test evaluation as an already observed result.',
              '- Musical coherence remains a human judgment. Inspect harmonic direction, cadence plausibility, '
              'repetition, and chromatic behavior in the fixed examples; frequency matching does not establish quality.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def evaluate(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    count = config['samples_per_mode']
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('samples_per_mode must be a positive integer')
    if set(config['checkpoints']) != {'gru'}:
        raise ValueError('Provide the approved GRU checkpoint')
    GenerationConfig(seed=config['seed'] + count * 2 - 1,
                     chord_temperature=config['chord_temperature'],
                     duration_temperature=config['duration_temperature'], max_events=config['max_events'])
    GenerationConfig(seed=config['seed'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty evaluation directory')
    data = (config_path.parent / config['data']).resolve()
    training, metadata = load_encoded_split(data, 'train')
    test, test_metadata = load_encoded_split(data, 'test')
    if test_metadata != metadata or set(metadata['split']['test']) & set(metadata['split']['train']):
        raise ValueError('Inconsistent test/train manifests')
    torch.set_num_threads(1)
    models = {}
    for kind, path in config['checkpoints'].items():
        checkpoint = (config_path.parent / path).resolve()
        model, saved = load_checkpoint(checkpoint)
        if saved['metadata'] != metadata or saved['monitor_kind'] != 'validation' or saved['model_type'] != kind:
            raise ValueError('Checkpoints must match frozen preprocessing and be validation-selected')
        models[kind] = (model, saved, checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    seeds = {mode: list(range(config['seed'] + i * config['samples_per_mode'],
                             config['seed'] + (i + 1) * config['samples_per_mode']))
             for i, mode in enumerate(['major', 'minor'])}
    frozen_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [data / 'metadata.json', data / 'chorales.json']}
    provenance = {'configuration': config, 'seed_list': seeds, 'test_ids': metadata['split']['test'],
                  'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                               'matplotlib': importlib.metadata.version('matplotlib'),
                               'numpy': importlib.metadata.version('numpy')},
                  'training_reference_ids': metadata['split']['train'], 'data_sha256': frozen_hashes,
                  'checkpoints': {kind: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                        'epoch': saved['epoch'], 'training_seed': saved['training_config']['seed']}
                                  for kind, (model, saved, p) in models.items()}}
    write_json(output / 'evaluation_manifest.json', provenance)
    references = decode_reference(training, metadata)
    predictive, generated, descriptive, distances, copying, sample_stats = {}, {}, {}, {}, {}, []
    descriptive['training'] = {mode: describe_samples([s for s in references if s['tonality'] == mode]) for mode in seeds}
    for kind, (model, saved, checkpoint) in models.items():
        start = time.perf_counter()
        print(f'{kind}: evaluating {len(test)} held-out chorales', flush=True)
        loader = make_loader(kind, test, metadata, model.config.to_dict(), 16)
        predictive[kind] = predictive_metrics(model, loader, metadata)
        write_json(output / 'prediction_metrics.json', predictive)
        generated[kind] = []
        for mode, mode_seeds in seeds.items():
            for i, seed in enumerate(mode_seeds):
                settings = GenerationConfig(seed=seed, chord_temperature=config['chord_temperature'],
                                            duration_temperature=config['duration_temperature'], max_events=config['max_events'])
                sample = generate(model, metadata, mode, settings)
                sample['checkpoint'] = provenance['checkpoints'][kind]
                sample['sample_name'] = f'{kind}_{mode}_seed{seed}'
                export_sample(sample, output / 'samples' / (sample['sample_name'] + '.csv'))
                generated[kind].append(sample)
                stats = {'model': kind, 'tonality': mode, 'seed': seed, 'sample_name': sample['sample_name'],
                         'termination_reason': sample['termination_reason'], 'truncated': sample['truncated'],
                         **sequence_statistics(sample), 'training_match': longest_training_match(sample['events'], references)}
                sample_stats.append(stats)
                if (i + 1) % 5 == 0:
                    print(f'{kind} {mode}: {i + 1}/{len(mode_seeds)} samples saved', flush=True)
        descriptive[kind], distances[kind], copying[kind] = {}, {}, {}
        for mode in seeds:
            samples = [s for s in generated[kind] if s['tonality'] == mode]
            reference = [s for s in references if s['tonality'] == mode]
            descriptive[kind][mode] = describe_samples(samples)
            # Do not treat a safety-cap cutoff as a cadence or true ending.
            endings = describe_samples([s for s in samples if not s['truncated']])
            for key in ['final_mode_tonic', 'final_either_tonic', 'final_dominant_to_mode_tonic']:
                descriptive[kind][mode][key] = endings.get(key, numeric_summary([]))
            a, b = pooled_counts(samples), pooled_counts(reference)
            distances[kind][mode] = {key: distribution_distance(a[key], b[key]) for key in a}
            stats = [s for s in sample_stats if s['model'] == kind and s['tonality'] == mode]
            copying[kind][mode] = {'longest_match': numeric_summary([s['training_match']['length'] for s in stats]),
                                   'matched_fraction': numeric_summary([s['training_match']['fraction_of_generated_events'] for s in stats])}
        print(f'{kind}: evaluation/generation finished in {time.perf_counter() - start:.1f}s', flush=True)
    write_json(output / 'sample_statistics.json', sample_stats)
    write_json(output / 'generated_statistics.json', {'descriptive': descriptive, 'distribution_distances': distances, 'copying': copying})
    counts_rows = []
    for source, samples in [('training', references), *generated.items()]:
        for mode in seeds:
            counts = pooled_counts([s for s in samples if s['tonality'] == mode])
            for category, counter in counts.items():
                total = sum(counter.values())
                for key, count in sorted(counter.items()):
                    counts_rows.append({'source': source, 'tonality': mode, 'category': category,
                                        'value': json.dumps(key), 'count': count, 'fraction': count / total})
    with (output / 'frequency_counts.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source', 'tonality', 'category', 'value', 'count', 'fraction'])
        writer.writeheader()
        writer.writerows(counts_rows)
    create_plots(output, predictive, generated, references, metadata)
    write_report(output, predictive, descriptive, distances, copying, generated)
    for kind, (model, saved, path) in models.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['checkpoints'][kind]['sha256']:
            raise RuntimeError('A checkpoint changed during evaluation')
    for name, digest in frozen_hashes.items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Preprocessing changed during evaluation')
    print(f'Evaluation report saved: {output / "REPORT.md"}', flush=True)
    return predictive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('GRU/configs/evaluation.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.config, args.output)


if __name__ == '__main__':
    main()

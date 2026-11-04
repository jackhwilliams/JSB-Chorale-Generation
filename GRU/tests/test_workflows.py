import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from GRU.data.datasets import load_encoded_split
from GRU.data.preprocessing import load_chorales, encode_chorale
from GRU.experiments.runner import resolve_experiments, run_experiments
from GRU.generation.runner import GenerationConfig, generate
from GRU.training.runner import load_checkpoint

class WorkflowTests(unittest.TestCase):
    def run_script(self, script, *args):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'GRU' / script), *args],
                                    cwd=cwd, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        remaining, values = result.stdout.strip(), []
        while remaining:
            value, end = json.JSONDecoder().raw_decode(remaining)
            values.append(value)
            remaining = remaining[end:].lstrip()
        return values[-1]

    def test_settings_cli_precedence_and_arbitrary_working_directory(self):
        result = self.run_script('train.py', '--dry-run', *['--hidden-size', '12', '--num-layers', '2'], '--seed', '123')
        self.assertEqual(result['model'], 'gru')
        self.assertEqual(result['training_config']['seed'], 123)
        self.assertEqual(result['model_config']['hidden_size'], 12)
        self.assertEqual(Path(result['data']), ROOT / 'GRU/artifacts/preprocessing')
        self.assertFalse(Path(result['output']).exists())
        result = self.run_script('inspect_model.py', *['--hidden-size', '12', '--num-layers', '2'])
        self.assertEqual(result['model'], 'gru')
        self.assertEqual(result['output_shapes']['chord_logits'][-1], 77)

    def test_generation_from_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as output:
            path = Path(output) / 'sample.csv'
            summary = self.run_script('generate.py', '--max-events', '2', '--output', str(path))
            self.assertLessEqual(summary['event_count'], 2)
            saved = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(saved['model_type'], 'gru')
            self.assertIn('/GRU/checkpoints/', saved['checkpoint']['path'].replace('\\', '/'))

    def test_all_dry_runs_do_not_write_outputs(self):
        for script in ['train.py', 'generate.py', 'evaluate.py', 'prepare_data.py',
                       'run_experiments.py', 'run_evaluation.py']:
            result = self.run_script(script, '--dry-run')
            self.assertFalse(Path(result['output']).exists(), script)

    def test_evaluation_defaults_to_validation(self):
        with tempfile.TemporaryDirectory() as output:
            folder = Path(output) / 'metrics'
            result = self.run_script('evaluate.py', '--output', str(folder))
            self.assertEqual(result['overall']['chorale_count'], 40)
            self.assertEqual(json.loads((folder / 'metrics.json').read_text())['split'], 'validation')
            self.assertAlmostEqual(result['overall']['balanced_loss'], 3.1166948, places=4)

    def test_selected_assets_match_checksums(self):
        hashes = json.loads((ROOT / 'GRU/tests/fixtures/asset_hashes.json').read_text())
        for relative, expected in hashes.items():
            path = ROOT / 'GRU' / relative
            contents = path.read_bytes() if path.suffix == '.pt' else path.read_text(encoding='utf8').encode('utf8')
            self.assertEqual(hashlib.sha256(contents).hexdigest(), expected, relative)

    def test_frozen_representation_matches_current_source(self):
        source = load_chorales(ROOT / 'data/chorale chords')
        metadata = json.loads((ROOT / 'GRU/artifacts/preprocessing/metadata.json').read_text())
        saved = json.loads((ROOT / 'GRU/artifacts/preprocessing/chorales.json').read_text())
        self.assertEqual([encode_chorale(c, metadata) for c in source], saved)

    def test_exact_old_checkpoint_sampling_regression(self):
        torch.set_num_threads(1)
        references = json.loads((ROOT / 'GRU/tests/fixtures/generation_reference.json').read_text())
        for variant, reference in references.items():
            model, saved = load_checkpoint(ROOT / 'GRU/checkpoints' / variant / 'best.pt')
            for key, expected in reference.items():
                mode, seed = key.split('_')
                result = generate(model, saved['metadata'], mode, GenerationConfig(seed=int(seed)))
                self.assertEqual(result['events'], expected['events'], (variant, key))
                self.assertEqual(result['termination_reason'], expected['termination_reason'])
                self.assertEqual(len(result['trace']), len(expected['trace']))
                for actual_step, expected_step in zip(result['trace'], expected['trace']):
                    self.assertEqual(set(actual_step), set(expected_step))
                    for field, value in expected_step.items():
                        if isinstance(value, float):
                            self.assertAlmostEqual(actual_step[field], value, places=5, msg=field)
                        else:
                            self.assertEqual(actual_step[field], value, field)

    def manifest(self, root, entries=None):
        root = Path(root)
        (root / 'training.json').write_text(json.dumps({'seed': 8675309}))
        (root / 'model.json').write_text('{}')
        if entries is None:
            entries = [dict(name='first', model='gru', model_config='model.json',
                            training_overrides={'max_epochs': 2}),
                       dict(name='second', model='gru', model_config='model.json')]
        path = root / 'experiments.json'
        path.write_text(json.dumps(dict(schema_version=1, training_config='training.json', experiments=entries)))
        return path

    def test_experiments_resolve_only_local_model_and_overrides(self):
        approved = resolve_experiments(ROOT / 'GRU/configs/experiments.json')
        self.assertEqual([e['model'] for e in approved], ['gru'])
        with tempfile.TemporaryDirectory() as folder:
            entries = resolve_experiments(self.manifest(folder))
            self.assertEqual([e['training_config']['max_epochs'] for e in entries], [2, 100])
            self.assertTrue(all(e['training_config']['seed'] == 8675309 for e in entries))

    def test_experiments_reject_invalid_names_models_and_objectives(self):
        invalid = [[], [dict(name='../escape', model='gru', model_config='model.json')],
                   [dict(name='a', model='gru', model_config='model.json')] * 2,
                   [dict(name='a', model='unsupported', model_config='model.json')],
                   [dict(name='a', model='gru', model_config='model.json',
                         training_overrides={'duration_weight': 2})]]
        for entries in invalid:
            with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                resolve_experiments(self.manifest(folder, entries))

    def test_experiments_preserve_runs_and_rank_by_validation(self):
        calls = []
        def simulated_fit(kind, data, folder, model_config, config, mode):
            calls.append((kind, config.seed, mode))
            folder.mkdir()
            loss = 4. if folder.name == 'first' else 3.
            (folder / 'run.json').write_text(json.dumps(dict(mode=mode, model_type=kind, parameters=100)))
            (folder / 'summary.json').write_text(json.dumps(dict(monitor_kind='validation', seed=config.seed,
                epochs=2, updates=10, best_epoch=1, elapsed_seconds=1.2, stop_reason='epoch_budget',
                best=dict(loss=loss, chord_loss=loss-1, duration_loss=1, chord_accuracy=.5, duration_accuracy=.7))))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest(root)
            with patch('GRU.experiments.runner.fit', side_effect=simulated_fit):
                ranked = run_experiments(manifest, ROOT / 'GRU/artifacts/preprocessing', root / 'results')
            self.assertEqual(calls, [('gru', 8675309, 'train')] * 2)
            self.assertEqual([r['name'] for r in ranked], ['second', 'first'])
            self.assertEqual([r['rank'] for r in ranked], [1, 2])
            self.assertTrue((root / 'results/comparison.csv').exists())
            self.assertIn('| 1 | second |', (root / 'results/comparison.md').read_text())
            with self.assertRaises(ValueError):
                run_experiments(manifest, ROOT / 'GRU/artifacts/preprocessing', root / 'results')

    def test_experiment_failure_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest(root)
            with patch('GRU.experiments.runner.fit', side_effect=RuntimeError('test failure')) as fit:
                with self.assertRaises(RuntimeError):
                    run_experiments(manifest, ROOT / 'GRU/artifacts/preprocessing', root / 'results')
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(json.loads((root / 'results/failure.json').read_text())['experiment'], 'first')

    def test_local_pipeline_does_not_import_other_architecture(self):
        code = "import sys; from GRU import launchers; assert not any(n.split('.')[0] == 'MLP' for n in sys.modules)"
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_edited_top_settings_and_cli_precedence(self):
        import contextlib
        import io
        path = ROOT / 'GRU/train.py'
        source = path.read_text().replace('HIDDEN_SIZE = 64', 'HIDDEN_SIZE = 11')
        for arguments, expected in [(['--dry-run'], 11), (['--dry-run', '--hidden-size', '7'], 7)]:
            output = io.StringIO()
            with patch.object(sys, 'argv', [str(path), *arguments]), contextlib.redirect_stdout(output):
                exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})
            settings = json.loads(output.getvalue())
            self.assertEqual(settings['model_config']['hidden_size'], expected)
            self.assertFalse(Path(settings['output']).exists())

    def test_full_report_pipeline_using_training_probes_only(self):
        from GRU.evaluation.runner import evaluate
        from GRU.evaluation import runner
        data = ROOT / 'GRU/artifacts/preprocessing'
        chorales, metadata = load_encoded_split(data, 'train')
        refs = (chorales[0], next(c for c in chorales if c['tonality_token'] != chorales[0]['tonality_token']))
        config = dict(data=str(data), checkpoints={'gru': str(ROOT / 'GRU/checkpoints/baseline/best.pt')},
                      seed=123, samples_per_mode=1, chord_temperature=1., duration_temperature=1., max_events=2)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / 'config.json'
            path.write_text(json.dumps(config))
            with patch.object(runner, 'load_encoded_split', return_value=(refs, metadata)) as load:
                metrics = evaluate(path, folder / 'report')
            self.assertEqual([call.args[1] for call in load.call_args_list], ['train', 'test'])
            self.assertEqual(set(metrics), {'gru'})
            self.assertEqual(metrics['gru']['overall']['chorale_count'], 2)
            for name in ['REPORT.md', 'test_prediction.png', 'generated_lengths.png',
                         'generated_frequencies.png', 'frequency_counts.csv', 'evaluation_manifest.json']:
                self.assertTrue((folder / 'report' / name).is_file(), name)
            self.assertEqual(len(list((folder / 'report/samples').glob('*.csv'))), 2)
            with self.assertRaises(ValueError):
                evaluate(path, folder / 'report')

    def test_system_python_selects_existing_environment(self):
        interpreter = Path(getattr(sys, '_base_executable', sys.executable))
        if not interpreter.exists() or not (ROOT / '.venv').is_dir():
            self.skipTest('This test requires a project .venv and its base interpreter')
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([str(interpreter), '-B', str(ROOT / 'GRU/inspect_model.py')],
                                    cwd=cwd, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['model'], 'gru')

if __name__ == '__main__':
    unittest.main()

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from GRU.data.datasets import load_encoded_split
from GRU.data.preprocessing import load_chorales
from GRU.model import GRUModel, GRUConfig
from GRU.generation.runner import GenerationConfig, export_sample, generate, generate_from_checkpoint, sample_class

def forced_logits(count, selected):
    logits = torch.full((1, count), -10000.0)
    logits[0, selected] = 0
    return logits

class GenerationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (_, cls.meta) = load_encoded_split(ROOT / 'GRU/artifacts/preprocessing', 'train')

    def setUp(self):
        torch.manual_seed(17)

    def model(self, kind):
        return GRUModel(self.meta, GRUConfig(hidden_size=8, chord_embedding_dim=4, duration_embedding_dim=2))

    def test_config_and_sampling_validation(self):
        self.assertEqual(GenerationConfig(**json.loads((ROOT / 'GRU/configs/generation.json').read_text())), GenerationConfig())
        for values in [{'max_events': 0}, {'max_events': True}, {'seed': -1}, {'chord_temperature': 0}, {'duration_temperature': float('nan')}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GenerationConfig(**values)
        for logits in [torch.tensor([float('nan')]), torch.zeros(2, 3), torch.tensor([])]:
            with self.assertRaises(ValueError):
                sample_class(logits, 1, torch.Generator())
        (chosen, probability) = sample_class(torch.tensor([0.0, 1.0]), 1e-300, torch.Generator())
        self.assertEqual(chosen, 1)
        self.assertEqual(probability, 1)

    def test_temperature_controls_probabilities_without_class_masking(self):
        logits = torch.tensor([0.0, 1.0, 2.0])
        for temperature in [0.5, 1.0, 2.0]:
            (selected, chance) = sample_class(logits, temperature, torch.Generator().manual_seed(123))
            expected = (logits.double() / temperature).softmax(-1)[selected].item()
            self.assertAlmostEqual(chance, expected)

    def test_eos_at_start_skips_duration_and_preserves_empty_sample(self):
        for kind in ['gru']:
            model = self.model(kind)
            eos = len(self.meta['chord_index_by_token'])
            with patch.object(model.components, 'predict_chords', return_value=forced_logits(77, eos)), patch.object(model.components, 'predict_durations') as durations:
                result = generate(model, self.meta, 'minor')
            self.assertEqual(result['events'], [])
            self.assertEqual(result['total_duration'], 0)
            self.assertEqual(result['termination_reason'], 'eos')
            self.assertFalse(result['truncated'])
            self.assertEqual(len(result['trace']), 1)
            durations.assert_not_called()

    def test_rolling_windows_gru_state_and_causal_elapsed_time(self):
        for kind in ['gru']:
            model = self.model(kind)
            dominant = self.meta['chord_index_by_token'].index(56)
            eos = len(self.meta['chord_index_by_token'])
            chord_logits = [forced_logits(77, i) for i in [0, dominant, eos]]
            duration_logits = [(forced_logits(24, i), torch.tensor([True])) for i in [0, 2]]
            method = 'encode_step'
            with patch.object(model.components, 'predict_chords', side_effect=chord_logits), patch.object(model.components, 'predict_durations', side_effect=duration_logits) as duration, patch.object(model, method, wraps=getattr(model, method)) as encode:
                result = generate(model, self.meta, 'major')
            self.assertEqual([e['chord_index'] for e in result['events']], [0, 56])
            self.assertEqual([e['duration'] for e in result['events']], [1, 3])
            self.assertEqual([r['elapsed_duration'] for r in result['trace']], [0, 1, 4])
            self.assertEqual(result['total_duration'], 4)
            self.assertEqual(result['termination_reason'], 'eos')
            self.assertEqual([c.args[1].item() for c in duration.call_args_list], [0, dominant])
            self.assertEqual(encode.call_count, 3)
            self.assertIsNone(encode.call_args_list[0].args[4])
            self.assertIsNotNone(encode.call_args_list[1].args[4])
            self.assertEqual([c.args[0].item() for c in encode.call_args_list], [76, 0, dominant])

    def test_event_cap_reports_truncation_without_invented_eos(self):
        for kind in ['gru']:
            model = self.model(kind)
            with patch.object(model.components, 'predict_chords', return_value=forced_logits(77, 0)), patch.object(model.components, 'predict_durations', return_value=(forced_logits(24, 0), torch.tensor([True]))):
                result = generate(model, self.meta, 'major', GenerationConfig(max_events=3))
            self.assertEqual(result['event_count'], 3)
            self.assertEqual(len(result['trace']), 3)
            self.assertEqual(result['termination_reason'], 'max_events')
            self.assertTrue(result['truncated'])
            self.assertTrue(all((not t['eos'] for t in result['trace'])))

    def test_seed_reproducibility_and_model_mode_restoration(self):
        for kind in ['gru']:
            model = self.model(kind).train()
            state = torch.get_rng_state().clone()
            config = GenerationConfig(seed=8675309, max_events=8)
            first = generate(model, self.meta, 'minor', config)
            second = generate(model, self.meta, 'minor', config)
            self.assertEqual(first, second)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_export_roundtrip_and_overwrite_protection(self):
        model = self.model('gru')
        with patch.object(model.components, 'predict_chords', return_value=forced_logits(77, 0)), patch.object(model.components, 'predict_durations', return_value=(forced_logits(24, 1), torch.tensor([True]))):
            result = generate(model, self.meta, 'major', GenerationConfig(max_events=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.csv'
            (csv_path, json_path) = export_sample(result, path)
            loaded = load_chorales(directory)[0]
            self.assertEqual([(e.chord_index, e.duration) for e in loaded.events], [(0, 2), (0, 2)])
            self.assertEqual(json.loads(json_path.read_text(encoding='utf8')), result)
            with self.assertRaises(ValueError):
                export_sample(result, path)
            self.assertEqual(csv_path.read_text().splitlines()[0], 'major')
            empty = dict(result, events=[], event_count=0, total_duration=0, termination_reason='eos', truncated=False)
            (empty_path, _) = export_sample(empty, Path(directory) / 'empty.csv')
            with empty_path.open() as stream:
                self.assertEqual(len(list(csv.reader(stream))), 2)

    def test_real_checkpoint_loading_and_provenance(self):
        for kind in ['gru']:
            checkpoint = ROOT / 'GRU/checkpoints/baseline/best.pt'
            first = generate_from_checkpoint(checkpoint, 'major', GenerationConfig(max_events=3))
            second = generate_from_checkpoint(checkpoint, 'major', GenerationConfig(max_events=3))
            self.assertEqual(first, second)
            self.assertEqual(first['model_type'], kind)
            self.assertEqual(first['checkpoint']['training_seed'], 8675309)
            self.assertEqual(first['checkpoint']['monitor_kind'], 'validation')
            self.assertEqual(len(first['checkpoint']['sha256']), 64)
if __name__ == '__main__':
    unittest.main()

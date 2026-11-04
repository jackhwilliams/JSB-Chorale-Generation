import io
import json
from pathlib import Path
import sys
import unittest
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from GRU.data.datasets import GRUDataset, GRUCollator, load_encoded_split
from GRU.model import GRUConfig, GRUModel
from GRU.models.shared import EventCrossEntropy

class GRUTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.chorales, cls.metadata) = load_encoded_split(ROOT / 'GRU/artifacts/preprocessing', 'train')

    def setUp(self):
        torch.manual_seed(29)

    def batch(self):
        return next(iter(DataLoader(GRUDataset(self.chorales[:2], self.metadata), batch_size=2, collate_fn=GRUCollator(self.metadata))))

    def test_approved_config_and_dynamic_depth(self):
        self.assertEqual(GRUConfig.from_json(ROOT / 'GRU/configs/baseline.json'), GRUConfig())
        for (width, depth, duration) in [(32, 1, [16]), (64, 2, [32, 16]), (128, 3, [])]:
            with self.subTest(width=width, depth=depth):
                config = GRUConfig(hidden_size=width, num_layers=depth, duration_hidden_sizes=duration, recurrent_dropout=0.2)
                model = GRUModel(self.metadata, config)
                self.assertEqual(model.input_dim, 42)
                self.assertEqual(model.recurrent.dropout, 0.2 if depth > 1 else 0)
                batch = self.batch()
                outputs = model(batch)
                self.assertEqual(outputs['chord_logits'].shape, (2, 70, 77))
                self.assertEqual(outputs['duration_logits'].shape, (2, 70, 24))
                self.assertEqual(outputs['hidden_state'].shape, (depth, 2, width))
                result = EventCrossEntropy(self.metadata)(outputs, batch)
                self.assertTrue(torch.isfinite(result['loss']))

    def test_unsorted_packed_batch_matches_each_chorale(self):
        model = GRUModel(self.metadata, GRUConfig(hidden_size=12, num_layers=2)).eval()
        batch = self.batch()
        with torch.no_grad():
            (context, state) = model.encode_context(batch)
            dataset = GRUDataset(self.chorales[:2], self.metadata)
            for i in range(2):
                solo = GRUCollator(self.metadata)([dataset[i]])
                (individual, individual_state) = model.encode_context(solo)
                length = batch['lengths'][i].item()
                torch.testing.assert_close(context[i, :length], individual[0], atol=1e-06, rtol=1e-05)
                torch.testing.assert_close(state[:, i], individual_state[:, 0], atol=1e-06, rtol=1e-05)
                self.assertEqual(context[i, length:].abs().sum().item(), 0)

    def test_padding_cannot_change_recurrence_or_loss(self):
        model = GRUModel(self.metadata, GRUConfig(hidden_size=12)).eval()
        batch = self.batch()
        changed = {k: v.clone() if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
        padding = ~batch['sequence_mask']
        changed['input_chords'][padding] = 0
        changed['input_durations'][padding] = 1
        changed['tonality'][padding] = 1
        changed['progress'][padding] = 100
        with torch.no_grad():
            (before, after) = (model(batch), model(changed))
        for key in before:
            torch.testing.assert_close(before[key], after[key])
        criterion = EventCrossEntropy(self.metadata)
        torch.testing.assert_close(criterion(before, batch)['loss'], criterion(after, changed)['loss'])

    def test_stepwise_matches_full_sequence(self):
        model = GRUModel(self.metadata, GRUConfig(hidden_size=12, num_layers=2)).eval()
        batch = GRUCollator(self.metadata)([GRUDataset(self.chorales[:1], self.metadata)[0]])
        history = {k: v for (k, v) in batch.items() if not k.startswith('target_')}
        with torch.no_grad():
            (full_context, full_hidden) = model.encode_context(history)
            hidden = None
            steps = []
            for t in range(batch['lengths'][0].item()):
                (context, hidden) = model.encode_step(batch['input_chords'][:, t], batch['input_durations'][:, t], batch['tonality'][:, t], batch['progress'][:, t], hidden)
                steps.append(context)
            torch.testing.assert_close(torch.stack(steps, dim=1), full_context, atol=1e-06, rtol=1e-05)
            torch.testing.assert_close(hidden, full_hidden, atol=1e-06, rtol=1e-05)

    def test_default_state_resets_and_optional_state_is_explicit(self):
        model = GRUModel(self.metadata, GRUConfig(hidden_size=8)).eval()
        batch = self.batch()
        zero = torch.zeros(1, 2, 8)
        with torch.no_grad():
            (first, _) = model.encode_context(batch)
            (explicit_zero, _) = model.encode_context(batch, zero)
            (nonzero, _) = model.encode_context(batch, torch.ones_like(zero))
            (repeated, _) = model.encode_context(batch)
        self.assertTrue(torch.equal(first, explicit_zero))
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, nonzero))

    def test_future_events_and_targets_do_not_enter_prior_chord_predictions(self):
        model = GRUModel(self.metadata, GRUConfig(hidden_size=12)).eval()
        batch = self.batch()
        changed = {k: v.clone() if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
        changed['input_chords'][:, 5:] = 0
        changed['input_durations'][:, 5:] = 1
        with torch.no_grad():
            (before, after) = (model(batch), model(changed))
        torch.testing.assert_close(before['chord_logits'][:, :5], after['chord_logits'][:, :5])
        changed = dict(batch)
        changed['target_chords'] = torch.zeros_like(batch['target_chords'])
        with torch.no_grad():
            new_targets = model(changed)
        self.assertTrue(torch.equal(before['chord_logits'], new_targets['chord_logits']))

    def test_backward_through_recurrence_embeddings_and_heads(self):
        model = GRUModel(self.metadata, GRUConfig(hidden_size=12, num_layers=2))
        batch = self.batch()
        outputs = model(batch)
        EventCrossEntropy(self.metadata)(outputs, batch)['loss'].backward()
        for parameter in [model.recurrent.weight_ih_l0, model.recurrent.weight_hh_l0, model.recurrent.weight_ih_l1, model.components.chord_head.weight, model.components.duration_head[-1].weight, model.components.encoder.chord_embedding.weight, model.components.encoder.duration_embedding.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_input_dropout_excludes_conditioning_and_eval_is_deterministic(self):
        model = GRUModel(self.metadata, GRUConfig(hidden_size=8, input_dropout=0.5))
        batch = self.batch()
        features = model._features(batch['input_chords'], batch['input_durations'], batch['tonality'], batch['progress'])
        self.assertTrue(torch.equal(features[..., -2], batch['tonality'].float()))
        self.assertTrue(torch.equal(features[..., -1], batch['progress']))
        model.eval()
        with torch.no_grad():
            (first, second) = (model(batch), model(batch))
        self.assertTrue(torch.equal(first['chord_logits'], second['chord_logits']))
        self.assertFalse(model.input_dropout.training)

    def test_config_and_state_dict_roundtrip(self):
        layers = [16, 8]
        config = GRUConfig(hidden_size=12, num_layers=2, duration_hidden_sizes=layers)
        layers[0] = 1
        self.assertEqual(config.duration_hidden_sizes, (16, 8))
        self.assertEqual(GRUConfig(**json.loads(json.dumps(config.to_dict()))), config)
        model = GRUModel(self.metadata, config).eval()
        batch = self.batch()
        stream = io.BytesIO()
        torch.save({'config': config.to_dict(), 'state_dict': model.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = GRUModel(self.metadata, GRUConfig(**saved['config'])).eval()
        restored.load_state_dict(saved['state_dict'])
        with torch.no_grad():
            (expected, actual) = (model(batch), restored(batch))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]))

    def test_invalid_config_shapes_lengths_and_missing_targets(self):
        for values in [{'hidden_size': 0}, {'num_layers': True}, {'num_layers': 1.5}, {'duration_hidden_sizes': 64}, {'duration_hidden_sizes': [0]}, {'input_dropout': 1}, {'recurrent_dropout': float('nan')}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GRUConfig(**values)
        model = GRUModel(self.metadata, GRUConfig(hidden_size=8))
        batch = self.batch()
        for lengths in [torch.tensor([0, 70]), torch.tensor([59.0, 70.0]), torch.tensor([59, 71])]:
            with self.assertRaises(ValueError):
                model.encode_context(dict(batch, lengths=lengths))
        with self.assertRaises(ValueError):
            model.encode_context(batch, torch.zeros(2, 2, 8))
        with self.assertRaises(ValueError):
            model({k: v for (k, v) in batch.items() if k != 'target_chords'})
if __name__ == '__main__':
    unittest.main()

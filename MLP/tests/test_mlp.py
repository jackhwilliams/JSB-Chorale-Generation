import io
import json
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from MLP.data.datasets import MLPDataset, load_encoded_split
from MLP.model import MLPConfig, MLPModel
from MLP.models.shared import EventCrossEntropy

class MLPTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.chorales, cls.metadata) = load_encoded_split(ROOT / 'MLP/artifacts/preprocessing', 'train')

    def setUp(self):
        torch.manual_seed(31)

    def batch(self, window):
        return next(iter(DataLoader(MLPDataset(self.chorales[:2], self.metadata, window), batch_size=3)))

    def test_approved_config_and_automatic_input_dimensions(self):
        config = MLPConfig.from_json(ROOT / 'MLP/configs/baseline.json')
        self.assertEqual(config, MLPConfig())
        model = MLPModel(self.metadata, config)
        self.assertEqual(model.input_dim, 642)
        self.assertEqual([(m.in_features, m.out_features) for m in model.backbone if isinstance(m, nn.Linear)], [(642, 128), (128, 64)])
        self.assertEqual([(m.in_features, m.out_features) for m in model.components.duration_head if isinstance(m, nn.Linear)], [(96, 64), (64, 24)])
        self.assertEqual(model.components.chord_head.in_features, 64)
        self.assertEqual([m.p for m in model.backbone if isinstance(m, nn.Dropout)], [0.2, 0.2])

    def test_window_and_independent_layer_configurations(self):
        for (window, hidden, duration) in [(4, [64], [32]), (8, [128, 64], [64, 32]), (16, [256, 128, 64], []), (32, [32, 16, 8, 4], [16, 8, 4])]:
            with self.subTest(window=window):
                config = MLPConfig(window_size=window, hidden_sizes=hidden, duration_hidden_sizes=duration, chord_embedding_dim=5, duration_embedding_dim=3)
                model = MLPModel(self.metadata, config)
                self.assertEqual(model.input_dim, window * 8 + 2)
                self.assertEqual([m.out_features for m in model.backbone if isinstance(m, nn.Linear)], hidden)
                self.assertEqual([m.out_features for m in model.components.duration_head if isinstance(m, nn.Linear)], duration + [24])
                outputs = model(self.batch(window))
                self.assertEqual(outputs['chord_logits'].shape, (3, 77))
                self.assertEqual(outputs['duration_logits'].shape, (3, 24))

    def test_backward_reaches_backbone_and_both_embeddings(self):
        batch = self.batch(4)
        model = MLPModel(self.metadata, MLPConfig(window_size=4))
        losses = EventCrossEntropy(self.metadata)(model(batch), batch)
        losses['loss'].backward()
        for parameter in [model.backbone[0].weight, model.backbone[3].weight, model.components.chord_head.weight, model.components.duration_head[-1].weight, model.components.encoder.chord_embedding.weight, model.components.encoder.duration_embedding.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_chord_context_has_no_target_dependency(self):
        model = MLPModel(self.metadata, MLPConfig(window_size=4)).eval()
        batch = self.batch(4)
        changed = dict(batch)
        changed['target_chords'] = (batch['target_chords'] + 1) % 76
        with torch.no_grad():
            (a, b) = (model(batch), model(changed))
        self.assertTrue(torch.equal(a['chord_logits'], b['chord_logits']))
        self.assertFalse(torch.equal(a['duration_logits'], b['duration_logits']))

    def test_single_example_eos_and_generation_interface(self):
        dataset = MLPDataset(self.chorales[:1], self.metadata, 4)
        model = MLPModel(self.metadata, MLPConfig(window_size=4)).eval()
        eos_batch = next(iter(DataLoader([dataset[-1]], batch_size=1)))
        outputs = model(eos_batch)
        self.assertFalse(outputs['duration_valid_mask'].any())
        self.assertEqual(EventCrossEntropy(self.metadata)(outputs, eos_batch)['duration_loss'].item(), 0)
        single = dataset[0]
        self.assertEqual(model(single)['chord_logits'].shape, (77,))
        history = {k: v for (k, v) in self.batch(4).items() if not k.startswith('target_')}
        context = model.encode_context(history)
        logits = model.components.predict_chords(context)
        self.assertEqual(logits.shape, (3, 77))
        selected = torch.zeros(3, dtype=torch.long)
        self.assertEqual(model(history, selected)['duration_logits'].shape, (3, 24))
        with self.assertRaises(ValueError):
            model(history)

    def test_dropout_eval_and_checkpoint_roundtrip(self):
        config = MLPConfig(window_size=4, hidden_sizes=[16, 8], duration_hidden_sizes=[8, 4])
        model = MLPModel(self.metadata, config).eval()
        batch = self.batch(4)
        expected = model(batch)
        self.assertTrue(torch.equal(expected['chord_logits'], model(batch)['chord_logits']))
        self.assertFalse(any((m.training for m in model.modules() if isinstance(m, nn.Dropout))))
        stream = io.BytesIO()
        torch.save({'config': config.to_dict(), 'state_dict': model.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        restored = MLPModel(self.metadata, MLPConfig(**checkpoint['config'])).eval()
        restored.load_state_dict(checkpoint['state_dict'])
        for (key, value) in restored(batch).items():
            self.assertTrue(torch.equal(value, expected[key]))

    def test_config_roundtrip_and_invalid_settings(self):
        layers = [128, 64]
        config = MLPConfig(hidden_sizes=layers)
        layers[0] = 1
        self.assertEqual(config.hidden_sizes, (128, 64))
        self.assertEqual(MLPConfig(**json.loads(json.dumps(config.to_dict()))), config)
        for values in [{'window_size': 0}, {'window_size': True}, {'hidden_sizes': []}, {'hidden_sizes': [1.5]}, {'duration_hidden_sizes': [0]}, {'duration_hidden_sizes': 64}, {'hidden_sizes': '128,64'}, {'dropout': 1}, {'dropout': float('nan')}, {'chord_embedding_dim': -1}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                MLPConfig(**values)
        model = MLPModel(self.metadata)
        with self.assertRaisesRegex(ValueError, 'W=16'):
            model(self.batch(4))
if __name__ == '__main__':
    unittest.main()

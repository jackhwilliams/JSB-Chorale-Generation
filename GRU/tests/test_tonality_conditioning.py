"""New conditioning path and compatibility with frozen baseline checkpoints."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from GRU.data.datasets import GRUDataset, GRUCollator, load_encoded_split
from GRU.generation.runner import GenerationConfig, generate, generate_from_checkpoint
from GRU.model import GRUConfig, GRUModel
from GRU.models.shared import EventCrossEntropy, FeatureConfig, SharedPredictionComponents
from GRU.training.runner import TrainingConfig, fit, load_checkpoint

class TonalityConditioningTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / 'GRU/artifacts/preprocessing'
        (cls.chorales, cls.metadata) = load_encoded_split(cls.data, 'train')

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(39)

    def config(self, **overrides):
        values = dict(hidden_size=8, chord_embedding_dim=4, duration_embedding_dim=2, duration_hidden_sizes=[8], tonality_conditioning='one_hot_direct')
        values.update(overrides)
        return GRUConfig(**values)

    def batch(self):
        return GRUCollator(self.metadata)(list(GRUDataset(self.chorales[:2], self.metadata)))

    def test_new_config_and_one_hot_inputs(self):
        config = GRUConfig.from_json(ROOT / 'GRU/configs/tonality.json')
        model = GRUModel(self.metadata, config)
        self.assertEqual(model.input_dim, 43)
        self.assertEqual(model.components.chord_head.in_features, 66)
        self.assertEqual(model.components.duration_head[0].in_features, 98)
        encoder = model.components.encoder
        features = encoder.conditioning(torch.tensor([0, 1]), torch.tensor([0.25, 0.5]))
        self.assertEqual(features.tolist(), [[1.0, 0.0, 0.25], [0.0, 1.0, 0.5]])
        self.assertEqual(GRUConfig(**config.to_dict()), config)
        with self.assertRaises(ValueError):
            GRUConfig(tonality_conditioning='unsupported')

    def test_both_heads_receive_mode_even_when_context_is_identical(self):
        heads = SharedPredictionComponents(8, self.metadata, FeatureConfig(4, 2, ()), tonality_encoding='one_hot', direct_tonality=True)
        with torch.no_grad():
            heads.chord_head.weight.zero_()
            heads.chord_head.bias.zero_()
            heads.chord_head.weight[0, 8] = 2
            heads.chord_head.weight[1, 9] = 2
            heads.duration_head[0].weight.zero_()
            heads.duration_head[0].bias.zero_()
            heads.duration_head[0].weight[0, 8] = 2
            heads.duration_head[0].weight[1, 9] = 2
        context = torch.zeros(2, 8)
        outputs = heads(context, torch.tensor([0, 0]), tonality=torch.tensor([0, 1]))
        self.assertEqual(outputs['chord_logits'].argmax(-1).tolist(), [0, 1])
        self.assertEqual(outputs['duration_logits'].argmax(-1).tolist(), [0, 1])
        for call in [lambda : heads.predict_chords(context), lambda : heads.predict_durations(context, torch.tensor([0, 0])), lambda : heads.predict_chords(context, torch.tensor([2, 0]))]:
            with self.assertRaises(ValueError):
                call()

    def test_batched_loss_and_gradient_flow_to_direct_mode_weights(self):
        model = GRUModel(self.metadata, self.config())
        batch = self.batch()
        minor = next((c for c in self.chorales if c['tonality_token'] == 1))
        batch = GRUCollator(self.metadata)(list(GRUDataset([self.chorales[0], minor], self.metadata)))
        outputs = model(batch)
        loss = EventCrossEntropy(self.metadata)(outputs, batch)['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for weights in [model.components.chord_head.weight, model.components.duration_head[0].weight]:
            self.assertTrue(torch.isfinite(weights.grad).all())
            for column in [8, 9]:
                self.assertGreater(weights.grad[:, column].abs().sum().item(), 0)
        self.assertGreater(model.recurrent.weight_ih_l0.grad.abs().sum().item(), 0)

    def test_stepwise_and_packed_contexts_still_agree(self):
        model = GRUModel(self.metadata, self.config(num_layers=2)).eval()
        batch = GRUCollator(self.metadata)([GRUDataset(self.chorales[:1], self.metadata)[0]])
        with torch.inference_mode():
            (context, state) = model.encode_context(batch)
            hidden = None
            steps = []
            for t in range(batch['lengths'][0].item()):
                (step, hidden) = model.encode_step(batch['input_chords'][:, t], batch['input_durations'][:, t], batch['tonality'][:, t], batch['progress'][:, t], hidden)
                steps.append(step)
            torch.testing.assert_close(context, torch.stack(steps, dim=1), atol=1e-06, rtol=1e-05)
            torch.testing.assert_close(state, hidden, atol=1e-06, rtol=1e-05)

    def test_generation_passes_requested_mode_to_both_heads(self):
        model = GRUModel(self.metadata, self.config())
        logits = torch.full((1, 77), -10000.0)
        logits[0, 0] = 0
        for (mode, token) in [('major', 0), ('minor', 1)]:
            with patch.object(model.components, 'predict_chords', return_value=logits) as chord, patch.object(model.components, 'predict_durations', wraps=model.components.predict_durations) as duration:
                result = generate(model, self.metadata, mode, GenerationConfig(max_events=2))
            self.assertEqual(result['event_count'], 2)
            for head in [chord, duration]:
                self.assertEqual(head.call_count, 2)
                self.assertTrue(all((c.kwargs['tonality'].tolist() == [token] for c in head.call_args_list)))

    def test_training_and_new_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = fit('gru', self.data, folder, self.config().to_dict(), TrainingConfig(overfit_updates=2))
            (model, saved) = load_checkpoint(Path(folder) / 'best.pt')
            self.assertTrue(summary['checkpoint_reload_verified'])
            self.assertTrue(model.components.direct_tonality)
            self.assertEqual(saved['model_config']['tonality_conditioning'], 'one_hot_direct')
            self.assertEqual(model.input_dim, 9)

    def test_legacy_checkpoint_preserves_recorded_generation(self):
        references = json.loads((ROOT / 'GRU/tests/fixtures/generation_reference.json').read_text(encoding='utf8'))
        for kind in ['gru']:
            path = ROOT / 'GRU/checkpoints/baseline/best.pt'
            saved = references['baseline']['major_8675309']
            result = generate_from_checkpoint(path, 'major', GenerationConfig(**saved['generation_config']))
            self.assertEqual(result['events'], saved['events'])
            self.assertEqual(len(result['trace']), len(saved['trace']))
            for actual, expected in zip(result['trace'], saved['trace']):
                for field, value in expected.items():
                    if isinstance(value, float):
                        self.assertAlmostEqual(actual[field], value, places=5)
                    else:
                        self.assertEqual(actual[field], value)
            self.assertEqual(result['termination_reason'], saved['termination_reason'])
if __name__ == '__main__':
    unittest.main()

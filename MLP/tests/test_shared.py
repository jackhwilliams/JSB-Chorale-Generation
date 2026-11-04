import io
import math
from pathlib import Path
import sys
import unittest
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from MLP.data.datasets import load_encoded_split, MLPDataset
from MLP.models.shared import EventCrossEntropy, FeatureConfig, OutputVocabulary, SharedPredictionComponents

class SharedComponentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.chorales, cls.meta) = load_encoded_split(ROOT / 'MLP/artifacts/preprocessing', 'train')

    def setUp(self):
        torch.manual_seed(17)

    def batch(self, kind):
        dataset = MLPDataset(self.chorales[:2], self.meta, 4)
        return next(iter(DataLoader(dataset, batch_size=4)))

    def test_mappings_exclude_nonmusical_outputs(self):
        vocab = OutputVocabulary(self.meta)
        self.assertEqual(len(vocab.chord_token_by_class), 77)
        self.assertEqual(len(vocab.duration_token_by_class), 24)
        for (classes, decode, encode) in [(torch.arange(77), vocab.chord_tokens, vocab.chord_targets), (torch.arange(24), vocab.duration_tokens, vocab.duration_targets)]:
            self.assertTrue(torch.equal(classes, encode(decode(classes))))
        special = self.meta['chord_special_tokens']
        self.assertEqual(vocab.chord_targets(torch.tensor([special['BOS'], special['PAD']])).tolist(), [-1, -1])
        self.assertEqual(vocab.duration_targets(torch.tensor([0])).item(), -1)
        self.assertEqual(vocab.duration_tokens(torch.tensor([0, 23])).tolist(), [1, 24])
        for invalid in [torch.tensor([-1]), torch.tensor([79]), torch.tensor([1.0])]:
            with self.assertRaises(ValueError):
                vocab.chord_targets(invalid)

    def test_shapes_conditioning_and_backward_on_real_batches(self):
        for kind in ['mlp']:
            batch = self.batch(kind)
            module = SharedPredictionComponents(12, self.meta, FeatureConfig(5, 3, (9, 7)))
            features = module.encoder(batch['input_chords'], batch['input_durations'])
            self.assertEqual(features.shape, (*batch['input_chords'].shape, 8))
            conditioning = module.encoder.conditioning(batch['tonality'], batch['progress'])
            self.assertEqual(conditioning.shape, (*batch['progress'].shape, 2))
            self.assertTrue(torch.equal(conditioning[..., 0], batch['tonality'].float()))
            context = torch.randn(*batch['target_chords'].shape, 12, requires_grad=True)
            outputs = module(context, batch['target_chords'])
            self.assertEqual(outputs['chord_logits'].shape, (*context.shape[:-1], 77))
            self.assertEqual(outputs['duration_logits'].shape, (*context.shape[:-1], 24))
            losses = EventCrossEntropy(self.meta)(outputs, batch)
            self.assertTrue(torch.allclose(losses['loss'], losses['chord_loss'] + losses['duration_loss']))
            self.assertEqual(losses['chord_count'].item(), batch['chord_loss_mask'].sum().item())
            self.assertEqual(losses['duration_count'].item(), batch['duration_loss_mask'].sum().item())
            losses['loss'].backward()
            self.assertTrue(torch.isfinite(context.grad).all())
            self.assertGreater(module.chord_head.weight.grad.abs().sum().item(), 0)
            self.assertGreater(module.duration_head[-1].weight.grad.abs().sum().item(), 0)
            self.assertGreater(module.encoder.chord_embedding.weight.grad.abs().sum().item(), 0)

    def test_uniform_logits_equal_manual_cross_entropy(self):
        batch = self.batch('mlp')
        module = SharedPredictionComponents(4, self.meta)
        outputs = module(torch.zeros(*batch['target_chords'].shape, 4), batch['target_chords'])
        outputs['chord_logits'] = torch.zeros_like(outputs['chord_logits'], requires_grad=True)
        outputs['duration_logits'] = torch.zeros_like(outputs['duration_logits'], requires_grad=True)
        result = EventCrossEntropy(self.meta)(outputs, batch)
        self.assertAlmostEqual(result['chord_loss'].item(), math.log(77), places=5)
        self.assertAlmostEqual(result['duration_loss'].item(), math.log(24), places=5)
        self.assertAlmostEqual(result['loss'].item(), math.log(77) + math.log(24), places=5)
        weighted = EventCrossEntropy(self.meta, chord_weight=2, duration_weight=0.5)(outputs, batch)
        self.assertAlmostEqual(weighted['loss'].item(), 2 * math.log(77) + 0.5 * math.log(24), places=5)

    def test_padding_and_eos_duration_have_zero_gradient(self):
        batch = self.batch('mlp')
        module = SharedPredictionComponents(4, self.meta)
        outputs = module(torch.randn(*batch['target_chords'].shape, 4), batch['target_chords'])
        for (name, mask) in [('chord_logits', batch['chord_loss_mask']), ('duration_logits', batch['duration_loss_mask'])]:
            logits = outputs[name].detach().clone()
            logits[~mask] = float('nan')
            outputs[name] = logits.requires_grad_()
        losses = EventCrossEntropy(self.meta)(outputs, batch)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        for (name, mask) in [('chord_logits', batch['chord_loss_mask']), ('duration_logits', batch['duration_loss_mask'])]:
            self.assertEqual(outputs[name].grad[~mask].abs().sum().item(), 0)
            self.assertGreater(outputs[name].grad[mask].abs().sum().item(), 0)

    def test_conditioning_changes_duration_but_not_chord_prediction(self):
        module = SharedPredictionComponents(2, self.meta, FeatureConfig(2, 2, ()))
        with torch.no_grad():
            module.encoder.chord_embedding.weight[0].zero_()
            module.encoder.chord_embedding.weight[1].fill_(1)
            module.duration_head[0].weight.zero_()
            module.duration_head[0].bias.zero_()
            module.duration_head[0].weight[0, 2] = 1
        context = torch.zeros(1, 2)
        (a, b) = (module(context, torch.tensor([0])), module(context, torch.tensor([1])))
        self.assertTrue(torch.equal(a['chord_logits'], b['chord_logits']))
        self.assertFalse(torch.equal(a['duration_logits'], b['duration_logits']))
        self.assertFalse(torch.equal(a['duration_logits'].softmax(-1), b['duration_logits'].softmax(-1)))

    def test_eos_padding_and_empty_masks(self):
        module = SharedPredictionComponents(4, self.meta)
        special = self.meta['chord_special_tokens']
        targets = torch.tensor([special['EOS'], special['PAD']])
        batch = {'target_chords': targets, 'target_durations': torch.zeros(2, dtype=torch.long), 'chord_loss_mask': torch.tensor([True, False]), 'duration_loss_mask': torch.tensor([False, False])}
        context = torch.randn(2, 4, requires_grad=True)

        def unexpected(*args):
            raise AssertionError('Duration network called for a nonmusical chord')
        handle = module.duration_head.register_forward_hook(unexpected)
        outputs = module(context, targets)
        handle.remove()
        losses = EventCrossEntropy(self.meta)(outputs, batch)
        self.assertEqual(losses['duration_loss'].item(), 0)
        losses['loss'].backward()
        self.assertTrue(torch.isfinite(context.grad).all())
        batch['chord_loss_mask'][:] = False
        batch['target_chords'][:] = special['PAD']
        outputs = module(context, batch['target_chords'])
        zero = EventCrossEntropy(self.meta)(outputs, batch)
        self.assertEqual(zero['loss'].item(), 0)
        zero['loss'].backward()

    def test_embedding_padding_stays_zero_after_update(self):
        module = SharedPredictionComponents(4, self.meta)
        pad = self.meta['chord_special_tokens']['PAD']
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        features = module.encoder(torch.tensor([pad, 0]), torch.tensor([0, 1]))
        features.sum().backward()
        optimizer.step()
        self.assertEqual(module.encoder.chord_embedding.weight[pad].abs().sum().item(), 0)
        self.assertEqual(module.encoder.duration_embedding.weight[0].abs().sum().item(), 0)
        self.assertGreater(module.encoder.chord_embedding.weight.grad[0].abs().sum().item(), 0)
        self.assertGreater(module.encoder.duration_embedding.weight.grad[1].abs().sum().item(), 0)

    def test_serialization_and_single_prediction(self):
        module = SharedPredictionComponents(4, self.meta, FeatureConfig(3, 2, (5,)))
        (context, chord) = (torch.randn(4), torch.tensor(0))
        expected = module(context, chord)
        self.assertEqual(expected['chord_logits'].shape, (77,))
        self.assertEqual(expected['duration_logits'].shape, (24,))
        stream = io.BytesIO()
        torch.save(module.state_dict(), stream)
        stream.seek(0)
        restored = SharedPredictionComponents(4, self.meta, module.config)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        actual = restored(context, chord)
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]))

    def test_reject_invalid_masks_and_conditioning(self):
        module = SharedPredictionComponents(4, self.meta)
        batch = self.batch('mlp')
        outputs = module(torch.randn(*batch['target_chords'].shape, 4), batch['target_chords'])
        batch['target_durations'][0] = 0
        with self.assertRaises(ValueError):
            EventCrossEntropy(self.meta)(outputs, batch)
        with self.assertRaises(ValueError):
            module.predict_durations(torch.randn(1, 4), torch.tensor([self.meta['chord_special_tokens']['BOS']]))
        with self.assertRaises(ValueError):
            module.encoder.conditioning(torch.tensor([2]), torch.tensor([0.0]))
        with self.assertRaises(ValueError):
            FeatureConfig(chord_embedding_dim=0)
        with self.assertRaises(ValueError):
            EventCrossEntropy(self.meta, duration_weight=-1)
if __name__ == '__main__':
    unittest.main()

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from GRU.model import GRUConfig
from GRU.models.shared import EventCrossEntropy
from GRU.training.runner import TrainingConfig, fit, load_checkpoint, load_encoded_split, make_loader, make_model, run_epoch, seed_everything

class TrainingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / 'GRU/artifacts/preprocessing'
        (cls.chorales, cls.meta) = load_encoded_split(cls.data, 'train')

    def setUp(self):
        seed_everything(8675309)

    def configuration(self, kind):
        return GRUConfig(hidden_size=8, duration_hidden_sizes=[8], input_dropout=0).to_dict()

    def test_default_seed_and_configuration_validation(self):
        config = TrainingConfig(**json.loads((ROOT / 'GRU/configs/training.json').read_text()))
        self.assertEqual(config, TrainingConfig())
        self.assertEqual(config.seed, 8675309)
        for values in [{'seed': True}, {'batch_size': 0}, {'learning_rate': 0}, {'patience': 0}, {'gradient_clip': float('nan')}, {'device': 'oops'}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingConfig(**values)

    def test_validation_aggregation_is_independent_of_batch_boundaries(self):
        for kind in ['gru']:
            config = self.configuration(kind)
            model = make_model(kind, self.meta, config)
            criterion = EventCrossEntropy(self.meta)
            metrics = []
            for size in [2, 7]:
                loader = make_loader(kind, self.chorales[:2], self.meta, config, size)
                metrics.append(run_epoch(model, loader, criterion, torch.device('cpu')))
            for key in ['loss', 'chord_loss', 'duration_loss', 'chord_accuracy', 'duration_accuracy']:
                self.assertAlmostEqual(metrics[0][key], metrics[1][key], places=5)
            self.assertEqual(metrics[0]['chord_count'], 129)
            self.assertEqual(metrics[0]['duration_count'], 127)
            self.assertFalse(model.training)
            self.assertTrue(all((p.grad is None for p in model.parameters())))

    def test_optimizer_updates_and_clipping(self):
        for kind in ['gru']:
            config = self.configuration(kind)
            model = make_model(kind, self.meta, config)
            initial = next(model.parameters()).detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loader = make_loader(kind, self.chorales[:2], self.meta, config, 64)
            with patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
                result = run_epoch(model, loader, EventCrossEntropy(self.meta), torch.device('cpu'), optimizer, gradient_clip=1.0, max_updates=1)
            self.assertEqual(result['updates'], 1)
            self.assertEqual(clip.call_count, 1)
            self.assertEqual(clip.call_args.args[1], 1.0)
            self.assertFalse(torch.equal(initial, next(model.parameters())))

    def test_overfit_pipeline_checkpoint_and_reproducibility(self):
        config = TrainingConfig(overfit_updates=3)
        for kind in ['gru']:
            (summaries, states) = ([], [])
            for repeat in range(2):
                with tempfile.TemporaryDirectory() as folder:
                    with patch('GRU.training.runner.load_encoded_split', wraps=load_encoded_split) as load:
                        summary = fit(kind, self.data, folder, self.configuration(kind), config)
                    self.assertEqual([c.args[1] for c in load.call_args_list], ['train'])
                    self.assertEqual(summary['training_ids'], ['001', '005'])
                    self.assertEqual(summary['updates'], 3)
                    self.assertTrue(summary['checkpoint_reload_verified'])
                    (restored, checkpoint) = load_checkpoint(Path(folder) / 'best.pt')
                    self.assertEqual(checkpoint['training_config']['seed'], 8675309)
                    self.assertEqual(checkpoint['training_config']['weight_decay'], 0)
                    self.assertEqual(checkpoint['metadata']['split'], self.meta['split'])
                    self.assertIn('optimizer_state', checkpoint)
                    self.assertIn('loader_rng_state', checkpoint)
                    self.assertEqual(checkpoint['model_config'].get('dropout', 0), 0)
                    self.assertEqual(checkpoint['model_config'].get('input_dropout', 0), 0)
                    self.assertTrue((Path(folder) / 'last.pt').exists())
                    summaries.append(summary)
                    states.append(restored.state_dict())
                    with self.assertRaisesRegex(ValueError, 'existing results'):
                        fit(kind, self.data, folder, self.configuration(kind), config)
            self.assertEqual({k: v for (k, v) in summaries[0].items() if k != 'elapsed_seconds'}, {k: v for (k, v) in summaries[1].items() if k != 'elapsed_seconds'})
            self.assertTrue(all((torch.equal(states[0][k], states[1][k]) for k in states[0])))

    def test_early_stopping_selects_best_without_training_corpus(self):

        def metrics(loss):
            return {'loss': loss, 'updates': 1, 'chord_accuracy': 0, 'duration_accuracy': 0}
        sequence = [metrics(9), metrics(8), metrics(3), metrics(7), metrics(3), metrics(6), metrics(4), metrics(3)]

        def tiny_load(directory, split):
            self.assertIn(split, ['train', 'validation'])
            return (self.chorales[:2], self.meta)
        with tempfile.TemporaryDirectory() as folder:
            with patch('GRU.training.runner.run_epoch', side_effect=sequence), patch('GRU.training.runner.load_encoded_split', side_effect=tiny_load):
                result = fit('gru', self.data, folder, self.configuration('gru'), TrainingConfig(max_epochs=10, patience=2), mode='train')
            self.assertEqual(result['epochs'], 3)
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(result['stop_reason'], 'early_stopping')
            (_, best) = load_checkpoint(Path(folder) / 'best.pt')
            (_, last) = load_checkpoint(Path(folder) / 'last.pt')
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(last['epoch'], 3)
if __name__ == '__main__':
    unittest.main()

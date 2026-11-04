from collections import Counter
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from MLP.data.datasets import MLPDataset, load_encoded_split
from MLP.model import MLPModel, MLPConfig
from MLP.evaluation.metrics import predictive_metrics, decode_reference, distribution_distance, pooled_counts, longest_training_match, sequence_statistics, describe_samples

def sample(chords, durations=None, name='x', mode='major'):
    durations = durations if durations is not None else [2] * len(chords)
    return {'chorale_id': name, 'tonality': mode, 'events': [dict(chord_index=c, duration=d, chord_label='V 7' if c == 59 else 'I') for (c, d) in zip(chords, durations)], 'event_count': len(chords), 'total_duration': sum(durations), 'termination_reason': 'eos', 'truncated': False}

class EvaluationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.chorales, cls.meta) = load_encoded_split(ROOT / 'MLP/artifacts/preprocessing', 'train')

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def test_uniform_predictive_metrics_and_joint_denominator(self):
        refs = [self.chorales[0], next((c for c in self.chorales if c['tonality_token'] == 1))]
        for kind in ['mlp']:
            model = MLPModel(self.meta, MLPConfig(window_size=4, hidden_sizes=[8]))
            loader = DataLoader(MLPDataset(refs, self.meta, 4), batch_size=7)
            original = model.components.forward

            def uniform(context, targets):
                outputs = original(context, targets)
                outputs['chord_logits'] = torch.zeros_like(outputs['chord_logits'])
                outputs['duration_logits'] = torch.zeros_like(outputs['duration_logits'])
                return outputs
            with patch.object(model.components, 'forward', side_effect=uniform):
                result = predictive_metrics(model, loader, self.meta)
            total = result['overall']
            (chords, durations) = (total['chord_count'], total['duration_count'])
            self.assertEqual(chords - durations, 2)
            self.assertAlmostEqual(total['chord_perplexity'], 77, places=4)
            self.assertAlmostEqual(total['balanced_loss'], math.log(77) + math.log(24), places=5)
            self.assertAlmostEqual(total['joint_nll_per_event_including_eos'], math.log(77) + durations / chords * math.log(24), places=5)
            self.assertAlmostEqual(total['mean_sequence_nll'], (chords * math.log(77) + durations * math.log(24)) / 2, places=3)
            for category in ['chord_count', 'duration_count', 'chord_correct', 'duration_correct']:
                self.assertEqual(total[category], result['major'][category] + result['minor'][category])
            self.assertTrue(model.training)
            self.assertTrue(all((p.grad is None for p in model.parameters())))

    def test_distribution_distance_known_and_empty_cases(self):
        self.assertEqual(distribution_distance(Counter(a=3), Counter(a=9)), {'total_variation': 0.0, 'js_divergence_bits': 0.0})
        separate = distribution_distance(Counter(a=1), Counter(b=1))
        self.assertEqual(separate, {'total_variation': 1.0, 'js_divergence_bits': 1.0})
        self.assertIsNone(distribution_distance(Counter(), Counter(a=1))['total_variation'])
        self.assertEqual(distribution_distance(Counter(a=1, b=3), Counter(a=3, b=1))['total_variation'], 0.5)

    def test_transitions_never_cross_sequence_boundaries(self):
        counts = pooled_counts([sample([0, 56]), sample([40, 0])])
        self.assertEqual(counts['transitions'], Counter({(0, 56): 1, (40, 0): 1}))
        self.assertNotIn((56, 40), counts['transitions'])
        self.assertEqual(counts['chords'][0], 2)
        self.assertEqual(counts['durations'][2], 4)

    def test_exact_reuse_requires_duration_and_is_contiguous(self):
        references = [sample([0, 40, 56, 0], name='ref', mode='minor')]
        generated = sample([59, 40, 56, 0, 59])
        result = longest_training_match(generated['events'], references)
        self.assertEqual(result['length'], 3)
        self.assertEqual(result['generated_start'], 1)
        self.assertEqual(result['training_start'], 1)
        self.assertEqual(result['training_chorale_id'], 'ref')
        self.assertEqual(result['training_tonality'], 'minor')
        self.assertEqual(result['fraction_of_generated_events'], 0.6)
        changed = sample([40, 56, 0], [1, 1, 1])
        self.assertEqual(longest_training_match(changed['events'], references)['length'], 0)
        self.assertEqual(longest_training_match([], references)['length'], 0)

    def test_repetition_ending_proxies_and_empty_sequences(self):
        s = sample([0, 0, 56, 0, 56, 0])
        stats = sequence_statistics(s)
        self.assertEqual(stats['unique_chords'], 2)
        self.assertEqual(stats['adjacent_chord_repetition'], 0.2)
        self.assertEqual(stats['repeated_chord_bigram_fraction'], 0.4)
        self.assertTrue(stats['final_mode_tonic'])
        self.assertTrue(stats['final_dominant_to_mode_tonic'])
        empty = sample([])
        self.assertIsNone(sequence_statistics(empty)['final_mode_tonic'])
        described = describe_samples([s, empty])
        self.assertEqual(described['empty_count'], 1)
        self.assertEqual(described['final_mode_tonic']['count'], 1)
        self.assertEqual(described['unique_complete_sequences'], 2)
        picardy = sequence_statistics(sample([59, 0], mode='minor'))
        self.assertTrue(picardy['final_either_tonic'])
        self.assertFalse(picardy['final_mode_tonic'])

    def test_reference_decoding_preserves_events_and_lengths(self):
        decoded = decode_reference(self.chorales[:2], self.meta)
        for (source, reference) in zip(self.chorales, decoded):
            self.assertEqual(reference['chorale_id'], source['chorale_id'])
            self.assertEqual(reference['event_count'], len(source['chord_tokens']) - 2)
            self.assertEqual(sum((e['duration'] for e in reference['events'])), source['total_duration'])
if __name__ == '__main__':
    unittest.main()

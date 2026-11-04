"""Alignment tests use a hand-computable sequence and the full saved corpus."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from MLP.data.datasets import MLPDataset, load_encoded_split
from MLP.data.preprocessing import Chorale, Event, encode_chorale

class DatasetTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.corpus, cls.meta) = load_encoded_split(ROOT / 'MLP/artifacts/preprocessing', 'train')

    def examples(self):
        short = Chorale('short', 'minor', (Event(1, 2, 'i'), Event(41, 4, 'iv')))
        long = Chorale('long', 'major', (Event(0, 4, 'I'), Event(56, 2, 'V'), Event(59, 6, 'V 7'), Event(0, 8, 'I')))
        return [encode_chorale(c, self.meta) for c in [short, long]]

    def test_hand_computed_mlp_windows(self):
        short = self.examples()[0]
        dataset = MLPDataset([short], self.meta, window_size=3)
        bos = self.meta['chord_special_tokens']['BOS']
        eos = self.meta['chord_special_tokens']['EOS']
        pad = self.meta['chord_special_tokens']['PAD']
        (first, second) = short['chord_tokens'][1:3]
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0]['input_chords'].tolist(), [pad, pad, bos])
        self.assertEqual(dataset[1]['input_chords'].tolist(), [pad, bos, first])
        self.assertEqual(dataset[2]['input_chords'].tolist(), [bos, first, second])
        self.assertEqual(dataset[0]['input_durations'].tolist(), [0, 0, 0])
        self.assertEqual(dataset[0]['context_mask'].tolist(), [False, False, True])
        self.assertEqual([dataset[i]['target_chords'].item() for i in range(3)], [first, second, eos])
        self.assertEqual([dataset[i]['duration_loss_mask'].item() for i in range(3)], [True, True, False])
        for (i, elapsed) in enumerate([0, 2, 6]):
            self.assertAlmostEqual(dataset[i]['progress'].item(), elapsed / 228)
            self.assertEqual(dataset[i]['tonality'].item(), 1)
        self.assertEqual(dataset[-1]['target_chords'].item(), eos)
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_window_sizes_and_chorale_boundaries(self):
        examples = self.examples()
        for window in [1, 4, 8, 16, 32]:
            dataset = MLPDataset(examples, self.meta, window)
            self.assertEqual(len(dataset), 8)
            for index in range(len(dataset)):
                self.assertEqual(dataset[index]['input_chords'].shape, (window,))
            item = dataset[3]
            self.assertEqual(item['chorale_id'], 'long')
            self.assertEqual(item['target_index'], 1)
            self.assertEqual(item['context_mask'].sum().item(), 1)
            self.assertEqual(item['progress'].item(), 0)
        for value in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                MLPDataset(examples, self.meta, value)
        self.assertEqual(len(MLPDataset([], self.meta)), 0)

    def test_causal_inputs_and_fresh_samples(self):
        original = self.examples()[1]
        changed = deepcopy(original)
        changed['chord_tokens'][2] = self.meta['chord_index_by_token'].index(41)
        changed['duration_tokens'][2] = self.meta['duration_by_token'].index(80)
        changed['progress'] = [9999] * len(changed['progress'])
        dataset = MLPDataset([original], self.meta, 3)
        before = dataset[1]
        after = MLPDataset([changed], self.meta, 3)[1]
        for key in ['input_chords', 'input_durations', 'tonality', 'progress']:
            self.assertTrue(torch.equal(before[key], after[key]), key)
        item = dataset[0]
        saved = item['input_chords'].clone()
        item['input_chords'].fill_(999)
        self.assertTrue(torch.equal(saved, dataset[0]['input_chords']))

    def test_full_corpus_alignment(self):
        total = 0
        for (split, count) in [('train', 300), ('validation', 40), ('test', 40)]:
            (corpus, meta) = load_encoded_split(ROOT / 'MLP/artifacts/preprocessing', split)
            self.assertEqual(len(corpus), count)
            dataset = MLPDataset(corpus, meta, 16)
            offset = 0
            for (ci, source) in enumerate(corpus):
                length = len(source['chord_tokens']) - 1
                elapsed = 0
                for i in range(length):
                    item = dataset[offset + i]
                    self.assertEqual(item['target_chords'].item(), source['chord_tokens'][i + 1])
                    self.assertEqual(item['target_durations'].item(), source['duration_tokens'][i + 1])
                    if i:
                        elapsed += meta['duration_by_token'][source['duration_tokens'][i]]
                    self.assertAlmostEqual(item['progress'].item(), elapsed / meta['progress']['scale'], places=5)
                    self.assertEqual(bool(item['duration_loss_mask']), i < length - 1)
                offset += length
            total += offset
        self.assertEqual(total, 24227 + 380)

    def test_invalid_encoded_boundaries(self):
        for (field, index, value) in [('chord_tokens', 0, 0), ('duration_tokens', 0, 1), ('chord_tokens', 1, self.meta['chord_special_tokens']['PAD'])]:
            bad = deepcopy(self.examples()[0])
            bad[field][index] = value
            with self.assertRaises(ValueError):
                MLPDataset([bad], self.meta)
        with self.assertRaises(ValueError):
            load_encoded_split(ROOT / 'MLP/artifacts/preprocessing', 'unknown')
if __name__ == '__main__':
    unittest.main()

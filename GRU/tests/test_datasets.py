"""Alignment tests use a hand-computable sequence and the full saved corpus."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from GRU.data.datasets import GRUCollator, GRUDataset, load_encoded_split
from GRU.data.preprocessing import Chorale, Event, encode_chorale

class DatasetTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.corpus, cls.meta) = load_encoded_split(ROOT / 'GRU/artifacts/preprocessing', 'train')

    def examples(self):
        short = Chorale('short', 'minor', (Event(1, 2, 'i'), Event(41, 4, 'iv')))
        long = Chorale('long', 'major', (Event(0, 4, 'I'), Event(56, 2, 'V'), Event(59, 6, 'V 7'), Event(0, 8, 'I')))
        return [encode_chorale(c, self.meta) for c in [short, long]]

    def test_gru_shift_and_right_padding(self):
        examples = self.examples()
        dataset = GRUDataset(examples, self.meta)
        self.assertEqual(len(dataset), 2)
        for (item, source) in zip(dataset, examples):
            self.assertEqual(item['input_chords'].tolist(), source['chord_tokens'][:-1])
            self.assertEqual(item['target_chords'].tolist(), source['chord_tokens'][1:])
            self.assertEqual(item['target_durations'].tolist(), source['duration_tokens'][1:])
        batch = GRUCollator(self.meta)([dataset[0], dataset[1]])
        self.assertEqual(batch['lengths'].tolist(), [3, 5])
        self.assertEqual(batch['input_chords'].shape, (2, 5))
        self.assertEqual(batch['sequence_mask'].tolist(), [[True, True, True, False, False], [True] * 5])
        self.assertEqual(batch['chord_loss_mask'].sum().item(), 8)
        self.assertEqual(batch['duration_loss_mask'].sum().item(), 6)
        self.assertEqual(batch['target_chords'][0, 2].item(), self.meta['chord_special_tokens']['EOS'])
        self.assertTrue(batch['chord_loss_mask'][0, 2].item())
        self.assertFalse(batch['duration_loss_mask'][0, 2].item())
        self.assertEqual(batch['input_chords'][0, 3:].tolist(), [self.meta['chord_special_tokens']['PAD']] * 2)
        self.assertEqual(batch['target_chords'][0, 3:].tolist(), [self.meta['chord_special_tokens']['PAD']] * 2)
        self.assertEqual(batch['target_durations'][0, 3:].tolist(), [0, 0])
        self.assertEqual(batch['progress'][0, 3:].tolist(), [0, 0])
        self.assertEqual(batch['input_chords'].dtype, torch.long)
        self.assertEqual(batch['progress'].dtype, torch.float32)
        self.assertEqual(batch['sequence_mask'].dtype, torch.bool)
        with self.assertRaises(ValueError):
            GRUCollator(self.meta)([])

    def test_masks_exclude_padding_and_eos_duration(self):
        dataset = GRUDataset(self.examples(), self.meta)
        batch = GRUCollator(self.meta)([dataset[0], dataset[1]])
        for (name, expected) in [('chord_loss_mask', 8), ('duration_loss_mask', 6)]:
            costs = torch.ones_like(batch['progress'])
            costs[~batch[name]] = 1000000
            self.assertEqual(costs[batch[name]].sum().item(), expected)

    def test_worker_process_collation(self):
        dataset = GRUDataset(self.examples(), self.meta)
        batch = list(DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=GRUCollator(self.meta)))[0]
        self.assertEqual(batch['chorale_id'], ['short', 'long'])
        self.assertEqual(batch['lengths'].tolist(), [3, 5])

    def test_causal_inputs_and_fresh_samples(self):
        original = self.examples()[1]
        changed = deepcopy(original)
        changed['chord_tokens'][2] = self.meta['chord_index_by_token'].index(41)
        changed['duration_tokens'][2] = self.meta['duration_by_token'].index(80)
        changed['progress'] = [9999] * len(changed['progress'])
        dataset = GRUDataset([original], self.meta)
        before = dataset[0]
        after = GRUDataset([changed], self.meta)[0]
        for key in ['input_chords', 'input_durations', 'tonality', 'progress']:
            self.assertTrue(torch.equal(before[key][:2], after[key][:2]), key)
        item = dataset[0]
        saved = item['input_chords'].clone()
        item['input_chords'].fill_(999)
        self.assertTrue(torch.equal(saved, dataset[0]['input_chords']))

    def test_full_corpus_alignment(self):
        total = 0
        for (split, count) in [('train', 300), ('validation', 40), ('test', 40)]:
            (corpus, meta) = load_encoded_split(ROOT / 'GRU/artifacts/preprocessing', split)
            self.assertEqual(len(corpus), count)
            dataset = GRUDataset(corpus, meta)
            offset = 0
            for (ci, source) in enumerate(corpus):
                length = len(source['chord_tokens']) - 1
                elapsed = 0
                for i in range(length):
                    item = dataset[ci]
                    self.assertEqual(item['target_chords'][i].item(), source['chord_tokens'][i + 1])
                    self.assertEqual(item['target_durations'][i].item(), source['duration_tokens'][i + 1])
                    if i:
                        elapsed += meta['duration_by_token'][source['duration_tokens'][i]]
                    self.assertAlmostEqual(item['progress'][i].item(), elapsed / meta['progress']['scale'], places=5)
                    self.assertEqual(bool(item['duration_loss_mask'][i]), i < length - 1)
                offset += length
            total += offset
        self.assertEqual(total, 24227 + 380)

    def test_invalid_encoded_boundaries(self):
        for (field, index, value) in [('chord_tokens', 0, 0), ('duration_tokens', 0, 1), ('chord_tokens', 1, self.meta['chord_special_tokens']['PAD'])]:
            bad = deepcopy(self.examples()[0])
            bad[field][index] = value
            with self.assertRaises(ValueError):
                GRUDataset([bad], self.meta)
        with self.assertRaises(ValueError):
            load_encoded_split(ROOT / 'GRU/artifacts/preprocessing', 'unknown')
if __name__ == '__main__':
    unittest.main()

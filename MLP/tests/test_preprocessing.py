import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from MLP.data.preprocessing import approved_split, build_metadata, decode_event, encode_chorale, load_chorales, padding_event, preprocess, validate_split

class PreprocessingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = ROOT / 'data/chorale chords'
        cls.chorales = load_chorales(cls.source)
        cls.split = approved_split(cls.chorales)
        cls.metadata = build_metadata(cls.chorales, cls.split)

    def test_inspected_corpus_and_source_duration(self):
        self.assertEqual(len(self.chorales), 380)
        self.assertEqual(sum((len(c.events) for c in self.chorales)), 24227)
        self.assertEqual(sum((c.total_duration for c in self.chorales)), 92144)
        for c in self.chorales:
            with (ROOT / 'data/data transposed chorales' / (c.chorale_id + '.csv')).open() as stream:
                source = list(csv.reader(stream))
            self.assertEqual(c.total_duration, len(source) - 1)
            self.assertEqual(c.tonality, source[0][1])

    def test_split_partition_coverage_and_approved_swaps(self):
        self.assertEqual(self.split, approved_split(tuple(reversed(self.chorales))))
        self.assertEqual([len(self.split[k]) for k in ['train', 'validation', 'test']], [300, 40, 40])
        self.assertTrue({'147', '335'} <= set(self.split['train']))
        self.assertTrue({'000', '004'} <= set(self.split['test']))
        by_id = {c.chorale_id: c for c in self.chorales}
        for (subset, expected) in [('train', 153), ('validation', 21), ('test', 21)]:
            self.assertEqual(sum((by_id[i].tonality == 'major' for i in self.split[subset])), expected)
        validate_split(self.chorales, self.split)
        invalid = {k: v[:] for (k, v) in self.split.items()}
        invalid['test'].append(invalid['train'][0])
        with self.assertRaises(ValueError):
            validate_split(self.chorales, invalid)
        missing = {k: v[:] for (k, v) in self.split.items()}
        missing['train'].remove('335')
        missing['test'].append('335')
        with self.assertRaisesRegex(ValueError, 'Training lacks'):
            validate_split(self.chorales, missing)

    def test_exact_roundtrip_and_causal_progress(self):
        for c in self.chorales:
            encoded = encode_chorale(c, self.metadata)
            decoded = [decode_event(a, b, self.metadata) for (a, b) in zip(encoded['chord_tokens'], encoded['duration_tokens'])]
            self.assertIsNone(decoded[0])
            self.assertIsNone(decoded[-1])
            self.assertEqual(tuple(decoded[1:-1]), c.events)
            self.assertEqual(encoded['total_duration'], c.total_duration)
            elapsed = 0
            for (i, event) in enumerate(c.events, 1):
                self.assertEqual(encoded['elapsed_duration'][i], elapsed)
                elapsed += event.duration
            self.assertEqual(encoded['elapsed_duration'][-1], elapsed)
            self.assertEqual(encoded['progress'][-1], elapsed / self.metadata['progress']['scale'])
            self.assertFalse(encoded['duration_loss_mask'][0])
            self.assertFalse(encoded['duration_loss_mask'][-1])
            self.assertTrue(all(encoded['duration_loss_mask'][1:-1]))
            self.assertFalse(encoded['chord_loss_mask'][0])
            self.assertTrue(encoded['chord_loss_mask'][-1])

    def test_vocabulary_specials_and_training_only_scale(self):
        meta = self.metadata
        self.assertEqual(len(meta['chord_index_by_token']), 76)
        self.assertEqual(len(meta['duration_by_token']), 25)
        self.assertEqual(len(set(meta['chord_special_tokens'].values())), 3)
        self.assertTrue(all((t >= 76 for t in meta['chord_special_tokens'].values())))
        self.assertIsNone(meta['duration_by_token'][meta['duration_placeholder_token']])
        pad = padding_event(meta)
        self.assertFalse(pad['duration_loss_mask'])
        self.assertFalse(pad['chord_loss_mask'])
        self.assertIsNone(decode_event(pad['chord_token'], pad['duration_token'], meta))
        with self.assertRaises(ValueError):
            decode_event(0, 0, meta)
        with self.assertRaises(ValueError):
            decode_event(meta['chord_special_tokens']['EOS'], 1, meta)
        from dataclasses import replace
        changed = tuple((replace(c, events=c.events * 100) if c.chorale_id in self.split['test'] else c for c in self.chorales))
        self.assertEqual(meta['progress'], build_metadata(changed, self.split)['progress'])
        longest = max(self.chorales, key=lambda c: c.total_duration)
        self.assertGreater(encode_chorale(longest, meta)['progress'][-1], 1)

    def test_saved_artifacts_reproduce_and_sources_unchanged(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.glob('*.csv')}
        with tempfile.TemporaryDirectory() as directory:
            (metadata, encoded) = preprocess(self.source, directory)
            saved_meta = json.loads((Path(directory) / 'metadata.json').read_text(encoding='utf8'))
            saved_data = json.loads((Path(directory) / 'chorales.json').read_text(encoding='utf8'))
            self.assertEqual(metadata, saved_meta)
            self.assertEqual(encoded, saved_data)
            self.assertEqual(saved_data, [encode_chorale(c, saved_meta) for c in self.chorales])
            self.assertEqual(saved_meta['source_sha256'], before)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.glob('*.csv')}
        self.assertEqual(before, after)
        with self.assertRaises(ValueError):
            preprocess(self.source, self.source)

    def test_loader_rejects_bad_data(self):
        for content in ['major\nwrong,duration,chord_label\n0,4,I\n', 'major\nchord_index,duration,chord_label\n0,0,I\n', 'major\nchord_index,duration,chord_label\n0,4,\n', 'major\nchord_index,duration,chord_label\n0,4,I\n0,2,V\n', 'major\nchord_index,duration,chord_label\n']:
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / '001.csv').write_text(content, encoding='utf8')
                with self.assertRaises(ValueError):
                    load_chorales(directory)
if __name__ == '__main__':
    unittest.main()

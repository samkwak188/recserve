import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('bundle', pathlib.Path(__file__).resolve().parents[1]/'scripts/bundle.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        for name, magic in [('catalog', 0x43415431), ('queries', 0x51525931), ('index', 0x48535732)]:
            (self.root/name).write_bytes(struct.pack('<Iii4f', magic, 2, 2, 1, 0, 0, 1))
        (self.root/'users').write_text('query_row,user_id\n0,12\n1,24\n')
        (self.root/'items').write_text('item_row,item_id\n0,123\n1,456\n')
        self.data = dict(schema_version=1, model_version='test', catalog='catalog', queries='queries',
                         index='index', user_map='users', item_map='items',
                         files={p.name: bundle.sha256(p) for p in self.root.iterdir()})
        self.manifest = self.root/'manifest.json'
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.data))

    def test_valid(self):
        self.assertEqual(bundle.verify(self.manifest)['model_version'], 'test')

    def test_corrupt(self):
        (self.root/'catalog').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): bundle.verify(self.manifest)

    def test_path_escape(self):
        self.data['files']['../escape'] = 'abc'
        self.save()
        with self.assertRaises(ValueError): bundle.verify(self.manifest)

    def test_missing_role(self):
        del self.data['files']['catalog']
        self.save()
        with self.assertRaises(ValueError): bundle.verify(self.manifest)

    def test_bad_mapping(self):
        (self.root/'users').write_text('query_row,user_id\n0,12\n1,12\n')
        self.data['files']['users'] = bundle.sha256(self.root/'users')
        self.save()
        with self.assertRaises(ValueError): bundle.verify(self.manifest)


if __name__ == '__main__':
    unittest.main()

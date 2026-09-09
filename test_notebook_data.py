"""Validate the actual self-contained Notebook setup cell without Kaggle or ML dependencies."""
import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest


class NotebookDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(Path(__file__).with_name('S6E9_Prototype.ipynb').read_text(encoding='utf-8'))
        cell = next(''.join(c['source']) for c in notebook['cells']
                    if c['cell_type'] == 'code' and 'def find_s6e9_data(' in ''.join(c['source']))
        tree = ast.parse(cell)
        definitions = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        scope = {}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook-data-cell>', 'exec'), scope)
        cls.find = staticmethod(scope['find_s6e9_data'])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def dataset(self, relative):
        folder = self.root / relative
        folder.mkdir(parents=True)
        features = ['id', 'Age', 'Annual_Income_USD', 'Daily_Commute_km']
        for filename, header in [('train.csv', features + ['Will_Buy_EV']), ('test.csv', features),
                                 ('sample_submission.csv', ['id', 'Will_Buy_EV'])]:
            with (folder / filename).open('w', encoding='utf-8', newline='') as stream:
                csv.writer(stream).writerow(header)
        return folder

    def test_current_and_legacy_paths(self):
        current = self.dataset('competitions/playground-series-s6e9')
        self.assertEqual(self.find(self.root), current)
        legacy = self.dataset('playground-series-s6e9')
        self.assertEqual(self.find(self.root), current)
        (current / 'train.csv').unlink()
        self.assertEqual(self.find(self.root), legacy)

    def test_nested_upload_directory(self):
        folder = self.dataset('datasets/owner/custom-upload/extracted')
        self.assertEqual(self.find(self.root), folder)

    def test_prediction_only_dataset_explains_fix(self):
        predictions = self.root / 'ps-s6e9-predictions'
        predictions.mkdir()
        (predictions / '0.94635.csv').write_text('id,Will_Buy_EV\n1,0.5\n')
        with self.assertRaisesRegex(FileNotFoundError, 'Add Input') as failure:
            self.find(self.root)
        self.assertIn('ps-s6e9-predictions', str(failure.exception))
        self.assertIn('train.csv / test.csv / sample_submission.csv', str(failure.exception))

    def test_incomplete_and_wrong_schema_are_not_selected(self):
        incomplete = self.dataset('incomplete')
        (incomplete / 'sample_submission.csv').unlink()
        wrong = self.dataset('wrong-competition')
        (wrong / 'train.csv').write_text('id,unrelated_target\n')
        with self.assertRaises(FileNotFoundError):
            self.find(self.root)

    def test_multiple_matches_require_explicit_choice(self):
        first = self.dataset('copy-one')
        self.dataset('copy-two')
        with self.assertRaisesRegex(ValueError, 'DATA_OVERRIDE'):
            self.find(self.root)
        self.assertEqual(self.find(self.root, override=first), first)

    def test_invalid_override_does_not_fall_back_silently(self):
        self.dataset('playground-series-s6e9')
        with self.assertRaisesRegex(FileNotFoundError, 'DATA_OVERRIDE'):
            self.find(self.root, override=self.root / 'missing')

    def test_missing_input_root(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Add Input'):
            self.find(self.root / 'not-mounted')


if __name__ == '__main__':
    unittest.main()

"""Validate the actual self-contained Notebook setup cell without Kaggle or ML dependencies."""
import ast
import csv
import json
import shutil
from pathlib import Path
import tempfile
import unittest

import build_stage_notebooks


NOTEBOOKS = ('S6E9_Prototype.ipynb', '01_search.ipynb',
             '02_freeze.ipynb', '03_finalize.ipynb')


class NotebookPackageTests(unittest.TestCase):
    def test_embedded_package_matches_repository_sources(self):
        root = Path(__file__).parent
        expected = {'SCRIPT': 'prototype.py', 'GPU_SETUP': 'gpu_setup.py',
                    'DIVERSITY_SCRIPT': 'diversity.py'}
        expected.update({f's6e9/{path.name}': str(path.relative_to(root)).replace('\\', '/')
                         for path in (root / 's6e9').glob('*.py')})
        for filename in NOTEBOOKS:
            with self.subTest(notebook=filename):
                notebook = json.loads((root / filename).read_text(encoding='utf-8'))
                source = next(''.join(cell['source']) for cell in notebook['cells']
                    if cell['cell_type'] == 'code'
                    and "PACKAGE = Path('/kaggle/working/s6e9')" in ''.join(cell['source']))
                assignments = {}
                for node in ast.parse(source).body:
                    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                        continue
                    function = node.value.func
                    if not isinstance(function, ast.Attribute) or function.attr != 'write_text':
                        continue
                    target = function.value
                    if isinstance(target, ast.Name):
                        assignments[target.id] = ast.literal_eval(node.value.args[0])
                    elif isinstance(target, ast.BinOp) and isinstance(target.op, ast.Div):
                        assignments['s6e9/' + ast.literal_eval(target.right)] = ast.literal_eval(node.value.args[0])
                self.assertEqual(set(assignments), set(expected))
                for key, source_filename in expected.items():
                    self.assertEqual(assignments[key],
                                     (root / source_filename).read_text(encoding='utf-8'))

    def test_staged_notebooks_are_generated_from_current_source(self):
        for filename, expected in build_stage_notebooks.generate().items():
            with self.subTest(notebook=filename):
                self.assertEqual((Path(__file__).parent / filename).read_text(encoding='utf-8'), expected)


class NotebookStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent
        notebook = json.loads((root / '02_freeze.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if cell['cell_type'] == 'code' and 'def restore_run(' in ''.join(cell['source']))
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'restore_run')
        cls.function = function

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / 'input'
        self.inputs.mkdir()
        self.run = self.root / 'working' / 's6e9_fast_v6_2'
        scope = {'Path': Path, 'shutil': shutil, 'RUN': self.run}
        exec(compile(ast.Module(body=[self.function], type_ignores=[]),
                     '<notebook-restore-cell>', 'exec'), scope)
        self.restore = scope['restore_run']

    def test_previous_output_is_copied_to_writable_run(self):
        prior = self.inputs / 'previous-output' / self.run.name
        prior.mkdir(parents=True)
        (prior / 'promoted.json').write_text('{}')
        (prior / 'cache').mkdir()
        (prior / 'cache' / 'prediction.npy').write_bytes(b'cached')
        self.assertEqual(self.restore('promoted.json', self.inputs), self.run)
        self.assertEqual((self.run / 'cache' / 'prediction.npy').read_bytes(), b'cached')
        (self.run / 'writable.txt').write_text('ok')

    def test_missing_or_ambiguous_previous_output_stops(self):
        with self.assertRaisesRegex(FileNotFoundError, 'previous-stage Notebook Output'):
            self.restore('promoted.json', self.inputs)
        for owner in ('first', 'second'):
            prior = self.inputs / owner / self.run.name
            prior.mkdir(parents=True)
            (prior / 'promoted.json').write_text('{}')
        with self.assertRaisesRegex(FileNotFoundError, 'exactly one'):
            self.restore('promoted.json', self.inputs)

    def test_each_stage_runs_only_its_declared_commands(self):
        root = Path(__file__).parent
        expected = {
            '01_search.ipynb': ["execute('search')"],
            '02_freeze.ipynb': ["execute('freeze')"],
            '03_finalize.ipynb': ["execute('audit')", "execute('finalize')"],
        }
        for filename, commands in expected.items():
            notebook = json.loads((root / filename).read_text(encoding='utf-8'))
            source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
            found = [command for command in ("execute('search')", "execute('freeze')",
                                              "execute('audit')", "execute('finalize')")
                     if command in source]
            self.assertEqual(found, commands)


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
        cls.require_t4 = staticmethod(scope['require_t4_pair'])

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

class NotebookSubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(Path(__file__).with_name('S6E9_Prototype.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'def export_submission(' in ''.join(c['source']))
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'export_submission')
        scope = {'Path': Path, 'shutil': shutil}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<notebook-export>', 'exec'), scope)
        cls.export = staticmethod(scope['export_submission'])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.working = Path(self.temporary.name)
        self.run = self.working / 's6e9_run'
        self.run.mkdir()

    def test_nested_submission_is_exported_to_working_root_unchanged(self):
        content = b'id,Will_Buy_EV\r\n668665,0.25\r\n668666,0.8\r\n'
        source = self.run / 'submission.csv'
        source.write_bytes(content)
        result = self.export(self.run, self.working)
        self.assertEqual(result, self.working / 'submission.csv')
        self.assertEqual(result.read_bytes(), content)
        self.assertEqual(source.read_bytes(), content)

    def test_missing_model_output_never_creates_a_dummy_submission(self):
        with self.assertRaises(FileNotFoundError):
            self.export(self.run, self.working)
        self.assertFalse((self.working / 'submission.csv').exists())

    def test_export_already_at_root_is_safe_to_repeat(self):
        source = self.working / 'submission.csv'
        source.write_bytes(b'id,Will_Buy_EV\n668665,0.25\n')
        self.assertEqual(self.export(self.working, self.working), source)
        self.assertEqual(self.export(self.working, self.working), source)


if __name__ == '__main__':
    unittest.main()

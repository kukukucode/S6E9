"""Generate staged Kaggle notebooks from the self-contained notebook."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).parent
SOURCE = ROOT / "S6E9_Prototype.ipynb"


def lines(text):
    return text.splitlines(keepends=True)


def markdown(cell_id, text):
    return {"id": cell_id, "cell_type": "markdown", "source": lines(text), "metadata": {}}


def code(cell_id, text):
    return {"id": cell_id, "cell_type": "code", "source": lines(text),
            "metadata": {"trusted": True}, "outputs": [], "execution_count": None}


def restore_cell(required, stage):
    return code(f"s6e9-{stage}-restore", f'''import shutil


def restore_run(required, input_root='/kaggle/input'):
    if RUN.exists():
        if not (RUN / required).is_file():
            raise RuntimeError(f'{{RUN}} exists but {{required}} is missing. Delete it or use a new RUN path.')
        return RUN
    root = Path(input_root)
    matches = sorted({{path.parent for path in root.rglob(required)
                      if path.parent.name == RUN.name}}) if root.is_dir() else []
    if len(matches) != 1:
        found = '\\n'.join(f'  - {{path}}' for path in matches) or '  (none)'
        raise FileNotFoundError(
            'Add exactly one previous-stage Notebook Output as Input. '
            f'Expected {{RUN.name}}/{{required}}; found:\\n{{found}}')
    shutil.copytree(matches[0], RUN)
    print('Restored:', matches[0], '->', RUN)
    return RUN


restore_run('{required}')
''')


def generate():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in source["cells"]}
    common_ids = [f"s6e9-cell-{index:02d}" for index in range(1, 6)]
    common = [deepcopy(cells[cell_id]) for cell_id in common_ids]
    stages = {
        "01_search.ipynb": [
            markdown("s6e9-search-title", "# S6E9: search\n\nT4 x2で候補探索を実行します。\n"),
            *deepcopy(common), deepcopy(cells["s6e9-cell-06"]), deepcopy(cells["s6e9-cell-07"]),
            markdown("s6e9-search-next",
                     "## 次工程\n\nSave Version完了後、このNotebookのOutputを02_freezeのInputに追加します。\n"),
        ],
        "02_freeze.ipynb": [
            markdown("s6e9-freeze-title", "# S6E9: freeze\n\n01_searchのOutputを復元し、候補とblendを固定します。\n"),
            *deepcopy(common), restore_cell("promoted.json", "freeze"),
            deepcopy(cells["s6e9-cell-08"]), deepcopy(cells["s6e9-cell-09"]),
            markdown("s6e9-freeze-next",
                     "## 次工程\n\nSave Version完了後、このNotebookのOutputを03_finalizeのInputに追加します。\n"),
        ],
        "03_finalize.ipynb": [
            markdown("s6e9-finalize-title", "# S6E9: finalize\n\n02_freezeのOutputを復元し、監査と提出CSV作成を実行します。\n"),
            *deepcopy(common), restore_cell("frozen.json", "finalize"),
            deepcopy(cells["s6e9-cell-10"]), deepcopy(cells["s6e9-cell-11"]),
            deepcopy(cells["s6e9-cell-12"]), deepcopy(cells["s6e9-cell-13"]),
        ],
    }
    ensemble_source = (ROOT / "ensemble.py").read_text(encoding="utf-8")
    ensemble_metadata = deepcopy(source["metadata"])
    ensemble_metadata["kaggle"]["accelerator"] = "none"
    ensemble_metadata["kaggle"]["isGpuEnabled"] = False
    ensemble_metadata["kaggle"]["isInternetEnabled"] = False
    stages["04_ensemble.ipynb"] = [
        markdown("s6e9-ensemble-title", "# S6E9: OOF ensemble\n\n複数の03_finalize OutputをCPUでcross-fit比較します。\n"),
        code("s6e9-ensemble-imports", '''from pathlib import Path
import importlib.util
import json
import subprocess
import sys

required = ['numpy', 'pandas', 'scipy', 'sklearn']
missing = [name for name in required if importlib.util.find_spec(name) is None]
assert not missing, f'Missing libraries: {missing}'
INPUT_ROOT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working/s6e9_ensemble_v1')
MIN_FOLD_WINS = 4
print('Inputs:', INPUT_ROOT, 'Output:', OUTPUT)
'''),
        code("s6e9-ensemble-script",
             "ENSEMBLE_SCRIPT = Path('/kaggle/working/ensemble.py')\n"
             f"ENSEMBLE_SCRIPT.write_text({ensemble_source!r}, encoding='utf-8')\n"),
        code("s6e9-ensemble-run", '''subprocess.run([sys.executable, '-u', str(ENSEMBLE_SCRIPT),
    '--input-root', str(INPUT_ROOT), '--output', str(OUTPUT),
    '--min-fold-wins', str(MIN_FOLD_WINS)], check=True)
'''),
        code("s6e9-ensemble-result", '''from IPython.display import FileLink, display

report = json.loads((OUTPUT / 'ensemble_report.json').read_text(encoding='utf-8'))
print(json.dumps(report, indent=2, ensure_ascii=False))
submission = OUTPUT / 'submission_ensemble.csv'
if report['accepted']:
    display(FileLink(str(submission)))
else:
    print('Cross-fit gate未通過のため、提出CSVは生成していません。')
'''),
    ]
    result = {}
    for filename, stage_cells in stages.items():
        metadata = ensemble_metadata if filename == "04_ensemble.ipynb" else deepcopy(source["metadata"])
        notebook = {"metadata": metadata,
                    "nbformat_minor": source["nbformat_minor"],
                    "nbformat": source["nbformat"], "cells": stage_cells}
        result[filename] = json.dumps(notebook, ensure_ascii=False,
                                      separators=(",", ":")) + "\n"
    return result


def main():
    for filename, content in generate().items():
        (ROOT / filename).write_text(content, encoding="utf-8", newline="\n")
        print(filename)


if __name__ == "__main__":
    main()

# S6E9

KaggleのNotebookで `S6E9_Prototype.ipynb` をImportし、コンペデータ・Internet・`GPU T4 x2`を有効にしてRun Allします。提出用CSVは `/kaggle/working/submission.csv` に生成されます。

LightGBM 12 trials、XGBoost 6 trials、固定CatBoost・RealMLP候補を評価します。採用GPU主モデルの3-seed平均と副モデル最大30%のblendは、development OOF全体と4-fold中3-fold以上で改善した場合だけ使います。診断は `seed_average_comparison.json`、`blend_comparison.json`、`tie_break_comparison.json` に保存します。

CIはUbuntu・Python 3.12で構文チェックと全pytestを実行します。GPU実機はCIに含めず、GPU割り当てはモックテストで確認します。

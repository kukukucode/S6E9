# S6E9

KaggleのNotebookで `S6E9_Prototype.ipynb` をImportし、コンペデータ・Internet・`GPU T4 x2`を有効にしてRun Allします。提出用CSVは `/kaggle/working/submission.csv` に生成されます。

seed 2026、LightGBM 12 trials、XGBoost 6 trialsです。GPU候補は2-foldで予選し、LGB/XGBの上位3件と固定CatBoost 1件を4-foldで確認します。domain候補もGPUで学習し、GPU候補がCPU基準を全体AUCと4-fold中3-fold以上で上回る場合だけ主モデルにします。rank blendと同点予測のtie-breakingにも同じ3-fold条件を使います。診断は `tie_break_comparison.json` などへ記録します。初回に生成したCUDA LightGBM wheelはNotebook Outputへ保存され、次回そのOutputをInputに追加すると再利用できます。

CIはUbuntu・Python 3.12で構文チェックと全pytestを実行します。GPU実機はCIに含めず、GPU割り当てはモックテストで確認します。

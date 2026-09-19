# S6E9

KaggleのNotebookで `S6E9_Prototype.ipynb` をImportし、コンペデータ・Internet・`GPU T4 x2`を有効にしてRun Allします。提出用CSVは `/kaggle/working/submission.csv` に生成されます。

seed 2026、LightGBM 12 trials、XGBoost 6 trialsです。GPU候補は2-foldで予選し、LGB/XGBの上位3件と固定CatBoost 1件を4-foldで確認します。rank blendは4-fold中3-fold以上で改善した場合だけ採用します。T4ごとのworkerを維持し、予選と4-foldの順位差もJSONへ記録します。初回に生成したCUDA LightGBM wheelはNotebook Outputへ保存され、次回そのOutputをInputに追加すると再利用できます。

CIはUbuntu・Python 3.12で構文チェックと全pytestを実行します。GPU実機はCIに含めず、GPU割り当てはモックテストで確認します。

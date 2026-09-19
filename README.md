# S6E9

KaggleのNotebookで `S6E9_Prototype.ipynb` をImportし、コンペデータ・Internet・`GPU T4 x2`を有効にしてRun Allします。提出用CSVは `/kaggle/working/submission.csv` に生成されます。

seed 2026、LightGBM 12 trials、XGBoost 6 trialsです。GPU候補は2-foldで予選し、各familyの上位3件だけを4-foldで確定します。T4ごとのworkerを維持してデータを再利用し、昇格候補をmanifestに固定します。初回に生成したCUDA LightGBM wheelはNotebook Outputへ保存され、次回そのOutputをInputに追加すると再利用できます。

CIはUbuntu・Python 3.12で構文チェックと全pytestを実行します。GPU実機はCIに含めず、GPU割り当てはモックテストで確認します。

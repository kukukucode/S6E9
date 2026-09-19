# S6E9

KaggleのNotebookで `S6E9_Prototype.ipynb` をImportし、コンペデータ・Internet・`GPU T4 x2`を有効にしてRun Allします。提出用CSVは `/kaggle/working/submission.csv` に生成されます。

seed 2026、LightGBM 12 trials、XGBoost 6 trialsです。GPU候補は2-foldで予選し、各familyの上位3件だけを4-foldで確定します。T4を1枚ずつ分離した2プロセスでfoldを処理し、CPUモデルは1プロセスで4 foldを処理します。

CIはUbuntu・Python 3.12で構文チェックと全pytestを実行します。GPU実機はCIに含めず、GPU割り当てはモックテストで確認します。

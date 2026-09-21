# S6E9

Kaggleで `S6E9_Prototype.ipynb` をImportし、コンペデータ・Internet・`GPU T4 x2`を有効にしてRun Allします。選択済み構成は `/kaggle/working/submission.csv`、同じ予測のprobability/rank版は `submission_probability.csv` と `submission_rank.csv` に生成されます。GPUや入力が違う場合は学習前に停止します。

Notebookの先頭で `s6e9/` packageを `/kaggle/working` に展開します。実装は `data / split / features / models / tuning / freeze / audit / finalize / artifact` に分割し、`prototype.py` と `diversity.py` は互換CLIとして残しています。

LightGBM 12 trials、XGBoost 6 trials、固定CatBoost候補を評価し、生成式由来の相互作用をdomain候補で比較します。1%刻みblendは粗い重みよりdevelopment OOF全体と4-fold中3-fold以上で改善した場合だけ使います。RealMLPは実測で遅く精度も低かったため既定工程から外しています。

CIはUbuntu・Python 3.12で構文チェックと全pytestを実行します。GPU実機はCIに含めず、GPU割り当てはモックテストで確認します。

9.20.2026現在 PB
596/2498

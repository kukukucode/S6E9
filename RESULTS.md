# S6E9 v5

Kaggleで `S6E9_Prototype.ipynb` をImportし、コンペInput・GPU・Internetを有効にしてRun All。

- 既存LGB設定を固定し、特徴量をキャッシュ。LGB CUDA 12候補＋XGB CUDA 6候補を全4-foldで比較。
- OOFのAUC・相関で選別し、少量の確率ブレンドを固定→holdout確認→最終5-fold。
- 2GPU並列は設定で有効化。途中打ち切り・CatBoost・rank blendは見送り。
- 新規runは `s6e9_diversity_v5`。旧runとは分離。探索回数は増えるため全体の高速化は未保証。
- CSVはKaggleで生成。提出用 `submission.csv` はGitHub管理対象外。`final_oof.csv` は精度比較用。
- 今回はコード更新のみ。学習・テスト・スコア検証は未実行。

仕様確認: [LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html#device_type)、[XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/)。

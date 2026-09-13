# S6E9 v5.1

Kaggleで `S6E9_Prototype.ipynb` をImportし、コンペInput・GPU・Internetを有効にしてRun All。

- 最大2GPUを自動検出。2枚ならfold並列、1枚なら逐次実行。GPU制限・UUIDにも対応。
- キャッシュは非圧縮優先、空き容量不足時は圧縮。準備・学習時間とサイズをJSONに記録。
- CPU 511ビンの基準モデルと、同設定のドメイン特徴追加版を比較（追加4 fits）。良い方を主力として少量ブレンド。
- LGB CUDA 12候補＋XGB CUDA 6候補、全4-fold。early stoppingは維持。途中打ち切り・多seed・bootstrap採否判定は見送り。
- 新規runは `s6e9_diversity_v5_1`。旧runとは分離。速度・精度改善はKaggleで要検証。
- CSVはKaggleで生成。提出用 `submission.csv` はGitHub管理対象外。`final_oof.csv` は精度比較用。
- 本番学習・Kaggle提出は実行しません。
- 71テスト通過。GPU検出・割り当ては模擬検証、特徴量とキャッシュの値の一致は小規模データで確認。

仕様確認: [CUDAのGPU指定](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)、[joblibの圧縮](https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html)。

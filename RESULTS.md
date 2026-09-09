# Kaggle GPU対応（v4）

現在の `S6E9_Prototype.ipynb` はKaggleのNVIDIA GPUを使うLightGBM CUDA設定です。
KaggleのSettings → AcceleratorでGPUを選択し、InternetをOnにして、GitHubから最新Notebookを再ImportしてRun Allしてください。
初期画面の見出しは「S6E9: 数値TE・桁特徴量＋Kaggle GPU」です。各20候補・3seedと書かれたNotebookは旧v2です。

CUDAビルドが未導入の場合に限り、同じLightGBMバージョンをCUDA有効でビルドします。初回セットアップには数分以上かかる場合があります。
学習前にmax_bin=511とカテゴリ列を含む小規模な実学習を行い、GPUを使えない場合は理由を表示します。CPUへの自動切り替えは行いません。
特徴量・学習設定・ビン数はv3と同じ。前処理・TE・分布差診断・CSV出力はCPUです。GPUは1枚使用します。
保存先は `/kaggle/working/s6e9_signals_v4_cuda`、提出CSVは `/kaggle/working/submission.csv`。CPUで実行する場合は `LGB_DEVICE='cpu'` にします。

**GPU実機での速度・AUCは未計測です。** 46テスト通過（CUDA設定・セットアップ分岐は模擬テスト、CPU学習・予測は実行）。CPU版は保存済みv3との予測一致を確認しました。
GPU計算ではCPU版と同じ予測・AUCになるとは限りません。以下の0.946004はv3のCPU実測値です。
v3の正確な学習コードは `runs/signals_v3/prototype_snapshot.py`、提出CSVと実験記録も同フォルダに保存しています。

方式・ビルド手順: [LightGBM CUDA公式ガイド](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-cuda-version)、[Pythonパッケージのビルド手順](https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst#build-cuda-version)。

---

# S6E9 改訂版の実測結果（2026-09-10）

LightGBM・1seedの最終5-fold OOFは **0.946003647**。初回LGBM＋XGBoostの 0.941987578 に対し **+0.004016068** でした。
このOOFは候補選択の影響を含む参考値です。**新しい提出CSVのPublic Scoreは未計測**で、0.94635超えはまだ確認していません。初回CSVのPublic Scoreはユーザー確認で0.94184です。

## 同じdevelopment 4-foldでの比較

| 候補 | 変更 | development AUC | 所要分 |
|---|---|---:|---:|
| 0 | 生特徴量（既存基準） | 0.941746841 | 2.66 |
| 1 | 元の数値・カテゴリのTE＋頻度 | 0.945118923 | 2.95 |
| 2 | 上記＋桁特徴量とそのTE・頻度 | 0.945194753 | 6.58 |
| 3 | 上記＋所得・通勤距離の複数幅ビン | 0.945548997 | 5.78 |
| 4 | 上記のTEを平滑化10/100の2種類へ | 0.945583620 | 6.55 |
| 5 | 上記＋LightGBM設定変更 | 0.945683481 | 5.50 |

最初の5候補は同じモデル設定です。数値TEと頻度は同時に追加しているため、両者それぞれの寄与は未分離です。平滑化20から10/100への変更も、2列化との合計効果です。
候補5は全4foldで基準モデルを上回りました。比較は学習668,665行のうちdevelopment 80%だけを使い、同じsplit seed=2026で行いました。

## 選抜したモデルと評価

- LightGBM / multiscale_dual、seed=2026、最終5fold平均、各535 trees。
- depth=5、31 leaves、min_child_samples=20、colsample=0.5、max_bin=511、learning_rate=0.04。
- 所得のビン幅は1/100/1000、通勤距離は1/5/10。桁は10^-4〜10^3。
- TEは平滑化10/100。学習行はラベルに依存しない内部4-foldでcross-fitし、各行自身のラベルとpriorを除外。検証・testには学習部分だけの統計を適用。
- 外部の元データや他者の提出CSVは使用していません。
- 候補固定後の参考holdout AUC: 0.945848887。固定LGBM基準: 0.941704580。差: +0.004144308。
- 同じholdoutは初回にも見ているため、新たな独立評価ではありません。holdout確認後の追加チューニングは実施していません。

## 軽量化と時間

比較6候補を含む全工程はCPU 4 threadsで **39.43分**。探索 30.09分、固定 8.9秒、参考holdout 2.71分、最終学習 6.49分でした。機種やライブラリで所要時間は変わります。

今回の実行は探索24回＋holdout2回＋最終追加4回の計30回の学習です。最終提出は5モデルで、初回の10モデルより少なくなりました。前回の20候補×2モデル・3seed設定との所要時間比較は未実測です。

- 同じ入力・コード・学習条件のfold/seed予測を再利用。探索seedの再学習、同一ベースライン、完了した最終foldの再学習を省きます。
- 参考holdout学習時にtest予測も保存し、同条件の最終fold4で再利用しました。
- 旧前処理の二重特徴量生成、上書きして捨てる学習TE、文字列変換の重複を削減。
- 新TEはカテゴリの整数コードとbincountで集計し、2種類の平滑化で集計を共用。
- 実データで最終処理を再実行し、再学習0回、**8.42秒**で同一CSVを再生成できました。
- 再開用cacheはローカルrunフォルダに残しますがGitには含めません。Kaggleのセッション消失後の継続には、そのrunフォルダを別途保存する必要があります。

## 検証と利用

37テスト通過。既存5種類の前処理が修正前と一致。実データの基準LGBMは全development OOF予測が初回と完全一致しました。
提出CSVは286,571行でテンプレートのID・列順に一致し、欠損なし、有限の0〜1確率です。保存済みモデル別予測との最大誤差は 1.11e-16。
v3実測時のNotebookに埋め込んだスクリプトは、実測に使ったprototype.pyとバイト単位で一致していました。実行済み環境はconfig.jsonに記録しました。

そのまま提出するファイル: `runs/signals_v3/submission.csv`。
Kaggleで学習を再現する場合はGitHubから `S6E9_Prototype.ipynb` をImportし、公式コンペデータをInputに追加してRun All。最終CSVは `/kaggle/working/submission.csv` にも保存します。
コード変更時は新しいrunフォルダを使ってください。初回の記録は `runs/initial/` に保存しています。

## 参考

[Naji — Pure LGBM Model CV 0.94606 LB 0.94637](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637) の数値・桁・複数幅ビン・TEの考え方を参考に独自実装しました。参照Notebookのコードや既存予測は取り込んでいません。
元データ特徴量、auto平滑化、Focal Lossブレンドは今回の比較には含めていません。

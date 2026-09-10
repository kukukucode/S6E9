# S6E9 v4.2

`S6E9_Prototype.ipynb` をKaggleにImportし、コンペInput・GPU・Internetを有効にしてRun All。

- LightGBM＋数値TE・頻度・桁・複数幅ビン。6候補→選抜→最終5-fold。
- GPUで完了した5候補の最高development AUC: **0.945547341**（ユーザーログ）。6候補目はCUDA不正メモリアクセスで停止。
- 対策: 255以下のビンはGPU、大きいビンはCPU。ビン数は維持。原因の断定・Kaggle実機での解消確認は未実施。
- 旧 `s6e9_signals_v4_1_cuda` が残っていれば、データ・コード・ライブラリ・OOF・試行DBを照合して完了5候補を新runへ移行。旧runは保持。今回の探索は20回の再学習を省略。
- 単一モデル時の重み探索を省略。再開用予測を再利用。CSVは提出用 `submission.csv` とモデル比較用 `final_oof.csv` のみ。
- 元のCPU版最終OOF: **0.946003647**。候補選択後の参考値。新しいPublic Score・0.94635超えは未確認。
- 54テスト通過。中断復元・CPU予測一致を確認。GPU部分は模擬検証。

出力: `/kaggle/working/s6e9_signals_v4_2_cuda`。提出用CSVは `/kaggle/working/submission.csv` にも保存。
`submission.csv` はKaggleで生成し、GitHubには登録しません。
中断再開時はセッションを停止せず、旧runを残して最新Notebookを再Importしてください。

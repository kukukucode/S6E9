# S6E9

Kaggleで `01_search.ipynb` → `02_freeze.ipynb` → `03_finalize.ipynb` の順に実行し、各Save VersionのOutputを次のNotebookのInputに追加します。コンペデータ・Internet・`GPU T4 x2`が必要です。最後に3種類の提出CSVが生成されます。複数の03 Outputがあれば、CPU専用の`04_ensemble.ipynb`でOOFアンサンブルを検証できます。`S6E9_Prototype.ipynb`は一括実行版です。

Notebookの先頭で `s6e9/` packageを `/kaggle/working` に展開します。実装は `data / split / features / models / tuning / freeze / audit / finalize / artifact` に分割し、`prototype.py` と `diversity.py` は互換CLIとして残しています。

LightGBM 12 trials、XGBoost 6 trials、CatBoost 4候補を評価します。CatBoostは2-foldで選別し、固定候補よりdevelopment OOF全体と4-fold中3-fold以上で改善した候補だけを採用・seed平均します。1%刻みblendにも同じ改善条件を使います。RealMLPは既定工程から外しています。

CIはUbuntu・Python 3.12で構文チェックと全pytestを実行します。GPU実機はCIに含めず、GPU割り当てはモックテストで確認します。

9.20.2026現在 PB
596/2498

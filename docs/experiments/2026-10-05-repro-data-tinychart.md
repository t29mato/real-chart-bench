# TinyChart@768 ChartQA chart-to-table 再現用データ (2026-10-05)

## ソース・ライセンス
- アノテーション: HF `mPLUG/TinyChartData` の `test.json` (Apache-2.0、全体約25.9GBのうちtest.jsonのみ5.4MB取得)。
- 画像: HF側は24GBのtarのみのため、ChartQA公式 (`vis-nlp/ChartQA`, `ChartQA Dataset/test/png/`) のGitHub rawからファイル名一致で取得(item の `image` パス末尾 `.../ChartQA/test/png/<name>.png` と一致、PNG署名を検証)。ChartQAのライセンスは GPL-3.0 と記載されている(要確認)。ローカル再現専用で再配布しない (`data/cache/` 配下)。

## 件数・選別
- TinyChart評価は id に `chartqa2table-` を含むデータセットを chart2table として採点 (`tinychart/eval/run_eval.py`)。test.json 9061件中 `chartqa2table-test_*` は **1509件**(画像1509枚、重複なし)。
- 1509 > 1000 のため **seed 20261005 の `random.Random.sample` で500件**を抽出(全件ではない)。公表値 93.78 は全1509件での値なので、500件サンプルとは誤差(標本変動)がある。
- 生成物: `data/cache/repro/tinychart_chartqa/items.jsonl`、`.../images/` (500枚、計約24MB)。スクリプト: `scripts/eval/repro/prepare_tinychart_chartqa.py`。

## プロンプト形式
- 全件同一: `<image>\nGenerate underlying data table of the chart.`(items.jsonl の `prompt_raw`。`<image>\n` を除いた `prompt` も併記)。
- `eval_model.py` は `conv_templates[conv_mode]` (デフォルト `phi`) にこの文字列を human 発話として入れ、max_new_tokens=1024 で生成。
- items.jsonl の項目: id, image, source_image, prompt_raw, prompt, gt_answer。

## 採点の仕様 (`tinychart/eval/eval_chart2table.py::chart2table_evaluator`)
- 入力は各件 `gt_answer` と `model_answer`。双方 `'title |\n' + text.strip().lower()` に変換。
- 形式: 1行目がヘッダ行、以降が行。セルは ` | ` (前後に半角スペース) 区切り、行は改行区切り。例: `Country | BMI\nVenezuela | 25.11`。
- `table_datapoints_precision_recall` のF1(転置/非転置の良い方、数値は相対誤差10%以内、文字列はANLS 0.5)を全件平均×100 (RMS_F1 = 論文のRMS-F1相当)。
- `gt_answer` は生の文字列で保持(スクリプト側で上記変換が必要)。予測も同じ ` | ` 区切り表で出力させる必要がある。

## 不確実な点
- 論文Table 2のRMS-F1 93.78が上記 `table_datapoints_f1` (公式コードの採点) と同一指標かは未確認(コード上は row版がコメントアウトされdatapoints版を使用)。
- 公式は768px入力のモデルチェックポイント(TinyChart-3B-768)を使う。ここでは評価モデル側は未準備。
- ChartQA画像がTinyChartData内のtar画像と同一かはバイト比較していない(パス・ファイル名のみ一致確認)。

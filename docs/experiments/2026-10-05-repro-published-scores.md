# 合成図で学習したモデルの論文値の再現(2026-10-05)

design §7.75 (A): この実験環境(RTX 4090、各モデル専用の venv、実論文の図で使ったのと同じ読み込み・推論コード)が、
各モデルの論文の報告値を、著者と同じテストデータ・著者の採点コードで再現できるか。

採点コード: TinyChart の `eval_chart2table.py`(DePlot の `metrics.py` と同一の実装、Apache-2.0)を無改変で
`scripts/eval/repro/third_party/` に置いて使う。RMS-F1 = `table_datapoints_precision_recall`、
RNSS = `table_number_accuracy`。表の書式(区切り記号)だけを採点コードの読める形にそろえ、値には触れない
(`scripts/eval/repro/score_repro.py`)。

## 結果

| モデル | テストデータ | 論文の値 | この環境 | 判定 |
|---|---|---|---|---|
| TinyChart@768 | ChartQA 表抽出 500問(TinyChartData test.json の chartqa2table 1,509問から抽出) | RMS-F1 93.78 | **RMS-F1 93.20**(RNSS 96.98) | 再現 |
| DePlot | PlotQA test 500図(4種類 × 125) | RNSS 97.1 / RMS-F1 94.2 | **RNSS 97.27 / RMS-F1 94.52** | 再現(パッチ上限 4,096) |
| DePlot | ChartQA 表抽出 500問 | 87.22(TinyChart 論文の比較行 "Deplot+Codex"、DePlot 論文の値ではない) | RMS-F1 92.98(RNSS 97.27) | 動作確認 |
| UniChart base-960 | ChartQA 表抽出 500問 | 91.10 / 94.01(表抽出向けに追加学習したモデル。重みは非公開) | RMS-F1 80.79(RNSS 87.08) | 公開モデルは論文値より低い(想定どおり) |
| ChartGemma | — | 表抽出の値は報告なし | 未実施(ChartQA の質問応答で代用予定) | — |
| Granite Vision 4.1 | — | Chart2CSV 71.9(GPT-4o 判定、採点プロンプト非公開、テストデータは再配布制限) | 厳密な再現は行わない(オーナー判断) | — |

## 途中で見つけて直したこと

1. **DePlot のパッチ上限.** Hugging Face のプロセッサーの既定値は 2,048 パッチ。この設定では PlotQA で RNSS 93.1 /
   RMS-F1 83.9 にしかならない。同じ100図で 4,096 にすると 97.5 / 94.3 となり、論文値が再現できた。論文の値はこの
   設定によるものと判断し、DePlot は 4,096 で統一した(`scripts/eval/chart2table/worker.py` の SPECS)。実論文97図も
   4,096 でやり直した: 点 F1 0.147 → 0.156、表として読めない図 29 → 19。設定を直しても実論文の図はほぼ読めない。
   2,048 の出力は `data/chart2table_predictions/archive/` と `data/repro_predictions/*-max_patches2048.jsonl` に残す。
2. **テスト問題に付いた指示文の取り違え.** TinyChart のテスト問題には TinyChart 用の指示文が付いており、再現用の実行
   スクリプトがそれを全モデルに渡していた。UniChart は学習した指示文(`<extract_data_table> <s_answer>`)を受け取れず、
   指示文をおうむ返しにして RMS-F1 18.7 になった。各モデルには自分の指示文を渡し、問題付きの指示文は TinyChart の
   ときだけ使うように直した(`run_repro.py --item-prompt`)。誤った指示文での出力は `*-WRONG-PROMPT` /
   `*-tinychart-prompt-*` として残す。実論文97図と ChartQA の動作確認は最初から各モデルの指示文なので影響なし。

## 結論

TinyChart と DePlot は、この環境で論文の値を誤差の範囲で再現できた。そのうえで、実論文97図での点 F1 は
TinyChart 0.050、DePlot 0.156 にとどまる。合成図・ウェブのグラフで学習したモデルが実論文の実験グラフを読めない
ことは、実験環境の問題ではない。

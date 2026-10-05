# 合成チャートモデルの公表スコアの再現計画と、マーカー型の既存合成データセット調査(2026-10-05)

調査日: 2026-10-05 / 方法: Web調査のみ(WebFetch / WebSearch、論文PDFは pdftotext で本文を確認)。データのダウンロードや実物の確認はしていない。

凡例(各主張の確度):
- **[公式]** = 公式ページ(GitHub README / HFカード / 公式サイト / 公式リポジトリのコード)で確認
- **[論文]** = 論文本文で確認
- **[推測]** = 一般知識や間接情報からの推定
- **[未確認]** = 取得できなかった、または記載がなかった

前提: 先行メモ `2026-10-05-synthetic-dataset-survey.md`(合成データ候補の一覧)と `2026-10-05-chart2table-sanity-chartqa.md`(ChartQA 30図での値の再現率チェック、論文の指標ではない)の続き。ここでは (A) 各モデルが公表している chart-to-table の数値を、論文と同じ指標・同じテストデータで再現する計画と、(B) マーカー型の既存合成データセットの選定を扱う。

---

## 結論(先出し)

**(A) 再現の目標値**

| モデル | 公表タスク / データ / 指標 | 公表値 | 確度 |
|---|---|---|---|
| DePlot | plot-to-table。PlotQA のテスト図すべて(v1/v2 で表は同一)。RNSS / RMS-F1 | **RNSS 97.1 / RMS-F1 94.2** | [論文] Table 4 |
| UniChart | Chart-to-Table。ChartQA(ファインチューニングして評価)。RNSS / RMS-F1 | **94.01 / 91.10**(ゼロショットの WebCharts は 60.73 / 43.21) | [論文] Table 2 |
| TinyChart-3B-768 | Chart-to-Table。ChartQA の表の正解。RMS-F1 | **93.78** | [論文] Table 2 |
| ChartGemma | chart-to-table の数値は**公表されていない**。一番近い公表値は ChartQA の relaxed accuracy | **aug 90.80 / human 69.52** | [論文](HTML版) |
| Granite Vision 4.1 4B | Chart2CSV。ChartNet の human-verified テスト集合。GPT-4o を judge にした 0–100 点 | **71.9** | [公式] HFカードの図 bench_c2c.png |

- RMS-F1 と RNSS の公式実装は `google-research/google-research/deplot/metrics.py`(Apache-2.0)にある [公式]。TinyChart の `eval_chart2table.py` は同じ実装(text_theta=0.5, number_theta=0.1)を使っている [公式]。UniChart の論文は RNSS と RMS で評価しているが、リポジトリに評価スクリプトは無い [公式]。
- 3モデル(DePlot / UniChart / TinyChart)は同じ実装の RMS-F1 で比べられる。ChartGemma は QA の relaxed accuracy でしか比べられない。Granite は judge のプロンプトが公開されておらず、厳密な再現はできない [未確認]。

**(B) マーカー型の合成データの推奨**

1. **PlotQA の `dot_line`(テスト 5,574 図)** [論文]。マーカーの形は6種類からランダムに選ばれ、1〜4系列で、点ごとに (x, y) と bbox がある。CC-BY-4.0 なので再配布できる [公式]。DePlot の公表値(PlotQA)と同じデータで、DePlot・TinyChart・UniChart・ChartGemma の学習データにも PlotQA が入っている。このため A と B を同じデータで比べられる。弱点は、x 軸がカテゴリ(年・国など)で、x の要素が2〜12個しかなく、線形軸だけであること [論文]。
2. **ChartNet `core_permissive` の Scatter Plot**(数値の x 軸を補う。Granite の学習分布でもある)。ライセンスは CDLA-Permissive-2.0 で再配布できる [公式]。図ごとに CSV とプロット用コードが付く [公式]。ただし散布図の件数、対数軸の有無、CSV が描かれたマーカーと1対1か、はいずれも [未確認]。
   - 代替: FigureQA の dot_line。x は連続値の float、2〜7系列、点ごとの x, y がある [論文/公式]。ただしデータの利用規約がダウンロード時の同意画面にしかなく、[未確認]。
   - 代替: ICDAR 2019 Adobe Synth の scatter(test 550)。実データを matplotlib で描いたもので、系列ごとの (x, y) がある [論文]。ただし配布元が落ちていて、ライセンスは [未確認]。

---

## Part A: 各モデルの公表している chart-to-table の結果

### A-1. DePlot(google/deplot)

- 論文: Liu et al., "DePlot: One-shot visual language reasoning by plot-to-table translation", Findings of ACL 2023, https://arxiv.org/abs/2212.10505
- 公表値 [論文 Table 4]: "Benchmarking plot-to-table conversion accuracy on the PlotQA dataset (all individual plots in PlotQA test sets)"

  | Model | RNSS | RMS_F1 |
  |---|---|---|
  | ChartOCR | 81.0 | 60.1 |
  | PaLI-17B (res. 224) + plot-to-table | 77.2 | 24.8 |
  | PaLI-17B (res. 588) + plot-to-table | 90.5 | 74.9 |
  | MatCha | 95.4 | 92.3 |
  | **DePlot** | **97.1** | **94.2** |

- テストデータ [論文]: PlotQA のテスト図の表。"plot-table pairs from v1 and v2 are identical"。PlotQA のテスト図は 33,657 枚 [論文 PlotQA Table 3]。DePlot 論文の Table 3 は PlotQA の v1/v2 を「33K tables」と記載。
- 学習データ [論文 Table 2]: 自作の合成データ 270K、ChartQA 22K、PlotQA 224K(各 1/3)。表は ChartQA と PlotQA の train split からだけ取っている。つまり **PlotQA のテストは学習分布の内側**。
- 論文の DePlot の chart-to-table の数値は **PlotQA だけ**。ChartQA の数値は他の論文に載っている。TinyChart の Table 2 に "Deplot+Codex" の行があり、Chart-to-Table の RMS-F1 は 87.22 [論文 TinyChart Table 2]。この値の出典は TinyChart の引用 [10]/[35] 周辺だが、出典元は [未確認]。
- 指標の定義 [論文 §3.1]: RMS は表を「(行ヘッダ, 列ヘッダ) → 値」の写像の集合として扱う。キーの類似度は正規化 Levenshtein 距離 NL_τ、値の類似度は相対誤差 D_θ = min(1, ‖p−t‖/‖t‖) で測る。Hungarian 法で対応づけて precision / recall / F1 を出す。表を転置したものも計算して、F1 の高い方を採る。
- 公式の評価コード [公式]: https://github.com/google-research/google-research/tree/master/deplot
  - `metrics.py`(Apache-2.0)
    - `table_datapoints_precision_recall(targets, predictions, text_theta=0.5, number_theta=0.1)` → RMS
    - `table_number_accuracy(targets, predictions)` → RNSS
    - 区切りは " | " と改行(`<0x0A>` も可)。先頭行は "title |"。
  - `evaluate_chart_to_table.py`: 正解 CSV と予測 CSV(zip)か、`target`/`prediction` を持つ jsonl を受け取り、RMS と RNSS を出力する。
- 再現上の注意:
  - PlotQA の annotations.json から**正解の表を作る手順は公開されていない** [未確認]。行ヘッダ(x の目盛りラベル)や列ヘッダ(凡例名)の表記がずれると、text_theta=0.5 を超えた時点でキーの一致が 0 になり、RMS が大きく下がる。RNSS はこの影響を受けないので、RNSS の 97.1 を主な比較対象にし、RMS-F1 は補助にするのが安全 [推測]。
  - 論文の付録 Table 11 に、dot と x ラベルがずれて正しい値を出せない例が載っている [論文]。dot_line の図で下振れしうる。
  - 推論は temperature 0(決定的)、最大系列長 512 [論文]。

### A-2. UniChart(ahmed-masry/unichart-base-960)

- 論文: Masry et al., "UniChart", EMNLP 2023, https://arxiv.org/abs/2305.14761
- 公表値 [論文 Table 2]: Chart-to-Table を (RNSS | RMS_F1) で報告。

  | モデル | ChartQA | WebCharts(ゼロショット) |
  |---|---|---|
  | MatCha | 85.21 \| 83.49 | 44.37 \| 17.94 |
  | UniChart | **94.01 \| 91.10** | **60.73 \| 43.21** |

  脚注: "All the results are calculated after finetuning UniChart pretrained checkpoint except for WebCharts (zero-shot)"。本文は "Chart-to-Table: we use ChartQA for both fine-tuning and evaluation"。
- 他の論文の値: ChartAssistant(arXiv 2401.02384)の Table 11 に、PlotQA の chart-to-table RMS-F1 として UniChart 70.8、MatCha 82.7、ChartAst-S 95.6 がある [論文(HTMLをWebFetch要約で確認)]。UniChart がどのチェックポイントで評価されたかは [未確認]。
- 公開チェックポイント [公式 README]: unichart-base-960(事前学習のみ。`<extract_data_table> <s_answer>` で表を出す)、chartqa-960、chart2text-statista-960、chart2text-pew-960、opencqa-960。**chart-to-table でファインチューニングしたチェックポイントは無い** [公式]。
- 評価コード: リポジトリ(MIT)に RNSS/RMS の評価スクリプトは無い [公式]。DePlot の `metrics.py` を使う。
- 事前学習コーパス [論文 Table 4]: 611,934 図。Statista、OWID、OECD、PlotQA(157,070)、Beagle、ChartInfo、データ拡張、ExcelChart400K、PewResearch など。**種類は bar / line / pie だけで、散布図は無い**。HF の `ahmed-masry/UniChart-pretrain-images` と `ahmed-masry/unichart-pretrain-data` で公開 [公式]。
- 再現上の注意: 公表値の 94.01/91.10 はファインチューニング後のモデルの値。私たちが使っている base-960 では**同じ値は期待できない**。ChartQA は事前学習で表の生成タスクに使われているので大崩れはしないと見込むが、数ポイント下がっても環境の問題とは言えない [推測]。比べるなら「base-960 で RMS-F1 が 80 台後半〜90 前後なら妥当」程度の判定にとどめる [推測]。

### A-3. TinyChart-3B-768(mPLUG/TinyChart-3B-768)

- 論文: Zhang et al., "TinyChart", arXiv 2404.16635(EMNLP 2024)
- 公表値 [論文 Table 2, Chart-to-Table RMS_F1]:

  | モデル | RMS-F1 |
  |---|---|
  | **TinyChart@768** | **93.78** |
  | TinyChart@512 | 92.93 |
  | ChartAst-13B | 91.60 |
  | ChartLlama | 90.00 |
  | Deplot+Codex | 87.22 |
  | Llava1.5 | 48.95 |

  アブレーション(ベンチマークデータだけで学習、768, r=84)では 88.90 [論文 Table 6]。
- テストデータ [論文 §4.2]: "We evaluate the performance of Chart-to-Table with the data table annotation provided by ChartQA following [10, 35]. We report RMS_F1 as the metric."
- 学習データ [論文 Table 1]: chart-to-table の学習データは ChartQA 19,373、**PlotQA 190,720**、Chart2Text-8k 8,305、DVQA 300,000、Statista 29,589。ほかに ChartLlama の 148,398 も入っている。
- データ配布 [公式 HF]: `mPLUG/TinyChartData`(Apache-2.0、25.9GB、train 1.52M / test 9.06k)。README には `data/tinychart_images, train.json, test.json` の構成で置くとある。test.json の中で chart-to-table の項目が何件あるかは [未確認]。
- 評価コード [公式]: `X-PLUG/mPLUG-DocOwl/TinyChart/tinychart/eval/eval_chart2table.py`(ファイルヘッダは Google Research の著作権表示で、DePlot の metrics と同じ実装。text_theta=0.5, number_theta=0.1)。
  - 入力は `gt_answer` / `model_answer`。どちらも小文字化し、先頭に "title |\n" を付ける。
  - 実行は `scripts/evaluate.sh` → `eval_model.py` → `run_eval.py`。
  - リポジトリは Apache-2.0 [公式]。
- 再現上の注意: 公開されている重みは論文と同じ TinyChart@768 なので、**5モデルの中で一番まっすぐ再現できる**。テスト項目とプロンプトも TinyChartData の test.json にある [公式/推測]。

### A-4. ChartGemma(ahmed-masry/chartgemma)

- 論文: Masry et al., "ChartGemma", arXiv 2407.04172(COLING 2025)
- **chart-to-table(RNSS/RMS)の数値は報告されていない** [論文(HTML要約)]。
  - ただし指示調整のタスクに "Chart-to-Markdown"(表を Markdown で生成する)が含まれる [論文]。
  - 指示調整データは 122,857 図。PlotQA、Statista、Pew、OECD、OWID、WebCharts から取っている [論文]。HF `ahmed-masry/ChartGemma` で公開 [公式]。
  - リポジトリは GPL-3.0 [公式]。
- 一番近い公表値 [論文(HTML要約)]: ChartQA の relaxed accuracy は aug 90.80 / human 69.52。ほかに ChartFC 70.33、ChartCheck T1/T2 71.50/74.31。
- 再現方針: ChartQA テストの QA を relaxed accuracy(5% 許容)で一部再現して、環境が正しいことを示す。chart-to-markdown の RMS-F1 も参考値として出すが、論文の値が無いので比較はできない(数値の再現率を見た既存の sanity check を補強する位置づけ)。

### A-5. Granite Vision 4.1 4B(ibm-granite/granite-vision-4.1-4b)

- 公表値 [公式 HFカードの bench_c2c.png を目視で読んだ]: "Chart2CSV — ChartNet human-verified test set · LLM-as-a-judge (GPT-4o) scores 0–100"

  | モデル | 点 |
  |---|---|
  | Gemini-3.1-Pro | 73.8 |
  | **Granite-Vision-4.1-4B** | **71.9** |
  | GPT-5.4 | 69.2 |
  | Claude-Opus-4.6 | 65.9 |
  | Qwen3.5-4B | 59.9 |
  | InternVL3.5-4B | 57.0 |
  | Ministral-3-8B | 47.7 |
  | Gemma4-E4B | 45.0 |

  タスクタグは `<chart2csv>`。モデルのライセンスは Apache-2.0 [公式]。
- テスト集合 [公式 ChartNet HF]: `human_verified` の test は 2,000 件、train は 92.6K。
  - ライセンスは "Restricted"。本文は "may not be used for commercial exploitation or commercial deployment"、"Provided solely for verification and evaluation of published results" [公式]。
  - 公表値の検証が目的なら利用できるが、**再配布はできない**と読むのが安全 [推測]。
- 評価コードと judge プロンプト: HFカードにも ChartNet 論文(arXiv 2603.27064)の読めた範囲にも、実装やプロンプト本文は無かった [未確認]。
- 参考: ChartNet 論文の held-out 2,000 件では、Granite-Vision-2B(ChartNet で学習)が 70.3、GPT-4o が 46.7 [論文(HTML要約)]。
- 再現上の注意:
  - judge の プロンプトが公開されていないので、71.9 の厳密な再現はできない。
  - 代わりに、同じ 2,000 件(のサブセット)で次の2つを並べる。(i) 自前の GPT-4o 系 judge(プロンプトは公開して固定)、(ii) DePlot の RMS-F1 / RNSS。そのうえで「Granite が ChartNet で高い値を出す環境である」ことを示す [推測]。
  - Granite は ChartNet で学習しているので、このテストは学習分布の内側 [論文/推測]。

### A-6. 再現計画(具体案)

サブセットの大きさ: RMS-F1 は図ごとの平均なので、図ごとの標準偏差を 0.15〜0.25 と仮定すると、n=300 で標準誤差 ±1.0〜1.5pt、n=500 で ±0.7〜1.1pt [推測]。公表値と ±2〜3pt 以内に入れば再現できたとみなす目安として、**各 300〜500 図**で足りる。

| モデル | データ / split | 件数案 | 指標・コード | 期待値(公表) |
|---|---|---|---|---|
| DePlot | PlotQA test。annotations.json から表を作る | 500(vbar / hbar / line / dot_line を層化。dot_line を必ず含める) | deplot/metrics.py の RNSS(主)と RMS-F1(補) | 97.1 / 94.2 |
| DePlot(補) | ChartQA test の表 | 300 | 同上 | RMS-F1 87.22(TinyChart 論文の DePlot+Codex 行) |
| UniChart base-960 | ChartQA test の表(`<extract_data_table>`) | 300〜500 | 同上 | 94.01 / 91.10(ファインチューニング版。base ではやや低いと予想) |
| TinyChart@768 | TinyChartData test.json の chart-to-table 項目(ChartQA 由来) | 300〜500、または全件 | TinyChart の eval_chart2table.py(= DePlot の RMS) | RMS-F1 93.78 |
| ChartGemma | ChartQA test の QA(aug / human) | 各 500 | relaxed accuracy 5% | 90.80 / 69.52 |
| Granite 4.1 4B | ChartNet human_verified test | 300〜500(全 2,000 件も可) | 自前 judge(プロンプト公開)+ RMS-F1 | 71.9(judge が違うので参考値) |

- 共通: 予測の表を " | " と改行の形に正規化してから、公式の `metrics.py` に渡す。正解の表を作る手順(PlotQA と ChartNet の CSV → markdown)はコードとテストとして公開する。
- ChartQA(GPL-3.0)、TinyChartData(Apache-2.0)、ChartNet human_verified(Restricted)はローカルの評価だけに使う。リポジトリには入れない [推測(ライセンスの読み)]。

---

## Part B: マーカー型 / 散布図を含む既存の合成データセット

| データセット | マーカー / 散布図 | 数値の x 軸? | 対数軸 | 多系列 | 正解の粒度 | データのライセンス | ローカル評価 / 約100枚の再配布 | 入手経路・サイズ |
|---|---|---|---|---|---|---|---|---|
| **PlotQA** | `dot_line`(散布図。マーカーは asterisk / circle / diamond / square / triangle / inverted triangle からランダム)[論文]。test 5,574、train 26,010、val 5,571 [論文 Table 3] | ×。x の要素は 2〜12 個の離散値(年・国など)[論文] | ×。"values on a linear scale" [論文] | 1〜4系列 [論文] | 点ごとの x, y と bbox。軸、凡例、plot 領域の bbox もある(annotations.json の models)[公式] | CC-BY-4.0 [公式 README(先行メモで確認)] | 可 / 可(クレジット表記が要る) | Google Drive(PNG + JSON)[公式]。サイズは [未確認] |
| **FigureQA** | dot_line と line [論文/公式] | ○。x は Float の配列(連続値)[公式 annotations_format.md / 論文] | ×(記載なし)[論文] | 2〜7系列(色名で区別)[論文] | 系列ごとの x, y と、線分または点の bbox [公式] | コードは MIT [公式]。データの規約はダウンロード時の同意画面にあり、内容は [未確認] | 評価はおそらく可 / 再配布は [未確認] | Microsoft Download。圧縮 3.53GB / 展開 5.78GB [公式]。test の注釈が公開されているかは [未確認](val を使うのが安全) |
| ICDAR 2019 Adobe Synth(CHART2019-S) | Scatter: train 41,703 / test 550 [論文 Table I]。マーカーのスタイルも変えている [論文] | ○の見込み(World Bank 等の指標どうしの散布)[推測] | [未確認] | [未確認] | Task 6b: 名前付きの (x, y) 系列 [論文] | "publicly released for academic use" [論文]。正式なライセンスは [未確認]。tc11 は 502 で取得できない | 評価はおそらく可 / 再配布は [未確認] | tc11.cvc.uab.es(落ちている)/ chartinfo.github.io |
| ICPR 2020 Adobe Synth | Scatter と Scatter-Line を含む 15 種類 [検索結果の二次情報] | [推測] ○ | [未確認] | [未確認] | 同上 | [未確認] | [未確認] | tc11(取得できない) |
| Scatteract(Bloomberg) | 合成の散布図。生成スクリプトで train 25,000 / test 500 を作る [公式 README] | ○ [推測] | [未確認] | [推測] 単系列 | 点ごとのピクセル座標(.idl)と目盛り [公式] | **LICENSE が無い**(GitHub API で license=null)[公式] | 生成器を自分で回すことになり、「自前で生成しない」方針に反する。データの再配布は不可寄り | github.com/bloomberg/scatteract |
| ChartQA | bar / line / pie のみ。散布図は無い [推測(論文の一般知識)] | × | × | ○ | 表(CSV) | GPL-3.0 [公式] | 評価は可 / 再配布は避ける | GitHub / HF |
| ChartX | 18 種類。**scatter は無い**(bubble はある)[公式] | – | – | – | CSV とコード | CC-BY-4.0 [公式] | 可 / 可 | Google Drive / HF U4R/ChartX。GT の誤りが報告されている(PlotPick, arXiv 2605.06021)[論文] |
| ChartBench | 2D scatter / 2D scatter smooth / 3D scatter が test に各 50、train に計 2,046 [論文] | ○ [推測] | [未確認] | [未確認] | QA(Acc+ の yes/no)が中心。図ごとの数値表が公開されているかは [未確認] | HF のメタデータは MIT、論文は CC BY 4.0 [公式] | 評価は可だが数値の正解が無ければ不適 | HF SincereX/ChartBench、10.4GB [公式] |
| ChartLlama データ | 種類の内訳は [未確認](HF のプレビューは box plot)| [未確認] | [未確認] | [未確認] | QA の会話だけ。表は見えない [公式] | リポジトリは MIT。データのライセンスは [未確認] | 不適 | HF listen2you002/ChartLlama-Dataset、18.5MB |
| MMC | arXiv の実図 + 既存データセット [公式] | – | – | – | 指示データと yes/no・選択式の QA | GPL-3.0 [公式] | 不適 | HF xywang1/MMC |
| EvoChart | 合成の学習データと、実図 650 枚の EvoChart-QA [論文 abstract] | [未確認] | [未確認] | [未確認] | QA 中心 [推測] | 論文は CC BY 4.0。データは [未確認] | 不適寄り | [未確認] |
| **ChartNet**(IBM, 2026) | 24 種類。Scatter Plot と Bubble を含む [公式]。件数と比率は [未確認] | ○ [推測] | [未確認](コードの中の set_xscale('log') で絞れる [推測]) | [未確認] | 図ごとの CSV とプロット用コード(コードから点を再構成できる)[公式] | `core_permissive` 2.51M は CDLA-Permissive-2.0。ほかのサブセットは Restricted(研究・評価のみ、商用不可)[公式] | permissive は 可 / 可(ライセンス文を添える)[推測] | HF ibm-granite/ChartNet、parquet 1万件あたり約1.66GB、全体 1.02TB [公式] |
| UniChart 事前学習コーパス | bar / line / pie のみ [論文 Table 4] | – | – | – | 表 | 出典ごとに異なる [論文] | 不適 | HF ahmed-masry/UniChart-pretrain-* |
| TinyChartData | ChartQA、PlotQA、DVQA、ChartLlama 等の再パッケージ [論文 Table 1] | PlotQA 由来の部分は × | × | ○ | QA 形式の会話(chart-to-table は表を文字列で持つ) | Apache-2.0 [公式] | 可 / 再パッケージなので元(PlotQA = CC-BY)のライセンスにも従う [推測] | HF mPLUG/TinyChartData、25.9GB |
| ChartGemma 学習データ | PlotQA ほか [論文] | 一部(PlotQA) | × | ○ | 指示と応答 | [未確認](リポジトリは GPL-3.0) | 不適 | HF ahmed-masry/ChartGemma |
| GenPlot(2023) | bar / scatter / line / dot。scatter 100,000 図、1図あたり 3〜86 点、x も y も数値 [論文] | ○ [論文] | [未確認] | [未確認] | 生成時のデータ [推測] | Kaggle で配布。ライセンスは [未確認] | [未確認] | Kaggle |
| Benetech Making Graphs Accessible(Kaggle, 2023) | scatter / dot / line / bar。generated と extracted の2種類 [検索結果] | scatter は ○ [推測] | [未確認] | 単系列中心 [推測] | data-series(x, y)[推測] | Kaggle のコンペ規約(本文は [未確認]) | 再配布は不可寄り [推測] | Kaggle |
| BIY(Feedzai, VIS 2025) | 合成の散布図 18,921 枚(クラスタ・外れ値の検出が目的)[論文] | ○ | × | クラスタ | 点ごとの座標 [論文] | **CC BY-NC-ND 4.0** [論文] | 評価は可 / ND なので改変した再配布は不可 | github.com/feedzai/biy-paper |
| PUB / ClimateViz / SciGraphQA など | line・scatter の数値の正解が無い、または対象外(先行メモのとおり) | – | – | – | – | – | – | – |

要点:

- マーカーと点ごとの数値の正解とライセンスの3つがそろい、しかもモデルの公表ベンチマークと重なるのは **PlotQA だけ**。
- 数値の x 軸(温度のような連続量)を持つマーカー図は FigureQA、ChartNet、Adobe Synth にある。ただしライセンスまたは規模の点で、どれも一長一短。
- 対数軸を明記しているマーカー型の既存合成データは見つからなかった [未確認]。対数軸は ChartNet のコードをフィルタすれば拾える可能性がある [推測]。

---

## 推奨

### (1) Part A の再現計画(優先順)

1. **TinyChart@768**: TinyChartData の test.json の chart-to-table 項目(ChartQA)を全件、または 500 件。公式の `eval_chart2table.py` で採点し、93.78 と比べる。一番まっすぐ再現できる。
2. **DePlot**: PlotQA test から 500 図(4種類を層化)。`deplot/metrics.py` で RNSS 97.1 / RMS-F1 94.2 と比べる。正解の表は annotations.json の models(name と x, y)から「x ラベル | 系列名…」の形で作る。RNSS を主な判定に使う。
3. **UniChart base-960**: ChartQA test の表を 300〜500 図。94.01 / 91.10 はファインチューニング版の値なので、base では数ポイント低くても許容と事前に宣言しておく。PlotQA の 500 図も同時に採点し、ChartAst が報告した 70.8 を参考値にする。
4. **ChartGemma**: ChartQA test の QA を aug / human 各 500。relaxed accuracy を 90.80 / 69.52 と比べる。chart-to-markdown の RMS-F1 は参考値として出す。
5. **Granite Vision 4.1 4B**: ChartNet human_verified test を 300〜500 件(Restricted なのでローカルのみ)。judge が公開されていないので、71.9 は参考値扱い。プロンプトを公開した自前 judge と RMS-F1 を並べて出す。

### (2) Part B: マーカー型の合成データ(合計 150〜200 図)

- **主: PlotQA の dot_line を 100 図**(test split、乱数シードを固定)。
  - 層化は系列数(1 / 2 / 3–4)× y の桁(小 / 大)× マーカーの形。
  - A の DePlot 再現と同じ PlotQA test の中から取る。A では4種類で全体の値を、B では dot_line だけの値を出して、「同じモデルが自分の学習分布の中のマーカー図ではどうか」を直接示す。
  - CC-BY-4.0 なので、100 図と正解をリポジトリに入れられる(出典とライセンスを明記)。
  - 最初のパイロットで実物を 20 図見て、次の3点を確認する: マーカーだけか線もあるか、x ラベルの型、bbox と値が1対1か [未確認]。
- **副: 数値の x 軸を補うために 50〜100 図**。次の順で選ぶ。
  1. ChartNet `core_permissive` の Scatter Plot(と Line でマーカーがあるもの)。parquet をストリーミングで読み、chart_type でフィルタする。コードの中の `marker=` や `scatter(`、`set_xscale('log')` / `loglog` で層化する。CSV と描画された点が1対1かは、コードを読んで機械的に検証できる図だけを採用する。CDLA-Permissive-2.0 なので再配布できる。Granite の学習分布でもある。
  2. ChartNet が使えなければ FigureQA の dot_line(validation split)。x は連続値、2〜7系列。ただし利用規約を確認できるまでリポジトリには入れず、ローカルの評価だけにする。
  3. Adobe Synth 2019 の scatter test(550 図)は、配布元とライセンスを確認できたときだけ使う(司令塔・オーナーの判断事項)。
- 採点: real-chart-bench と同じ点単位の F1(データ空間)をそのまま使う。正解は PlotQA なら models の x, y、ChartNet なら CSV またはコードから復元した点列。
- 比較の筋書き: 同じモデルが、(a) 自分のベンチマークでは公表値を再現し(A)、(b) 合成のマーカー図でもそれなりに取れ(B)、(c) 実論文の図では大きく落ちる、という3段で示す。

## 司令塔・オーナーの判断が要る点

1. ChartNet human_verified(Restricted、"verification and evaluation of published results")を、Granite の再現のためにローカルで使ってよいか。
2. FigureQA と Adobe Synth のデータ利用規約は [未確認]。再配布しないローカル評価に限るかどうか。
3. Granite の Chart2CSV 再現で使う自前 judge のモデルとコスト(GPT-4o 系を使うかどうか)。
4. UniChart の公表値はファインチューニング版の値で、base-960 では一致しない前提で進めてよいか。

## 参照 URL

- DePlot 論文: https://arxiv.org/abs/2212.10505
- DePlot コード・指標: https://github.com/google-research/google-research/tree/master/deplot
  - https://raw.githubusercontent.com/google-research/google-research/master/deplot/metrics.py
  - https://raw.githubusercontent.com/google-research/google-research/master/deplot/evaluate_chart_to_table.py
- UniChart: https://arxiv.org/abs/2305.14761 、https://github.com/vis-nlp/UniChart
- ChartAssistant(PlotQA chart-to-table の比較値): https://arxiv.org/html/2401.02384
- TinyChart: https://arxiv.org/abs/2404.16635 、https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart
  - https://raw.githubusercontent.com/X-PLUG/mPLUG-DocOwl/main/TinyChart/tinychart/eval/eval_chart2table.py
  - https://huggingface.co/datasets/mPLUG/TinyChartData
- ChartGemma: https://arxiv.org/abs/2407.04172 、https://github.com/vis-nlp/ChartGemma
- Granite Vision 4.1: https://huggingface.co/ibm-granite/granite-vision-4.1-4b(bench_c2c.png)
- ChartNet: https://huggingface.co/datasets/ibm-granite/ChartNet 、https://arxiv.org/abs/2603.27064
- PlotQA: https://arxiv.org/abs/1909.00997 、https://github.com/NiteshMethani/PlotQA(PlotQA_Dataset.md)
- FigureQA: https://arxiv.org/abs/1710.07300 、https://github.com/Maluuba/FigureQA(docs/annotations_format.md)、https://www.microsoft.com/en-us/research/project/figureqa-dataset/download/
- ICDAR 2019 CHART-Infographics: https://par.nsf.gov/biblio/10188725 、https://chartinfo.github.io/ 、http://tc11.cvc.uab.es/datasets/CHART2019-S_1
- ICPR 2020: https://research.adobe.com/publication/icpr-2020-competition-on-harvesting-raw-tables-from-infographics
- Scatteract: https://github.com/bloomberg/scatteract
- ChartX: https://github.com/UniModal4Reasoning/ChartVLM
- ChartBench: https://arxiv.org/abs/2312.15915 、https://huggingface.co/datasets/SincereX/ChartBench
- ChartLlama: https://github.com/tingxueronghua/ChartLlama-code 、https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset
- MMC: https://github.com/FuxiaoLiu/MMC
- EvoChart: https://arxiv.org/abs/2409.01577
- GenPlot: https://arxiv.org/abs/2306.11699
- Benetech(Kaggle): https://www.kaggle.com/competitions/benetech-making-graphs-accessible
- BIY: https://arxiv.org/abs/2510.06071 、https://github.com/feedzai/biy-paper
- PlotPick: https://arxiv.org/abs/2605.06021

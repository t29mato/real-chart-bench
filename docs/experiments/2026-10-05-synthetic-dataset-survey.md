# 合成チャートデータセット調査(synthetic-vs-real ギャップ測定用)

調査日: 2026-10-05 / 方法: Web調査(WebFetch/WebSearch)のみ。データのダウンロード・実物確認はしていない。

凡例: **[確認]** = 公式ページ(GitHub / HF / 公式サイト / arXiv abstract)の記述を直接確認。**[推測]** = 論文・二次情報・一般知識からの推定。**[未確認]** = 取得失敗または記載なし。

## 要件の再掲

数値x軸・数値y軸(対数軸含む)の line / scatter、複数系列、マーカー。指標はデータ空間にマップした点単位F1。画像ごとに系列別の数値(x,y、できればマーカー単位)の正解と、できれば軸範囲/スケールが必要。評価利用と、小サブセット(200枚以下 + GT)のリポジトリ内再配布が可能なライセンスが必要。

## 結論(先出し)

1. 要件を全て満たす既存データセットは見つからなかった。特に **対数軸 + マーカー単位GT + 数値x軸** を同時に保証するものは確認できなかった。
2. **第1推奨: PlotQA**(line / dot_line のみ使用)。CC-BY-4.0 を公式READMEで確認でき、再配布条件が最もクリア。系列別 (x,y) と軸情報がJSONにある。ただし x が数値軸ではなくカテゴリ的(年など)の可能性が高く(推測)、対数軸は無いと推測。
3. **第2推奨(対数軸・マーカー・数値xのカバー): 自前 matplotlib 生成セット**。ICDAR 2019 Adobe Synth(CHART2019-S)の生成レシピ(実データ由来の系列 + matplotlib、軸/凡例/系列GTを出力)に倣い、PlotQA/FigureQA の生成方針(ランダム化)と AutoChart のマーカー10種ランダム化を取り込む。自前生成ならライセンス問題が無く、200枚以下をそのままrepoに入れられる。
4. 補助候補: **ChartNet `core_permissive`**(CDLA-Permissive-2.0、図ごとにCSV + matplotlib等のコード)。コード中に `set_yscale('log')` 等がある図を機械的に抽出すれば対数軸・散布図を拾える可能性があるが、サイズが巨大(1.02TB)で、CSVが「プロットされたマーカーそのもの」かは未検証。要パイロット。

## 候補比較表

| データセット | 規模 | 関連チャート | GT | データのライセンス | 評価利用/200枚再配布 | 備考 |
|---|---|---|---|---|---|---|
| PlotQA | 224,377 plots / 28.9M QA [確認] | line, dot_line(+bar) [確認]。1-4系列 [確認] | JSONに models(name, color, bbox, 各データ点の x,y値)と general_figure_info(title, x/y axis, legend, plot_info) [確認] | CC-BY-4.0(データ)、MIT(コード・モデル) [確認: GitHub README] | 可(帰属表示必須) | PNG + JSON、Google Drive配布 [確認]。x軸型・対数軸・マーカーは未確認 |
| CHART2019-S (ICDAR 2019 Adobe Synth) | line 約41.9k train / 1k test(line)、全体で多数 [確認: 二次情報]。LineFormer論文は約38k line と記述 [確認: 二次情報] | line, scatter, bar, box [確認] | 実データ由来、matplotlib生成。Task 4(軸)/6.b(データ抽出)用GT [確認: 二次情報] | "publicly released for academic use" [確認: Adobe研究ページ]。正式なライセンス名は [未確認](tc11.cvc.uab.es は502で取得不可) | 評価は可の可能性が高いが、再配布は不明 → 司令塔/オーナーが主催者(UB/Adobe)に確認要 | 軸/凡例/プロット要素GTまであり、我々の指標に最も近い設計 [推測] |
| ChartDataset2023(Mendeley Data yf2wz2n8nx) | 25,894枚 [確認] | Area, Bar, Box, Donut, Line, Pie, Scatter [確認] | 要素抽出アルゴリズムによる自動アノテーション。数値データGTかは [未確認](画素/要素レベルと推測) | CC BY 4.0 [確認: Mendeley] | 可だが、数値GTが無ければ不適 | 数値抽出向けではない可能性が高い |
| FigureQA | 規模は公式ページに記載なし [確認]。(論文では約100万QA/約12万図 [推測]) | line, dot line(+bar, pie) [確認] | yes/no QAと要素注釈。数値データは公開ページに記載なし [確認] | 公式ページに記載なし [確認]、別途 Maluuba のDLページ/リポジトリ要確認 [未確認] | 不可寄り(GT不適合) | 色名つき系列・数値軸なし。対象外 |
| ChartQA | 約10k〜100k [確認: HF] | line(+bar, pie)。Pew等の実チャート + 合成混在 [確認] | 図ごとの underlying data table(CSV) [確認] | GPL-3.0 [確認: GitHub, HF] | 評価は可だが GPL-3.0 は200枚再配布に伝染性の懸念 → 避ける | カテゴリx中心、系列値のみで座標GTなし。対象外寄り |
| LineEX | 合成430K枚 [確認: 論文/CVF] | 科学風 line | 点列GT | 「採択後に公開予定」と記載 [確認]。公式リポジトリ・ライセンスは見つからず [未確認] | 不明 | GitHubのURLを検索で特定できず(Mayank-Singh-Lab/LineEX は404) |
| LineFormer 学習データ | AdobeSynth19 + UB-PMC22 + LineEX [確認] | line | 上記由来 | リポジトリREADMEにライセンス記載なし、学習データの配布もなし [確認] | 不可(独自配布なし) | データは既存セットの寄せ集め |
| DePlot / MatCha 事前学習データ | - | - | - | READMEに合成事前学習データ公開の記載なし [確認] | 不可 | 未公開と見なす |
| ChartX | 48K、検証4,848/テスト1,152 [確認] | line, bar, pie ほか18種(散布図は明示なし) [確認] | 図ごとにCSV + Pythonコード [確認] | CC-BY-4.0 [確認: GitHub] | 可 | PlotPickが実GTの誤り(値の取り違え・列入れ替え)を報告 [確認: arXiv 2605.06021]。lineはカテゴリx中心 [推測] |
| ChartBench | 276,996行、10.4GB [確認] | area, bar, box, line, pie, scatter, radar ほか [確認] | Yes/No形式の真偽QA。数値データGTではない [確認] | MIT [確認: HF] | 評価利用は不適(GT不適合) | 対象外 |
| ChartNet (IBM Granite, 2026) | core 1.7M、core_permissive 2.51M行、全体1.02TB [確認] | 24種、6ライブラリ [確認]。line/scatter/log の有無・比率は未確認 | プロットコード + 画像 + データ表(CSV) + 要約 + QA [確認] | `core_permissive` は CDLA-Permissive-2.0、他サブセットは別制約 [確認: HF] | permissiveサブセットのみ可。200枚再配布可(CDLA-Permissive、帰属・ライセンス文表示) [推測] | Parquet、各1.66GB超 [確認]。LLM生成コードの図で、CSV=描画マーカー値とは限らない [推測] |
| ChartGen-200K (IBM) | 222.5K、27種(scatter, line含む) [確認: 検索結果] | scatter, line | 画像 + Pythonコード(データはコード内) | HFが401で取得不可 [未確認] | 不明 | 数値GTはコードから再構成が必要 |
| AutoChart | 公開 [確認] | scatter, line, bar(マーカー10種ランダム) [確認: 検索結果/論文] | 元の統計表(World Bank等) | 論文は arXiv非独占ライセンス、データの明示ライセンスなし [確認] | 不明 | 対数軸なし [確認]。生成レシピの参考にはなる |
| PUB | - | time series, histogram, violin, box, cluster [確認] | 生成パラメータ | CC BY 4.0(論文) [確認] | データ配布先が特定できず | line/scatter(数値)ではなく対象外寄り |
| WB-ChartExtract (2026, Berkane ほか) | 1,000〜10,000件 [確認: HF] | World Bank指標のチャート。チャート種別は未確認 | 351以上のCSV [確認] | CC-BY-4.0 [確認: HF] | 可 | HFビューアはスキーマ不整合で失敗中 [確認]。ChartQAの7倍のデータ点数 [確認: 論文]。実データ由来だが描画は合成(「合成」とみなせる)。時系列中心でx数値が年 [推測] |
| CharXiv(対照用の実チャート) | 2,320枚(val 1,000 / test 1,320) [確認] | 実際のarXiv図(line, scatter, 他) [確認] | 数値の元データGTは無い(QA中心) [確認: フィールドにoriginal_idのみ] | CC-BY-SA-4.0 [確認: HF] | ShareAlike条件あり。GTが無いため今回は対象外 | 実図の対照としてのみ有用 |

## 推奨と提案サンプル

### 推奨A: PlotQA(line / dot_line)から100枚

- 使う理由: ライセンスがCC-BY-4.0と明確で、再配布可。系列別のxy値と軸情報がJSONにある(公式ページ確認)。
- 事前確認事項(最初の30分のパイロット):
  - test split から line/dot_line を20枚だけ取り出し、x値が数値かカテゴリかを確認する。
  - 軸スケール(対数があるか)を確認する。
  - dot_line のマーカーと注釈データ点が1対1かを確認する。
- 層化(合計100枚):
  - 単一系列 50 / 複数系列(2-4) 50
  - line 50 / dot_line 50(= 線のみ vs マーカーあり)
  - y値のレンジ(桁)が小/大 で各層を半々
- 対数軸の層は PlotQA では取れない(推測)。そこは推奨Bで補う。

### 推奨B: 自前 matplotlib 合成セット(100枚 + 予備)

PlotQA だけでは「対数軸・数値x軸・マーカー単位」の比較ができないため、既存のレシピに倣って小規模に自作する。

- 参考にするレシピ:
  - ICDAR 2019 Adobe Synth: 実世界データ源から系列を作り matplotlib で描画。軸/凡例/系列のGTを出力する(論文: "ICDAR 2019 Competition on Harvesting Raw Tables from Infographics")。
  - LineFormer 論文付録/Chart-RCNN 系: matplotlib で 720x720、目盛り数5-10、目盛り範囲を [0,1]/[0,100]/[1000,10000] 等からランダム選択(検索結果の記述 [確認: 二次情報])。
  - AutoChart: scatterのマーカーを10種、色を20種からランダム。
  - PlotQA: 1-4系列、データの値域・ラベルのランダム化。
- 層化(合計100枚): 単一/複数系列 × 線形/対数(x, y, 両対数) × 線のみ/マーカーのみ/線+マーカー。1セル約8枚(12セル)+端数。
- GTは描画に使った数値配列(x, y)と axis limits / scale をそのままJSONに保存する。ライセンスは自分たちのもの(CC-BY-4.0等)にできる。

### 補助: ChartNet core_permissive からのフィルタ抽出(任意)

- コード列に `set_xscale('log')` / `set_yscale('log')` または `loglog` を含み、かつ `scatter`/`plot` を含む行だけをストリーミングで探し、数十枚を取り出す。
- 要検証: CSVのx,yが描画マーカーと一致するか。LLM生成なので画質・スタイルの偏りもある。100枚のうち20-30枚を目安にパイロットする。

## 主な未確認事項(司令塔・オーナー判断が必要)

1. CHART2019-S の正式ライセンス(tc11.cvc.uab.es が取得不可)。再配布するなら主催者に確認が必要。評価のみなら「学術利用向けに公開」の範囲で足りる可能性が高い(推測)。
2. LineEX の公式データ置き場とライセンスを特定できなかった。
3. FigureQA のライセンス表記(DLページ/リポジトリのLICENSE未確認)。ただし数値GTが無いため優先度は低い。
4. ChartQA は GPL-3.0 のため、200枚をrepoに含めると GPL の影響を受けうる。使うなら評価のみでローカルに限定する。
5. PlotQA の x 軸が数値軸かカテゴリか、対数軸が無いことは未確認(推測)。パイロットで確認する。
6. ChartGen-200K / Mendeley ChartDataset2023 のGT形式(数値GTの有無)は未確認。

## 参照URL

- PlotQA: https://github.com/NiteshMethani/PlotQA (PlotQA_Dataset.md 含む)
- CHART2019-S: https://tc11.cvc.uab.es/datasets/CHART2019-S_1 、https://research.adobe.com/publication/icdar-2019-competition-on-harvesting-raw-tables-from-infographics-chart-infographics
- ChartDataset2023: https://data.mendeley.com/datasets/yf2wz2n8nx
- LineFormer: https://github.com/TheJaeLal/LineFormer 、https://arxiv.org/abs/2305.01837
- LineEX: https://openaccess.thecvf.com/content/WACV2023/html/P._LineEX_Data_Extraction_From_Scientific_Line_Charts_WACV_2023_paper.html
- ChartX: https://github.com/UniModal4Reasoning/ChartVLM
- ChartBench: https://huggingface.co/datasets/SincereX/ChartBench
- ChartNet: https://huggingface.co/datasets/ibm-granite/ChartNet 、https://arxiv.org/abs/2603.27064
- ChartQA: https://github.com/vis-nlp/ChartQA 、https://huggingface.co/datasets/ahmed-masry/ChartQA
- FigureQA: https://www.microsoft.com/en-us/research/project/figureqa-dataset/
- CharXiv: https://huggingface.co/datasets/princeton-nlp/CharXiv
- WB-ChartExtract: https://huggingface.co/datasets/tberkane/WB-ChartExtract 、https://arxiv.org/abs/2605.27298
- PlotPick: https://arxiv.org/html/2605.06021
- AutoChart: https://arxiv.org/abs/2108.06897
- PUB: https://arxiv.org/abs/2409.02617
- ChartGen: https://arxiv.org/abs/2507.19492
- DePlot: https://github.com/google-research/google-research/tree/master/deplot

# 評価指標の文献調査(2026-10-07)

オーナー質問「どの論文のものが最も一般的な手法として捉えられている？」に対する調査記録。
論文への反映は 01-intro.md §1.2、03-evaluation.md §3.5.3 / §3.5.4。

## 結論

**分野によって別の標準がある。単一の標準は存在しない。**

| 系統 | 標準 | 出典 | 正規化の分母 |
|---|---|---|---|
| チャート→表(VLM / LLM) | RMS | DePlot (Liu et al. 2023) | 正解値の大きさ `min(1, |g-p|/|g|)` |
| チャート解析(文書解析コンペ) | CHART-Infographics Task 6/7 | Davila et al. 2019-2024 | 正解点の共分散(マハラノビス) |
| 点単位しきい値一致(散布図) | Scatteract 式(2) | Cliche et al. 2017 | 正解点の値幅 |

本研究の主指標は3番目の系統に属する。

## 一次資料で本文を確認したもの

- **Scatteract**(ar5iv HTML, arXiv 1704.06687)
  - 式(2): `|Xpred-Xtrue|/ΔXtrue <= 0.02 AND |Ypred-Ytrue|/ΔYtrue <= 0.02`、x と y を個別判定
  - 対応づけ: 全ペアの距離を計算し、各正解点に最も近い予測点から確定、確定した点は双方から除く
  - F1 > 0.8 を「抽出成功」と定義。合成散布図の 89% で成功
  - **FigureSeer について**: 「FigureSeer や従来の手法はピクセルからチャート座標への変換を含んでいない」と明記
- **CHART-Infographics の指標定義**(chartinfo.github.io/metrics/metric.pdf)
  - Point Set(散布図): 正解点の共分散 V で正規化したマハラノビス距離、γ で割って 1 で打ち切り
  - `V^-1` の対角成分に `400 / mean^2` の上限 → 「最大の罰(=1)は予測値が正解の 5γ% 以上違うときにのみ起こる」
  - コスト行列を 1 で正方にパディングし、ハンガリアン法。スコア = `1 - cost/max(N,M)`
  - 系列名の寄与: `D = 1 - max(metric^β, (1-L(n1,n2)^α) * metric)`
  - **我々の実装(scripts/eval/repro/third_party/chartinfo/metric6b.py)の読みと一致**
- **From Pixels to Insights(総説, arXiv 2403.12027)**
  - チャート→表の指標は RNSS / RMS / SCRM の3つのみ。SOTA 表(Table VI)は `RMS F1` で統一
- **EpiCurveBench**(arXiv 2605.27195)
  - 実図 1,000 枚(折れ線 71.4% / 棒 15.7% / 混在 12.9%)
  - ECS: Edit Distance with Real Penalty + DP 整列。差は **`|pred-true|/(y_max-y_min)`**、しきい値は y 軸レンジの 1%
  - RMS / SCRM 批判: 「時間方向のずれを完全な不一致として罰する」。4つの VLM を ~5点の帯に圧縮してしまう
  - 最良モデルでも 52.3% ECS
- **ExChart-Bench**(arXiv 2606.29808)
  - 3,600 対(実図 744 / 合成 2,856)、5 種。出所 ChartQAPro / CharXiv / web / Vega-Lite / Matplotlib
  - Adaptive MAPE: 誤差を **図内の最大絶対値**で割る(正解値や平均で割ると小さい値が過大に効く)
  - GLM-4.5V 5.94%、Qwen2.5-VL 7B 16.01%、自家 ExChart 7B 4.87%
- **PlotPick**(arXiv 2605.06021)
  - 数値 F1、許容は**値に対する相対 5%**、ラベルなしの集合として照合
  - 評価は **合成のみ**(ChartX 300 / PlotQA 529)。「どちらのベンチマークも出版された生物医学の図に似ていない」と自ら明記
  - WebPlotDigitizer や人手との比較はしていない(論文が限界として明記)
- **LineFormer**(検索結果の本文引用)
  - 「FigureSeer と線形計画の手法は点ごとの差を 2% のしきい値で二値化する」
  - LineFormer 自身は「微小なずれを捉える精度が高い」として**連続値の積分**を採用 → しきい値方式を意図的に退けている

## 未検証(要旨・検索結果のみ。引用前に本文確認が必要)

- **PaperUnPlot**(TaDA 2026 / VLDB Workshops、Bertillo ら)
  - 実図 950 対、10 種、PubMed Central + arXiv。35.6 万枚のチャート分布分析つき
  - **NMS**: 「RMS は数値誤差を軸レンジではなく対象値の大きさで正規化し、対数軸を考慮しない」→ 両方を修正
  - GPT-4o / GPT-4o Mini / Gemini 2.5 Flash / Claude Sonnet 4.5 を評価。全モデルが合成 > 実図
  - **PDF が解析できず本文未確認**(TaDA のサイト、2MB)
- **MetalThermoChartIE / PolyCompChartIE**(OpenReview vj8dqNrzEe)
  - 材料科学の図。**RCLS**(Relative Coordinate-Label Similarity)を提案
  - LLaMA 3.2-Vision 11B / Qwen2.5-VL-7B を in-domain で fine-tune
  - **OpenReview のブラウザ検証で取得不能**。本研究の領域に最も近いので優先して確認すべき
- **Turan & Sparks, IMMI 15:212-223 (2026)**「Revolutionizing Scientific Figure Decoding」
  - 材料科学の図。ChatGPT 4o / Perplexity SONAR / Gemini 2.5 Pro
  - 指標は point count error% と mean X/Y error
  - Springer が認証リダイレクト、chemRxiv の PDF が 403
- **FigureSeer**(Siegel et al., ECCV 2016)の評価基準
  - 2% しきい値を使っていたかは二次資料が食い違う(上記 LineFormer vs Scatteract)
  - PDF が複数経路で取得できず(403 / 404)

## 引用回数(Semantic Scholar、レート制限で部分取得のみ)

| 論文 | 引用 | 年 |
|---|---|---|
| PlotQA | 498 | 2019 |
| ChartOCR | 155 | 2021 |
| Scatteract | 79 | 2017 |

DePlot / ChartQA / LineFormer / CHART コンペは 429 で取得できなかった。
**引用回数を「どれが標準か」の根拠には使っていない**(根拠は総説の記述と近年の論文が報告している指標)。

## 本研究への含意

1. **3.5.2 の結論は独立に支持されている。** PaperUnPlot(NMS)と EpiCurveBench(ECS)は、どちらも軸レンジで正規化する。EpiCurveBench の式は本研究と実質同一
2. **したがって「軸レンジで割るべき」は本研究の貢献ではない。** 貢献は、どの分母がどれだけ良いかを実データで測った部分(等方性・安定性・順位)。先行研究はいずれも定性的な理由づけ
3. **4.4(合成 > 実図)も本研究固有ではない。** CHART-Infographics が2020年に、PaperUnPlot / ExChart-Bench / EpiCurveBench が2026年に同じ結論
4. **規模で明確に負けている。** 94 対 950 / 1,000 / 5万。差分はマーカー位置を点単位で測ること、正解が独立 DB 由来であること、CC BY で再配布できることの3点に絞るべき
5. **RMS を主指標に使えない理由を明記した。** 本コーパスは `prop_y` が1図内で同一なので、RMS の行見出しに相当するものが存在しない

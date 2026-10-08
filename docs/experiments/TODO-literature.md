# 取得できなかった論文 — 別ネットワークから再試行する

2026-10-07 の文献調査(`2026-10-07-metric-literature-survey.md`)で本文が取れなかったもの。
このマシン・このネットワークからは 403 / 認証リダイレクト / ブラウザ検証で止まった。
**2026-10-08、自宅ネットワークから再試行して4本取れた。** 取れたものには ✅ を付け、
確認できた内容を本文(03-evaluation §3.5.3 / §3.5.4)に反映済み。残り2本は依然 403。

決め手は2つだった。**curl に通常のブラウザの User-Agent を付ける**(FigureSeer の S3 が
これで通った)ことと、**PDF をローカルで抽出する**(`scripts/tools/pdf_text.py`。
WebFetch は PDF を解析できない)ことである。

取れたら `2026-10-07-metric-literature-survey.md` の「未検証」節を本文確認済みに移し、
論文の該当箇所(01-intro §1.2 / 03-evaluation §3.5.3-§3.5.4)の記述を確定させる。

## 優先度 高 — 本研究の位置づけに直接関わる

### 1. ❌ MetalThermoChartIE / PolyCompChartIE(RCLS)— **まだ取れない**
- **なぜ重要**: 材料科学の図のベンチマーク。名前からして金属熱物性で、**本研究の領域に最も近い**。
  規模・正解の作り方・指標が分かれば、1.2 の差分主張を書き換える必要があるかもしれない。
- 題名: Information Extraction from Diverse Charts in Materials Science
- URL: https://openreview.net/forum?id=vj8dqNrzEe / https://openreview.net/pdf?id=vj8dqNrzEe
- 失敗理由: OpenReview のブラウザ検証画面で停止
- 確認したいこと: RCLS の定義(RMS との差)、両データセットの図数と出所、実図か合成か、
  公開ライセンス、熱電材料や Starrydata への言及、評価したモデルと数値

### 2. ✅ PaperUnPlot(NMS)— **取得・確認済み**
- **なぜ重要**: 実図 950 対、PubMed Central + arXiv。**軸レンジ正規化という本研究 3.5.2 の結論に
  先行している**。引用必須。35.6 万枚のチャート分布分析も 1.2 の根拠に使える。
- 題名: PaperUnPlot: Benchmarking Chart-to-Table in the Wild(Bertillo ら、TaDA 2026 / VLDB Workshops)
- URL: https://tabular-data-analysis.github.io/tada2026/papers/TaDA26_13.pdf (2MB)
- 失敗理由: PDF が解析できなかった(ダウンロードは成功、テキスト抽出が不能)
- 確認したいこと: NMS の正確な定義(軸レンジで割る式そのもの)、950 図の内訳と
  ライセンス、正解の作り方(誰が注釈したか)、評価したモデルと数値、
  「既存ベンチマークの systematic blind spot」の具体的な中身
- 代替: 著者に arXiv 版があるか確認。GitHub https://github.com/giorgiomelch/about_chart_to_table が関連の可能性

### 3. ✅ FigureSeer(Siegel et al., ECCV 2016)— **取得・確認済み。未決着を解消**
- **なぜ重要**: 2% しきい値の起源が Scatteract(2017)か FigureSeer(2016)かで**二次資料が食い違う**。
  LineFormer は「FigureSeer は 2% で二値化する」と書き、Scatteract は「FigureSeer は
  ピクセル→チャート座標の変換を含まない」と書いている。論文 3.5.4 で未決着としている箇所。
- URL(全滅): http://ai2-website.s3.amazonaws.com/publications/Siegel16eccv.pdf (403)、
  https://link.springer.com/chapter/10.1007/978-3-319-46478-7_41 (認証)
- 失敗理由: S3 が 403、Springer が認証、他の経路は 404
- 確認したいこと: 評価節の正確な基準。しきい値があるか、何で正規化しているか、
  チャート座標に変換しているか

## 優先度 中

### 4. ❌ Turan & Sparks, IMMI 15:212-223 (2026)— **まだ取れない**
- 題名: Revolutionizing Scientific Figure Decoding: Benchmarking LLM Data Extraction Performance
- **なぜ重要**: 材料科学の図で LLM を評価した先行研究。4章の位置づけに関わる。
- URL: https://link.springer.com/article/10.1007/s40192-026-00443-8 (Springer 認証リダイレクト)
  chemRxiv preprint: https://chemrxiv.org/doi/10.26434/chemrxiv-2025-pcq8d (PDF が 403)
- 確認したいこと: 図数と出所、実図か、point count error% と mean X/Y error の正規化、
  データセットは公開されているか、評価した3モデルの数値

### 5. ✅ CHART-Info 2024 の論文 — **取得・確認済み。食い違いは解消**
- **なぜ重要**: 「5万枚超」という検索結果の数字と、**我々の監査で数えた 36,182 図が食い違う**。
  訓練のみか訓練+評価か、15 クラスの内訳、データ点が入っている図の定義を確認したい。
  我々の監査では 36,182 図中データ値ありが 5,427(15.0%)、うち x が数値のものが 2,214。
- URL: https://cvit.iiit.ac.in/images/ConferencePapers/2024/chart_info.pdf (未試行)
  https://link.springer.com/chapter/10.1007/978-3-031-78495-8_19 (Springer)
- 確認したいこと: 公称の総図数の定義、Task 6/7 に使える図数の公称値、
  我々の監査(`scripts/eval/audit_chartinfo_annotations.py`)との一致

### 6. ✅ DePlot の RMS の原典の式 — **取得・確認済み**
- **なぜ重要**: 3.5.3 で RMS の定義を二次資料(総説と後続論文)から書いている。
  `min(1, |g-p|/|g|)` と「(行見出し, 列見出し, 値)」が原典の表現か確認したい。
- URL: https://arxiv.org/pdf/2212.10505(PDF は取得したが解析不能)
- 代替: ACL Anthology の HTML 版があるはず。arXiv の ar5iv 版 https://ar5iv.labs.arxiv.org/html/2212.10505

## 取得方法のメモ

- **ar5iv がよく効く**: `https://ar5iv.labs.arxiv.org/html/<arXiv ID>` は HTML なので解析できる。
  Scatteract はこれで本文が取れた。arXiv にあるものはまずこれを試す。
- **arXiv の `/html/` 版**も新しい論文では使える(EpiCurveBench、ExChart-Bench、PlotPick はこれで取れた)。
- **PDF は基本的に解析できない**。このマシンに `pdftotext` / `pypdf` / `poppler-utils` が無い。
  自作の最小抽出器 `/tmp/pdftext.py` は chartinfo の metric.pdf では成功したが、
  画像主体の PDF では失敗する。**poppler-utils を入れるのが本筋**(要オーナー承認)。


---

## 2026-10-08 に確認できたこと

### FigureSeer — 2% しきい値の起源が決着した

原論文の評価節:

> 予測経路の点は、正解との**正規化した差**がしきい値を下回れば真陽性とする。
> `(yᵢ − y'ᵢ) < th`(実験では `th = 0.02`)

**2% は FigureSeer(2016)に始まる。** ただし y だけを予測経路の x 位置で比べる
**経路の指標**であり、x と y を別々にチャート座標で判定する Scatteract(2017)とは
測っているものが違う。LineFormer の記述は正しく、Scatteract の「FigureSeer は
ピクセル→チャート座標の変換を含まない」も正しい。両立する。→ 論文 §3.5.4 に反映。

### CHART-Info 2024 — 我々の監査と完全に一致した

論文の Table 3「タスクごとに使える図」:

| | Task1 | Task2 | Task3 | Task4 | Task5 | **Task6** | Task7 |
|---|---|---|---|---|---|---|---|
| 訓練 | **36,182** | 8,343 | 8,343 | 6,965 | 7,065 | **5,427** | 5,427 |
| 評価 | 15,093 | 3,280 | 3,280 | 3,128 | 3,128 | 2,676 | 2,676 |

**我々が数えた 36,182 と 5,427 はどちらも Table 3 と一致する。** 食い違って見えたのは、
検索結果の「5万枚超」が訓練+評価(36,182 + 15,093 = 51,275)で、こちらは訓練分だけを
落としていたためである。データ抽出に使えるのは **8,103 / 51,275 = 15.8%**。

**もう一つ:** 2024年の論文は「**タスク6と7のベースラインは本稿の範囲外**」と明記している。
我々が測っている課題には、彼らの最新データセットに**公表ベースラインが無い**。

### DePlot — 割り当ては見出し駆動

RMS の対応づけは `1 − NL_τ(行見出し‖列見出し)` をコストとして**見出しだけ**で行い、
そのうえで組になった項目の類似度を計算する。本研究の実装は原典どおりで、
論文 4.9 の順位の入れ替わりは**忠実な実装の上で起きている**。

### PaperUnPlot — 本文で数字を確認

950対 / 10種 / PMC + arXiv / 合成の対照つき。NMS は「軸レンジではなく対象値の大きさで
正規化し、対数軸を考慮しない」という RMS の2点を直す。8モデルを評価し、
**Gemini 2.5 Flash が最良、Claude Sonnet 4.5 が次**。全モデルが合成図で実図より高い。
Gemini は実図と合成図でほぼ同点、Claude は合成図で大きく上がる。

## まだ取れない2本

OpenReview(MetalThermoChartIE)と chemRxiv(Turan & Sparks)は、ブラウザの
User-Agent を付けても 403 を返す。**人がブラウザで開いて保存するしかない。**
前者は材料科学の図が対象で本研究に最も近いので、優先度は高いままである。

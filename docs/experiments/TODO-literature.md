# 取得できなかった論文 — 別ネットワークから再試行する

2026-10-07 の文献調査(`2026-10-07-metric-literature-survey.md`)で本文が取れなかったもの。
このマシン・このネットワークからは 403 / 認証リダイレクト / ブラウザ検証で止まった。
**オーナーが別マシン・別ネットワークから取得を試す**(2026-10-08 予定)。

取れたら `2026-10-07-metric-literature-survey.md` の「未検証」節を本文確認済みに移し、
論文の該当箇所(01-intro §1.2 / 03-evaluation §3.5.3-§3.5.4)の記述を確定させる。

## 優先度 高 — 本研究の位置づけに直接関わる

### 1. MetalThermoChartIE / PolyCompChartIE(RCLS)
- **なぜ重要**: 材料科学の図のベンチマーク。名前からして金属熱物性で、**本研究の領域に最も近い**。
  規模・正解の作り方・指標が分かれば、1.2 の差分主張を書き換える必要があるかもしれない。
- 題名: Information Extraction from Diverse Charts in Materials Science
- URL: https://openreview.net/forum?id=vj8dqNrzEe / https://openreview.net/pdf?id=vj8dqNrzEe
- 失敗理由: OpenReview のブラウザ検証画面で停止
- 確認したいこと: RCLS の定義(RMS との差)、両データセットの図数と出所、実図か合成か、
  公開ライセンス、熱電材料や Starrydata への言及、評価したモデルと数値

### 2. PaperUnPlot(NMS)
- **なぜ重要**: 実図 950 対、PubMed Central + arXiv。**軸レンジ正規化という本研究 3.5.2 の結論に
  先行している**。引用必須。35.6 万枚のチャート分布分析も 1.2 の根拠に使える。
- 題名: PaperUnPlot: Benchmarking Chart-to-Table in the Wild(Bertillo ら、TaDA 2026 / VLDB Workshops)
- URL: https://tabular-data-analysis.github.io/tada2026/papers/TaDA26_13.pdf (2MB)
- 失敗理由: PDF が解析できなかった(ダウンロードは成功、テキスト抽出が不能)
- 確認したいこと: NMS の正確な定義(軸レンジで割る式そのもの)、950 図の内訳と
  ライセンス、正解の作り方(誰が注釈したか)、評価したモデルと数値、
  「既存ベンチマークの systematic blind spot」の具体的な中身
- 代替: 著者に arXiv 版があるか確認。GitHub https://github.com/giorgiomelch/about_chart_to_table が関連の可能性

### 3. FigureSeer(Siegel et al., ECCV 2016)
- **なぜ重要**: 2% しきい値の起源が Scatteract(2017)か FigureSeer(2016)かで**二次資料が食い違う**。
  LineFormer は「FigureSeer は 2% で二値化する」と書き、Scatteract は「FigureSeer は
  ピクセル→チャート座標の変換を含まない」と書いている。論文 3.5.4 で未決着としている箇所。
- URL(全滅): http://ai2-website.s3.amazonaws.com/publications/Siegel16eccv.pdf (403)、
  https://link.springer.com/chapter/10.1007/978-3-319-46478-7_41 (認証)
- 失敗理由: S3 が 403、Springer が認証、他の経路は 404
- 確認したいこと: 評価節の正確な基準。しきい値があるか、何で正規化しているか、
  チャート座標に変換しているか

## 優先度 中

### 4. Turan & Sparks, IMMI 15:212-223 (2026)
- 題名: Revolutionizing Scientific Figure Decoding: Benchmarking LLM Data Extraction Performance
- **なぜ重要**: 材料科学の図で LLM を評価した先行研究。4章の位置づけに関わる。
- URL: https://link.springer.com/article/10.1007/s40192-026-00443-8 (Springer 認証リダイレクト)
  chemRxiv preprint: https://chemrxiv.org/doi/10.26434/chemrxiv-2025-pcq8d (PDF が 403)
- 確認したいこと: 図数と出所、実図か、point count error% と mean X/Y error の正規化、
  データセットは公開されているか、評価した3モデルの数値

### 5. CHART-Info 2024 の論文(Davila, Lazarus ら)
- **なぜ重要**: 「5万枚超」という検索結果の数字と、**我々の監査で数えた 36,182 図が食い違う**。
  訓練のみか訓練+評価か、15 クラスの内訳、データ点が入っている図の定義を確認したい。
  我々の監査では 36,182 図中データ値ありが 5,427(15.0%)、うち x が数値のものが 2,214。
- URL: https://cvit.iiit.ac.in/images/ConferencePapers/2024/chart_info.pdf (未試行)
  https://link.springer.com/chapter/10.1007/978-3-031-78495-8_19 (Springer)
- 確認したいこと: 公称の総図数の定義、Task 6/7 に使える図数の公称値、
  我々の監査(`scripts/eval/audit_chartinfo_annotations.py`)との一致

### 6. DePlot の RMS の原典の式
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

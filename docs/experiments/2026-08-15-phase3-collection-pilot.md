# 2026-08-15 Phase 3パイロット: CC-BY確定論文バッチでの収集パイプライン実測

- Status: done
- 関連設計: [`docs/design/benchmark-architecture.md`](../design/benchmark-architecture.md) §7.9〜§7.11
- 再現スクリプト: [`scripts/pilot/phase3_collect_cc_by_batch.py`](../../scripts/pilot/phase3_collect_cc_by_batch.py)(real-chart-bench自身のadapter/usecase実装を使用、モックなし)
- 参照: deep-digitizerワーカーのパイロット `docs/experiments/2026-08-15-pilot-figure-pairing.md`(is_oa全般での歩留まり実測。本パイロットはCC-BY確定分に絞った追試)

## 目的

Phase 2で確定した「ライセンス再配布可(CC-BY)判定パイプライン」の出力を実際に投入し、
(a) PDF取得、(b) 図候補抽出、(c) ground truth(Starrydata XY値)のmanifest化、
の3段階を**自リポジトリの実装コード**(モックではなく`HttpPdfFetchAdapter`/`PyMuPdfFigureExtractor`/
`build_ground_truth_for_paper`)に実データを通して検証する。

## 手順・パラメータ

1. `ThermoelectricMaterials`全論文からPhase 2と同一シード(`seed=7`)で500件抽出し、OpenAlexへ
   バッチ問い合わせ(license + `best_oa_location.pdf_url`)
2. `classify_license(license, is_oa=...)`でREDISTRIBUTABLEと判定され、かつ`pdf_url`を持つ論文を
   先頭から30件選定(実際に取得できたのは29件)
3. 各論文に対し `HttpPdfFetchAdapter.fetch(pdf_url)` を実行(リクエスト間隔0.5秒)
4. 取得成功(`status=OK`)したPDFに対し `PyMuPdfFigureExtractor.extract()` を実行
   (埋め込みラスター画像 + ページ全体レンダリングfallback、閾値150×150px、150dpi)
5. 全29論文に対し、Starrydataの実curveデータを`parse_curve_row`でパースし
   `build_ground_truth_for_paper()` でFigureRecord/GroundTruthCurveを生成
   (PDF取得の成否とは独立。ground truthはStarrydata側のデータのみで完結する)

## 結果

### PDF取得歩留まり(CC-BY確定分、n=29)

| ステータス | 件数 | 割合 |
|---|---|---|
| `ok` | 11 | 37.9% |
| `http_error`(403/404等) | 14 | 48.3% |
| `not_a_pdf`(paywall/HTML interstitial) | 4 | 13.8% |

**deep-digitizerパイロットとの比較**:

| 対象母集団 | PDF取得成功率 |
|---|---|
| deep-digitizer: `is_oa=true`全般(bronze/green/gold/hybrid/diamond混在、n=45) | 28.9% |
| 本パイロット: `license=CC-BY`確定分のみ(n=29) | **37.9%** |

CC-BYに絞ることで取得成功率が約9ポイント改善。CC-BY(主にgold/hybrid OA)は出版社が正式に
機械可読PDFを直接ホストしている割合が高く、bronze/green(自己アーカイブ・無料閲覧のみ)に比べ
ボット対策/認証画面に阻まれにくいという仮説と整合する。ただしそれでも**約6割は取得失敗**であり、
Phase 3量産時にはUnpaywall API併用・publisher別ハンドリング等の頑健化投資を要検討
(deep-digitizerの提言と同じ結論に達した)。

### 図候補抽出(PDF取得成功11件)

- 1論文あたり平均 **14.5枚**の候補画像(埋め込み画像+ページレンダリングの合計)
- 抽出処理自体はエラーなし(11/11)

### Ground truthmanifest生成(全29論文、PDF取得の成否と無関係)

| 指標 | 件数 |
|---|---|
| ground truthを生成できた論文 | 29 / 29(100%) |
| FigureRecord(`figure_id`単位) | 101 |
| GroundTruthCurve | 365 |

Phase 2で調査した同一30論文サンプルの実測(364曲線)とほぼ一致(誤差は無作為抽出30件目の
有無によるもの)。**ground truthのmanifest化はPDF取得・図抽出の成否に依存しない**ため、
CC-BY確定論文であれば即座に100%の歩留まりでground truthを確保できることを確認した。

## わかったこと(考察)

1. **ライセンスをCC-BYに絞ることはPDF取得歩留まりの実質的な改善策になる**(28.9% → 37.9%)。
   ライセンス判定を先に行うPhase 1の設計(§1.2フロー: ライセンス判定 → PDF取得)の順序は正しい。
2. **それでも6割強はPDF取得に失敗する**。§1.3の判定ロジックだけでは解決しない別レイヤーの課題
   ("再配布可能" と "実際にアクセスできる" は別問題)。v0の規模計画(`benchmark-architecture.md` §7.9の約569論文見積り)は、
   PDF取得歩留まり(概算35〜40%)を追加で織り込むと、**画像付きペアが確保できる論文は
   実質200〜230論文程度**という、より保守的な見積りに修正すべき。
3. **ground truth(XY値)とfigure画像(PDF)の収集は完全に分離できる**。ground truthはCC-BY確定
   論文であれば100%の歩留まりで即座に確保可能なので、v0データセットは「ground truth
   のみ確保済み(画像は後日/一部欠落)」という段階的な充実が可能な設計にしておくと、
   PDF取得のボトルネックがv0全体をブロックしない。
4. **画像とfigure_idの自動ペアリングは依然未解決**(deep-digitizer §7.10と同じ結論)。
   本パイロットでも、抽出した候補画像プール(平均14.5枚/論文)のどれがどの`figure_id`に
   対応するかの自動判定は実装していない。次の技術調査事項として持ち越す
   (候補: 図キャプションOCR+レイアウト解析、軽量VLMでのパネル位置推定 — deep-digitizerの
   次アクションと共通)。

## v0データセット規模計画への反映

- 当初見積り(§7.9): 約569論文 ≒ 数千〜1万曲線(ground truthベース)
- **本パイロットで判明した追加制約**: 画像付きペアが必要な場合、PDF取得歩留まり約35〜40%を
  乗じると実質200〜230論文相当
- **推奨方針**: v0を「ground truth manifestは569論文全量(画像ペアリングは別途)」と
  「画像ペア確保済みサブセット(200〜230論文相当)」の2階層で構成する。評価ハーネス
  (§4)は画像必須のため、当面の実運用は後者のサブセットが実質的なv0となる。

## 次アクション

- 画像↔figure_idペアリングの技術調査(deep-digitizerと合同で検討したい: 司令塔確認事項)
- PDF取得頑健化(Unpaywall API併用、publisher別リトライ戦略)の投資要否を司令塔に確認
- 上記の技術調査結果を踏まえ、実際の569論文全量に対する収集を実行してv0 manifestを確定する

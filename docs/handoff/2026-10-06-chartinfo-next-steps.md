# 次の作業(別マシン向け)— CHART-Infographics での本測定

2026-10-06 時点。このセッションで分かったことと、次にやることをまとめる。

## 状況

本研究のデータセット(94図)の「新しいデータセットを作った」という主張は弱い。
CHART-Infographics(ICPR 2024)がデータ抽出に使える図を 5,427枚持ち、x が数値のものが 2,214枚、
**うち 958枚(43%)が材料科学**で、同種の測定プロットである。系列の網羅でも彼らのほうが良い
(96.9% 対 こちらは 18.7% を除外して揃える)。詳細は論文 7.13 と
`docs/experiments/2026-10-06-chartinfo-dataset-audit.md`。

一方で、**実図のデータ抽出は 2020年を最後に測られていない**。ICDAR 2023 は中止、ICPR 2024 は
ベースラインなし、2025/2026 の告知なし(論文 8.6)。測定の空白は実在する。

したがって方針は、貢献をデータセットから**測定**へ移し、本研究のデータセットは
**出所の違う独立な検証セット**として位置づける(論文 1.4 貢献5、4.8、7.13)。

## やること

### 1. CHART-Infographics の scatter 592図で本測定(最優先)

パイロット10図は完了済み(論文 4.8)。Opus 5.5 完全自動で、彼らの式 0.877 / データ 0.873、
本研究の点 F1 0.952。2020年の最良 0.710 を、より不利な条件で上回った。

本測定の構成:

- 対象: ICPR 2024 訓練セットの scatter で、データ値があり x が数値のもの **592図**
  (line 1,622図は scatter の結果を見てから判断)
- 条件: 主条件1「完全自動」。彼らの Task 6b は上流タスクの正解を与える条件なので、
  **条件が違うことを必ず明記する**(こちらが不利)
- モデル: Claude(Opus 5.5 / Sonnet 5.5 / Fable 5.1)+ GPT-6.1-Sol。ローカル VLM も入れば
  配備制約の議論(6章)が彼らのデータでも言える
- 採点: **両方の指標で**。`scripts/eval/score_chartinfo_pilot.py` がそのまま使える
  (彼らの公式 `metric6b.py` は `scripts/eval/repro/third_party/chartinfo/` に無改変で置いてある)
- プロンプト: `data/chartinfo_pilot/prompt_v2.md` を使う。系列名の表記規約
  (`[unnamed data series #N]`、`*C`、LaTeX 風記号)が入っている版。この規約を伝えないと
  名前スコアが 0.406 → 0.889 の差になる(論文 4.8)

**データの入手:**

```
curl -sL "https://www.dropbox.com/scl/fi/vy7gyemald8z6rohx9mxx/CHARTINFO_2024_Train.zip?rlkey=ki3q4bb02rzdpdbih17we63gm&dl=1" -o chartinfo2024.zip   # 1.87GB
python3 scripts/eval/audit_chartinfo_annotations.py chartinfo2024.zip   # 内訳の確認
```

**注意点(パイロットで判明):**

- **密マーカー図が彼らのデータの 28.8%** を占める(本研究は 14.9%)。彼らはこれを区別していない。
  パイロットの最大の失敗(fig_09、適合率 0.520)もこれだった。本研究の密マーカー基準
  (`domain/marker_detection.py`)を彼らのデータにも当て、主表と副表を分けるべきか検討する
- 系列ラベルが凡例ボックスではなく**プロット内に書かれている図**で、名前を拾えない
  (パイロット fig_07、名前 0.167)。これは規約の問題ではなく本当の失敗
- 彼らの正解は軸範囲を持たない。本研究の点 F1 を当てるとき、正規化に使う「軸レンジ」は
  正解点の広がりで代用している(`score_chartinfo_pilot.py` の `our_score` に注記あり)。
  task4 の目盛ピクセル位置から実際の軸範囲を復元すれば、より正確になる

### 2. LineFormer を n=106 → 現行94図で再実行

starrydata/starrydata2#511 のまま未着手。職場の RTX 4090 で。

### 3. 線形図サブセット

starrydata/starrydata2#529。CHART-Infographics の line 1,622図が使えるので、
Starrydata 側で充放電曲線を探す前に、まずこちらで測るほうが速いかもしれない。

## このセッションで追加したもの

- `scripts/eval/audit_chartinfo_annotations.py` — 配布 ZIP の注釈率を数える(展開不要)
- `scripts/eval/score_chartinfo_pilot.py` — 両方の指標で採点
- `scripts/eval/repro/third_party/chartinfo/metric6b.py` — 彼らの公式採点コード(無改変)
- `data/chartinfo_pilot/` — パイロットの生回答・正解・鍵・プロンプト2版
- `docs/experiments/2026-10-06-chartinfo-dataset-audit.md` — 注釈率の実測記録
- 論文 1.4 貢献5 の書き換え、4.8、7.13、8.6 の追加

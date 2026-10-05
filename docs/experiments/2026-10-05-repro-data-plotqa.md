# PlotQA テスト分割の再現用データ準備 (2026-10-05)

## ソース・ライセンス
- PlotQA (Methani et al., WACV 2020), https://github.com/NiteshMethani/PlotQA (`PlotQA_Dataset.md` の Google Drive リンク、gdown で取得)。
- ライセンス (README 引用): "All the datasets created as part of this work are released under a CC-BY-4.0 license" / "All models & code are released under an MIT license"。データ=CC-BY-4.0、コード=MIT。
- ダウンロード: test annotations 248 MB (`annotations.json`)、test images 876 MB (tar.gz, `png/<image_index>.png`)。画像アーカイブは必要分のみ展開後に削除。注釈は `~/.cache/real-chart-bench/plotqa/` に保持。
- スクリプト: `scripts/eval/repro/prepare_plotqa.py` (冪等。seed 20261005)。

## テスト分割の件数
計 33,657 プロット: vbar_categorical 11,242 / hbar_categorical 11,292 / line 5,549 / dot_line 5,574。全プロットにタイトルあり。系列数は 1〜4 (dot_line: 1=1139, 2=1091, 3=1781, 4=1563)。

## 選択
1. `data/cache/repro/plotqa/deplot_repro.jsonl`: 4 種 x 125 = 500 件 (画像は `data/cache/repro/plotqa/images/`)。
2. `data/cache/repro/plotqa/dot_line_100.jsonl` と `data/synthetic/plotqa_dot_line/`: dot_line 100 件、系列数 1/2/3/4 を 25 件ずつ。画像 100 枚、`ground_truth.json`、`ATTRIBUTION.md` (CC-BY-4.0 表記)。(1)と(2)は独立抽選で 2 枚重複 (展開画像は計 598 枚)。

## 表フォーマットの判断 (DePlot 公式 `metrics.py` の `_parse_table` を確認)
- 公式は `text.lower().splitlines()` 後、先頭行が `title |` で始まればタイトル、各行を `" | "` で分割、最初の行をヘッダ。`<0x0A>` は T5 トークナイザの改行表現で、評価前に `\n` に戻される想定 (公式コードでは未確認、推定)。target は `" <0x0A> "` 連結で出力しているので、`\n` 評価時は置換すること。
- 形式: `TITLE | <title> <0x0A> <軸ラベル> | <系列名...> <0x0A> <カテゴリ> | <値...> ...`。小文字化されるので TITLE の大小は無影響。
- ヘッダ先頭セル: vbar/line/dot_line は x 軸ラベル、hbar はカテゴリ軸が y 軸なので y 軸ラベル (DePlot 論文/PlotQA の正確な変換規則は不明、推定)。
- 数値は注釈の値そのまま (整数値の float は整数表記)。丸めなし。凡例名は `name` を使用。
- 注釈の癖: 目盛り/ラベル配列は全て 2 回繰り返し (前半を採用)。hbar は model.x が値、model.y がカテゴリ (逆転)。line/dot_line の model.x は 0..n-1 の添字で、実ラベルは x_axis.major_labels。

## dot_line の x はカテゴリカルか
注釈上の x は添字だが、実ラベルは文字列 (年)。テスト 5,574 件全てで数値にパース可能 (年が大半)。ground_truth.json には `x_labels_raw`、`x_labels_numeric`、各系列の `x_raw` (添字)、`y`、マーカー bbox(px)、y/x 軸の major tick 値・ラベル・bbox(px)、プロット bbox を保持。y 軸の範囲は明示フィールドが無く、tick 値から導く。

## 不確実な点
- DePlot の PlotQA 目標表の厳密な生成規則 (軸ラベルの扱い、数値の丸め、hbar の向き) は未公開で、上記は推定。RNSS/RMS-F1 は転置も許容するため向きの影響は小さいはず。
- 画像は RGBA PNG。

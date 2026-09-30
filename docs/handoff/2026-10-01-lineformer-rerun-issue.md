# LineFormer を現行データセット(n=106)で再実行する

## なぜ必要か

リーダーボードに載っている LineFormer の数値は **古いデータセットのもの**で、他の行と比較できない。

| 行 | dataset_version | n |
|---|---|---|
| LineFormer (pretrained) | `v0-eval-pilot-n42-comparable-n38` | 38 |
| Claude 4モデル / naive-cv / achromatic-cv | `v0-eval-pilot-n106` | 106 |

`results/lineformer-pretrained-comparable-subset.json` の 0.6329 は、2026-08-29 に n42 セットで実行した結果から、その後レジストリを去った図を差し引いて**算術的に再計算しただけ**で、再実行ではない。同一サブセットの naive-cv が 0.6044 なので現状は「LineFormer が +0.03」に見えるが、**n=106 での比較は存在しない**。

論文では「クラウドLLMに入れられないデータには専用ローカルモデルという選択肢がある」という配備制約の議論を予定しており、その論拠は LineFormer の実測に依存する。古い母集団の数字では支えられない。

## 環境

- **職場の GeForce RTX 4090 で実行する。**
- 自宅マシンは GPU 非搭載。Colab は使わない — 配備制約(ローカル実行可能であること)を論じる節の証拠を、Google のマシンで取ると主張が崩れる。
- mmcv / mmdetection のビルドが macOS で失敗した経緯は design §7.16、Colab 側の失敗記録は §7.33〜7.35。**Linux + CUDA なら素直に通るはず。**

## 手順

1. リポジトリを clone / pull(`main`、コミット `579a8b2` 以降)
2. `notebooks/lineformer_colab.ipynb` の `LineFormerModelRunner` を Colab 依存から切り離してローカル実行する
   - 事前学習済み重み: LineFormer (ICDAR2023)
   - finetune 版も走らせるなら HF `t29mato/lineformer-battery-finetuned`
3. 評価対象は `select_verified_pairings(load_registry(...))` が返す **106図**。ハードコードしない(この数は今後も動く)
4. 出力は他のベースラインと同じ payload 形式:
   - `model_id`, `model_name`, `dataset_version`(`v0-eval-pilot-n106`), `run_at`, `n_figures`, `mean_summary_score`, `per_figure[]`
   - `per_figure` の各要素に `figure_id`(`{paper_id}-{figure_id}` 形式), `summary_score`, `match_rate`, `mean_curve_distance`, `mean_coverage_ratio`, `error`
5. `results/lineformer-pretrained-n106.json` として保存
6. 同一条件で比較できるよう、**軸レンジは ExtractionTask 経由で与える**(Claude 各モデルと naive-cv に与えたのと同じ情報)

## 注意点

- **単位空間の移行が 2026-09-30 に入っている**(30軸・29図)。registry と ground_truth は論文の印字単位に揃っており、以前のスナップショットとは数値空間が違う。古いキャッシュを使わないこと
- **画像が差し替わった図が4つある**(40067 / 40587 / 45818 はパネル分割、13761 は90度回転補正)。`registry.json` の `image_path` を必ず読むこと
- `axis_pixel_candidates.json` のピクセル座標は、この4図について `pixel_coords_stale_since: 2026-09-30` が立っている。**位置情報として使わない**

## 完了条件

- [ ] `results/lineformer-pretrained-n106.json` が `dataset_version: v0-eval-pilot-n106` で存在する
- [ ] リーダーボードの全行が同一 dataset_version になる
- [ ] 古い n42 の行は履歴として残すが、リーダーボードの現行表からは外す
- [ ] design doc に実行環境(GPU、ドライバ、mmcv/mmdet のバージョン)を記録する — 再現性のため

## 参考: 現行リーダーボード(n=106、軸レンジあり条件)

| 手法 | score | match_rate | curve_dist | coverage |
|---|---|---|---|---|
| Claude Fable 5 | 0.9789 | 0.9855 | 0.0403 | 0.9916 |
| Claude Opus 5 | 0.9781 | 0.9844 | 0.0401 | 0.9900 |
| Claude Sonnet 5 | 0.9623 | 0.9934 | 0.0917 | 0.9852 |
| naive-cv (色相) | 0.7390 | 0.7205 | 0.3808 | 0.8772 |
| achromatic-cv (輝度) | 0.6604 | 0.5329 | 0.5250 | 0.9732 |
| Claude Haiku 4.5 | 0.5591 | 0.4818 | 0.4445 | 0.6400 |
| **LineFormer** | **未測定 (n=38 の 0.6329 は別母集団)** | | | |

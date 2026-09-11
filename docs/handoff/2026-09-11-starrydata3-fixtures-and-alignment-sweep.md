# 引き継ぎ: Starrydata3 フィクスチャ束と GT↔画像アライメント掃引 (2026-09-11)

作業ディレクトリ `/tmp/.../scratchpad/` はセッションローカルで**残らない**ため、
再開に必要なものはすべてリポジトリに入れてある。`/var/tmp/` の納品物も
エクスポータから再生成できる(下記)。

## 1. いまの状態

`main` = `fafd6e3`、push 済み、テスト551件・ruff・import-linter green。

| コミット | 内容 |
|---|---|
| `1092c0d` | Starrydata3 へ E2E フィクスチャ束(9枚)を作成・引き渡し |
| `b03d0b6` | verified 全件の GT↔画像アライメント掃引 |
| `af6c38e` | 同一論文の複数図を追加(9枚 → 21枚) |
| `fafd6e3` | 候補プール枚数の更新(26 → 52) |

## 2. Starrydata3 への引き渡し物

**納品先**: `/var/tmp/real-chart-bench-e2e-fixtures-2026-09-09/`
(`fixtures.json` + `images/` 21枚 + `overlays/` 21枚 + `README.md`)

**再生成**: `python scripts/export/build_starrydata3_e2e_fixtures.py <出力先>`
(pillow が要る。`uv run --with pillow python ...`)

**リポジトリ側の記録**: `docs/interop/starrydata3-e2e-fixtures-2026-09-09.json` / `.md`、
経緯は `docs/interop/README.md`。

**中身**: 実図21枚・全2208点。各点が**データ座標とピクセル座標の両方**を持つ。
軸校正は印字目盛の実測ピクセル(`axis_pixel_candidates.json` の `owner_reviewed` のみ)、
点のピクセルは `PixelCalibration.to_pixel`(本作業で追加)による**逆写像で導出**。
複数図を持つ論文は5本: 28331(5枚)、10939 / 27759 / 446(各3枚)、22102(2枚)。

**壊さないための仕掛け**:

- `tests/adapter/test_starrydata3_e2e_fixture_bundle.py` が、committed な束が
  registry / 軸読み / ground_truth から乖離したら落ちる。
  **軸座標の再修正が入ったらここが落ちる** —— 落ちたら束を再生成し、**先方に連絡する**。
  実際 `2c8ea3c`(軸読み43件受理)と競合したが、収録21枚には影響なしだった。
- エクスポータは `excluded_reason` / `gt_suspect_status` が付いた図を渡そうとすると停止する。

## 3. アライメント掃引

`python scripts/eval/sweep_gt_image_alignment.py --json <出力>`(pillow / scipy / numpy)

GT点を軸読みでピクセルに写像し、最近傍インクまでの距離を、**同じ画像上で「対角2%だけ
わざと外した配置」と比較**する(絶対閾値は中空マーカー・密インクで破綻するため)。
既知の `gt_wrong` 3件を独立に再発見できることで妥当性を確認済み。

結果と解釈は `docs/experiments/2026-09-09-gt-image-alignment-sweep.md`。
**§4「やってはいけないこと」は必ず読むこと** —— 補正フィットの係数を原因の手がかりに
してはいけない(縮退する。この罠に実際にはまって誤報告した)。

## 4. 次にやること(優先度順)

1. **`44283/39578` の人手確認。** 掃引が上げた唯一の新規。
   `gt_suspect` / `llm_flagged` として記録済み(`scripts/eval/record_alignment_sweep_2026_09_09.py`)。
   4本中3本のGT曲線がどの描画系列にも乗らない。**採点除外はしていない** —— オーナー判断待ち。
   昇格させる場合は `excluded_reason` と `data/gt_issues/` への上流報告も併せて判断が要る。
   なおこの図の軸読み自体が `llm_candidate` なので、軸側の誤りも排除できていない。
2. **`verified` の意味を設計ドキュメントに明記するか。**
   「ペアリングの検証(曲線数・レンジ・大小関係)であって点ごとの一致ではない」ことが、
   Starrydata3 に指摘されるまで文章化されていなかった。§7.19 への追記が自然。
3. **掃引をCIに載せるか。** 現状は手動実行。
4. **Starrydata3 からの追加要請に備える。** 選定条件を満たす候補が21枚の外に52枚あり、
   18668 は単独で11枚出せる。先方は「1論文で多数の図」を並べたときの挙動を見たがっている。

## 5. 注意

- **`uv.lock` が未追跡のまま**(`.gitignore` にも無い)。`uv run` が生成したもので、
  追跡するかどうかの方針が未確定のためコミットしていない。
- 先方への報告で**2回、確認不足の誤りを出した**(`83/9049` を「調査未着手」「GT誤り」と
  報告 → 実際は 2026-09-07 にオーナー判断済み・GT誤りの証拠なし)。
  **registry の `excluded_reason` と `docs/handoff/` を先に読むこと。**
  訂正は先方・`docs/interop/README.md`・納品 JSON の3箇所に反映済み。
- 先方(Starrydata3)は束を使って**自分たちのバグを2件見つけている**
  (SI保存値に図単位のラベル、`°C` U+00B0+C を単位パーサが未対応)。
  この束は向こうの回帰テストの土台になっているので、**内容を変えたら必ず連絡する**。

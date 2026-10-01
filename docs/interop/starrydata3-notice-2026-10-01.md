# Starrydata3 への連絡: E2E フィクスチャから 1 図を外してください(2026-10-01)

**対象**: 2026-09-11 にお渡しした E2E 回帰テスト用フィクスチャ(21 図)のうち
**`27759_25222`**(論文 27759 の figure 16(a)、Power factor の対数 y 軸)。

**お願い**: この図を E2E テストから外してください。他の 20 図は変更ありません
(座標・画像とも 09-11 版とバイト単位で同一です)。

**理由**: この図には 5 系列が描かれていますが、正解データは 4 系列しか持っていません。
5 本目の「Ref. (Y0.56Al0.57B14)」(紫のひし形)は他論文の比較試料で、Starrydata は
自論文のデータしか収録しないためデジタイズされていません。お渡しした 4 系列の値・
ピクセル座標は正しいままですが、正解として不完全なので、5 本目も拾うデジタイザが
誤りと判定されてしまいます。real-chart-bench 側でもこの図を採点対象から外しました
(`excluded_reason: gt_incomplete`、design §7.65)。

**差し替え版**: `/var/tmp/real-chart-bench-e2e-fixtures-2026-10-01/`
(`fixtures.json` 20 図 + `images/` + `overlays/` + `README.md`)。`fixtures.json` の
`excluded` に外した理由を記載しています。再生成は
`python scripts/export/build_starrydata3_e2e_fixtures.py <出力先>`。

# 再開メモ(アカウント切り替え後、2026-10-08)

## 状態

- **CHART-Infographics scatter(582図、30図ずつ20回、プロンプト v2、完全自動)**

  | モデル | 状態 |
  |---|---|
  | GPT-6.1-Sol | 1〜20回目 完了 |
  | Sonnet 5.5 | 1〜20回目 完了(19回目は切り替え直前に完了予定) |
  | Opus 5.5 | 1〜6回目 完了、**7〜20回目が残り** |
  | Fable 5.1 | 1〜5回目 完了、**6〜20回目が残り** |

  - 封印ディレクトリ: `/tmp/claude-1000/-home-mato-projects-real-chart-bench/ed9a3441-f1b2-45c7-9ac0-2ab22648919d/scratchpad/chartinfo/batchNN/<model>/`(準備済み)。/tmp が消えていたら `scripts/eval/prepare_chartinfo_batch.py` で作り直す(同じシードで同じ図になる)。
  - 回収と採点: `scripts/eval/archive_score_chartinfo.sh <work> batchNN <model>`。自己申告の逸脱は `data/chartinfo_runs/NOTES.md` に追記する。
- **方式D v3(Claude Opus が司令塔)**
  - 主条件2: 完了(点 F1 0.933)
  - **主条件1 の2バッチが残り**: `~/.cache/real-chart-bench/orchestrator/claude-sealed-v3/noaxis/claude-opus-5-5/part{1,2}/`
  - 起動文: `Read <DIR>/INSTRUCTIONS.md and follow it exactly. Work only inside <DIR>.`
  - 回収と採点: `archive_orchestrator_claude.py ~/.cache/real-chart-bench/orchestrator/claude-sealed-v3 --v3` → `score_llm_predictions.py orch3-claude-noaxis`
- **ローカルエージェント(Codex + ollama Qwen3.8-27B)の全94図**: ドライバ `scripts/eval/run_codex_local_batches.sh` が裏で続行中(GPU)。Claude の使用量とは無関係。

## 使用量の運用(オーナー指示)

- Claude のセッション使用量が **80% を超えないこと**。75% で止める見張りを使う。
  - 見張り: `/tmp/.../scratchpad/usage_watch75.sh`。herdr のペイン w3:pD で `/usage` を3分ごとに読む。
- **同時に動かすのは4つまで。** 消費の目安:
  - Sonnet: 1回(30図)で 3〜4%
  - Opus / Fable: 1回で 10% 前後
- 止めた回は、残っている答えを保ったまま「未回答の図だけ埋める」指示で再開する。

## 次に気をつけること

- **同じ回の各モデルの封印ディレクトリが隣り合っている。** 他モデルの答えが見える作りなので、次に回を用意するときはモデルごとに別の場所へ分ける。
  - batch05 の Fable と Opus の一致は、丸めた格子上の値だけと確認済み。
- **論文への反映は保留中**(オーナー指示)。

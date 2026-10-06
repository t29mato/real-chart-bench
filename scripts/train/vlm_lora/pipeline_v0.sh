#!/usr/bin/env bash
# 方式B v0 runs after run (a) "synth" has finished training (docs/design/local-model.md):
#   1. benchmark run of (a)                     -> data/local_vlm_run_ft/qwen3.5-9b-qlora-synth/
#   2. train (b) "synth-real": continue from (a) with Starrydata real figures mixed in
#   3. Starrydata validation split: base / (a) / (b)   (tuning data, not the benchmark)
#   4. benchmark run of (b)                     -> data/local_vlm_run_ft/qwen3.5-9b-qlora-synth-real/
# Every GPU step goes through flock (run_ft.sh / run_train.sh). Re-running skips finished work.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RUNS="$HOME/.cache/real-chart-bench/vlm-lora-runs"
TD="$HOME/.cache/real-chart-bench/train-data"
FT="$REPO/scripts/eval/local_vlm/run_ft.sh"

until grep -q '"done": true' "$RUNS/synth/state.json" 2>/dev/null; do sleep 60; done

"$FT" bench "$RUNS/synth/adapter" "$REPO/data/local_vlm_run_ft/qwen3.5-9b-qlora-synth" noaxis,pixcal \
  > "$RUNS/bench-synth.log" 2>&1

"$HERE/run_train.sh" --data "$TD/plotqa" "$TD/synth-materials" "$TD/starrydata" --weights 1 2 1 \
  --init-adapter "$RUNS/synth/adapter" --out "$RUNS/synth-real" --steps 300 --accum 8 \
  --lr 5e-5 --warmup 20 --save-every 50 --val-n 48 --wall-minutes 60 > "$RUNS/synth-real.log" 2>&1 || exit 1

for a in none synth synth-real; do
  ad=none; [ "$a" != none ] && ad="$RUNS/$a/adapter"
  "$FT" val "$ad" "$RUNS/eval/real-$a" "$RUNS/synth-real" 100 starrydata > "$RUNS/eval-real-$a.log" 2>&1
done

"$FT" bench "$RUNS/synth-real/adapter" "$REPO/data/local_vlm_run_ft/qwen3.5-9b-qlora-synth-real" \
  noaxis,pixcal > "$RUNS/bench-synth-real.log" 2>&1
echo "[pipeline_v0] done"

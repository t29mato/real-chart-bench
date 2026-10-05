"""Runs one chart-to-table model over the test items of a published benchmark,
to reproduce the model's published number in this environment (design §7.75 (A)).

Inference only: one record per item with the raw output, appended to --out
(resumable). Scoring is score_repro.py, with the authors' own scorer.

Each item is {"id", "image", "gt_table", "prompt"?}; an item's own "prompt"
(e.g. TinyChart's test.json) overrides the model's standard prompt.

Runs in the model's venv; imports the same Runner as the real-figure runs
(scripts/eval/chart2table/worker.py), so loading and generation settings are
identical.

Usage: <venv python> run_repro.py <model> <items.jsonl> <out.jsonl>
"""

import json
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chart2table"))

from worker import SPECS, Runner  # noqa: E402


def main() -> None:
    model, items_path, out_path = sys.argv[1:4]
    items = [json.loads(line) for line in open(items_path) if line.strip()]
    done = set()
    try:
        with open(out_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    except FileNotFoundError:
        pass
    todo = [i for i in items if i["id"] not in done]
    print(f"[start] {model}: {len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)
    runner = Runner(model, SPECS[model], "cuda:0")
    t_run = time.time()
    with open(out_path, "a") as out:
        for n, item in enumerate(todo, 1):
            t0 = time.time()
            rec = {"id": item["id"], "model": model, "prompt": item.get("prompt"), "error": None}
            try:
                rec["raw_text"] = runner.run(item["image"], item.get("prompt"))
            except Exception:  # noqa: BLE001 -- one item must not stop the run
                rec["raw_text"] = None
                rec["error"] = traceback.format_exc()
            rec["seconds"] = round(time.time() - t0, 3)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                eta = (time.time() - t_run) / n * (len(todo) - n)
                print(f"[{len(done) + n}/{len(items)}] eta {eta / 60:.1f} min", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

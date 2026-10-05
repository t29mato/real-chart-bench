"""Runs one chart-to-table model over the test items of a published benchmark,
to reproduce the model's published number in this environment (design §7.75 (A)).

Inference only: one record per item with the raw output, appended to --out
(resumable). Scoring is score_repro.py, with the authors' own scorer.

Each item has an image path ("image"), an id ("id", or PlotQA's
"image_index") and optionally its own "prompt" (TinyChart's test.json has
one). The model's own documented prompt is used unless --item-prompt is
given: an item's prompt belongs to the model whose test set it is, and
giving it to another model (2026-10-05: UniChart got TinyChart's prompt and
echoed it back) measures the prompt, not the model.

Runs in the model's venv; imports the same Runner as the real-figure runs
(scripts/eval/chart2table/worker.py), so loading and generation settings are
identical.

Usage: <venv python> run_repro.py <model> <items.jsonl> <out.jsonl> [--item-prompt]
"""

import json
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chart2table"))

from worker import SPECS, Runner  # noqa: E402


def item_id(item: dict) -> str:
    return str(item["id"] if "id" in item else item["image_index"])


def main() -> None:
    model, items_path, out_path = sys.argv[1:4]
    use_item_prompt = "--item-prompt" in sys.argv[4:]
    items = [json.loads(line) for line in open(items_path) if line.strip()]
    done = set()
    try:
        with open(out_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    except FileNotFoundError:
        pass
    todo = [i for i in items if item_id(i) not in done]
    print(f"[start] {model}: {len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)
    runner = Runner(model, SPECS[model], "cuda:0")
    t_run = time.time()
    with open(out_path, "a") as out:
        for n, item in enumerate(todo, 1):
            t0 = time.time()
            prompt = item.get("prompt") if use_item_prompt else None
            rec = {
                "id": item_id(item),
                "model": model,
                "prompt": prompt or SPECS[model]["prompt"],
                "error": None,
            }
            try:
                rec["raw_text"] = runner.run(item["image"], prompt)
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

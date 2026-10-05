"""Sanity check (design §7.75): does each chart-to-table model, run in THIS
environment, read its own kind of chart? If it does, a low score on
real-chart-bench is the model meeting real figures, not a broken setup.

Uses ChartQA's test charts and their ground-truth tables (vis-nlp/ChartQA on
GitHub, GPL-3.0; downloaded to data/cache/, used locally, not redistributed).
DePlot, UniChart, ChartGemma and TinyChart all train or evaluate on ChartQA.

Check, per chart: the numeric cells of the ground-truth table and of the
model's table, matched one-to-one within 5% relative (|p - g| <= 0.05 * |g|,
or exact for g = 0); recall = matched / ground-truth values, precision =
matched / predicted values. Order and labels are ignored -- this asks only
"are the right numbers there", which is the part a broken environment (wrong
weights, wrong preprocessing, wrong prompt) would destroy.

Runs in the model's venv (imports the worker's Runner, so loading and
inference are exactly the real run's).

Usage: <venv python> sanity_chartqa.py <model> [n=30]
"""

import csv
import io
import json
import pathlib
import random
import sys
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from worker import SPECS, Runner  # noqa: E402

CACHE = REPO / "data/cache/chartqa_sanity"
RAW = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test/"
API = "https://api.github.com/repos/vis-nlp/ChartQA/contents/ChartQA%20Dataset/test/tables"
SEED = 20261005


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "real-chart-bench"})
    return urllib.request.urlopen(req, timeout=60).read()


def _charts(n: int) -> list[str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    listing = CACHE / "listing.json"
    if not listing.exists():
        names = sorted(x["name"][:-4] for x in json.loads(_get(API)) if x["name"].endswith(".csv"))
        listing.write_text(json.dumps(names))
    names = json.loads(listing.read_text())
    chosen = random.Random(SEED).sample(names, n)
    for name in chosen:
        for kind, ext in (("png", "png"), ("tables", "csv")):
            path = CACHE / f"{name}.{ext}"
            if not path.exists():
                path.write_bytes(_get(RAW + f"{kind}/{urllib.parse.quote(name)}.{ext}"))
    return chosen


def _number(text: str):
    s = text.strip().replace(",", "").replace("%", "").replace("$", "").replace("−", "-")
    try:
        return float(s)
    except ValueError:
        return None


def _values_from_csv(text: str) -> list[float]:
    rows = list(csv.reader(io.StringIO(text)))
    return [v for row in rows[1:] for c in row[1:] if (v := _number(c)) is not None]


def _values_from_prediction(text: str, row_separator: str | None) -> list[float]:
    text = (text or "").replace("<0x0A>", "\n")
    if row_separator:
        text = text.replace(row_separator, "\n")
    values = []
    for line in text.splitlines():
        cells = line.split("|") if "|" in line else next(csv.reader(io.StringIO(line)), [])
        values += [v for c in cells[1:] if (v := _number(c)) is not None]
    return values


def _match(pred: list[float], gt: list[float]) -> int:
    remaining = list(pred)
    n = 0
    for g in gt:
        tol = 0.05 * abs(g)
        best = None
        for i, p in enumerate(remaining):
            if (abs(p - g) <= tol) if g else p == 0:
                best = i
                break
        if best is not None:
            remaining.pop(best)
            n += 1
    return n


def main() -> None:
    model = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    charts = _charts(n)
    import torch

    runner = Runner(model, SPECS[model], "cuda:0")
    sep = " & " if model == "unichart" else None
    out = CACHE / f"{model}.jsonl"
    rows = []
    with out.open("w") as f:
        for name in charts:
            t0 = time.time()
            text = runner.run(str(CACHE / f"{name}.png"))
            gt = _values_from_csv((CACHE / f"{name}.csv").read_text())
            pred = _values_from_prediction(text, sep)
            m = _match(pred, gt)
            row = {
                "chart": name,
                "n_gt": len(gt),
                "n_pred": len(pred),
                "matched": m,
                "recall": m / len(gt) if gt else None,
                "precision": m / len(pred) if pred else 0.0,
                "seconds": round(time.time() - t0, 2),
                "raw_text": text,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rec = [r["recall"] for r in rows if r["recall"] is not None]
    prec = [r["precision"] for r in rows]
    print(
        f"{model}: ChartQA test, {len(rows)} charts -- mean value recall "
        f"{sum(rec) / len(rec):.3f}, precision {sum(prec) / len(prec):.3f} "
        f"(within 5%), torch {torch.__version__}"
    )


if __name__ == "__main__":
    main()

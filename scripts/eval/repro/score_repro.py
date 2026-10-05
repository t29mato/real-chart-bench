"""Scores a reproduction run (run_repro.py output) with the authors' own
table metrics (scripts/eval/repro/third_party/tinychart_eval_chart2table.py,
= DePlot's metrics.py): RMS-F1 (`table_datapoints_precision_recall`) and
RNSS (`table_number_accuracy`). Design §7.75 (A).

Tables reach the scorer in the form both official drivers use: a first
"title | <title>" line (DePlot's evaluate_chart_to_table.py builds targets
that way; TinyChart's chart2table_evaluator prepends an empty "title |"),
then one " | "-separated row per line, lowercased. An existing title line is
kept as the title; otherwise an empty one is added.

Before that, only the *layout* of a prediction is normalized to the scorer's
" | " cells / one row per line, because each model writes tables its own way:
DePlot's "<0x0A>" row token, UniChart's " & " rows, Markdown pipes and rule
rows, a CSV (no "|") line, ``` fences. Values are never touched. The same
normalization is applied to targets (PlotQA targets use "<0x0A>").

Usage: <chart2table venv python> score_repro.py <items.jsonl> <predictions.jsonl>
"""

import csv
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))

import tinychart_eval_chart2table as official  # noqa: E402

_RULE = re.compile(r"^\|?\s*:?-{2,}")
_TITLE = re.compile(r"^title\s*\|", re.I)


def normalize_layout(text: str, row_separator: str | None = None) -> str:
    text = (text or "").replace("<0x0A>", "\n")
    text = re.sub(r"```[a-zA-Z]*", "", text)
    if row_separator:
        text = text.replace(row_separator, "\n")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _RULE.match(line):
            continue
        if not rows and _TITLE.match(line):
            # DePlot-style "TITLE | <title>" first line: the scorer's title
            rows.append("title | " + line.split("|", 1)[1].strip())
            continue
        if "|" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
        else:
            cells = [c.strip() for c in next(csv.reader(io.StringIO(line)))]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _with_title(table: str) -> str:
    table = table.strip().lower()
    return table if table.startswith("title |") else "title |\n" + table


def main() -> None:
    items_path, preds_path = sys.argv[1:3]
    rows = [json.loads(line) for line in open(items_path) if line.strip()]
    items = {str(r["id"] if "id" in r else r["image_index"]): r for r in rows}
    preds = [json.loads(line) for line in open(preds_path) if line.strip()]
    model = preds[0]["model"]
    sep = " & " if model == "unichart" else None
    refs, hyps, n_err = [], [], 0
    for p in preds:
        item = items[p["id"]]
        gt = item.get("gt_answer") or item.get("gt_table") or item.get("target")
        n_err += p["error"] is not None
        refs.append([_with_title(normalize_layout(gt))])
        hyps.append(_with_title(normalize_layout(p.get("raw_text") or "", sep)))
    rms = official.table_datapoints_precision_recall(refs, hyps)
    rnss = official.table_number_accuracy(refs, hyps)
    out = {
        "model": model,
        "items": str(items_path),
        "n": len(preds),
        "n_errors": n_err,
        "rms_f1": round(rms["table_datapoints_f1"], 2),
        "rms_precision": round(rms["table_datapoints_precision"], 2),
        "rms_recall": round(rms["table_datapoints_recall"], 2),
        "rnss": round(rnss["numbers_match"], 2),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()

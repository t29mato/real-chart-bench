"""The fixed tools of 方式D as a command line (docs/design/local-model.md
「方式D: 司令塔 + 道具」). Prints the tool's result as JSON.

  python scripts/tools/rcb_tool.py <tool> --image fig.png --params '{"color": "#d62728"}' \\
      [--save results/r1.json] [--dets-dir DIR] [--given-calibration cal.json] [--no-auto]

Tools: dominant_colors, mask, symbol_extract, line_extract, marker_detector,
tick_calibration, to_values, render_overlay (adapter/orchestrator_tools.py), and

  answer --image fig.png --fig fig_001.png --predictions predictions.json \\
      --params '{"series": [{"from": "results/r3.json", "index": -1, "label": ""}],
                 "calibration": "results/r1.json"}' [--pixels]

which writes one figure's answer into predictions.json from saved results:
the named series, converted to values through the named calibration (or left
in image pixels with --pixels). No number of an answer is typed by hand.

A parameter that names an earlier result is the path of its saved JSON
(``--save``). The same script runs from a sealed directory, where the
package sits next to it (tools/real_chart_bench/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "real_chart_bench").is_dir():
    sys.path.insert(0, str(HERE))
else:
    sys.path.insert(0, str(HERE.parents[1] / "src"))

from real_chart_bench.adapter.orchestrator_tools import TOOLS, ToolBox, ToolError  # noqa: E402
from real_chart_bench.domain.digitizer_tools import series_to_values  # noqa: E402
from real_chart_bench.domain.orchestration import assemble_final as _assemble  # noqa: E402
from real_chart_bench.domain.orchestration import parse_action  # noqa: E402


def _resolver(base: Path):
    def resolve(ref: str) -> dict:
        p = Path(ref)
        if not p.is_absolute():
            p = base / p
        if not p.exists():
            raise ToolError(f"no saved result {ref} (save results with --save)")
        return json.loads(p.read_text())

    return resolve


def _answer(args, resolve) -> dict:
    params = json.loads(args.params or "{}")
    action = parse_action(json.dumps({"action": "final", **params}), list(TOOLS))
    refs = {s["from"] for s in action.series} | ({action.calibration} if action.calibration
                                                  else set())
    results = {r: resolve(r) for r in refs}
    series, cal = _assemble(action, results, need_calibration=not args.pixels)
    if cal is not None:
        series = series_to_values(series, cal)
    pred_path = Path(args.predictions)
    preds = json.loads(pred_path.read_text()) if pred_path.exists() else {}
    preds[args.fig] = series
    pred_path.write_text(json.dumps(preds, indent=1) + "\n")
    return {"kind": "answer", "fig": args.fig, "n_series": len(series),
            "n_points": sum(len(s["x"]) for s in series),
            "units": "pixels" if args.pixels else "values", "figures_in_file": len(preds)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tool", choices=[*TOOLS, "answer"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--params", default="{}", help="JSON object")
    ap.add_argument("--save", help="also write the result JSON here")
    ap.add_argument("--dets-dir", help="detector cache (sha256-keyed JSON files)")
    ap.add_argument("--given-calibration", help="JSON calibration given in this condition")
    ap.add_argument("--no-auto", action="store_true", help="no automatic tick calibration")
    ap.add_argument("--out-dir", help="where render_overlay writes when no out is given")
    ap.add_argument("--fig", help="answer: the task id")
    ap.add_argument("--predictions", help="answer: predictions.json to update")
    ap.add_argument("--pixels", action="store_true", help="answer: keep image pixels")
    args = ap.parse_args()
    resolve = _resolver(Path.cwd())
    try:
        if args.tool == "answer":
            if not args.fig or not args.predictions:
                raise ToolError("answer needs --fig and --predictions")
            out = _answer(args, resolve)
        else:
            given = (json.loads(Path(args.given_calibration).read_text())
                     if args.given_calibration else None)
            tb = ToolBox(Path(args.image), dets_dir=args.dets_dir and Path(args.dets_dir),
                         given_calibration=given, allow_auto_calibration=not args.no_auto,
                         out_dir=args.out_dir and Path(args.out_dir))
            out = tb.run(args.tool, json.loads(args.params or "{}"), resolve)
    except (ToolError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(2)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(out) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()

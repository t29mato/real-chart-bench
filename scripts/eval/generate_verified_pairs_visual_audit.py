"""Generate a Markdown visual audit of every verified_pairs entry.

For each `status == "verified"` entry in `data/verified_pairs/registry.json`,
renders a plot of the digitized ground-truth curves (from
`data/verified_pairs/ground_truth.json`) using the entry's own axis
calibration (`x_range`/`y_range`/`x_scale`/`y_scale`), and places it next to
the original chart crop plus the entry's metadata and evidence text in one
Markdown file. Intended for a human to eyeball each pair and confirm the
digitized curve actually matches what the original chart shows -- this is a
review aid, not a scoring tool (see `run_baselines.py` / the domain metrics
for that).

Output is written under `data/verified_pairs/audit/` (gitignored --
regeneratable from committed registry.json + ground_truth.json + images).

Usage: python scripts/eval/generate_verified_pairs_visual_audit.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
GROUND_TRUTH_PATH = REPO_ROOT / "data/verified_pairs/ground_truth.json"
PAPERS_PATH = REPO_ROOT / "data/manifest/v0/papers.json"
AUDIT_DIR = REPO_ROOT / "data/verified_pairs/audit"
PLOTS_DIR = AUDIT_DIR / "plots"
MARKDOWN_PATH = AUDIT_DIR / "visual-audit.md"

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _slug(entry: dict) -> str:
    ref = entry["figure_reference"].replace(" ", "_").replace("(", "").replace(")", "")
    return f"{entry['paper_id']}_{entry['figure_id']}_{ref}"


def _render_ground_truth_plot(entry: dict, curves: list[dict], out_path: Path) -> str | None:
    """Returns a warning string if anything looked off (e.g. dropped points), else None."""
    warning = None
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=110)
    x_scale = entry.get("x_scale", "linear")
    y_scale = entry.get("y_scale", "linear")
    dropped_nonpositive = 0

    for i, curve in enumerate(curves):
        xs, ys = curve["x"], curve["y"]
        if y_scale == "log":
            pairs = [(x, y) for x, y in zip(xs, ys) if y > 0]
            dropped_nonpositive += len(xs) - len(pairs)
            xs, ys = ([p[0] for p in pairs], [p[1] for p in pairs])
        if x_scale == "log":
            pairs = [(x, y) for x, y in zip(xs, ys) if x > 0]
            xs, ys = ([p[0] for p in pairs], [p[1] for p in pairs])
        color = _COLORS[i % len(_COLORS)]
        label = curve.get("prop_y", f"curve {i}")
        if sum(1 for c in curves if c.get("prop_y") == label) > 1:
            label = f"{label} #{i}"
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1, color=color, label=label)

    if dropped_nonpositive:
        warning = f"{dropped_nonpositive} non-positive y-value point(s) omitted (log-y scale)"

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    x_range = entry["x_range"]
    y_range = entry["y_range"]
    try:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
    except ValueError:
        pass  # e.g. non-positive limit on a log axis; let matplotlib autoscale instead
    unit_x = curves[0].get("unit_x", "") if curves else ""
    unit_y = curves[0].get("unit_y", "") if curves else ""
    prop_x = curves[0].get("prop_x", "x") if curves else "x"
    ax.set_xlabel(f"{prop_x} ({unit_x})" if unit_x else prop_x)
    ax.set_ylabel(unit_y or "y")
    ax.legend(fontsize=6, loc="best")
    ax.set_title(f"ground truth: {len(curves)} curve(s)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return warning


def main() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    papers_by_id = {p["paper_id"]: p for p in json.loads(PAPERS_PATH.read_text())}

    verified = [e for e in registry if e["status"] == "verified"]
    verified.sort(key=lambda e: (e["paper_id"], e["figure_reference"]))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in verified:
        curves = ground_truth.get(entry["figure_id"], [])
        slug = _slug(entry)
        plot_path = PLOTS_DIR / f"{slug}.png"
        warning = _render_ground_truth_plot(entry, curves, plot_path)
        rows.append((entry, curves, plot_path, warning))

    lines: list[str] = []
    lines.append("# Verified Pairs Visual Audit")
    lines.append("")
    lines.append(
        f"Generated from `data/verified_pairs/registry.json` "
        f"({len(verified)} `status=\"verified\"` entries) + "
        f"`data/verified_pairs/ground_truth.json`. Regenerate with "
        f"`python scripts/eval/generate_verified_pairs_visual_audit.py`."
    )
    lines.append("")
    lines.append(
        "For each entry: **left = original chart crop**, **right = the digitized "
        "ground-truth curve(s) re-plotted** using the entry's own `x_range`/`y_range`/"
        "scale. Check the box once you've confirmed the right plot's shape, series "
        "count, and value ranges actually match the left image."
    )
    lines.append("")
    n_log_y = sum(1 for e in verified if e.get("y_scale") == "log")
    lines.append(f"- Total verified entries: **{len(verified)}**")
    lines.append(f"- log-y entries: {n_log_y}")
    n_warnings = sum(1 for *_, w in rows if w)
    lines.append(f"- Entries with a rendering warning (see ⚠️ below): {n_warnings}")
    lines.append("")
    lines.append("## Index")
    lines.append("")
    for entry, _curves, _plot_path, warning in rows:
        anchor = _slug(entry).lower()
        flag = " ⚠️" if warning else ""
        lines.append(
            f"- [ ] [{entry['paper_id']} / fig {entry['figure_reference']}]"
            f"(#{anchor}){flag}"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    for entry, curves, plot_path, warning in rows:
        slug = _slug(entry)
        anchor = slug.lower()
        paper = papers_by_id.get(entry["paper_id"], {})
        doi = paper.get("doi", "?")
        img_rel = os.path.relpath(REPO_ROOT / entry["image_path"], AUDIT_DIR)
        plot_rel = os.path.relpath(plot_path, AUDIT_DIR)

        lines.append(f'<a id="{anchor}"></a>')
        lines.append(f"## {entry['paper_id']} — Figure {entry['figure_reference']}")
        lines.append("")
        lines.append("- [ ] **目視確認OK**")
        lines.append("")
        if warning:
            lines.append(f"> ⚠️ **{warning}**")
            lines.append("")
        lines.append(
            "| paper_id | figure_id | DOI | license | x_range | y_range | scale | "
            "verified_at |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        x_scale = entry.get("x_scale", "linear")
        y_scale = entry.get("y_scale", "linear")
        lines.append(
            f"| {entry['paper_id']} | {entry['figure_id']} | "
            f"[{doi}](https://doi.org/{doi}) | {entry.get('license_id', '?')} | "
            f"{entry['x_range']} | {entry['y_range']} | x:{x_scale}/y:{y_scale} | "
            f"{entry.get('verified_at', '?')} |"
        )
        lines.append("")
        lines.append(f"n curves in ground_truth.json: {len(curves)}")
        lines.append("")
        lines.append("<table><tr>")
        lines.append(
            f'<td width="50%"><b>original chart</b><br>'
            f'<img src="{img_rel}" width="100%"></td>'
        )
        lines.append(
            f'<td width="50%"><b>digitized ground truth (re-plotted)</b><br>'
            f'<img src="{plot_rel}" width="100%"></td>'
        )
        lines.append("</tr></table>")
        lines.append("")
        lines.append(f"**evidence** (from registry.json): {entry['evidence']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    MARKDOWN_PATH.write_text("\n".join(lines))
    print(f"wrote {MARKDOWN_PATH} ({len(verified)} entries, {len(rows)} plots in {PLOTS_DIR})")


if __name__ == "__main__":
    main()

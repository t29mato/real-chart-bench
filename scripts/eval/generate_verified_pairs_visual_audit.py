"""Generate a Markdown visual audit of every verified_pairs entry.

For each `status == "verified"` entry in `data/verified_pairs/registry.json`,
this produces one Markdown section with:

1. the original chart crop,
2. (when `data/verified_pairs/axis_pixel_candidates.json` has a matching
   entry) a **pixel-calibration overlay** -- the original image with the
   LLM-judged tick pixel positions drawn on top, so a human can visually
   confirm the calibration lines actually land on the printed tick marks,
3. the digitized ground-truth curves (from `ground_truth.json`) re-plotted.
   Starrydata stores every curve in SI units, but papers almost always
   display a rescaled unit (uV/K, S/cm, mOhm.cm, ...). Two sources are
   tried, in order of trustworthiness:

   a. **axis-pixel-derived** (primary): when `axis_pixel_candidates.json`
      has a matching entry, the printed axis's own tick *values* (not just
      their pixel positions) let us derive the exact SI -> paper-display
      factor with no unit-string guessing: divide the printed tick value
      by the SI-unit registry x_range/y_range at that same point. Trusted
      only when it agrees independently at both the min and max end of the
      axis (see `_derive_factor`); a disagreement is surfaced as a
      top-of-file "needs attention" flag rather than silently applied.
   b. **evidence-text-derived** (fallback, only when (a) is unavailable):
      most entries' `evidence` field already states the exact factor a
      human/agent verified against the image while writing the entry
      (e.g. "Converting GT y to the chart's uOhm cm units (x1e8)"). A
      narrow regex looks for that specific phrasing. This is *not*
      independently cross-checked the way (a) is -- it's trusting text
      that was already validated once, not re-deriving it -- so it's
      labeled distinctly ("evidence-derived, unverified") in the output.

   Where neither source applies, the re-plot falls back to raw SI units
   and is labeled as such.

This is a review aid, not a scoring tool (see `run_baselines.py` / the
domain metrics for that) -- it never modifies registry.json or
ground_truth.json, both stay in their SI-unit, "always derive" form used
by the evaluation harness (design 7.33); this script only re-expresses a
copy of the curve data for human eyeballing.

Output is written under `data/verified_pairs/audit/` (gitignored --
regeneratable from committed registry.json + ground_truth.json +
axis_pixel_candidates.json + images).

Usage: python scripts/eval/generate_verified_pairs_visual_audit.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
GROUND_TRUTH_PATH = REPO_ROOT / "data/verified_pairs/ground_truth.json"
PAPERS_PATH = REPO_ROOT / "data/manifest/v0/papers.json"
AXIS_PIXEL_CANDIDATES_PATH = REPO_ROOT / "data/verified_pairs/axis_pixel_candidates.json"
AUDIT_DIR = REPO_ROOT / "data/verified_pairs/audit"
PLOTS_DIR = AUDIT_DIR / "plots"
OVERLAYS_DIR = AUDIT_DIR / "overlays"
MARKDOWN_PATH = AUDIT_DIR / "visual-audit.md"
REVIEW_HTML_PATH = AUDIT_DIR / "review.html"

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_FACTOR_AGREEMENT_TOL = 0.03  # 3% relative disagreement between endpoint-derived factors

# Matches the "(xN)" / "(divide by N)" / "(/N)" phrasing evidence texts use
# when documenting the y-axis unit conversion they manually verified against
# the source image, e.g. "Converting GT y to the chart's uOhm cm units
# (x1e8): ...". Only ever describes the *y* axis in this corpus (x is
# almost always Temperature/K, unconverted) -- see the module docstring.
_EVIDENCE_FACTOR_RE = re.compile(
    r"\(x(?P<mult>1e-?\d+|\d+(?:\.\d+)?)\)"
    r"|\(divide by (?P<div>1e-?\d+|\d+(?:\.\d+)?)\)"
    r"|\((?:/|÷)\s*(?P<div2>1e-?\d+|\d+(?:\.\d+)?)\)",
    re.IGNORECASE,
)


def _evidence_text_factor(evidence: str) -> dict | None:
    """Fallback factor source when no axis-pixel data exists for an entry:
    parse the explicit conversion factor most evidence texts already state
    (see AGENTS.md's evidence-style convention -- "state the actual numbers
    you cross-checked"). Unlike `_derive_factor`, this is NOT independently
    re-verified here, just trusting text a human/agent already validated
    once against the image -- callers must label it as such.
    """
    m = _EVIDENCE_FACTOR_RE.search(evidence)
    if not m:
        return None
    if m.group("mult") is not None:
        factor = float(m.group("mult"))
    else:
        divisor = float(m.group("div") or m.group("div2"))
        factor = 1.0 / divisor
    return {
        "kind": "multiplicative",
        "factor": factor,
        "offset": None,
        "confident": None,  # not independently cross-checked -- see docstring
        "detail": f"parsed from evidence text: {m.group(0)}",
    }


def _slug(entry: dict) -> str:
    ref = entry["figure_reference"].replace(" ", "_").replace("(", "").replace(")", "")
    return f"{entry['paper_id']}_{entry['figure_id']}_{ref}"


def _derive_factor(
    reg_range: list[float], label_min: float, label_max: float, scale: str = "linear"
) -> dict:
    """Derive the SI -> paper-display relationship for one axis.

    Most properties in this domain (resistivity, conductivity, Seebeck
    coefficient, ...) rescale multiplicatively: display = k * SI. Temperature
    is the one common exception -- degC vs K is *additive* (degC = K -
    273.15), which a multiplicative-factor fit will always misreport as
    "disagreement" even when both numbers are correct. A third pattern shows
    up on a handful of log-y Arrhenius plots (e.g. paper 46278, design 7.42):
    the axis prints the raw log10 value itself (-1, -2, ... -6) instead of
    decade labels (10^-1, 10^-2, ...) -- display = log10(SI), not a linear
    relationship at all, so neither the multiplicative nor additive fit
    applies (and the negative label values on a physically-positive quantity
    are the tell). This function checks all three and reports whichever one
    actually fits.

    Returns a dict with `kind` ("multiplicative" | "additive" | "log10" |
    "indeterminate"), `factor` (the multiplicative k, always populated for
    applying to a plot, 1.0 when the fit isn't multiplicative), `offset`
    (populated only when `kind == "additive"`), `confident` (bool), and
    `detail` (str explaining the cross-check).
    """
    lo, hi = reg_range
    span = hi - lo
    if span == 0:
        return {
            "kind": "indeterminate",
            "factor": 1.0,
            "offset": None,
            "confident": False,
            "detail": "degenerate registry range (span=0)",
        }

    if scale == "log" and lo > 0 and (label_min < 0 or label_max < 0):
        import math

        fit = abs(math.log10(lo) - label_min) < 0.5 and abs(math.log10(hi) - label_max) < 0.5
        return {
            "kind": "log10",
            "factor": 1.0,
            "offset": None,
            "confident": True if fit else None,
            "detail": (
                f"axis prints raw log10 values ({label_min:g}..{label_max:g}) against a "
                f"positive-SI log-scale range ({lo:g}..{hi:g}) -- not a linear factor, "
                f"design 7.42's known pattern"
                + (
                    ""
                    if fit
                    else " (log10(range) doesn't match the printed values though -- "
                    "check manually)"
                )
            ),
        }

    k_span = (label_max - label_min) / span
    k_lo = (label_min / lo) if lo != 0 else None
    k_hi = (label_max / hi) if hi != 0 else None
    endpoint_ks = [k for k in (k_lo, k_hi) if k is not None]

    mult_factor, mult_agree, mult_detail = None, False, None
    if len(endpoint_ks) == 2:
        a, b = endpoint_ks
        denom = max(abs(a), abs(b), 1e-30)
        mult_agree = abs(a - b) / denom <= _FACTOR_AGREEMENT_TOL
        mult_factor = (a + b) / 2 if mult_agree else k_span
        mult_detail = f"k_min={a:.6g}, k_max={b:.6g}, k_span={k_span:.6g}"
    elif len(endpoint_ks) == 1:
        (k,) = endpoint_ks
        denom = max(abs(k), abs(k_span), 1e-30)
        mult_agree = abs(k - k_span) / denom <= _FACTOR_AGREEMENT_TOL
        mult_factor = k if mult_agree else k_span
        mult_detail = f"k_endpoint={k:.6g}, k_span={k_span:.6g}"
    else:
        mult_factor, mult_detail = k_span, f"k_span={k_span:.6g} (both range endpoints are 0)"

    if mult_agree:
        return {
            "kind": "multiplicative",
            "factor": mult_factor,
            "offset": None,
            "confident": True,
            "detail": mult_detail,
        }

    # multiplicative fit failed -- check whether it's actually additive
    # (display = SI + offset), the degC-vs-K signature.
    offset_lo = label_min - lo
    offset_hi = label_max - hi
    offset_denom = max(abs(offset_lo), abs(offset_hi), 1.0)
    additive_agree = abs(offset_lo - offset_hi) / offset_denom <= _FACTOR_AGREEMENT_TOL * 3
    if additive_agree:
        return {
            "kind": "additive",
            "factor": 1.0,
            "offset": (offset_lo + offset_hi) / 2,
            "confident": True,
            "detail": (
                f"offset_min={offset_lo:.6g}, offset_max={offset_hi:.6g} "
                f"(fits display = SI + offset, e.g. degC = K - 273.15, "
                f"not a unit-scale mismatch)"
            ),
        }

    # Neither hypothesis fit. This is very often just the "registry range is
    # a hair wider than the outermost printed label" pattern (e.g. axis
    # framed to 900 to leave room for a data point at 865, only 300-800
    # labeled) rather than an actual unit-space bug -- but a wrong best-
    # effort factor would visibly *distort* the re-plot's axis, which reads
    # as worse than not converting at all. Fall back to raw SI (factor=1)
    # rather than force a number neither check could confirm; the mismatch
    # is still surfaced via `confident=False` for the "needs attention" list.
    return {
        "kind": "indeterminate",
        "factor": 1.0,
        "offset": None,
        "confident": False,
        "detail": mult_detail + " (unreliable -- re-plot uses raw SI instead of this factor)",
    }


def _render_ground_truth_plot(
    entry: dict,
    curves: list[dict],
    out_path: Path,
    k_x: float,
    k_y: float,
    off_x: float,
    off_y: float,
    converted: bool,
) -> str | None:
    """Re-plot the ground-truth curves as `value * k + offset` per axis.

    `offset` is non-zero only for an additive axis relationship (e.g. degC =
    K - 273.15); it's applied after the multiplicative factor so both can
    combine, though in practice only one of the two is ever non-trivial for
    a given axis in this domain. Returns a warning string, if any.
    """
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
        xs = [x * k_x + off_x for x in xs]
        ys = [y * k_y + off_y for y in ys]
        color = _COLORS[i % len(_COLORS)]
        label = curve.get("prop_y", f"curve {i}")
        if sum(1 for c in curves if c.get("prop_y") == label) > 1:
            label = f"{label} #{i}"
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1, color=color, label=label)

    if dropped_nonpositive:
        warning = f"{dropped_nonpositive} non-positive y-value point(s) omitted (log-y scale)"

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    x_range = [v * k_x + off_x for v in entry["x_range"]]
    y_range = [v * k_y + off_y for v in entry["y_range"]]
    try:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
    except ValueError:
        pass
    unit_x = curves[0].get("unit_x", "") if curves else ""
    unit_y = curves[0].get("unit_y", "") if curves else ""
    prop_x = curves[0].get("prop_x", "x") if curves else "x"
    x_suffix = "" if k_x == 1 and off_x == 0 else f" x{k_x:.4g}+{off_x:.4g}"
    y_suffix = "" if k_y == 1 and off_y == 0 else f" x{k_y:.4g}+{off_y:.4g}"
    ax.set_xlabel(f"{prop_x} ({unit_x}{x_suffix})" if unit_x else prop_x)
    ax.set_ylabel(f"{unit_y}{y_suffix}" if unit_y else "y")
    ax.legend(fontsize=6, loc="best")
    title = "ground truth, paper units" if converted else "ground truth, RAW SI units (unconverted)"
    ax.set_title(f"{title} -- {len(curves)} curve(s)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return warning


def _render_pixel_overlay(entry: dict, axp: dict, out_path: Path) -> None:
    """Draw the axis_pixel_candidates tick pixel positions on top of the source image."""
    img_path = REPO_ROOT / entry["image_path"]
    img = Image.open(img_path)
    w, h = img.size
    fig, ax = plt.subplots(figsize=(5, 5 * h / w if w else 5), dpi=110)
    ax.imshow(img)
    bbox = axp["pixel_bbox_mean"]

    def _vline(px, label):
        if px is None:
            return
        ax.axvline(px, color="lime", linewidth=1, linestyle="--")
        ax.text(
            px, h * 0.02, str(label),
            color="lime", fontsize=7, ha="center", va="top", rotation=90,
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.5},
        )

    def _hline(px, label):
        if px is None:
            return
        ax.axhline(px, color="magenta", linewidth=1, linestyle="--")
        ax.text(
            w * 0.02, px, str(label),
            color="magenta", fontsize=7, ha="left", va="center",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.5},
        )

    _vline(bbox["x_min_px"], axp["x_min_label"])
    _vline(bbox["x_max_px"], axp["x_max_label"])
    _hline(bbox["y_min_px"], axp["y_min_label"])
    _hline(bbox["y_max_px"], axp["y_max_label"])
    if bbox["x_min_px"] is None and bbox["x_max_px"] is None:
        ax.text(
            w * 0.5, h * 0.5, "x-axis not visible\nin this crop",
            color="yellow", fontsize=9, ha="center", va="center",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
        )
    disagreement = axp.get("model_disagreement_px", 0)
    legend_handles = [
        Patch(color="lime", label="x tick (min/max)"),
        Patch(color="magenta", label="y tick (min/max)"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="lower right")
    ax.set_title(
        f"pixel calibration -- {axp['status']}, model disagreement {disagreement:.2g}px",
        fontsize=8,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_review_html(
    rows: list[tuple], papers_by_id: dict, attention_keys: set[tuple[str, str]]
) -> None:
    """Writes a single self-contained HTML page for one-by-one manual review.

    Plain local file, not an Artifact -- opened directly via file:// in a
    browser, so none of the Artifact sandbox's constraints apply (no size
    cap beyond being a reasonable local file, no CSP blocking localStorage
    or downloads). Images are referenced by path relative to this file
    (which lives in AUDIT_DIR alongside plots/ and overlays/), not
    embedded, since embedding ~330 images as data URIs would bloat a single
    HTML file for no benefit when the images already sit right next to it
    on disk.

    Review state (OK / flagged + a free-text note per entry, for "this one
    looks wrong, here's why") is kept in the browser's localStorage, keyed
    by entry id, so it survives reloads without needing a backend -- plus
    an Export button that downloads the current state as JSON, since
    localStorage for file:// pages is Chrome-reliable but not guaranteed
    across every browser.
    """
    entries = []
    for entry, curves, plot_path, overlay_path, axp, factor_detail, factor_source, warning in rows:
        slug = _slug(entry)
        anchor = slug.lower()
        paper = papers_by_id.get(entry["paper_id"], {})
        img_rel = os.path.relpath(REPO_ROOT / entry["image_path"], AUDIT_DIR)
        plot_rel = os.path.relpath(plot_path, AUDIT_DIR)
        overlay_rel = os.path.relpath(overlay_path, AUDIT_DIR) if overlay_path else None

        k_x = k_y = 1.0
        off_x = off_y = 0.0
        if factor_source == "axis-pixel":
            fx, fy = factor_detail["x"], factor_detail["y"]

            def _short(f: dict) -> str:
                if f["kind"] == "additive":
                    return f"+{f['offset']:.4g}"
                if f["kind"] in ("log10", "indeterminate"):
                    return "raw SI"
                return f"×{f['factor']:.4g}"

            factor_summary = f"x: {_short(fx)}, y: {_short(fy)} (axis-pixel derived)"
            k_x, off_x = fx["factor"], (fx["offset"] or 0.0)
            k_y, off_y = fy["factor"], (fy["offset"] or 0.0)
        elif factor_source == "evidence-text":
            fy = factor_detail["y"]
            factor_summary = f"y: ×{fy['factor']:.4g} (evidence-text, unverified)"
            k_y = fy["factor"]
        else:
            factor_summary = "raw SI (no conversion source)"

        x_range_display = [v * k_x + off_x for v in entry["x_range"]]
        y_range_display = [v * k_y + off_y for v in entry["y_range"]]
        is_converted = (k_x, off_x, k_y, off_y) != (1.0, 0.0, 1.0, 0.0)

        entries.append(
            {
                "id": anchor,
                "paper_id": entry["paper_id"],
                "figure_reference": entry["figure_reference"],
                "doi": paper.get("doi", "?"),
                "license_id": entry.get("license_id", "?"),
                "x_range": entry["x_range"],
                "y_range": entry["y_range"],
                "x_range_display": x_range_display if is_converted else None,
                "y_range_display": y_range_display if is_converted else None,
                "x_scale": entry.get("x_scale", "linear"),
                "y_scale": entry.get("y_scale", "linear"),
                "verified_at": entry.get("verified_at", "?"),
                "evidence": entry["evidence"],
                "n_curves": len(curves),
                "img": img_rel,
                "overlay": overlay_rel,
                "plot": plot_rel,
                "warning": warning,
                "axp_status": axp["status"] if axp else None,
                "factor_summary": factor_summary,
                "needs_attention": (entry["paper_id"], entry["figure_id"]) in attention_keys,
            }
        )

    data_json = json.dumps(entries, ensure_ascii=False)
    template = (Path(__file__).parent / "_review_html_template.html").read_text()
    html = template.replace("__ENTRIES_JSON__", data_json)
    REVIEW_HTML_PATH.write_text(html)


def main() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    papers_by_id = {p["paper_id"]: p for p in json.loads(PAPERS_PATH.read_text())}
    axis_candidates_raw = json.loads(AXIS_PIXEL_CANDIDATES_PATH.read_text())
    axp_by_key = {
        (a["paper_id"], a["figure_id"]): a for a in axis_candidates_raw if "_meta" not in a
    }

    verified = [e for e in registry if e["status"] == "verified"]
    verified.sort(key=lambda e: (e["paper_id"], e["figure_reference"]))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    n_with_axis_data = 0
    n_with_evidence_text_factor = 0
    attention = []  # entries whose derived factor disagreed at the two endpoints

    for entry in verified:
        curves = ground_truth.get(entry["figure_id"], [])
        slug = _slug(entry)
        axp = axp_by_key.get((entry["paper_id"], entry["figure_id"]))
        if axp is not None and axp.get("status") == "excluded":
            axp = None

        k_x, k_y, off_x, off_y = 1.0, 1.0, 0.0, 0.0
        factor_detail = None
        factor_source = "none"
        overlay_path = None

        if axp is not None:
            n_with_axis_data += 1
            factor_source = "axis-pixel"
            no_label = {
                "kind": "indeterminate",
                "factor": 1.0,
                "offset": None,
                "confident": None,
                "detail": "no printed label for this axis in the source crop (see notes)",
            }
            x_scale = entry.get("x_scale", "linear")
            y_scale = entry.get("y_scale", "linear")
            fx = (
                _derive_factor(entry["x_range"], axp["x_min_label"], axp["x_max_label"], x_scale)
                if axp["x_min_label"] is not None and axp["x_max_label"] is not None
                else no_label
            )
            fy = (
                _derive_factor(entry["y_range"], axp["y_min_label"], axp["y_max_label"], y_scale)
                if axp["y_min_label"] is not None and axp["y_max_label"] is not None
                else no_label
            )
            factor_detail = {"x": fx, "y": fy}
            k_x, off_x = fx["factor"], (fx["offset"] or 0.0)
            k_y, off_y = fy["factor"], (fy["offset"] or 0.0)
            if fx["confident"] is False or fy["confident"] is False:
                attention.append((entry, fx, fy))

            overlay_path = OVERLAYS_DIR / f"{slug}.png"
            _render_pixel_overlay(entry, axp, overlay_path)
        else:
            # x is unconverted either way (always Temperature/K in this
            # corpus); only y benefits from the evidence-text fallback.
            fy = _evidence_text_factor(entry["evidence"])
            if fy is not None:
                n_with_evidence_text_factor += 1
                factor_source = "evidence-text"
                factor_detail = {"x": None, "y": fy}
                k_y = fy["factor"]

        plot_path = PLOTS_DIR / f"{slug}.png"
        warning = _render_ground_truth_plot(
            entry, curves, plot_path, k_x, k_y, off_x, off_y, factor_source != "none"
        )
        rows.append(
            (entry, curves, plot_path, overlay_path, axp, factor_detail, factor_source, warning)
        )

    lines: list[str] = []
    lines.append("# Verified Pairs Visual Audit")
    lines.append("")
    lines.append(
        f"Generated from `data/verified_pairs/registry.json` "
        f"({len(verified)} `status=\"verified\"` entries) + "
        f"`data/verified_pairs/ground_truth.json` + "
        f"`data/verified_pairs/axis_pixel_candidates.json`. Regenerate with "
        f"`python scripts/eval/generate_verified_pairs_visual_audit.py`."
    )
    lines.append("")
    lines.append(
        "Each entry shows, left to right: **(1) the original chart crop**, "
        "**(2) the pixel-calibration overlay** (green = x-axis tick pixel positions, "
        "magenta = y-axis tick pixel positions, from `axis_pixel_candidates.json` -- "
        "only present for entries that have axis-pixel ground truth), and "
        "**(3) the digitized ground-truth curve(s) re-plotted in the paper's own "
        "display units** (derived from the axis-pixel tick label values, not guessed -- "
        "see the module docstring). Where no axis-pixel data exists yet, (2) is omitted "
        "and (3) falls back to raw SI units, clearly labeled."
    )
    lines.append("")
    n_log_y = sum(1 for e in verified if e.get("y_scale") == "log")
    lines.append(f"- Total verified entries: **{len(verified)}**")
    lines.append(
        f"- Entries with axis-pixel ground truth (unit-converted + calibration-checkable): "
        f"**{n_with_axis_data}** / {len(verified)}"
    )
    lines.append(
        f"- Entries additionally unit-converted from evidence text (unverified fallback, "
        f"y-axis only): **{n_with_evidence_text_factor}** / {len(verified)}"
    )
    n_raw_si = len(verified) - n_with_axis_data - n_with_evidence_text_factor
    lines.append(f"- Entries still shown in raw SI units (no conversion source): {n_raw_si}")
    lines.append(f"- log-y entries: {n_log_y}")
    n_render_warnings = sum(1 for *_, w in rows if w)
    lines.append(f"- Entries with a re-plot rendering warning: {n_render_warnings}")
    lines.append(f"- Entries with a factor-agreement mismatch (see below): {len(attention)}")
    lines.append("")

    if attention:
        lines.append("## ⚠️ Needs attention: axis range / tick label disagreement")
        lines.append("")
        lines.append(
            "For these entries, the registry's `x_range`/`y_range` and the axis-pixel "
            "candidate's tick label values don't scale by a single consistent factor at "
            "both the min and max end. This usually means one of the two doesn't actually "
            "describe the same axis extent (mismatched panel, a misread tick, or the "
            "registry range extending past the outermost printed tick) -- **not "
            "necessarily an error**, but worth a manual look. The re-plot below still uses "
            "a best-effort factor (the whole-range ratio) so it's in the right ballpark."
        )
        lines.append("")
        for entry, fx, fy in attention:
            anchor = _slug(entry).lower()
            lines.append(
                f"- [ ] [{entry['paper_id']} / fig {entry['figure_reference']}](#{anchor})"
            )
            if fx["confident"] is False:
                lines.append(f"  - x-axis: {fx['detail']}")
            if fy["confident"] is False:
                lines.append(f"  - y-axis: {fy['detail']}")
        lines.append("")

    lines.append("## Index")
    lines.append("")
    for entry, _curves, _plot_path, _overlay_path, axp, _fd, factor_source, warning in rows:
        anchor = _slug(entry).lower()
        flags = ""
        if warning:
            flags += " ⚠️render"
        if factor_source == "none":
            flags += " 🚫noconversion"
        elif factor_source == "evidence-text":
            flags += " 📝text-derived"
        elif axp is not None and axp.get("status") == "llm_candidate":
            flags += " 🟡unverified-axis"
        lines.append(
            f"- [ ] [{entry['paper_id']} / fig {entry['figure_reference']}]"
            f"(#{anchor}){flags}"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    for entry, curves, plot_path, overlay_path, axp, factor_detail, factor_source, warning in rows:
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
        def _describe(f: dict) -> str:
            if f["kind"] == "additive":
                return f"+{f['offset']:.6g} (degC/K-style offset)"
            if f["kind"] == "log10":
                return "n/a (raw-log10-printed axis, see note)"
            if f["kind"] == "indeterminate":
                if f["confident"] is False:
                    return "raw SI (unreliable factor, see 'needs attention' above)"
                return "n/a (no label)"
            return f"x{f['factor']:.6g}"

        if factor_source == "axis-pixel":
            fx, fy = factor_detail["x"], factor_detail["y"]
            if fx["confident"] is False or fy["confident"] is False:
                conf_note = "⚠️ see 'needs attention' section above"
            elif fx["confident"] and fy["confident"]:
                conf_note = "✅ confirmed at both axis endpoints"
            else:
                conf_note = "ℹ️ not a simple linear factor on one axis (see above)"
            disagreement_px = axp.get("model_disagreement_px")
            disagreement_str = f"{disagreement_px:.2g}px" if disagreement_px is not None else "n/a"
            lines.append(
                f"> Axis-pixel status: **{axp['status']}** "
                f"(model disagreement {disagreement_str}). "
                f"x: {_describe(fx)}, y: {_describe(fy)} -- {conf_note}."
            )
            lines.append("")
        elif factor_source == "evidence-text":
            fy = factor_detail["y"]
            lines.append(
                f"> 📝 No axis-pixel ground truth for this entry -- y converted from a factor "
                f"**parsed out of the evidence text** instead (y: {_describe(fy)}, "
                f"{fy['detail']}). This was validated once by whoever wrote the entry but is "
                f"**not independently re-checked here** -- treat with a bit less confidence "
                f"than the axis-pixel-derived entries above. x-axis is unconverted (raw SI == "
                f"the paper's units for temperature in this domain)."
            )
            lines.append("")
        else:
            lines.append(
                "> 🚫 No axis-pixel ground truth and no parseable evidence-text factor for "
                "this entry -- re-plot below is raw SI units (unconverted), which may look "
                "nothing like the original chart's axis numbers even though the underlying "
                "data is correct (compare curve *shape*, not axis labels)."
            )
            lines.append("")

        x_scale = entry.get("x_scale", "linear")
        y_scale = entry.get("y_scale", "linear")
        lines.append(
            "| paper_id | figure_id | DOI | license | x_range (SI) | y_range (SI) | scale | "
            "verified_at |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
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
            f'<td width="33%"><b>original chart</b><br>'
            f'<img src="{img_rel}" width="100%"></td>'
        )
        if overlay_path is not None:
            overlay_rel = os.path.relpath(overlay_path, AUDIT_DIR)
            lines.append(
                f'<td width="33%"><b>pixel-calibration overlay</b><br>'
                f'<img src="{overlay_rel}" width="100%"></td>'
            )
        else:
            lines.append('<td width="33%"><b>pixel-calibration overlay</b><br>(not available)</td>')
        lines.append(
            f'<td width="33%"><b>digitized ground truth (re-plotted)</b><br>'
            f'<img src="{plot_rel}" width="100%"></td>'
        )
        lines.append("</tr></table>")
        lines.append("")
        lines.append(f"**evidence** (from registry.json): {entry['evidence']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    MARKDOWN_PATH.write_text("\n".join(lines))

    attention_keys = {(e["paper_id"], e["figure_id"]) for e, _fx, _fy in attention}
    _write_review_html(rows, papers_by_id, attention_keys)

    print(
        f"wrote {MARKDOWN_PATH} and {REVIEW_HTML_PATH} ({len(verified)} entries, "
        f"{n_with_axis_data} with axis-pixel data, "
        f"{n_with_evidence_text_factor} with evidence-text-derived factor, "
        f"{len(attention)} flagged for attention)"
    )


if __name__ == "__main__":
    main()

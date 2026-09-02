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
      has a matching entry, each axis's registry range vs. printed tick
      labels is classified by `domain/pairing_checks.py::
      classify_range_disagreement` (design 7.49/7.52) into one of five
      verdicts -- BENIGN_MARGIN, UNIT_SPACE_DIFFERENCE, AXIS_SCALE_FACTOR,
      REAL_MISMATCH, INDETERMINATE -- and `display_conversion` turns that
      verdict into the `(factor, offset)` actually safe to apply for the
      re-plot (see both functions' docstrings). This REPLACES the old
      `_derive_factor`, which *fitted* a factor from the endpoints and
      flagged any disagreement as "needs attention" -- 45 of 111 entries,
      of which an independent triage
      (docs/experiments/2026-09-02-flagged-entries-triage.md) found 43
      were not errors at all, just the fitting method's inability to tell
      a benign one-sided framing margin from a real unit bug. Since
      commit 31bd7e9 (design 7.47), registry.json/ground_truth.json are
      stored in each paper's *display* units, so the a-priori expected
      registry-vs-label relationship is the identity map, not something to
      fit.
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
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from real_chart_bench.domain.curve import ScaleType  # noqa: E402
from real_chart_bench.domain.pairing_checks import (  # noqa: E402
    AxisPixelCalibration,
    RangeDisagreementVerdict,
    Verdict,
    classify_range_disagreement,
    display_conversion,
)
from real_chart_bench.domain.unit_conversion import (  # noqa: E402
    IncompatibleUnitsError,
    UnitParseError,
    si_to_display_factor,
)

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

# Verdicts safe to label the re-plot axis with the printed unit *text*: in
# all three, `display_conversion` returns a conversion that's either the
# identity (registry already in display units, BENIGN_MARGIN) or one
# actually derived from/confirmed against the printed axis itself
# (UNIT_SPACE_DIFFERENCE, AXIS_SCALE_FACTOR). REAL_MISMATCH/INDETERMINATE
# are deliberately excluded -- labeling with the printed unit there would
# silently imply a conversion that was never confirmed.
_SAFE_TO_LABEL_VERDICTS = (
    Verdict.BENIGN_MARGIN,
    Verdict.UNIT_SPACE_DIFFERENCE,
    Verdict.AXIS_SCALE_FACTOR,
)

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
    you cross-checked"). Unlike the axis-pixel-derived path (now
    `domain/pairing_checks.py`), this is NOT independently re-verified here,
    just trusting text a human/agent already validated once against the
    image -- callers must label it as such.
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


class AxisEvaluation(NamedTuple):
    """One axis's (registry vs. printed-label) evaluation for one entry:
    the `RangeDisagreementVerdict` from `classify_range_disagreement`
    (`None` when this axis has no printed label captured in this crop --
    nothing to classify, not the same as an INDETERMINATE verdict), plus the
    raw inputs that produced it -- needed again by `display_conversion` and
    for rendering.
    """

    verdict: RangeDisagreementVerdict | None
    label_min: float | None
    label_max: float | None
    reg_lo: float
    reg_hi: float
    gt_unit: str | None
    printed_unit: str | None


def _axis_scale(entry: dict, axis: str) -> ScaleType:
    return ScaleType.LOG if entry.get(f"{axis}_scale", "linear") == "log" else ScaleType.LINEAR


def _axis_calibration(
    axp: dict, axis: str, img_w: int, img_h: int
) -> AxisPixelCalibration | None:
    """Builds the pixel calibration `classify_range_disagreement`'s rule (d)
    needs to project a registry endpoint into the image and check it lands
    inside it (design 7.49's paper 10939/figure 1528 case -- the one rule
    this can't be skipped for)."""
    bbox = axp["pixel_bbox_mean"]
    lo_px, hi_px = bbox.get(f"{axis}_min_px"), bbox.get(f"{axis}_max_px")
    if lo_px is None or hi_px is None:
        return None
    extent = img_w if axis == "x" else img_h
    return AxisPixelCalibration(label_lo_px=lo_px, label_hi_px=hi_px, image_extent_px=extent)


def _evaluate_axis(
    entry: dict, axp: dict | None, curves: list[dict], axis: str, img_w: int, img_h: int
) -> AxisEvaluation:
    """Runs `classify_range_disagreement` for one axis ("x" or "y") of one
    verified_pairs entry, wiring up everything the domain function needs
    from this script's own data files: registry.json's range + scale,
    axis_pixel_candidates.json's printed labels + pixel positions + printed
    unit text, and ground_truth.json's curve values + unit (`gt_unit`/
    `printed_unit` are the genuinely independent third constraint -- see
    design 7.52 -- so they're always passed when available).
    """
    reg_lo, reg_hi = entry[f"{axis}_range"]
    gt_unit = curves[0].get(f"unit_{axis}") if curves else None
    printed_unit = axp.get(f"{axis}_axis_unit") if axp else None
    label_min = axp.get(f"{axis}_min_label") if axp else None
    label_max = axp.get(f"{axis}_max_label") if axp else None

    if axp is None or label_min is None or label_max is None:
        return AxisEvaluation(None, label_min, label_max, reg_lo, reg_hi, gt_unit, printed_unit)

    gt_extents: list[float] = []
    for curve in curves:
        values = curve.get(axis) or []
        if values:
            gt_extents.append(min(values))
            gt_extents.append(max(values))

    verdict = classify_range_disagreement(
        label_min=label_min,
        label_max=label_max,
        reg_lo=reg_lo,
        reg_hi=reg_hi,
        scale=_axis_scale(entry, axis),
        gt_extents=gt_extents,
        calibration=_axis_calibration(axp, axis, img_w, img_h),
        gt_unit=gt_unit,
        printed_unit=printed_unit,
    )
    return AxisEvaluation(verdict, label_min, label_max, reg_lo, reg_hi, gt_unit, printed_unit)


def _axis_conversion(axis_eval: AxisEvaluation) -> tuple[float, float]:
    """`(factor, offset)` to apply to registry-space values for this axis's
    re-plot -- `display_conversion` dispatched on the verdict, or the
    identity when this axis had no printed label to classify at all (never
    guess a conversion with nothing to check it against)."""
    if axis_eval.verdict is None:
        return 1.0, 0.0
    return display_conversion(
        axis_eval.verdict.verdict,
        label_min=axis_eval.label_min,
        label_max=axis_eval.label_max,
        reg_lo=axis_eval.reg_lo,
        reg_hi=axis_eval.reg_hi,
        gt_unit=axis_eval.gt_unit,
        printed_unit=axis_eval.printed_unit,
    )


def _printed_unit_for_label(axis_eval: AxisEvaluation) -> str | None:
    """The printed unit text to use for the re-plot's axis label, or `None`
    to fall back to Starrydata's raw SI unit name instead -- only for a
    verdict where the applied conversion is actually confirmed to land in
    that printed unit's space (see `_SAFE_TO_LABEL_VERDICTS`). "-" (this
    corpus's placeholder for "dimensionless / no unit captured") is not a
    real unit string to print."""
    if axis_eval.verdict is None or axis_eval.verdict.verdict not in _SAFE_TO_LABEL_VERDICTS:
        return None
    unit = axis_eval.printed_unit
    return None if unit in (None, "-") else unit


def _describe_axis_eval(axis_eval: AxisEvaluation) -> str:
    """One-line human-readable verdict description for the per-entry
    Markdown section."""
    if axis_eval.verdict is None:
        return "n/a (no printed label for this axis in the source crop)"
    descriptions = {
        Verdict.BENIGN_MARGIN: "benign_margin (same unit space, ordinary framing margin)",
        Verdict.UNIT_SPACE_DIFFERENCE: (
            "unit_space_difference (display-unit conversion backlog -- normal, see design 7.47)"
        ),
        Verdict.AXIS_SCALE_FACTOR: (
            "axis_scale_factor (axis-label multiplier, e.g. '1000/T' -- normal)"
        ),
        Verdict.REAL_MISMATCH: "⚠️ REAL_MISMATCH",
        Verdict.INDETERMINATE: "❓ INDETERMINATE",
    }
    return descriptions[axis_eval.verdict.verdict]


def _render_ground_truth_plot(
    entry: dict,
    curves: list[dict],
    out_path: Path,
    k_x: float,
    k_y: float,
    off_x: float,
    off_y: float,
    converted: bool,
    printed_x_unit: str | None = None,
    printed_y_unit: str | None = None,
) -> str | None:
    """Re-plot the ground-truth curves as `value * k + offset` per axis.

    `offset` is non-zero only for an additive axis relationship (e.g. degC =
    K - 273.15); it's applied after the multiplicative factor so both can
    combine, though in practice only one of the two is ever non-trivial for
    a given axis in this domain. Returns a warning string, if any.

    `printed_x_unit`/`printed_y_unit` (from axis_pixel_candidates.json's
    design-7.46 fields, when captured) are the *actual unit text printed on
    the original chart's axis*, e.g. "μV/K" -- when available, the axis
    label uses that directly, since it's what the human is comparing
    against. Without it, the label falls back to Starrydata's raw SI unit
    name plus the applied factor as a suffix (still correct, just less
    immediately legible).
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

    if printed_x_unit:
        ax.set_xlabel(f"{prop_x} ({printed_x_unit})")
    else:
        x_suffix = "" if k_x == 1 and off_x == 0 else f" x{k_x:.4g}+{off_x:.4g}"
        ax.set_xlabel(f"{prop_x} ({unit_x}{x_suffix})" if unit_x else prop_x)

    if printed_y_unit:
        ax.set_ylabel(printed_y_unit)
    else:
        y_suffix = "" if k_y == 1 and off_y == 0 else f" x{k_y:.4g}+{off_y:.4g}"
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


def _dimensional_unit_check(
    si_unit: str | None, printed_unit: str | None, numeric_factor: float, factor_source: str
) -> str | None:
    """Cross-checks the evidence-text-derived numeric SI->display factor
    against one predicted purely from the two unit *names* (dimensional
    analysis, `unit_conversion.si_to_display_factor`) -- a second,
    independent verification route for exactly the kind of "Starrydata's raw
    value doesn't actually match physical reality" or "registry's
    calibration doesn't match the stated unit" bug that a numeric-tick-only
    check can't see (see design 7.46). Only used for the evidence-text
    fallback path now -- axis-pixel-derived entries get this same
    cross-check, and more, from `classify_range_disagreement`'s rule (e)
    directly (design 7.52). Returns a short human-readable verdict string,
    or None if the check isn't applicable (no printed unit captured, or no
    SI unit on record).
    """
    if not printed_unit or not si_unit:
        return None
    try:
        predicted = si_to_display_factor(si_unit, printed_unit)
    except IncompatibleUnitsError as exc:
        return f"⚠️ dimension mismatch: {exc}"
    except UnitParseError:
        return None  # printed unit text too irregular to parse -- not a finding

    if factor_source == "none" or numeric_factor == 1.0:
        return (
            f"ℹ️ dimensional analysis predicts ×{predicted:.4g} "
            f"(no independent numeric factor to compare against)"
        )
    if abs(predicted / numeric_factor - 1) <= 0.05:
        return f"✅ confirmed by dimensional analysis (×{predicted:.4g})"
    return (
        f"⚠️ dimensional analysis predicts ×{predicted:.4g} but the evidence-text-derived "
        f"factor is ×{numeric_factor:.4g} -- worth a second look"
    )


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
    for (
        entry,
        curves,
        plot_path,
        overlay_path,
        axp,
        x_eval,
        y_eval,
        factor_detail,
        factor_source,
        warning,
    ) in rows:
        slug = _slug(entry)
        anchor = slug.lower()
        paper = papers_by_id.get(entry["paper_id"], {})
        img_rel = os.path.relpath(REPO_ROOT / entry["image_path"], AUDIT_DIR)
        plot_rel = os.path.relpath(plot_path, AUDIT_DIR)
        overlay_rel = os.path.relpath(overlay_path, AUDIT_DIR) if overlay_path else None

        k_x = k_y = 1.0
        off_x = off_y = 0.0
        x_verdict = y_verdict = None
        x_verdict_reason = y_verdict_reason = None
        if factor_source == "axis-pixel":
            k_x, off_x = _axis_conversion(x_eval)
            k_y, off_y = _axis_conversion(y_eval)
            x_verdict = x_eval.verdict.verdict.value if x_eval.verdict else None
            y_verdict = y_eval.verdict.verdict.value if y_eval.verdict else None
            x_verdict_reason = (
                x_eval.verdict.reason if x_eval.verdict else "no printed label for x-axis"
            )
            y_verdict_reason = (
                y_eval.verdict.reason if y_eval.verdict else "no printed label for y-axis"
            )

            def _fmt_k(k: float, off: float) -> str:
                suffix = "" if off == 0 else f"{off:+.4g}"
                return f"×{k:.4g}{suffix}"

            factor_summary = (
                f"x: {x_verdict or 'n/a'} ({_fmt_k(k_x, off_x)}), "
                f"y: {y_verdict or 'n/a'} ({_fmt_k(k_y, off_y)}) "
                "(domain/pairing_checks.py)"
            )
        elif factor_source == "evidence-text":
            fy = factor_detail["y"]
            factor_summary = f"y: ×{fy['factor']:.4g} (evidence-text, unverified)"
            k_y = fy["factor"]
        else:
            factor_summary = "raw SI (no conversion source)"

        x_range_display = [v * k_x + off_x for v in entry["x_range"]]
        y_range_display = [v * k_y + off_y for v in entry["y_range"]]
        is_converted = (k_x, off_x, k_y, off_y) != (1.0, 0.0, 1.0, 0.0)

        printed_y_unit = axp.get("y_axis_unit") if axp else None
        si_unit_y = curves[0].get("unit_y") if curves else None
        unit_check = (
            None
            if factor_source == "axis-pixel"
            else _dimensional_unit_check(si_unit_y, printed_y_unit, k_y, factor_source)
        )

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
                "printed_y_unit": printed_y_unit,
                "unit_check": unit_check,
                "x_verdict": x_verdict,
                "y_verdict": y_verdict,
                "x_verdict_reason": x_verdict_reason,
                "y_verdict_reason": y_verdict_reason,
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

    # Axes actually classified by classify_range_disagreement, bucketed by
    # verdict: (entry, axis_name, RangeDisagreementVerdict). BENIGN_MARGIN is
    # tracked only as a count -- it is explicitly NOT "needs attention" (the
    # entire point of design 7.49/7.52 replacing the old fitted-factor
    # "needs attention" flag, which fired on 45/111 entries, 43 of which an
    # independent triage found were not errors).
    n_benign_margin = 0
    real_mismatch: list[tuple[dict, str, RangeDisagreementVerdict]] = []
    indeterminate: list[tuple[dict, str, RangeDisagreementVerdict]] = []
    unit_space_difference: list[tuple[dict, str, RangeDisagreementVerdict]] = []
    axis_scale_factor: list[tuple[dict, str, RangeDisagreementVerdict]] = []

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
        x_eval = y_eval = None

        if axp is not None:
            n_with_axis_data += 1
            factor_source = "axis-pixel"

            img_path = REPO_ROOT / entry["image_path"]
            img_w, img_h = Image.open(img_path).size

            x_eval = _evaluate_axis(entry, axp, curves, "x", img_w, img_h)
            y_eval = _evaluate_axis(entry, axp, curves, "y", img_w, img_h)
            k_x, off_x = _axis_conversion(x_eval)
            k_y, off_y = _axis_conversion(y_eval)

            for axis_name, axis_eval in (("x", x_eval), ("y", y_eval)):
                if axis_eval.verdict is None:
                    continue
                v = axis_eval.verdict.verdict
                if v is Verdict.BENIGN_MARGIN:
                    n_benign_margin += 1
                elif v is Verdict.UNIT_SPACE_DIFFERENCE:
                    unit_space_difference.append((entry, axis_name, axis_eval.verdict))
                elif v is Verdict.AXIS_SCALE_FACTOR:
                    axis_scale_factor.append((entry, axis_name, axis_eval.verdict))
                elif v is Verdict.REAL_MISMATCH:
                    real_mismatch.append((entry, axis_name, axis_eval.verdict))
                else:
                    indeterminate.append((entry, axis_name, axis_eval.verdict))

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
        printed_x_unit = _printed_unit_for_label(x_eval) if x_eval is not None else None
        printed_y_unit = _printed_unit_for_label(y_eval) if y_eval is not None else None
        warning = _render_ground_truth_plot(
            entry,
            curves,
            plot_path,
            k_x,
            k_y,
            off_x,
            off_y,
            factor_source != "none",
            printed_x_unit,
            printed_y_unit,
        )
        rows.append(
            (
                entry,
                curves,
                plot_path,
                overlay_path,
                axp,
                x_eval,
                y_eval,
                factor_detail,
                factor_source,
                warning,
            )
        )

    n_axes_evaluated = (
        n_benign_margin
        + len(unit_space_difference)
        + len(axis_scale_factor)
        + len(real_mismatch)
        + len(indeterminate)
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
        "display units** where the axis's classification (`domain/pairing_checks.py::"
        "classify_range_disagreement`, design 7.49/7.52) confirms it's safe to. Where "
        "no axis-pixel data exists yet, (2) is omitted and (3) falls back to raw SI "
        "units, clearly labeled."
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
    lines.append(
        f"- Axis-pixel-derived range classifications "
        f"(`domain/pairing_checks.py::classify_range_disagreement`, {n_axes_evaluated} axes "
        f"evaluated): benign_margin=**{n_benign_margin}**, "
        f"unit_space_difference=**{len(unit_space_difference)}**, "
        f"axis_scale_factor=**{len(axis_scale_factor)}**, "
        f"real_mismatch=**{len(real_mismatch)}**, indeterminate=**{len(indeterminate)}**"
    )
    lines.append("")

    def _attention_list(
        title: str, intro: str, items: list[tuple[dict, str, RangeDisagreementVerdict]]
    ) -> None:
        if not items:
            return
        lines.append(title)
        lines.append("")
        lines.append(intro)
        lines.append("")
        for entry, axis_name, verdict in items:
            anchor = _slug(entry).lower()
            lines.append(
                f"- [ ] [{entry['paper_id']} / fig {entry['figure_reference']}](#{anchor}) "
                f"-- {axis_name}-axis: {verdict.reason}"
            )
        lines.append("")

    _attention_list(
        "## ⚠️ Needs attention: REAL_MISMATCH",
        "For these axes, the registry range does not encode the relationship the axis's own "
        "unit strings and/or GT curve extents declare -- this is the group that actually "
        "warrants a human look (replaces the old fitted-factor 'needs attention' flag, which "
        "fired on 45/111 entries -- an independent triage found 43 of those were not errors, "
        "just the old method's inability to tell a benign framing margin from a real bug; "
        "see docs/experiments/2026-09-02-flagged-entries-triage.md).",
        real_mismatch,
    )
    _attention_list(
        "## ❓ Indeterminate (unable to classify)",
        "For these axes, the available data (printed unit strings, GT extents, pixel "
        "calibration) wasn't enough to confidently classify the registry-vs-label "
        "relationship -- not a known-benign pattern, but not confirmed as an error either. "
        "Worth a quieter look than REAL_MISMATCH above.",
        indeterminate,
    )
    _attention_list(
        "## ℹ️ Display-unit conversion backlog (UNIT_SPACE_DIFFERENCE)",
        "For these axes, the registry range and GT curve are internally self-consistent but "
        "in a different (still-SI, or Kelvin-vs-Celsius) unit space than the printed axis -- "
        "the design-7.47 migration to display units hasn't reached these entries yet. "
        "**Not an error** -- informational backlog tracking only.",
        unit_space_difference,
    )
    _attention_list(
        "## ℹ️ Axis-label scale factors (AXIS_SCALE_FACTOR)",
        "For these axes, the printed axis carries its own scale-factor annotation (e.g. "
        "'1000/T', 'sigma x10^4') on top of a dimensionally-identical unit -- the endpoints "
        "cleanly imply a power-of-ten multiplier that is the axis label's own declared "
        "choice, not a unit or endpoint error. **Normal, no action needed.**",
        axis_scale_factor,
    )

    lines.append("## Index")
    lines.append("")
    for entry, _curves, _plot_path, _overlay_path, axp, _xe, _ye, _fd, factor_source, warning in (
        rows
    ):
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

    for (
        entry,
        curves,
        plot_path,
        overlay_path,
        axp,
        x_eval,
        y_eval,
        factor_detail,
        factor_source,
        warning,
    ) in rows:
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

        if factor_source == "axis-pixel":
            flagged = any(
                ev is not None
                and ev.verdict is not None
                and ev.verdict.verdict in (Verdict.REAL_MISMATCH, Verdict.INDETERMINATE)
                for ev in (x_eval, y_eval)
            )
            conf_note = (
                "⚠️ see 'needs attention' section above"
                if flagged
                else "classified via domain/pairing_checks.py"
            )
            disagreement_px = axp.get("model_disagreement_px")
            disagreement_str = f"{disagreement_px:.2g}px" if disagreement_px is not None else "n/a"
            lines.append(
                f"> Axis-pixel status: **{axp['status']}** "
                f"(model disagreement {disagreement_str}). "
                f"x: {_describe_axis_eval(x_eval)}, y: {_describe_axis_eval(y_eval)} -- "
                f"{conf_note}."
            )
            lines.append("")
            for axis_name, axis_eval in (("x", x_eval), ("y", y_eval)):
                if axis_eval.verdict is not None:
                    lines.append(f"> - {axis_name}-axis: {axis_eval.verdict.reason}")
            lines.append("")
        elif factor_source == "evidence-text":
            fy = factor_detail["y"]
            lines.append(
                f"> 📝 No axis-pixel ground truth for this entry -- y converted from a factor "
                f"**parsed out of the evidence text** instead (y: x{fy['factor']:.6g}, "
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

    attention_keys = {
        (e["paper_id"], e["figure_id"]) for e, _axis, _v in (real_mismatch + indeterminate)
    }
    _write_review_html(rows, papers_by_id, attention_keys)

    print(
        f"wrote {MARKDOWN_PATH} and {REVIEW_HTML_PATH} ({len(verified)} entries, "
        f"{n_with_axis_data} with axis-pixel data, "
        f"{n_with_evidence_text_factor} with evidence-text-derived factor, "
        f"{n_axes_evaluated} axes classified: benign_margin={n_benign_margin}, "
        f"unit_space_difference={len(unit_space_difference)}, "
        f"axis_scale_factor={len(axis_scale_factor)}, real_mismatch={len(real_mismatch)}, "
        f"indeterminate={len(indeterminate)})"
    )


if __name__ == "__main__":
    main()

"""Exports every model's extracted curves, per figure, for the results explorer
page: ground truth, each model's curves in data space, which ground-truth
series each predicted curve was matched to, and the per-figure scores.

Curves are rebuilt from what each model actually produced -- LLM answers from
data/llm_run_v2/calibrated/ (with the same unit rescale the scorer
applies), LineFormer from its raw pixel output, the CV baselines by re-running
them -- and re-scored here. Each recomputed score is checked against the
published results/*.json, so the page cannot show curves that are not the
ones that were scored.

Every model x figure also carries its point evaluation at the primary tau
(design §7.67): which ground-truth points were found or missed, which
predicted points matched or were extra, and the matched pairs -- for both the
calibrated condition (axis ranges given) and, for the LLMs, noaxis.

LLM-style runs kept outside the repo (e.g. local VLMs) can be added with
--extra-llm-archive (same layout as data/llm_run_v2) and --extra-results (their
scored payloads, <model>-v0-r2.json / -v0-r2-noaxis.json); they are verified
against those payloads the same way and marked as local runs.

Output: <out>/data.json and <out>/images/<figure_id>.<ext>
Usage: python scripts/viz/export_results_explorer.py [--out DIR]
           [--extra-llm-archive DIR --extra-results DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eval"))

from run_baselines import build_dataset  # noqa: E402
from score_llm_predictions import PREDICTION_RESCALE_V2_CALIBRATED, parse_curves  # noqa: E402

from real_chart_bench.adapter.achromatic_cv_extractor import AchromaticCvModelRunner  # noqa: E402
from real_chart_bench.adapter.lineformer_model_runner import (  # noqa: E402
    LineFormerPrediction,
    PrecomputedLineFormerModelRunner,
    image_key,
)
from real_chart_bench.adapter.naive_cv_extractor import NaiveCvModelRunner  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.curve import Curve  # noqa: E402
from real_chart_bench.domain.evaluation import evaluate_figure  # noqa: E402
from real_chart_bench.domain.pixel_calibration import PixelCalibration  # noqa: E402
from real_chart_bench.domain.point_metrics import evaluate_points  # noqa: E402
from real_chart_bench.usecase.evaluate_dataset import (  # noqa: E402
    POINT_NORM,
    PRIMARY_POINT_TAU,
    axis_frame_for_task,
    matcher_for_task,
)

MAX_POINTS = 400  # per curve, for display only; scoring uses every point

# the 2026-10-01 run (axis ranges given) -- design 7.66
LLM_MODELS = {
    "claude-fable-5-1": "Claude Fable 5.1",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}
MODELS = [
    # (id, name, kind, published results file)
    *[(m, n, "llm", f"{m}-v0-r2.json") for m, n in LLM_MODELS.items()],
    # the current LineFormer run's file is named after its figure count
    (
        "lineformer",
        "LineFormer (pretrained)",
        "dedicated",
        next(
            p.name
            for p in (REPO / "results").glob("lineformer-pretrained-n*.json")
            if p.stem[23:].isdigit()
        ),
    ),
    ("naive-cv", "naive-cv (hue)", "cv", "naive-cv-v0.json"),
    ("achromatic-cv", "achromatic-cv (luminance)", "cv", "achromatic-cv-v0.json"),
]


def _thin(xs, ys):
    step = max(1, -(-len(xs) // MAX_POINTS))
    return [round(v, 6) for v in xs[::step]], [round(v, 6) for v in ys[::step]]


def _llm_answers(
    task_by_fid: dict,
    models,
    condition: str = "calibrated",
    run: pathlib.Path = REPO / "data/llm_run_v2",
) -> dict[str, dict[str, list]]:
    """model -> figure_id ("paper-fig") -> list[Curve], scorer-identical.

    Only the calibrated condition is rescaled, exactly as score_llm_predictions
    does for v2 (the noaxis models read the printed units themselves)."""
    key = json.loads((run / "_key.json").read_text())
    out = {m: {} for m in models}
    for m in models:
        raw = {}
        for f in sorted((run / condition / m).glob("part*.predictions.json")):
            raw |= json.loads(f.read_text())
        for task_id, answer in raw.items():
            k = key[task_id]
            fid = f"{k['paper_id']}-{k['figure_id']}"
            if fid not in task_by_fid:
                continue  # excluded from scoring since the run
            factors = (
                PREDICTION_RESCALE_V2_CALIBRATED.get(k["figure_id"])
                if condition == "calibrated"
                else None
            )
            if factors:
                answer = [
                    {
                        "x": [v * factors.get("x", 1.0) for v in c.get("x", [])],
                        "y": [v * factors.get("y", 1.0) for v in c.get("y", [])],
                    }
                    for c in answer
                ]
            out[m][fid] = parse_curves(answer, task_by_fid[fid].x_scale)
    return out


def _axis_overlay(entry: dict | None, pairing) -> dict | None:
    """Tick-mark pixel positions, usable only if they still describe this
    image and these ranges (unit migration / re-crops can invalidate them)."""
    if entry is None or entry.get("pixel_coords_stale_since"):
        return None
    labels = tuple(entry[k] for k in ("x_min_label", "x_max_label", "y_min_label", "y_max_label"))
    ranges = (*pairing.x_range, *pairing.y_range)
    if None in labels or not entry.get("pixel_bbox_mean"):
        return None
    if any(abs(a - b) > 1e-9 * max(1.0, abs(b)) for a, b in zip(labels, ranges, strict=True)):
        return None
    if entry["image_path"] != pairing.image_path:
        return None
    return entry["pixel_bbox_mean"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "data/cache/results_explorer")
    parser.add_argument(
        "--extra-llm-archive",
        type=pathlib.Path,
        help="LLM run outside the repo, laid out like data/llm_run_v2 (shown as local runs)",
    )
    parser.add_argument(
        "--extra-results",
        type=pathlib.Path,
        help="scored payloads for --extra-llm-archive (<model>-v0-r2[-noaxis].json)",
    )
    args = parser.parse_args()
    if (args.extra_llm_archive is None) != (args.extra_results is None):
        parser.error("--extra-llm-archive and --extra-results go together")
    extra = _extra_models(args.extra_llm_archive, args.extra_results)
    models = MODELS + [(m, n, "llm", f) for m, n, f in extra]
    local_ids = {m for m, _, _ in extra}
    results_dir = {mid: REPO / "results" for mid, _, _, _ in MODELS} | {
        m: args.extra_results for m in local_ids
    }
    (args.out / "images").mkdir(parents=True, exist_ok=True)

    items, _ = build_dataset()
    items = [i for i in items if not i.figure_id.startswith("synthetic-")]
    task_by_fid = {i.figure_id: i.task for i in items}
    registry = {
        f"{p.paper_id}-{p.figure_id}": p
        for p in load_registry(REPO / "data/verified_pairs/registry.json")
    }
    axis = {
        f"{e['paper_id']}-{e['figure_id']}": e
        for e in json.loads((REPO / "data/verified_pairs/axis_pixel_candidates.json").read_text())
        if "paper_id" in e
    }

    lf_preds = [
        LineFormerPrediction.from_record(json.loads(line))
        for line in (REPO / _lineformer_raw_predictions()).open()
        if line.strip()
    ]
    lf_raw = {p.image_key: p for p in lf_preds}
    runners = {
        "lineformer": PrecomputedLineFormerModelRunner(lf_preds),
        "naive-cv": NaiveCvModelRunner(),
        "achromatic-cv": AchromaticCvModelRunner(),
    }
    llm = _llm_answers(task_by_fid, LLM_MODELS)
    llm_noaxis = _llm_answers(task_by_fid, LLM_MODELS, "noaxis")
    if extra:
        ids = [m for m, _, _ in extra]
        llm |= _llm_answers(task_by_fid, ids, "calibrated", args.extra_llm_archive)
        llm_noaxis |= _llm_answers(task_by_fid, ids, "noaxis", args.extra_llm_archive)
    published = {
        mid: {p["figure_id"]: p for p in _per_figure(results_dir[mid] / f)}
        for mid, _, _, f in models
    }
    # noaxis: LLMs only -- the other extractors are always handed the ranges
    noaxis_models = [(mid, name, f) for mid, name, kind, f in models if kind == "llm"]
    published_noaxis = {
        mid: {p["figure_id"]: p for p in _per_figure(results_dir[mid] / _noaxis_file(f))}
        for mid, _, f in noaxis_models
    }

    figures, mismatches = [], []
    for item in items:
        fid, task = item.figure_id, item.task
        pairing = registry[fid]
        ext = "png" if task.image_bytes[:4] == b"\x89PNG" else "jpg"
        (args.out / "images" / f"{fid}.{ext}").write_bytes(task.image_bytes)

        gt = item.ground_truth
        fig = {
            "id": fid,
            "image": f"images/{fid}.{ext}",
            "x_range": list(task.x_range),
            "y_range": list(task.y_range),
            "x_scale": task.x_scale.value,
            "y_scale": task.y_scale.value,
            "figure_kind": getattr(getattr(pairing, "figure_kind", None), "value", None),
            "axis_px": _axis_overlay(axis.get(fid), pairing),
            "gt": [
                {
                    **dict(zip(("x", "y"), _thin(c.x_values, c.y_values), strict=True)),
                    "label": c.series_label,
                }
                for c in gt
            ],
            # every ground-truth point, in each Curve's own (x-sorted) order --
            # the indices the point evaluation's statuses and pairs refer to
            "gt_points": [_points(c) for c in gt],
            "models": {},
            "noaxis": {},
        }
        lf = lf_raw[image_key(task.image_bytes)]
        fig["image_size"] = [lf.width, lf.height]
        fig["lineformer_px"] = [
            [[round(x, 1), round(y, 1)] for x, y in s[:: max(1, -(-len(s) // MAX_POINTS))]]
            for s in lf.series
        ]

        frame = axis_frame_for_task(task)
        for cond, cond_models, answers, pubs in (
            ("calibrated", [(m, k) for m, _, k, _ in models], llm, published),
            ("noaxis", [(m, "llm") for m, _, _ in noaxis_models], llm_noaxis, published_noaxis),
        ):
            out = fig["models"] if cond == "calibrated" else fig["noaxis"]
            for mid, kind in cond_models:
                pub = pubs[mid].get(fid)
                try:
                    pred = answers[mid].get(fid) if kind == "llm" else runners[mid].extract(task)
                except Exception as exc:  # noqa: BLE001 -- shown on the page, same as the scorer
                    pred, err = None, f"{type(exc).__name__}: {exc}"
                else:
                    err = None if pred is not None else "no answer"
                # design 7.67: the primary metric, checked like summary_score;
                # no answer is scored as nothing predicted, as the scorer does
                pt = evaluate_points(pred or [], gt, frame, PRIMARY_POINT_TAU, POINT_NORM)
                pub_pt = _primary_point(pub)
                if pub and (pub_pt is None or abs(pt.point_f1 - pub_pt["point_f1"]) > 1e-9):
                    mismatches.append(
                        (cond, mid, fid, "point_f1", pt.point_f1, pub_pt and pub_pt["point_f1"])
                    )
                if pred is None:
                    out[mid] = {
                        "curves": [],
                        "error": err,
                        **_scores(pub),
                        "point": _point_detail(pt, [], gt),
                    }
                    continue
                entry, ev = _package(pred, gt, matcher_for_task(task))
                if pub and abs(ev.summary_score - pub["summary_score"]) > 1e-6:
                    mismatches.append((cond, mid, fid, ev.summary_score, pub["summary_score"]))
                out[mid] = {**entry, **_scores(pub), "point": _point_detail(pt, pred, gt)}

        # Exploratory, not a leaderboard row: LineFormer's same raw pixels,
        # mapped through the tick-mark positions instead of the full frame.
        # Separates "found the series" from "placed the axis" (design §7.64).
        if fig["axis_px"] is not None:
            a = fig["axis_px"]
            cal = PixelCalibration(
                pixel_bbox=(a["x_min_px"], a["y_max_px"], a["x_max_px"], a["y_min_px"]),
                x_range=task.x_range,
                y_range=task.y_range,
                x_scale=task.x_scale,
                y_scale=task.y_scale,
            )
            pred = []
            for s in lf.series:
                if s:
                    pts = [cal.to_data(x, y) for x, y in s]
                    pred.append(
                        Curve(
                            x_values=tuple(p[0] for p in pts),
                            y_values=tuple(p[1] for p in pts),
                            x_scale=task.x_scale,
                        )
                    )
            entry, ev = _package(pred, gt, matcher_for_task(task))
            pt = evaluate_points(pred, gt, frame, PRIMARY_POINT_TAU, POINT_NORM)
            fig["models"]["lineformer-axis"] = {
                "point": _point_detail(pt, pred, gt),
                **entry,
                "score": ev.summary_score,
                "point_f1": pt.point_f1,
                "point_recall": pt.point_recall,
                "point_precision": pt.point_precision,
                "match_rate": ev.match_rate,
                "dist": ev.mean_curve_distance,
                "cov": ev.mean_coverage_ratio,
            }
        figures.append(fig)

    if mismatches:
        for m in mismatches[:10]:
            print("score mismatch", m, file=sys.stderr)
        raise SystemExit(f"{len(mismatches)} recomputed score(s) differ from results/*.json")

    summary = []
    for mid, name, kind, f in models:
        res = json.loads((results_dir[mid] / f).read_text())
        rows = [published[mid][fig["id"]] for fig in figures if fig["id"] in published[mid]]
        summary.append(
            {
                "id": mid,
                "name": name,
                "kind": kind,
                "local": mid in local_ids,
                "mean": res["mean_summary_score"],
                **_primary_macro(res),
                "n": len(rows),
                **{
                    k: sum(r[k] for r in rows) / len(rows)
                    for k in ("match_rate", "mean_curve_distance", "mean_coverage_ratio")
                },
            }
        )
    axis_figs = [f for f in figures if "lineformer-axis" in f["models"]]
    summary.append(
        {
            "id": "lineformer-axis",
            "name": "LineFormer (tick-calibrated, exploratory)",
            "kind": "exploratory",
            "local": False,
            "mean": sum(f["models"]["lineformer-axis"]["score"] for f in axis_figs)
            / len(axis_figs),
            **{
                k: sum(f["models"]["lineformer-axis"][k] for f in axis_figs) / len(axis_figs)
                for k in ("point_f1", "point_recall", "point_precision")
            },
            "n": len(axis_figs),
            **{
                k: sum(f["models"]["lineformer-axis"][v] for f in axis_figs) / len(axis_figs)
                for k, v in (
                    ("match_rate", "match_rate"),
                    ("mean_curve_distance", "dist"),
                    ("mean_coverage_ratio", "cov"),
                )
            },
        }
    )
    dataset_version = json.loads((REPO / "results/naive-cv-v0.json").read_text())["dataset_version"]
    summary_noaxis = []
    for mid, name, f in noaxis_models:
        res = json.loads((results_dir[mid] / _noaxis_file(f)).read_text())
        summary_noaxis.append(
            {
                "id": mid,
                "name": name,
                "kind": "llm",
                "local": mid in local_ids,
                "mean": res["mean_summary_score"],
                **_primary_macro(res),
                "n": res["n_figures"],
            }
        )
    payload = {
        "dataset_version": dataset_version,
        "point_tau": PRIMARY_POINT_TAU,
        "point_norm": POINT_NORM,
        # kept for the calibrated view; "conditions" has both leaderboards
        "models": summary,
        "conditions": {
            "calibrated": {"label": "軸レンジあり", "models": summary},
            "noaxis": {"label": "軸レンジなし", "models": summary_noaxis},
        },
        "figures": figures,
    }
    (args.out / "data.json").write_text(json.dumps(payload, separators=(",", ":")))
    size = (args.out / "data.json").stat().st_size / 1e6
    n_axis = sum(f["axis_px"] is not None for f in figures)
    print(
        f"wrote {args.out}/data.json ({size:.1f} MB): {len(figures)} figures, "
        f"{n_axis} with usable axis pixel positions; all scores (summary_score and "
        f"point_f1, both conditions) match the results files; "
        f"{len(local_ids)} local model(s) added"
    )


def _lineformer_raw_predictions() -> str:
    lf_file = next(f for mid, _, _, f in MODELS if mid == "lineformer")
    return json.loads((REPO / "results" / lf_file).read_text())["raw_predictions"]


def _per_figure(results_file: pathlib.Path) -> list[dict]:
    return json.loads(results_file.read_text())["per_figure"]


def _noaxis_file(calibrated_file: str) -> str:
    return calibrated_file.removesuffix(".json") + "-noaxis.json"


def _extra_models(archive: pathlib.Path | None, results: pathlib.Path | None):
    """(id, display name, results file) for each model in an extra LLM archive."""
    if archive is None:
        return []
    out = []
    for d in sorted(p for p in (archive / "calibrated").iterdir() if p.is_dir()):
        f = f"{d.name}-v0-r2.json"
        if not (results / f).exists() or not (results / _noaxis_file(f)).exists():
            raise SystemExit(f"{d.name}: no scored payload {f} (and -noaxis) in {results}")
        name = json.loads((results / f).read_text())["model_name"]
        out.append((d.name, name.split("（")[0].strip(), f))
    return out


def _sig(v: float) -> float | None:
    return float(f"{v:.7g}") if math.isfinite(v) else None


def _points(c: Curve) -> list[list[float | None]]:
    return [[_sig(x), _sig(y)] for x, y in zip(c.x_values, c.y_values, strict=True)]


def _point_detail(pt, pred, gt) -> dict:
    """One model's point evaluation on one figure, for drawing.

    pred: every predicted point (each Curve's x-sorted order); pred_ok / gt_ok:
    1 = matched / found, 0 = extra / missed; pairs: [pred curve, pred point,
    gt curve, gt point]; series: [pred curve or None, gt curve or None, n]."""
    p_ix = {id(c): i for i, c in enumerate(pred)}
    g_ix = {id(c): j for j, c in enumerate(gt)}
    pred_ok = [[0] * len(c) for c in pred]
    gt_ok = [[0] * len(c) for c in gt]
    pairs, series = [], []
    for s in pt.series:
        i = p_ix[id(s.predicted)] if s.predicted is not None else None
        j = g_ix[id(s.ground_truth)] if s.ground_truth is not None else None
        series.append([i, j, s.n_matched])
        for pi, gi in s.matched_pairs:
            pred_ok[i][pi] = gt_ok[j][gi] = 1
            pairs.append([i, pi, j, gi])
    return {
        "f1": pt.point_f1,
        "recall": pt.point_recall,
        "precision": pt.point_precision,
        "n_gt": pt.n_ground_truth,
        "n_pred": pt.n_predicted,
        "n_matched": pt.n_matched,
        "series": series,
        "pred": [_points(c) for c in pred],
        "pred_ok": pred_ok,
        "gt_ok": gt_ok,
        "pairs": pairs,
    }


def _package(pred, gt, matcher):
    ev = evaluate_figure(pred, gt, matcher)
    match_of = {}
    for m in ev.matches:
        if m.predicted is not None and m.ground_truth is not None:
            match_of[id(m.predicted)] = (
                next(i for i, g in enumerate(gt) if g is m.ground_truth),
                m.comparison.distance,
                m.comparison.coverage_ratio,
            )
    curves = []
    for c in pred:
        xs, ys = _thin(c.x_values, c.y_values)
        gi, dist, cov = match_of.get(id(c), (None, None, None))
        curves.append({"x": xs, "y": ys, "gt": gi, "dist": dist, "cov": cov})
    return {"curves": curves, "error": None}, ev


def _primary_point(pub: dict | None) -> dict | None:
    """A published per_figure row's point metrics at the primary tau."""
    if pub is None or "point" not in pub:
        return None
    return pub["point"]["by_tau"].get(f"{PRIMARY_POINT_TAU:g}")


def _primary_macro(res: dict) -> dict:
    block = res.get("point_metrics")
    if not block:
        return {}
    macro = block["by_tau"][f"{block['primary_tau']:g}"]["macro"]
    return {k: macro[k] for k in ("point_f1", "point_recall", "point_precision")}


def _scores(pub: dict | None) -> dict:
    if pub is None:
        return {"score": None}
    pt = _primary_point(pub) or {}
    return {
        "point_f1": pt.get("point_f1"),
        "point_recall": pt.get("point_recall"),
        "point_precision": pt.get("point_precision"),
        "score": pub["summary_score"],
        "match_rate": pub["match_rate"],
        "dist": pub["mean_curve_distance"],
        "cov": pub["mean_coverage_ratio"],
    }


if __name__ == "__main__":
    main()

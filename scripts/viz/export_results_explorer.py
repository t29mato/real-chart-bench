"""Exports every model's extracted curves, per figure, for the results explorer
page: ground truth, each model's curves in data space, which ground-truth
series each predicted curve was matched to, and the per-figure scores.

Curves are rebuilt from what each model actually produced -- LLM answers from
data/llm_run_v3/ (the current Claude rows, design §7.73 (2)) and
data/llm_run_v2/ (kept as history, with the same unit rescale the scorer
applies), each read from exactly the official parts (part1, part2) as the
scorer reads them, LineFormer from its raw pixel output, the CV baselines by re-running
them -- and re-scored here. Each recomputed score is checked against the
published results/*.json, so the page cannot show curves that are not the
ones that were scored.

Every model x figure also carries its point evaluation at the primary tau
(design §7.67): which ground-truth points were found or missed, which
predicted points matched or were extra, and the matched pairs -- for both the
calibrated condition (axis ranges given) and, for the LLMs, noaxis.

Every figure carries its marker density (design §7.72): the median
within-series nearest-neighbour spacing of its ground-truth points and whether
it is dense (< 2 tau). Dense figures are left out of the point leaderboard and
ranked on summary_score in a second one; their point view is reference only.
Both the per-figure density and both leaderboards' means are checked against
the results files.

The local VLM runs (design §7.69 / §7.73 (3)) are read from
data/local_vlm_run_v3/ (current, v3 prompt) and data/local_vlm_run_v2/
(history) -- the parsed answers in each jsonl, as the scorer reads them -- and
verified against results/<model>-v0-local-v3[-noaxis].json and
-local-v2[-noaxis].json the same way. Whether a row ran locally comes from the
results file's `execution`.

Output: <out>/data.json and <out>/images/<figure_id>.<ext>
Usage: python scripts/viz/export_results_explorer.py [--out DIR]
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
from score_llm_predictions import (  # noqa: E402
    LOCAL_V2_ARCHIVE,
    LOCAL_V3_ARCHIVE,
    MODELS_LOCAL_V2,
    PREDICTION_RESCALE_V2_CALIBRATED,
    V2_ARCHIVE,
    V3_ARCHIVE,
    parse_curves,
)

from real_chart_bench.adapter.achromatic_cv_extractor import AchromaticCvModelRunner  # noqa: E402
from real_chart_bench.adapter.agent_run_archive import (  # noqa: E402
    OFFICIAL_PARTS,
    load_agent_run_parts,
)
from real_chart_bench.adapter.lineformer_model_runner import (  # noqa: E402
    LineFormerPrediction,
    PrecomputedLineFormerModelRunner,
    image_key,
)
from real_chart_bench.adapter.local_vlm_run import load_local_vlm_run  # noqa: E402
from real_chart_bench.adapter.naive_cv_extractor import NaiveCvModelRunner  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.curve import Curve, curves_for_curve_scoring  # noqa: E402
from real_chart_bench.domain.evaluation import evaluate_figure  # noqa: E402
from real_chart_bench.domain.pixel_calibration import PixelCalibration  # noqa: E402
from real_chart_bench.domain.point_metrics import (  # noqa: E402
    DENSE_SPACING_TAU_FACTOR,
    evaluate_points,
)
from real_chart_bench.usecase.evaluate_dataset import (  # noqa: E402
    POINT_NORM,
    PRIMARY_POINT_TAU,
    axis_frame_for_task,
    marker_density_for,
    matcher_for_task,
)

MAX_POINTS = 400  # per curve, for display only; scoring uses every point

_CLAUDE = {
    "claude-fable-5-1": "Claude Fable 5.1",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}
# the 2026-10-04 v3 run (markers-only prompt, design 7.73 (2)): the current
# Claude rows. Explorer ids get a "-v3" suffix so they do not collide with v2.
LLM_MODELS_V3 = {f"{m}-v3": f"{n}（v3 プロンプト）" for m, n in _CLAUDE.items()}
# the 2026-10-01 run (v2 prompt) -- design 7.66; kept as history
LLM_MODELS = {m: f"{n}（v2 プロンプト）" for m, n in _CLAUDE.items()}
MODELS = [
    # (id, name, kind, published results file)
    *[(m, n, "llm", f"{m.removesuffix('-v3')}-v0-r3.json") for m, n in LLM_MODELS_V3.items()],
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
    # local VLMs on the v3 tasks (design 7.73 (2)/(3)): the current local rows
    *[
        (f"{m}-v3", f"{n}（v3 プロンプト・単発）", "llm", f"{m}-v0-local-v3.json")
        for m, n in MODELS_LOCAL_V2.items()
    ],
    # local VLMs, same tasks as the 2026-10-01 run -- kept as history
    *[
        (m, f"{n}（v2 プロンプト・単発）", "llm", f"{m}-v0-local-v2.json")
        for m, n in MODELS_LOCAL_V2.items()
    ],
]


def _thin(xs, ys):
    step = max(1, -(-len(xs) // MAX_POINTS))
    return [round(v, 6) for v in xs[::step]], [round(v, 6) for v in ys[::step]]


def _llm_answers(
    task_by_fid: dict,
    models,
    condition: str = "calibrated",
    local: bool = False,
    run: pathlib.Path = V2_ARCHIVE,
    local_archive: pathlib.Path = LOCAL_V2_ARCHIVE,
) -> dict[str, dict[str, list]]:
    """model -> figure_id ("paper-fig") -> list[Curve], scorer-identical.

    Only the v2 calibrated condition is rescaled, exactly as
    score_llm_predictions does (the noaxis models read the printed units
    themselves; the v3 tasks were built from today's registry). Local VLMs
    answered the same tasks (v2, or v3 with ``run=V3_ARCHIVE``); their answers
    are the parsed series in <local_archive>/<model>/<condition>.jsonl. A model
    id may carry a "-v3" suffix (explorer-only); the archive dir is the bare
    model id."""
    key = json.loads((run / "_key.json").read_text())
    rescale = PREDICTION_RESCALE_V2_CALIBRATED if run == V2_ARCHIVE else {}
    out = {m: {} for m in models}
    for m in models:
        if local:
            raw = load_local_vlm_run(
                local_archive / m.removesuffix("-v3") / f"{condition}.jsonl"
            ).answers
        else:
            raw = load_agent_run_parts(
                run / condition / m.removesuffix("-v3"), OFFICIAL_PARTS
            )
        for task_id, answer in raw.items():
            k = key[task_id]
            fid = f"{k['paper_id']}-{k['figure_id']}"
            if fid not in task_by_fid:
                continue  # excluded from scoring since the run
            factors = rescale.get(k["figure_id"]) if condition == "calibrated" else None
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
    args = parser.parse_args()
    models = MODELS
    results_dir = {mid: REPO / "results" for mid, _, _, _ in MODELS}
    # where each row ran, as its results file records it (design 7.69)
    local_ids = {
        mid
        for mid, _, _, f in models
        if json.loads((results_dir[mid] / f).read_text()).get("execution") == "local"
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
    llm = _llm_answers(task_by_fid, LLM_MODELS_V3, run=V3_ARCHIVE)
    llm_noaxis = _llm_answers(task_by_fid, LLM_MODELS_V3, "noaxis", run=V3_ARCHIVE)
    llm |= _llm_answers(task_by_fid, LLM_MODELS)
    llm_noaxis |= _llm_answers(task_by_fid, LLM_MODELS, "noaxis")
    local_v3 = {f"{m}-v3": n for m, n in MODELS_LOCAL_V2.items()}
    for cond, answers in (("calibrated", llm), ("noaxis", llm_noaxis)):
        answers |= _llm_answers(
            task_by_fid, local_v3, cond, local=True, run=V3_ARCHIVE, local_archive=LOCAL_V3_ARCHIVE
        )
    llm |= _llm_answers(task_by_fid, MODELS_LOCAL_V2, "calibrated", local=True)
    llm_noaxis |= _llm_answers(task_by_fid, MODELS_LOCAL_V2, "noaxis", local=True)
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
        density = marker_density_for(item, PRIMARY_POINT_TAU)
        spacing = _sig(density.median_nn_spacing)
        fig = {
            "id": fid,
            "image": f"images/{fid}.{ext}",
            "x_range": list(task.x_range),
            "y_range": list(task.y_range),
            "x_scale": task.x_scale.value,
            "y_scale": task.y_scale.value,
            "figure_kind": getattr(getattr(pairing, "figure_kind", None), "value", None),
            "axis_px": _axis_overlay(axis.get(fid), pairing),
            # design 7.72: dense figures are not point-scored
            "marker_density": {"median_nn_spacing": spacing, "dense": density.dense},
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
                pub_density = (pub or {}).get("marker_density")
                if pub and (
                    pub_density is None
                    or pub_density["dense"] != density.dense
                    or not _close(pub_density["median_nn_spacing"], density.median_nn_spacing)
                ):
                    mismatches.append((cond, mid, fid, "marker_density", density, pub_density))
                try:
                    pred = answers[mid].get(fid) if kind == "llm" else runners[mid].extract(task)
                except Exception as exc:  # noqa: BLE001 -- shown on the page, same as the scorer
                    pred, err = None, f"{type(exc).__name__}: {exc}"
                else:
                    err = None if pred is not None else "no answer"
                if pred is not None:
                    # evaluate_model_on_dataset scores a figure whose curve
                    # comparison raises (e.g. x <= 0 on a log axis) as an
                    # empty answer on every metric; mirror that here.
                    # Non-finite points no longer raise: curve scoring
                    # ignores them (owner decision 2026-10-05)
                    try:
                        entry, ev = _package(pred, gt, matcher_for_task(task))
                    except Exception as exc:  # noqa: BLE001
                        pred, err = None, f"{type(exc).__name__}: {exc}"
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
        row = {
            "id": mid,
            "name": name,
            "kind": kind,
            "local": mid in local_ids,
            "mean": res["mean_summary_score"],
            **_primary_macro(res),
            **_dense_block(res),
            "n": len(rows),
            **{
                k: sum(r[k] for r in rows) / len(rows)
                for k in ("match_rate", "mean_curve_distance", "mean_coverage_ratio")
            },
        }
        mismatches += _check_board(row, figures, "models", mid)
        summary.append(row)
    axis_figs = [f for f in figures if "lineformer-axis" in f["models"]]
    axis_point = [f for f in axis_figs if not f["marker_density"]["dense"]]
    axis_dense = [f for f in axis_figs if f["marker_density"]["dense"]]

    def axis_mean(figs, k):
        return sum(f["models"]["lineformer-axis"][k] for f in figs) / len(figs) if figs else None

    summary.append(
        {
            "id": "lineformer-axis",
            "name": "LineFormer (tick-calibrated, exploratory)",
            "kind": "exploratory",
            "local": False,
            "mean": axis_mean(axis_figs, "score"),
            **{
                k: axis_mean(axis_point, k)
                for k in ("point_f1", "point_recall", "point_precision")
            },
            "n_point": len(axis_point),
            "dense": {
                "n": len(axis_dense),
                "mean": axis_mean(axis_dense, "score"),
                "match_rate": axis_mean(axis_dense, "match_rate"),
                "mean_curve_distance": axis_mean(axis_dense, "dist"),
                "mean_coverage_ratio": axis_mean(axis_dense, "cov"),
            },
            "n": len(axis_figs),
            **{
                k: axis_mean(axis_figs, v)
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
        row = {
            "id": mid,
            "name": name,
            "kind": "llm",
            "local": mid in local_ids,
            "mean": res["mean_summary_score"],
            **_primary_macro(res),
            **_dense_block(res),
            "n": res["n_figures"],
        }
        mismatches += _check_board(row, figures, "noaxis", mid)
        summary_noaxis.append(row)
    if mismatches:
        for m in mismatches[:10]:
            print("leaderboard mismatch", m, file=sys.stderr)
        raise SystemExit(f"{len(mismatches)} leaderboard mean(s) differ from results/*.json")
    payload = {
        "dataset_version": dataset_version,
        "point_tau": PRIMARY_POINT_TAU,
        "point_norm": POINT_NORM,
        "dense_spacing": DENSE_SPACING_TAU_FACTOR * PRIMARY_POINT_TAU,
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
        f"{n_axis} with usable axis pixel positions, "
        f"{sum(f['marker_density']['dense'] for f in figures)} dense-marker; all scores "
        f"(summary_score, point_f1, marker density and both leaderboards' means, both "
        f"conditions) match the results files; "
        f"{len(local_ids)} of them local"
    )


def _lineformer_raw_predictions() -> str:
    lf_file = next(f for mid, _, _, f in MODELS if mid == "lineformer")
    return json.loads((REPO / "results" / lf_file).read_text())["raw_predictions"]


def _per_figure(results_file: pathlib.Path) -> list[dict]:
    return json.loads(results_file.read_text())["per_figure"]


def _noaxis_file(calibrated_file: str) -> str:
    return calibrated_file.removesuffix(".json") + "-noaxis.json"


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
    # curve scoring sees each answer without its non-finite points, and not at
    # all when fewer than two finite points are left (owner decision
    # 2026-10-05); the page draws what was scored
    scored_of = {}
    for c in pred:
        kept = curves_for_curve_scoring([c])
        scored_of[id(c)] = kept[0] if kept else None
    ev = evaluate_figure([s for s in scored_of.values() if s is not None], gt, matcher)
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
        s = scored_of[id(c)]
        xs, ys = _thin(s.x_values, s.y_values) if s is not None else ([], [])
        gi, dist, cov = match_of.get(id(s), (None, None, None))
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


def _close(a: float | None, b: float) -> bool:
    """A stored spacing (None for inf) against a recomputed one."""
    if a is None:
        return not math.isfinite(b)
    return math.isfinite(b) and abs(a - b) <= 1e-9 * max(1.0, abs(b))


def _dense_block(res: dict) -> dict:
    """design 7.72: the point table's figure count and the dense-marker block."""
    out = {"n_point": (res.get("point_metrics") or {}).get("n_figures")}
    block = res.get("dense_marker_metrics")
    if block:
        out["dense"] = {
            "n": block["n_figures"],
            "mean": block["mean_summary_score"],
            "match_rate": block["mean_match_rate"],
            "mean_curve_distance": block["mean_curve_distance"],
            "mean_coverage_ratio": block["mean_coverage_ratio"],
        }
    return out


def _check_board(row: dict, figures: list, key: str, mid: str) -> list:
    """Both leaderboards' means, recomputed from the per-figure entries the page
    shows, against the published blocks (design 7.72)."""
    point = [f[key][mid]["point"]["f1"] for f in figures
             if mid in f[key] and not f["marker_density"]["dense"]]
    dense = [f[key][mid]["score"] for f in figures
             if mid in f[key] and f["marker_density"]["dense"]]
    out = []
    if len(point) != row.get("n_point") or (
        point and abs(sum(point) / len(point) - row["point_f1"]) > 1e-9
    ):
        out.append((key, mid, "point table", row.get("n_point"), len(point)))
    d = row.get("dense") or {}
    if len(dense) != d.get("n") or (dense and abs(sum(dense) / len(dense) - d["mean"]) > 1e-9):
        out.append((key, mid, "dense table", d.get("n"), len(dense)))
    return out


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

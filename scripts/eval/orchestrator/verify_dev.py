"""Do verify's signals separate good from bad extractions? Measured on
development data only (docs/design/local-model.md 方式D「検証とやり直し」):
the detector's validation split (synthetic: PlotQA + the materials generator,
400 images), the stress validation set (300) and the Starrydata validation
papers (13 real figures, papers outside the benchmark). Nothing here reads
the benchmark.

  .venv/bin/python scripts/eval/orchestrator/verify_dev.py run      # CPU, ~10 min
  .venv/bin/python scripts/eval/orchestrator/verify_dev.py analyze  # tables, thresholds

run: for every figure, a fixed set of tool extractions (the detector at its
default and other settings, and a colour pipeline: dominant colours ->
symbol_extract), each scored against the label (pooled pixel point F1,
radius 2% of the plot box's long side, as dev_score.py) and checked by
verify (signals only from the image). Condition 1: the automatic
calibration, its calibration signals, and whether it agrees with the label's
ticks (1% of the tick span, as 方式C's success rate).

analyze: AUC / rank correlation of each signal against "needs redo"
(F1 < 0.8), thresholds chosen on val_synth only, then checked on stress and
val_real; and the redo loop simulated with a fixed ladder of tools.

The detector's raw detections for these images come from the 方式A cache
(scripts/tools/export_detector_cache.py --set val_synth|val_real|stress ->
~/.cache/real-chart-bench/orchestrator/dev-dets).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.orchestrator_tools import ToolBox, ToolError  # noqa: E402
from real_chart_bench.domain.digitizer_tools import fit_ticks  # noqa: E402
from real_chart_bench.domain.marker_detection import pixel_point_f1  # noqa: E402
from real_chart_bench.domain.orchestration import MAX_RETRIES, retry_decision  # noqa: E402
from real_chart_bench.domain.verification import (  # noqa: E402
    Thresholds,
    calibration_reasons,
    calibration_signals,
    image_score,
    score,
    verdict,
    verify,
)

CACHE = Path.home() / ".cache/real-chart-bench"
PKL = CACHE / "detector/cache/real-b1-cpu.pkl"
DETS = CACHE / "orchestrator/dev-dets"
OUT = CACHE / "orchestrator/verify-dev"
REPORT = REPO / "data/local_orchestrator_runs/verify_dev_report.json"
SETS = ("val_synth", "stress", "val_real")
BAD_F1 = 0.8

# the candidates: the tool calls an orchestrator would try (fixed in advance)
DETECTOR = {
    "det_default": {},
    "det_t025": {"threshold": 0.25},
    "det_t060": {"threshold": 0.6},
    "det_ls1280": {"long_side": 1280},
    "det_ls1024_t030": {"long_side": 1024, "threshold": 0.3},
}
# the redo ladder: what to try next when verify says redo (fixed in advance)
LADDER = ["det_default", "det_frame", "det_ls1280", "det_t025", "colors", "det_ls1024_t030",
          "det_t060"]


def _resolver(store):
    def resolve(ref):
        if ref not in store:
            raise ToolError(f"no result {ref}")
        return store[ref]
    return resolve


def colour_pipeline(tb: ToolBox, frame, store) -> dict:
    """dominant colours inside the frame -> symbol_extract per colour."""
    resolve = _resolver(store)
    mask = {"frame": frame, "frame_margin": 3} if frame else None
    cols = tb.run("dominant_colors", {"k": 8, **({"mask": mask} if mask else {})},
                  resolve)["colors"]
    chosen = [c for c in cols if not c["achromatic"] and c["fraction"] >= 0.01]
    if not chosen:
        chosen = [c for c in cols if c["achromatic"]][:1]
    short = min(frame[2] - frame[0], frame[3] - frame[1]) if frame else min(tb.size)
    series = []
    for c in chosen[:6]:
        r = tb.run("symbol_extract", {"color": c["color"], "distance_pct": 5,
                                      "min_diameter_px": max(3, 0.008 * short),
                                      "max_diameter_px": max(8, 0.08 * short),
                                      **({"mask": mask} if mask else {})}, resolve)
        series += [s for s in r["series"] if len(s["x"]) >= 2]
    return {"kind": "points", "series": series, "n_series": len(series),
            "n_points": sum(len(s["x"]) for s in series)}


def summarize(sig: dict) -> dict:
    ser = [s for s in sig["series"] if s["n"]]
    n = sum(s["n"] for s in ser) or 1
    con = [s for s in ser if s["contrast"] is not None]
    return {
        "n_points": sig["n_points"],
        "n_series": len(ser),
        "min_contrast": min((s["contrast"] for s in con), default=None),
        "mean_contrast": (sum(s["contrast"] * s["n"] for s in con) / sum(s["n"] for s in con)
                          if con else None),
        "mean_hit": (sum(s["hit"] * s["n"] for s in con) / sum(s["n"] for s in con)
                     if con else None),
        "det_support": (sig.get("detector") or {}).get("support"),
        "det_coverage": (sig.get("detector") or {}).get("coverage"),
        "det_f1": (sig.get("detector") or {}).get("f1"),
        "min_self_match": min((s["self_match"] for s in ser if s.get("self_match") is not None),
                              default=None),
        "missed_share": sig["missed"]["share"],
        "lookalike_share": sig["lookalikes"]["n"] / (sig["lookalikes"]["n"] + sig["n_points"])
        if sig["n_points"] else 0.0,
        "unexplained_share": sig["unexplained"]["share"],
        "unexplained_same": sig["unexplained"]["same_color"] / (
            sig["unexplained"]["same_color"] + sig["n_points"]) if sig["n_points"] else 0.0,
        "dup_share": sum(s["duplicates"] for s in ser) / n,
        "cross_dup_share": sig["cross_duplicates"] / n,
        "outside_share": sum(s["outside"] for s in ser) / n,
        "color_consistency": (sum((s["color_consistency"] or 0) * s["n"] for s in ser) / n),
        "score": score(sig),
    }


def label_agreement(auto: dict, label_axis: dict) -> float:
    """|auto - label| at the label's outermost ticks, as a share of the label's
    value span (decades on a log axis), with the label's least-squares line
    (Starrydata's label ticks are OCR centres: one tick alone is noisy).
    inf when the scales differ."""
    if auto["scale"] != label_axis["scale"]:
        return math.inf
    ticks = [[t["px"], t["value"]] for t in label_axis["ticks"]]
    try:
        ls, li = fit_ticks(ticks, label_axis["scale"])
        a_s, a_i = (float(v) for v in auto["fit"]) if auto.get("fit") else fit_ticks(
            auto["ticks"], auto["scale"])
    except ValueError:
        return math.inf
    pxs = [p for p, _ in ticks]
    lo, hi = min(pxs), max(pxs)
    tl = [(p - li) / ls for p in (lo, hi)]
    ta = [(p - a_i) / a_s for p in (lo, hi)]
    span = abs(tl[1] - tl[0])
    return max(abs(a - b) for a, b in zip(ta, tl, strict=True)) / span if span else math.inf


def run_figure(job: dict) -> dict:
    lab = job["label"]
    img = Path(lab["_path"])
    tb = ToolBox(img, dets_dir=DETS, allow_auto_calibration=True)
    store: dict = {}
    frame = tb.frame_of(None)
    text = tb.text_boxes()
    peaks = tb.marker_peaks()
    truth = [tuple(p) for s in lab["series"] for p in s["points_px"]]
    bx0, by0, bx1, by1 = lab["plot_bbox"]
    radius = 0.02 * max(bx1 - bx0, by1 - by0)
    rows = []
    cands = {}
    for name, params in DETECTOR.items():
        try:
            cands[name] = tb.run("marker_detector", params, _resolver(store))
        except ToolError:
            continue
    try:
        cands["colors"] = colour_pipeline(tb, frame, store)
    except ToolError:
        pass
    # an orchestrator's slip: the default answer without its largest series
    d0 = cands.get("det_default")
    if d0 and len(d0["series"]) > 1:
        big = max(range(len(d0["series"])), key=lambda i: len(d0["series"][i]["x"]))
        cands["det_drop1"] = {**d0, "series": [s for i, s in enumerate(d0["series"])
                                               if i != big]}
    if frame:
        cands["det_frame"] = tb.run("marker_detector", {"mask": {"frame": frame,
                                                                 "frame_margin": -3}},
                                    _resolver(store))
    for name, res in cands.items():
        pred = [(x, y) for s in res["series"] for x, y in zip(s["x"], s["y"], strict=True)]
        f1 = pixel_point_f1(pred, truth, radius)
        sig = verify(tb.rgb, res["series"], frame, text_boxes=text, detections=peaks)
        v = verdict(sig)
        rows.append({"set": job["set"], "fig": img.name, "cand": name, "f1": f1,
                     "n_truth": len(truth), **summarize(sig), "accept_default": v["accept"],
                     "signals": sig})
    # condition 1: the automatic calibration and its self-consistency
    cal_row = None
    try:
        c = tb.run("tick_calibration", {"mode": "auto"}, _resolver(store))
    except ToolError as exc:
        c = {"ok": False, "message": str(exc)}
    if c.get("ok"):
        cal = c["calibration"]
        agree = {ax: label_agreement(cal[ax], lab["axes"][ax]) for ax in ("x", "y")}
        sig = calibration_signals(cal, c.get("tick_marks"))
        cal_row = {"set": job["set"], "fig": img.name, "agree": agree,
                   "correct": max(agree.values()) <= 0.01, "signals": sig}
    else:
        cal_row = {"set": job["set"], "fig": img.name, "agree": None, "correct": False,
                   "failed": True}
    return {"rows": rows, "cal": cal_row}


def cmd_run(workers: int) -> None:
    cache = pickle.loads(PKL.read_bytes())
    jobs = [{"set": s, "label": row["label"]} for s in SETS for row in cache["sets"][s]]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cals = [], []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        for i, out in enumerate(pool.map(run_figure, jobs, chunksize=2)):
            rows += out["rows"]
            cals.append(out["cal"])
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    (OUT / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "calibration.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cals))
    print(f"{len(rows)} candidates, {len(cals)} calibrations -> {OUT}")


# ------------------------------------------------------------------ analysis


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a positive scores above a negative), ties half."""
    if not pos or not neg:
        return float("nan")
    pairs = 0.0
    for p in pos:
        for n in neg:
            pairs += 1.0 if p > n else 0.5 if p == n else 0.0
    return pairs / (len(pos) * len(neg))


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


# signal -> direction (+1: larger means better)
SIGNALS = {"score": 1, "image_score": 1, "det_f1": 1, "det_support": 1, "det_coverage": 1,
           "min_contrast": 1, "mean_contrast": 1, "mean_hit": 1, "min_self_match": 1,
           "missed_share": -1, "lookalike_share": -1, "unexplained_share": -1,
           "unexplained_same": -1, "dup_share": -1, "cross_dup_share": -1,
           "outside_share": -1, "color_consistency": 1}


def with_scores(rows: list[dict], use_detector: bool = True) -> list[dict]:
    """Rows with the domain's current score (and image_score) recomputed
    from the stored signals; use_detector False drops the detector peaks
    (the path taken where no detection cache exists)."""
    out = []
    for r in rows:
        sig = r["signals"] if use_detector else {
            k: v for k, v in r["signals"].items() if k != "detector"}
        out.append({**r, "signals": sig, "score": score(sig), "image_score": image_score(sig)})
    return out


def signal_table(rows: list[dict]) -> list[dict]:
    out = []
    for name, sgn in SIGNALS.items():
        vals = [(r[name] if r[name] is not None else -1.0 if sgn > 0 else 2.0, r["f1"])
                for r in rows if r["n_points"]]
        good = [sgn * v for v, f in vals if f >= BAD_F1]
        bad = [sgn * v for v, f in vals if f < BAD_F1]
        out.append({"signal": name, "auc_good_vs_bad": round(auc(good, bad), 3),
                    "spearman_f1": round(sgn * spearman([v for v, _ in vals],
                                                        [f for _, f in vals]), 3)})
    return out


def _balanced(acc: list[bool], good: list[bool]) -> float:
    tp = sum(a and g for a, g in zip(acc, good, strict=True))
    tn = sum((not a) and (not g) for a, g in zip(acc, good, strict=True))
    return 0.5 * (tp / max(1, sum(good)) + tn / max(1, len(good) - sum(good)))


GRID = [round(0.05 * k, 2) for k in range(6, 20)]  # 0.30 .. 0.95


def choose_score_threshold(rows: list[dict]) -> float:
    """The accept threshold on the score: balanced accuracy of accept vs
    F1 >= BAD_F1, on val_synth only."""
    good = [r["f1"] >= BAD_F1 for r in rows]
    best = max(GRID, key=lambda t: (_balanced([r["score"] >= t for r in rows], good), -t))
    return best


def choose_cal_thresholds(cals: list[dict], base: Thresholds) -> Thresholds:
    ok = [c for c in cals if not c.get("failed")]
    good = [c["correct"] for c in ok]
    best = None
    for res in (0.005, 0.01, 0.02, 0.05, 1.0):
        for mg in (0.0, 0.5, 0.7, 0.9):
            for lm in (0.0, 0.5, 0.7, 0.9):
                for gd in (0.1, 0.2, 0.5, 1.0):
                    th = Thresholds(**{**base.as_dict(), "max_residual": res,
                                       "min_marks_on_grid": mg, "min_labels_on_marks": lm,
                                       "max_grid_dev": gd})
                    acc = [not calibration_reasons(c["signals"], th) for c in ok]
                    bal = _balanced(acc, good)
                    if best is None or bal > best[0] + 1e-9:
                        best = (bal, th)
    return best[1]


def confusion(rows, th) -> dict:
    acc = [verdict(r["signals"], th)["accept"] for r in rows]
    good = [r["f1"] >= BAD_F1 for r in rows]
    tp = sum(a and g for a, g in zip(acc, good, strict=True))
    fp = sum(a and not g for a, g in zip(acc, good, strict=True))
    fn = sum((not a) and g for a, g in zip(acc, good, strict=True))
    tn = sum((not a) and not g for a, g in zip(acc, good, strict=True))
    return {"n": len(rows), "good": sum(good), "accept_good": tp, "accept_bad": fp,
            "redo_good": fn, "redo_bad": tn,
            "mean_f1_accepted": round(sum(r["f1"] for r, a in zip(rows, acc, strict=True) if a)
                                      / max(1, sum(acc)), 3),
            "mean_f1_redo": round(sum(r["f1"] for r, a in zip(rows, acc, strict=True)
                                      if not a) / max(1, len(acc) - sum(acc)), 3)}


def _by_fig(rows):
    by: dict = {}
    for r in rows:
        by.setdefault(r["fig"], {})[r["cand"]] = r
    return by


def simulate(rows: list[dict], th: Thresholds) -> dict:
    """The redo loop of domain/orchestration.retry_decision with a fixed
    ladder of tool calls standing in for the orchestrator: the detector
    default first, then the next ladder entry while verify says redo, at
    most MAX_RETRIES redos, then the best-scoring attempt."""
    base, loop, oracle_tried, oracle_all, n_redo, tries = [], [], [], [], 0, []
    for cands in _by_fig(rows).values():
        ladder = [cands[c] for c in LADDER if c in cands]
        attempts = []
        for c in ladder:
            v = verdict(c["signals"], th)
            attempts.append({"key": c["cand"], "accept": v["accept"], "score": v["score"],
                             "row": c})
            dec = retry_decision(attempts, MAX_RETRIES)
            if dec["action"] != "redo":
                break
        pick = attempts[dec["pick"]]["row"] if dec["pick"] is not None else attempts[-1]["row"]
        base.append(ladder[0]["f1"])
        loop.append(pick["f1"])
        n_redo += len(attempts) > 1
        tries.append(len(attempts))
        oracle_tried.append(max(a["row"]["f1"] for a in attempts))
        oracle_all.append(max(c["f1"] for c in cands.values()))
    n = len(base)
    return {"figures": n, "default_f1": round(sum(base) / n, 4),
            "loop_f1": round(sum(loop) / n, 4),
            "oracle_tried_f1": round(sum(oracle_tried) / n, 4),
            "oracle_all_f1": round(sum(oracle_all) / n, 4),
            "redone": n_redo, "mean_tries": round(sum(tries) / n, 3),
            "worse": sum(b > lp + 1e-9 for b, lp in zip(base, loop, strict=True)),
            "better": sum(lp > b + 1e-9 for b, lp in zip(base, loop, strict=True))}


def selection(rows: list[dict]) -> dict:
    """Per figure, answer the ladder candidate a signal ranks first (ties:
    the ladder's order) -- how well does each signal choose between tools?"""
    by = _by_fig(rows)
    out = {}
    for name, sgn in SIGNALS.items():
        tot = 0.0
        for cands in by.values():
            ladder = [cands[c] for c in LADDER if c in cands]

            def key(i, ladder=ladder, name=name, sgn=sgn):
                v = ladder[i][name]
                return (sgn * v if v is not None else -9.0, -i)
            tot += ladder[max(range(len(ladder)), key=key)]["f1"]
        out[name] = round(tot / len(by), 4)
    out["(det_default)"] = round(sum(c["det_default"]["f1"] for c in by.values()) / len(by), 4)
    out["(oracle)"] = round(sum(max(r["f1"] for r in c.values()) for c in by.values())
                            / len(by), 4)
    return out


def slips(rows: list[dict], th: Thresholds) -> dict:
    """An orchestrator's slip -- the default answer without its largest
    series: how often verify accepts the default and sends the slip back."""
    n = acc_default = redo_slip = 0
    for cands in _by_fig(rows).values():
        if "det_drop1" not in cands:
            continue
        n += 1
        acc_default += verdict(cands["det_default"]["signals"], th)["accept"]
        redo_slip += not verdict(cands["det_drop1"]["signals"], th)["accept"]
    return {"figures": n, "default_accepted": acc_default, "slip_sent_back": redo_slip}


def cal_table(cals: list[dict], th: Thresholds) -> dict:
    ok = [c for c in cals if not c.get("failed")]
    out = {"figures": len(cals), "calibrated": len(ok),
           "correct": sum(c["correct"] for c in ok)}
    acc = [not calibration_reasons(c["signals"], th) for c in ok]
    out["accept_correct"] = sum(a and c["correct"] for a, c in zip(acc, ok, strict=True))
    out["accept_wrong"] = sum(a and not c["correct"] for a, c in zip(acc, ok, strict=True))
    out["redo_correct"] = sum((not a) and c["correct"] for a, c in zip(acc, ok, strict=True))
    out["redo_wrong"] = sum((not a) and not c["correct"] for a, c in zip(acc, ok, strict=True))
    sig_auc = {}
    for key, sgn in (("residual", -1), ("grid_dev", -1), ("marks_on_grid", 1),
                     ("labels_on_marks", 1)):
        good, bad = [], []
        for c in ok:
            vals = [c["signals"][ax].get(key) for ax in ("x", "y")]
            if any(v is None for v in vals):
                continue
            v = min(sgn * x for x in vals)  # the worse axis
            (good if c["correct"] else bad).append(v)
        sig_auc[key] = {"auc": round(auc(good, bad), 3), "n": len(good) + len(bad)}
    out["signal_auc"] = sig_auc
    return out


def cmd_analyze() -> None:
    raw = [json.loads(x) for x in (OUT / "candidates.jsonl").read_text().splitlines()]
    cals = [json.loads(x) for x in (OUT / "calibration.jsonl").read_text().splitlines()]
    rows = with_scores(raw)
    img_rows = with_scores(raw, use_detector=False)
    report: dict = {"bad_f1": BAD_F1, "ladder": LADDER, "max_retries": MAX_RETRIES}
    synth = [r for r in rows if r["set"] == "val_synth" and r["n_points"]]
    synth_img = [r for r in img_rows if r["set"] == "val_synth" and r["n_points"]]
    th = Thresholds(**{**Thresholds().as_dict(),
                       "min_score": choose_score_threshold(synth),
                       "min_score_image": choose_score_threshold(synth_img)})
    th = choose_cal_thresholds([c for c in cals if c["set"] == "val_synth"], th)
    report["thresholds_chosen_on_val_synth"] = th.as_dict()
    report["thresholds_in_code"] = Thresholds().as_dict()
    for s in SETS:
        rs = [r for r in rows if r["set"] == s]
        ri = [r for r in img_rows if r["set"] == s]
        report[s] = {
            "signals": signal_table(rs),
            "selection": selection(rs),
            "candidates_mean_f1": {c: round(sum(r["f1"] for r in rs if r["cand"] == c)
                                            / max(1, sum(r["cand"] == c for r in rs)), 4)
                                   for c in [*LADDER, "det_drop1"]},
            "confusion": confusion(rs, th),
            "confusion_image_only": confusion(ri, th),
            "loop": simulate(rs, th),
            "loop_image_only": simulate(ri, th),
            "slips": slips(rs, th),
            "calibration": cal_table([c for c in cals if c["set"] == s], th),
        }
    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "run":
        cmd_run(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    else:
        cmd_analyze()


if __name__ == "__main__":
    main()

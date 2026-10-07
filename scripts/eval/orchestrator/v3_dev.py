"""Development evidence for 方式D v3 (docs/design/local-model.md「v3: 検証の
誤検知と道具の追加」). Nothing here reads the benchmark.

Sets (all outside the benchmark):
  val_synth, stress, val_real  -- 方式A's validation split, as in verify_dev.py
  v3_tune, v3_check            -- synthetic stress cases of the v2 failure types
                                  (gen_v3_stress.py); thresholds may only be
                                  chosen on v3_tune (and val_synth)
  chartinfo                    -- CHART-Info line charts (chartinfo_dev.py);
                                  for points only those drawn with markers
                                  (the detector finds >= 70% of the vertices)

  run <tag> [workers]   run with the `real_chart_bench` on PYTHONPATH (the HEAD
                        snapshot for "before", the worktree for "after"):
                        per figure the candidates of verify_dev.py plus the
                        label itself ("truth", jittered 0.5 px), the label
                        without 40% of each series' points ("drop40") and
                        the label without its largest series ("drop1"); and
                        (when present) the marker_blobs tool; each scored by
                        pixel point F1 and verified; plus the automatic
                        calibration against the label's ticks.
  compare <before> <after>   tables: the "needs redo" signal (F1 < 0.8) --
                        precision / recall / false alarms -- per set and per
                        failure type; the calibration check; auto calibration
                        correct / wrong / unreadable; the tools' F1.

Writes ~/.cache/real-chart-bench/orchestrator/dev-v3/runs/<tag>.jsonl and the
compare report to data/local_orchestrator_runs/v3_dev_report.json.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CACHE = Path.home() / ".cache/real-chart-bench"
DEV = CACHE / "orchestrator/dev-v3"
PKL = CACHE / "detector/cache/real-b1-cpu.pkl"
OLD_DETS = CACHE / "orchestrator/dev-dets"
NEW_DETS = DEV / "dets"
REPORT = HERE.parents[2] / "data/local_orchestrator_runs/v3_dev_report.json"
BAD_F1 = 0.8


def _slim(sig: dict) -> dict:
    out = {k: v for k, v in sig.items() if k not in ("series",)}
    for k in ("unexplained", "lookalikes"):
        if isinstance(out.get(k), dict):
            out[k] = {kk: vv for kk, vv in out[k].items() if not isinstance(vv, list)}
    if isinstance(out.get("detector"), dict):
        out["detector"] = {k: v for k, v in out["detector"].items() if k != "uncovered"}
    return out


def jobs() -> list[dict]:
    out = []
    cache = pickle.loads(PKL.read_bytes())
    for s in ("val_synth", "stress", "val_real"):
        out += [{"set": s, "label": row["label"], "dets": str(OLD_DETS)}
                for row in cache["sets"][s]]
    for s, sub in (("v3_tune", "stress/tune"), ("v3_check", "stress/check"),
                   ("chartinfo", "chartinfo")):
        root = DEV / sub
        for line in (root / "labels.jsonl").read_text().splitlines():
            lab = json.loads(line)
            lab["_path"] = str(root / lab["image"])
            out.append({"set": s, "label": lab, "dets": str(NEW_DETS)})
    return out


def _f1(series, truth, radius):
    from real_chart_bench.domain.marker_detection import pixel_point_f1

    pred = [(x, y) for s in series for x, y in zip(s["x"], s["y"], strict=True)]
    return pixel_point_f1(pred, truth, radius)


def run_figure(job: dict) -> dict:
    import verify_dev

    from real_chart_bench.adapter.orchestrator_tools import ToolBox, ToolError
    from real_chart_bench.domain.verification import calibration_signals, verdict, verify

    verify_dev.DETS = Path(job["dets"])
    lab = job["label"]
    img = Path(lab["_path"])
    tb = ToolBox(img, dets_dir=Path(job["dets"]), allow_auto_calibration=True)
    store: dict = {}
    res = verify_dev._resolver(store)
    frame = tb.frame_of(None)
    text = tb.text_boxes()
    peaks = tb.marker_peaks()
    truth = [tuple(p) for s in lab["series"] for p in s["points_px"]]
    bx0, by0, bx1, by1 = lab["plot_bbox"]
    radius = 0.02 * max(bx1 - bx0, by1 - by0)
    meta = lab.get("v3") or {}
    marker_chart = True
    if job["set"] == "chartinfo":  # markers drawn at the vertices?
        conf = [(x, y) for x, y, s in (peaks or []) if s >= 0.5]
        hit = sum(1 for tx, ty in truth
                  if any(math.hypot(tx - x, ty - y) <= 0.015 * max(bx1 - bx0, by1 - by0)
                         for x, y in conf))
        marker_chart = bool(truth) and hit >= 0.7 * len(truth)
    cands = {}
    if marker_chart:
        for name, params in verify_dev.DETECTOR.items():
            if name in ("det_default", "det_ls1280", "det_t025"):
                try:
                    cands[name] = tb.run("marker_detector", params, res)
                except ToolError:
                    pass
        try:
            cands["colors"] = verify_dev.colour_pipeline(tb, frame, store)
        except ToolError:
            pass
        if "blob_extract" in getattr(sys.modules[ToolBox.__module__], "TOOLS", ()):
            try:
                cands["blobs"] = blob_pipeline(tb, frame, store)
            except ToolError:
                pass
        rng = random.Random(img.name)
        tser = [{"label": f"t{i}", "x": [p[0] + rng.uniform(-.5, .5) for p in s["points_px"]],
                 "y": [p[1] + rng.uniform(-.5, .5) for p in s["points_px"]]}
                for i, s in enumerate(lab["series"]) if s["points_px"]]
        cands["truth"] = {"series": tser}
        cands["drop40"] = {"series": [{**s, "x": [x for k, x in enumerate(s["x"]) if k % 5 >= 2],
                                       "y": [y for k, y in enumerate(s["y"]) if k % 5 >= 2]}
                                      for s in tser]}
        if len(tser) > 1:
            big = max(range(len(tser)), key=lambda i: len(tser[i]["x"]))
            cands["drop1"] = {"series": [s for i, s in enumerate(tser) if i != big]}
    rows = []
    for name, r in cands.items():
        series = [s for s in r["series"] if s["x"]]
        if not series:
            continue
        sig = verify(tb.rgb, series, frame, text_boxes=text, detections=peaks)
        v = verdict(sig)
        rows.append({"set": job["set"], "fig": img.name, "cand": name, "kind": meta.get("points"),
                     "f1": _f1(series, truth, radius), "accept": v["accept"],
                     "score": v["score"], "signals": _slim(sig)})
    cal_row = {"set": job["set"], "fig": img.name, "kind": meta.get("axis")}
    try:
        c = tb.run("tick_calibration", {"mode": "auto"}, res)
    except ToolError as exc:
        c = {"ok": False, "message": str(exc)}
    if c.get("ok") and lab.get("axes"):
        cal = c["calibration"]
        agree = {ax: verify_dev.label_agreement(cal[ax], lab["axes"][ax]) for ax in ("x", "y")}
        cal_row |= {"status": "correct" if max(agree.values()) <= 0.01 else "wrong",
                    "agree": {k: (None if not math.isfinite(v) else round(v, 4))
                              for k, v in agree.items()},
                    "signals": calibration_signals(cal, c.get("tick_marks"))}
    else:
        cal_row |= {"status": "unreadable", "message": c.get("message")}
    return {"rows": rows, "cal": cal_row}


def blob_pipeline(tb, frame, store) -> dict:
    """dominant colours inside the frame -> blob_extract per colour."""
    res = __import__("verify_dev")._resolver(store)
    mask = {"frame": frame, "frame_margin": 3} if frame else None
    cols = tb.run("dominant_colors", {"k": 8, **({"mask": mask} if mask else {})},
                  res)["colors"]
    chosen = [c for c in cols if not c["achromatic"] and c["fraction"] >= 0.01]
    if not chosen:
        chosen = [c for c in cols if c["achromatic"]][:1]
    short = min(frame[2] - frame[0], frame[3] - frame[1]) if frame else min(tb.size)
    series = []
    for c in chosen[:6]:
        r = tb.run("blob_extract", {"color": c["color"], "distance_pct": 5,
                                    "min_diameter_px": max(3, 0.008 * short),
                                    "max_diameter_px": max(8, 0.08 * short),
                                    **({"mask": mask} if mask else {})}, res)
        series += [s for s in r["series"] if len(s["x"]) >= 2]
    return {"kind": "points", "series": series}


def cmd_run(tag: str, workers: int) -> None:
    js = jobs()
    out = DEV / "runs" / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f, ProcessPoolExecutor(
            workers, mp_context=mp.get_context("spawn")) as pool:
        for i, r in enumerate(pool.map(run_figure, js, chunksize=2)):
            f.write(json.dumps(r) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(js)}", flush=True)
    print(f"{len(js)} figures -> {out}")


# ------------------------------------------------------------------ compare


def load(tag: str):
    rows, cals = [], []
    for line in (DEV / "runs" / f"{tag}.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows += r["rows"]
        cals.append(r["cal"])
    return rows, cals


def redo_table(rows: list[dict]) -> dict:
    bad = [not r["accept"] for r in rows]
    need = [r["f1"] < BAD_F1 for r in rows]
    tp = sum(b and n for b, n in zip(bad, need, strict=True))
    fp = sum(b and not n for b, n in zip(bad, need, strict=True))
    fn = sum((not b) and n for b, n in zip(bad, need, strict=True))
    good = len(rows) - sum(need)
    return {"n": len(rows), "needs_redo": sum(need), "redo": sum(bad),
            "precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3),
            "false_alarm_rate": round(fp / max(1, good), 3)}


def cal_check_table(cals: list[dict]) -> dict:
    from real_chart_bench.domain.verification import Thresholds, calibration_reasons

    ok = [c for c in cals if c.get("status") in ("correct", "wrong")]
    flag = [bool(calibration_reasons(c["signals"], Thresholds())) for c in ok]
    wrong = [c["status"] == "wrong" for c in ok]
    tp = sum(f and w for f, w in zip(flag, wrong, strict=True))
    fp = sum(f and not w for f, w in zip(flag, wrong, strict=True))
    return {"calibrated": len(ok), "wrong": sum(wrong), "flagged": sum(flag),
            "flag_wrong": tp, "flag_correct": fp,
            "false_alarm_rate": round(fp / max(1, len(ok) - sum(wrong)), 3),
            "recall_wrong": round(tp / max(1, sum(wrong)), 3)}


def status_table(cals: list[dict]) -> dict:
    out = {"n": len(cals)}
    for s in ("correct", "wrong", "unreadable"):
        out[s] = sum(c.get("status") == s for c in cals)
    return out


def cmd_compare(before: str, after: str) -> None:
    from real_chart_bench.domain.verification import verdict

    report: dict = {"bad_f1": BAD_F1, "before": before, "after": after}
    data = {t: load(t) for t in (before, after)}
    sets = sorted({r["set"] for r in data[after][0]})
    for t, (rows, cals) in data.items():
        if t == before:  # "before" judged with the code it ran with
            pass
        else:
            for r in rows:  # re-judge with the code's current thresholds
                r["accept"] = verdict(r["signals"])["accept"]
    for s in sets:
        sec: dict = {}
        for t, (rows, cals) in data.items():
            rs = [r for r in rows if r["set"] == s]
            sec[t] = {"redo": redo_table(rs),
                      "by_kind": {k: redo_table([r for r in rs if r["kind"] == k])
                                  for k in sorted({r["kind"] for r in rs if r["kind"]})},
                      "tools_f1": {c: round(sum(r["f1"] for r in rs if r["cand"] == c)
                                            / max(1, sum(r["cand"] == c for r in rs)), 4)
                                   for c in sorted({r["cand"] for r in rs})},
                      "cal_check": cal_check_table([c for c in cals if c["set"] == s]),
                      "auto_calibration": status_table([c for c in cals if c["set"] == s])}
            kinds = sorted({c.get("kind") for c in cals if c["set"] == s and c.get("kind")})
            if kinds:
                sec[t]["auto_calibration_by_kind"] = {
                    k: status_table([c for c in cals if c["set"] == s and c.get("kind") == k])
                    for k in kinds}
        report[s] = sec
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 24)
    elif cmd == "compare":
        cmd_compare(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()

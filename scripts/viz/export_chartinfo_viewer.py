"""Data for the CHART-Infographics batch viewer: per figure, the image, the
ground-truth points (data values and their pixel positions), each model's
points projected onto the image, which points matched under this project's
point F1 (tau 2%), and both scores.

Pixel positions of a model's points come from an axis map fitted per figure
on the ground truth's own (pixel, value) pairs (task6 "visual elements" vs
"data series"), linear or log10 whichever fits better; figures where no map
fits are shown without overlay.

Usage: python scripts/viz/export_chartinfo_viewer.py <zip> <batch_dir> <out.json>
"""

from __future__ import annotations

import base64
import io
import json
import math
import pathlib
import sys
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eval"))
from score_chartinfo_pilot import their_score  # noqa: E402

MODELS = ["claude-opus-5-5", "claude-fable-5-1", "gpt-6.1-sol", "claude-sonnet-5-5"]
TAU = 0.02


def fit_axis(vals, pix):
    best = None
    for scale in ("linear", "log"):
        v = np.array(vals, float)
        if scale == "log":
            if (v <= 0).any():
                continue
            v = np.log10(v)
        if np.ptp(v) == 0:
            continue
        a, b = np.polyfit(v, pix, 1)
        res = float(np.max(np.abs(a * v + b - np.array(pix))))
        if best is None or res < best[3]:
            best = (scale, a, b, res)
    return best


def to_px(fit, v):
    scale, a, b, _ = fit
    if scale == "log":
        if v <= 0:
            return None
        v = math.log10(v)
    return a * v + b


def matches(pred, gt):
    """This project's point F1 (score_chartinfo_pilot.our_score), with the
    per-point outcome kept."""

    def pts(series):
        return [
            [
                (float(p["x"]), float(p["y"]))
                for p in s.get("data", [])
                if isinstance(p.get("x"), int | float) and isinstance(p.get("y"), int | float)
            ]
            for s in series
        ]

    g, p = pts(gt), pts(pred)
    flat = [q for s in g for q in s]
    xs, ys = [q[0] for q in flat], [q[1] for q in flat]
    sx, sy = (max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0
    norm = lambda s: np.array([[(x - min(xs)) / sx, (y - min(ys)) / sy] for x, y in s])  # noqa: E731
    gm = [[False] * len(s) for s in g]
    pm = [[False] * len(s) for s in p]
    pair = {}
    if p and g:
        cost = np.ones((len(p), len(g)))
        cache = {}
        for i, ps in enumerate(p):
            for j, gs in enumerate(g):
                if not ps or not gs:
                    continue
                d = np.linalg.norm(norm(ps)[:, None] - norm(gs)[None], axis=2)
                ri, ci = linear_sum_assignment(d)
                ok = [(r, c) for r, c in zip(ri, ci) if d[r, c] <= TAU]
                cache[(i, j)] = ok
                cost[i, j] = 1 - 2 * len(ok) / (len(ps) + len(gs))
        for i, j in zip(*linear_sum_assignment(cost)):
            pair[int(i)] = int(j)
            for r, c in cache.get((i, j), []):
                pm[i][r] = True
                gm[j][c] = True
    m = sum(map(sum, gm))
    n_p = sum(map(len, p))
    rec, prec = m / len(flat), (m / n_p if n_p else 0.0)
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return (
        gm,
        pm,
        pair,
        {"f1": f1, "recall": rec, "precision": prec, "n_gt": len(flat), "n_pred": n_p},
    )


def main() -> None:
    from PIL import Image

    z = zipfile.ZipFile(sys.argv[1])
    batch = pathlib.Path(sys.argv[2])
    key = json.loads((batch / "_key.json").read_text())
    gt_all = json.loads((batch / "ground_truth.json").read_text())
    preds = {
        m: json.loads((batch / f"{m}.predictions.json").read_text())
        for m in MODELS
        if (batch / f"{m}.predictions.json").exists()
    }
    figs = []
    for fig in sorted(key):
        k = key[fig]
        ann = json.loads(z.read(k["source"]))["task6"]["output"]
        vis = (ann.get("visual elements") or {}).get("scatter points") or []
        gt = gt_all[fig]
        im = Image.open(io.BytesIO(z.read(k["image"]))).convert("RGB")
        w, h = im.size
        s = min(1.0, 900 / w)
        im = im.resize((round(w * s), round(h * s)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
        pairs_x, pairs_y = [], []
        gt_series = []
        for si, ser in enumerate(gt):
            pix = vis[si] if si < len(vis) and len(vis[si]) == len(ser["data"]) else None
            pts = []
            for pi, p in enumerate(ser["data"]):
                px = pix[pi] if pix else None
                if px:
                    pairs_x.append((p["x"], px["x"]))
                    pairs_y.append((p["y"], px["y"]))
                pts.append(
                    {"x": p["x"], "y": p["y"], "px": [px["x"] * s, px["y"] * s] if px else None}
                )
            gt_series.append({"name": ser.get("name", ""), "points": pts})
        fx = (
            fit_axis([a for a, _ in pairs_x], [b for _, b in pairs_x])
            if len(pairs_x) >= 2
            else None
        )
        fy = (
            fit_axis([a for a, _ in pairs_y], [b for _, b in pairs_y])
            if len(pairs_y) >= 2
            else None
        )
        overlay = bool(fx and fy and fx[3] < 6 and fy[3] < 6)
        models = {}
        for m, P in preds.items():
            pred = [ser for ser in P.get(fig, []) if isinstance(ser, dict)]
            gm, pm, pair, ours = matches(pred, gt)
            comb, nm, data = their_score(pred, gt) if pred else (0.0, 0.0, 0.0)
            series = []
            for i, ser in enumerate([q for q in pred]):
                pts = []
                valid = [
                    p
                    for p in ser.get("data", [])
                    if isinstance(p.get("x"), int | float) and isinstance(p.get("y"), int | float)
                ]
                for r, p in enumerate(valid):
                    px = None
                    if overlay:
                        X, Y = to_px(fx, p["x"]), to_px(fy, p["y"])
                        px = [X * s, Y * s] if X is not None and Y is not None else None
                    pts.append(
                        {
                            "x": p["x"],
                            "y": p["y"],
                            "px": px,
                            "hit": bool(pm[i][r]) if i < len(pm) and r < len(pm[i]) else False,
                        }
                    )
                series.append({"name": ser.get("name", ""), "points": pts, "gt": pair.get(i)})
            models[m] = {
                "series": series,
                "gt_hit": gm,
                "their": {"combined": comb, "name": nm, "data": data},
                "ours": ours,
            }
        figs.append(
            {
                "fig": fig,
                "source": k["image"].split("/")[-1],
                "w": im.size[0],
                "h": im.size[1],
                "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                "overlay": overlay,
                "axes": {"x": fx[0] if fx else None, "y": fy[0] if fy else None},
                "gt": gt_series,
                "models": models,
            }
        )
    pathlib.Path(sys.argv[3]).write_text(
        json.dumps({"batch": batch.name, "tau": TAU, "figures": figs})
    )
    print(
        len(figs),
        "figures;",
        sum(f["overlay"] for f in figs),
        "with overlay;",
        round(pathlib.Path(sys.argv[3]).stat().st_size / 1e6, 1),
        "MB",
    )


if __name__ == "__main__":
    main()

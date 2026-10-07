"""Synthetic stress cases for 方式D v3 (docs/design/local-model.md
「v3: 検証の誤検知と道具の追加」): the failure TYPES seen in the v2 runs,
drawn from random data (no benchmark figure is used or copied).

Point kinds (what verify and the extraction tools must cope with):
  gapline     markers joined by thick lines that stop short of each marker
              (the Origin style: line segments are separate blobs)
  errbar      error bars with caps (short segments aligned with the points)
  fit         markers plus a smooth fit curve that does not pass through them
  whitecross  filled markers split by a white + or x into four blobs
  overlap     series whose markers overlap (offset 0.3-0.9 marker diameters),
              and same-series neighbours that touch
Axis kinds (what the calibration check and tick reading must cope with):
  minorN      minor ticks with N subdivisions (2, 4, 5, 10, 11)
  neg         negative tick labels
  half        0.5-step labels (0.0, 0.5, 1.0, ...)
  sci         labels printed as a x 10^n
  reversed    a reversed axis
  cropped     the image cut through the x tick labels

Each image i comes from numpy's default_rng(seed + i). Labels follow
labels.jsonl (image, plot_bbox, series[].points_px, axes.{x,y}.ticks) plus
"v3": {"points": kind, "axis": kind}. The "tune" set (seed 31000) is the one
thresholds may be chosen on; "check" (seed 32000) only reports.

  .venv/bin/python scripts/eval/orchestrator/gen_v3_stress.py [--n 160] [--workers 16]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.synth_chart_labels import display_to_image  # noqa: E402

OUT = Path.home() / ".cache/real-chart-bench/orchestrator/dev-v3/stress"
SEEDS = {"tune": 31000, "check": 32000}
SS = 4
POINT_KINDS = ["gapline", "errbar", "fit", "whitecross", "overlap"]
AXIS_KINDS = ["minorN", "neg", "half", "sci", "reversed", "cropped", "plain"]
MARKERS = ["o", "s", "^", "D", "v"]
COLOURS = ["#1f3fd0", "#d01f1f", "#1f9f2f", "#202020", "#8a2be2", "#e07000"]


def _fmt_sci(v, _pos):
    if v == 0:
        return "0.0"
    e = int(np.floor(np.log10(abs(v))))
    return f"{v / 10 ** e:.1f}$\\times$10$^{{{e}}}$"


def render(i: int, seed: int):
    rng = np.random.default_rng(seed + i)
    pk = POINT_KINDS[i % len(POINT_KINDS)]
    ak = AXIS_KINDS[int(rng.integers(len(AXIS_KINDS)))]
    dpi = int(rng.choice([80, 100, 120, 150]))
    w_in = float(rng.uniform(4.0, 7.0))
    h_in = w_in * float(rng.uniform(0.65, 0.9))
    W, H = round(w_in * dpi), round(h_in * dpi)
    w_in, h_in = W / dpi, H / dpi
    ms = float(rng.uniform(6, 13))  # points
    n_series = int(rng.integers(2, 5))
    n_pts = int(rng.integers(6, 14))
    # values: printed range per axis kind
    x = np.linspace(0, 1, n_pts)
    x_lo, x_hi = (300.0, 900.0) if rng.random() < 0.5 else (0.0, 10.0)
    xs = x_lo + (x_hi - x_lo) * x
    ys = []
    base = rng.uniform(0.2, 0.8)
    a, b, c = rng.uniform(-0.6, 0.6, 3)
    shared = base + a * x + b * x**2 + c * np.sin(3 * x)
    # overlap: one curve, series offset by 0.3-0.9 marker diameters (the
    # marker is ms*dpi/72 px; the plot is roughly 0.75 of the image height)
    d_frac = (ms * dpi / 72) / (0.75 * H)
    for s in range(n_series):
        a, b, c = rng.uniform(-0.6, 0.6, 3)
        y = base + 0.25 * s + a * x + b * x**2 + c * np.sin(3 * x)
        if pk == "overlap":
            span = max(1e-6, float(np.ptp(shared)) + 0.3)
            y = shared + s * span * d_frac * rng.uniform(0.3, 0.9, n_pts)
        ys.append(y + rng.normal(0, 0.01 if pk != "overlap" else 0.0, n_pts))
    y_scale = {"neg": -100.0, "half": 3.0, "sci": 3e4}.get(
        ak, float(rng.choice([1.0, 10.0, 500.0])))
    y_off = -100.0 if ak == "neg" else 0.0
    ys = [y_off + y_scale * y for y in ys]
    rc = {"font.size": float(rng.uniform(8, 13)), "axes.linewidth": float(rng.uniform(0.8, 2.0)),
          "xtick.direction": str(rng.choice(["in", "out"])),
          "ytick.direction": str(rng.choice(["in", "out"])),
          "axes.formatter.useoffset": False}
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi * SS)
    with plt.rc_context(rc):
        ax = fig.add_subplot(111)
        lw = float(rng.uniform(1.5, 3.5))
        series = []
        for s in range(n_series):
            col = COLOURS[s % len(COLOURS)]
            mk = MARKERS[s % len(MARKERS)] if pk != "whitecross" else str(rng.choice(["o", "s"]))
            kw = dict(marker=mk, markersize=ms, color=col, linestyle="none", zorder=3)
            if pk == "gapline":
                gap = 0.6 + float(rng.uniform(0.0, 0.5))
                series.append((xs, ys[s], col, mk))
                ax.plot(xs, ys[s], **kw)
                continue
            if pk == "errbar":
                err = np.abs(np.ptp(ys[s]) + 1e-9) * rng.uniform(0.05, 0.15, n_pts)
                ax.errorbar(xs, ys[s], yerr=err, capsize=float(rng.uniform(3, 8)),
                            elinewidth=float(rng.uniform(1.0, 2.0)),
                            capthick=float(rng.uniform(1.0, 2.0)),
                            linestyle="-" if rng.random() < 0.5 else "none", lw=lw * 0.6, **{
                                k: v for k, v in kw.items() if k != "linestyle"})
            else:
                ax.plot(xs, ys[s], **kw)
            if pk == "fit":
                t = np.linspace(xs[0], xs[-1], 200)
                p = np.polyfit(xs, ys[s], 2)
                ax.plot(t, np.polyval(p, t), color=str(rng.choice([col, "#000000"])), lw=lw,
                        linestyle=str(rng.choice(["-", "--"])), zorder=2)
            if pk == "whitecross":
                ax.plot(xs, ys[s], marker=str(rng.choice(["+", "x"])), markersize=ms,
                        markeredgewidth=ms * float(rng.uniform(0.1, 0.18)), color="white",
                        linestyle="none", zorder=4)
            series.append((xs, ys[s], col, mk))
        ax.margins(x=0.08, y=0.12)
        ax.autoscale_view()
        if ak == "minorN":
            n = int(rng.choice([2, 4, 5, 10, 11]))
            ax.xaxis.set_minor_locator(AutoMinorLocator(n))
            ax.yaxis.set_minor_locator(AutoMinorLocator(n))
            ax.tick_params(which="minor", length=3)
        if ak == "half":
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}"))
        if ak == "sci":
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_sci))
        if ak == "reversed":
            ax.invert_xaxis() if rng.random() < 0.5 else ax.invert_yaxis()
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Quantity (a.u.)")
        fig.tight_layout(pad=0.6)
        fig.canvas.draw()
        Wh, Hh = fig.canvas.get_width_height()
        if pk == "gapline":  # segments that stop short of each marker
            for xs_, ys_, col, _mk in series:
                disp = ax.transData.transform(np.column_stack([xs_, ys_]))
                r_disp = (ms * dpi * SS / 72) * gap
                for (ax0, ay0), (ax1, ay1) in zip(disp, disp[1:], strict=False):
                    d = float(np.hypot(ax1 - ax0, ay1 - ay0))
                    if d <= 2 * r_disp:
                        continue
                    ux, uy = (ax1 - ax0) / d, (ay1 - ay0) / d
                    p0 = ax.transData.inverted().transform((ax0 + ux * r_disp, ay0 + uy * r_disp))
                    p1 = ax.transData.inverted().transform((ax1 - ux * r_disp, ay1 - uy * r_disp))
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=lw, zorder=2,
                            scalex=False, scaley=False)
            fig.canvas.draw()
        ticks = {}
        for which, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            lo, hi = sorted(ax.get_xlim() if which == "x" else ax.get_ylim())
            out = []
            for loc in axis.get_majorticklocs():
                if not lo <= loc <= hi:
                    continue
                pt = (loc, ax.get_ylim()[0]) if which == "x" else (ax.get_xlim()[0], loc)
                dx, dy = ax.transData.transform(pt)
                px = display_to_image(dx, dy, height=Hh, scale=SS)[0 if which == "x" else 1]
                out.append({"px": round(float(px), 3), "value": float(loc)})
            ticks[which] = {"scale": "linear", "ticks": out}
        pts = []
        for xs_, ys_, col, mk in series:
            disp = ax.transData.transform(np.column_stack([xs_, ys_]))
            pp = [list(display_to_image(float(a), float(b), height=Hh, scale=SS)) for a, b in disp]
            pts.append({"color": col, "mpl_marker": mk,
                        "points_px": [[round(a, 3), round(b, 3)] for a, b in pp]})
        bb = ax.get_window_extent()
        label_mid = None
        tls = [t for t in ax.xaxis.get_ticklabels() if t.get_visible() and t.get_text()]
        if tls:
            e = tls[0].get_window_extent()
            label_mid = display_to_image(0, (e.y0 + e.y1) / 2, height=Hh, scale=SS)[1]
        x0, y0 = display_to_image(bb.x0, bb.y1, height=Hh, scale=SS)
        x1, y1 = display_to_image(bb.x1, bb.y0, height=Hh, scale=SS)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi * SS)
    plt.close(fig)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((W, H), Image.BOX)
    if ak == "cropped" and label_mid is not None:  # cut through the x tick labels
        img = img.crop((0, 0, W, int(round(label_mid))))
    label = {"width": img.size[0], "height": img.size[1],
             "plot_bbox": [round(v, 3) for v in (x0, y0, x1, y1)], "series": pts,
             "axes": ticks, "v3": {"points": pk, "axis": ak, "marker_px": round(ms * dpi / 72, 2)}}
    return label, img


def write_one(job):
    i, seed, out = job
    label, img = render(i, seed)
    name = f"{seed + i}.png"
    img.save(Path(out) / "images" / name, optimize=True)
    label["image"] = f"images/{name}"
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    for name, seed in SEEDS.items():
        out = args.out / name
        (out / "images").mkdir(parents=True, exist_ok=True)
        with Pool(args.workers) as pool:
            labels = pool.map(write_one, [(i, seed, str(out)) for i in range(args.n)])
        (out / "labels.jsonl").write_text("".join(json.dumps(lab) + "\n" for lab in labels))
        print(f"{name}: {len(labels)} -> {out}")


if __name__ == "__main__":
    main()

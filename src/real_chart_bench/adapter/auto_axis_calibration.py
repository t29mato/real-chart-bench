"""Automatic axis calibration of one figure image, by rules and local OCR
only (docs/design/local-model.md, 方式C, and rule 2: never by Claude/GPT).

Frames and tick marks come from pixel rules (domain/axis_frame.py), tick
label text from Tesseract (adapter/tesseract_ocr.py), the (pixel, value)
line from domain/tick_calibration.fit_axis. With every option off,
`calibrate_frame` is the Starrydata label builder's
(scripts/train/starrydata_build.py).

Options (all developed on the synthetic validation split, scripts/train/calib_val.py):
- split_merged: tightly spaced labels that Tesseract reads as one word
  ("300320340") are split into an arithmetic run and cut at the ink gaps;
- right_axes: a y axis drawn (or labelled) on the right of the frame;
- superscripts: log axes print 10^n with a small raised exponent that
  Tesseract drops ("10") or runs into the base ("103"). The raised glyphs
  are found in the label's ink and OCR'd alone (`_exponent_rescue`), and
  evenly spaced "10..." labels are numbered as consecutive decades, the
  exponents that were read voting for the offset (decade_readings).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from real_chart_bench.adapter.tesseract_ocr import ocr_words
from real_chart_bench.domain.axis_frame import (
    Frame,
    detect_axis_frames,
    detect_right_axis_frames,
    detect_ticks,
    exponent_span,
    label_glyphs,
    mirror_frame,
    outward_tick_extent,
    split_at_widest_gaps,
)
from real_chart_bench.domain.tick_calibration import (
    AxisFit,
    apply_glyphs,
    dark_background,
    decade_readings,
    fit_axis,
    labels_for_axis,
    readings_for_axis,
    scaled_fit,
    scaled_frame,
    split_merged_labels,
    touches_border,
)

DARK = 140  # frame/tick binarisation (axes are black)


def load_rgb(path: Path) -> np.ndarray | None:
    try:
        im = Image.open(io.BytesIO(Path(path).read_bytes()))
        im.load()
    except Exception:  # noqa: BLE001
        return None
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    if im.mode == "CMYK":
        arr = np.asarray(im.convert("RGB"))
        return 255 - arr if arr.mean() < 80 else arr
    return np.asarray(im.convert("RGB"))


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.uint8)


_DECADE_WORD = re.compile(r"^10[-\d]{0,3}$")


def _read_exponent(crop: np.ndarray) -> str | None:
    for psm in (7, 8):
        text = "".join(e[0] for e in ocr_words(crop, upscale=6, psm=psm)).replace("−", "-")
        if re.fullmatch(r"-?\d{1,2}", text):
            return text
    return None


def _exponent_rescue(strip: np.ndarray, words: list) -> list:
    """Words that look like a "10^n" label whose exponent was dropped ("10")
    or run into the base ("103", "10-2"): find the raised glyphs right of the
    base in the label's ink (domain.axis_frame.exponent_span), OCR them on
    their own, much enlarged, and write the label as "10^n". The search box
    stays inside the label strip, so it never sees the axis line. Boxes are
    the strip's pixels."""
    out = []
    h, w = strip.shape
    for t, a, b, c, d, conf in words:
        if _DECADE_WORD.match(t.strip()):
            hh = d - b
            x0, y0 = max(0, int(a)), max(0, int(b - 0.6 * hh))
            x1, y1 = min(w, int(c + 1.2 * hh) + 1), min(h, int(d) + 2)
            region = strip[y0:y1, x0:x1]
            span = exponent_span(region < DARK) if region.size else None
            if span is not None:
                ex0, ey0, ex1, ey1 = span
                pad = 2
                crop = region[max(0, ey0 - pad) : ey1 + pad, max(0, ex0 - pad) : ex1 + pad]
                exp = _read_exponent(crop) if crop.size else None
                if exp is not None:
                    t = f"10^{exp}"
        out.append((t, a, b, c, d, conf))
    return out


def _split_merged(gray: np.ndarray, words: list) -> list:
    """Words that are several tick labels run together ("300320340") become
    one word per label. Each label's box is cut at the widest blank gaps of
    the word's ink (the space between labels), else in proportion to its
    characters."""
    out = []
    for t, a, b, c, d, conf in words:
        parts = split_merged_labels(t)
        if not parts:
            out.append((t, a, b, c, d, conf))
            continue
        ia, ib = max(0, int(a)), max(0, int(b))
        ink = gray[ib : int(d) + 1, ia : int(c) + 1] < DARK
        spans = split_at_widest_gaps(ink, len(parts)) if ink.size else None
        if spans is not None:
            out += [(p, ia + s0, b, ia + s1, d, conf) for p, (s0, s1) in zip(parts, spans,
                                                                             strict=True)]
            continue
        n = sum(len(p) for p in parts)
        x = a
        for p in parts:
            x_end = x + (c - a) * len(p) / n
            out.append((p, x, b, x_end, d, conf))
            x = x_end
    return out


_NUMERIC = re.compile(r"^[-−\d.]+$")


def _split_line(gray: np.ndarray, words: list) -> list:
    """The x-axis label line as a whole: when Tesseract groups the labels
    into words with wrong boxes ("220230240250" boxed over a third of the
    line), the joined text of the first numeric line is split into one
    arithmetic run and the labels are placed at the widest ink gaps of the
    whole line. Words are returned unchanged when that does not work."""
    nums = [w for w in words if _NUMERIC.match(w[0])]
    if not nums:
        return words
    top = min(nums, key=lambda w: w[2])
    line = sorted((w for w in nums if w[2] < top[4] and w[4] > top[2]), key=lambda w: w[1])
    parts = split_merged_labels("".join(w[0] for w in line))
    if not parts or len(parts) <= len(line):
        return words
    a, b = max(0, int(min(w[1] for w in line))), max(0, int(min(w[2] for w in line)))
    c, d = int(max(w[3] for w in line)) + 1, int(max(w[4] for w in line)) + 1
    ink = gray[b:d, a:c] < DARK
    spans = split_at_widest_gaps(ink, len(parts), min_ratio=1.5) if ink.size else None
    if spans is None:
        return words
    rest = [w for w in words if w not in line]
    conf = min(w[5] for w in line)
    return rest + [(p, a + s0, b, a + s1, d, conf)
                   for p, (s0, s1) in zip(parts, spans, strict=True)]


def _fix_glyphs(gray: np.ndarray, words: list, lo: int, hi: int) -> list:
    """v3: a minus sign or a decimal point the OCR dropped, restored from the
    label's own ink (domain.axis_frame.label_glyphs). The crop reaches a bit
    left of the word (a detached minus sign), never outside the label strip
    [lo, hi)."""
    out = []
    for t, a, b, c, d, conf in words:
        if _NUMERIC.match(t.strip()):
            hh = d - b
            x0 = max(lo, int(a - 0.8 * hh))
            crop = gray[max(0, int(b)) : int(d) + 1, x0 : min(hi, int(c) + 1)] < DARK
            if crop.size:
                t = apply_glyphs(t, label_glyphs(crop))
        out.append((t, a, b, c, d, conf))
    return out


def _mirror_words(words: list, width: int) -> list:
    return [(t, width - 1 - c, b, width - 1 - a, d, conf) for t, a, b, c, d, conf in words]


def calibrate_frame(
    gray: np.ndarray,
    dark: np.ndarray,
    frame: Frame,
    *,
    superscripts: bool = False,
    y_side: str = "left",
    split_merged: bool = False,
    v3: bool = False,
) -> dict:
    """x / y AxisFit (or None) of one frame, and the OCR words used.

    y_side "right": the frame's y axis is its right edge (x1), its labels
    to the right of it; read in the mirrored geometry, so the left-axis
    rules apply unchanged (the OCR itself always sees the unmirrored image).

    v3 (方式D「v3」, off by default so earlier pipelines are unchanged): words
    cut by the image border are left out (unreadable rather than guessed),
    dropped minus signs and decimal points are restored from the ink, "a x
    10^n" labels whose "x" the OCR lost are read, a log fit must be
    plausible, and a reversed axis is read when nothing else fits. (Snapping
    corner labels to the frame's edges was tried and dropped: on the stress
    set it moved labels off their ticks.)"""
    x0, y0, x1, y1 = frame
    fw, fh = x1 - x0, y1 - y0
    h, w = gray.shape
    xt, _ = detect_ticks(dark, frame)
    below, _ = outward_tick_extent(dark, frame)
    if y_side == "right":
        mdark = dark[:, ::-1]
        mframe = mirror_frame(frame, w)
        _, yt = detect_ticks(mdark, mframe)
        _, out_y = outward_tick_extent(mdark, mframe)
        tx0, tx1 = min(w, int(x1 + 2 + out_y)), min(w, int(x1 + 0.4 * fw))
        y_frame = mframe
    else:
        _, yt = detect_ticks(dark, frame)
        _, out_y = outward_tick_extent(dark, frame)
        tx0, tx1 = max(0, int(x0 - 0.4 * fw)), max(0, int(x0 - 2 - out_y))
        y_frame = frame
    # x strip: under the axis line and its outward ticks
    sx0, sx1 = max(0, int(x0 - 0.1 * fw)), min(w, int(x1 + 0.1 * fw))
    sy0, sy1 = min(h, int(y1 + 2 + below + 1)), min(h, int(y1 + 0.2 * fh + 14 + below))
    # y strip: beside the y axis line and its outward ticks
    ty0, ty1 = max(0, int(y0 - 0.1 * fh)), min(h, int(y1 + 0.1 * fh))

    def read(axis, strip, ox, oy, ticks, direction, fr, mirror):
        """Block mode (psm 6) reads a label column best; sparse mode (psm
        11) rescues scattered labels. Keep whichever fits more ticks."""
        best, best_n = (None, [], 0), -1
        # v3 leaves out words cut by the image border; when that leaves the
        # axis unreadable they are kept for a second read (design
        # pairing-automation.md 12.7): the fit still needs three agreeing ticks
        for keep_border in ((False, True) if v3 else (False,)):
            for psm in (6, 11):
                raw = ocr_words(strip, psm=psm)
                if superscripts:
                    raw = _exponent_rescue(strip, raw)
                ws = [(t, a + ox, b + oy, c + ox, d + oy, conf) for t, a, b, c, d, conf in raw]
                if v3 and not keep_border:
                    ws = [wd for wd in ws if not touches_border(wd[1:5], (w, h))]
                if split_merged:
                    if axis == "x":
                        ws = _split_line(gray, ws)
                    ws = _split_merged(gray, ws)
                if v3:  # after splitting: a point restored inside "300320340"
                    # would make the merged labels unsplittable (28331, design
                    # local-model.md「自動校正の改善」)
                    ws = _fix_glyphs(gray, ws, ox, ox + strip.shape[1])
                mw = _mirror_words(ws, w) if mirror else ws
                rd = readings_for_axis(mw, fr, axis, ticks=ticks)
                if superscripts:
                    # equally spaced "10..." labels as consecutive decades
                    dec = dict(decade_readings(labels_for_axis(mw, fr, axis, ticks=ticks),
                                               direction=direction))
                    rd = [(px, rs + [r for r in dec.get(px, []) if r not in rs]) for px, rs in rd]
                fit = fit_axis(rd, direction=direction, allow_reversed=v3, plausible_log=v3,
                               sci=v3)
                n = len(fit.ticks) if fit else 0
                if n > best_n:
                    best, best_n = (fit, ws, len(rd)), n
                # stop on a good fit that explains every label read; a fit that
                # leaves labels unexplained (a mis-split merged word) lets the
                # other reading mode try (14649: 4 of 6 labels, psm 11 reads 7)
                if n >= 4 and n >= len(rd):
                    return best
            if best_n >= 3:
                break
        return best

    xf, xw, n_xr = read("x", gray[sy0:sy1, sx0:sx1], sx0, sy0, xt, +1, frame, False)
    yf, yw, n_yr = read(
        "y", gray[ty0:ty1, tx0:tx1], tx0, ty0, yt, -1, y_frame, y_side == "right"
    )
    words = [w[:5] for w in xw + yw]
    return {"x_fit": xf, "y_fit": yf, "words": words, "n_xr": n_xr, "n_yr": n_yr}


@dataclass(frozen=True)
class ImageCalibration:
    frame: Frame
    x_fit: AxisFit | None
    y_fit: AxisFit | None
    n_frames: int
    y_side: str = "left"

    @property
    def ok(self) -> bool:
        return self.x_fit is not None and self.y_fit is not None


def _frame_area(f: Frame) -> float:
    return (f[2] - f[0]) * (f[3] - f[1])


def _same_plot_box(f: Frame, g: Frame) -> bool:
    ix = min(f[2], g[2]) - max(f[0], g[0])
    iy = min(f[3], g[3]) - max(f[1], g[1])
    return ix > 0 and iy > 0 and ix * iy > 0.8 * min(_frame_area(f), _frame_area(g))


def _candidate_frames(dark: np.ndarray, right_axes: bool) -> list[tuple[Frame, str]]:
    """Bottom-left frames first (largest first); a bottom-right L only when
    it is not the same plot box seen from its other corner."""
    uniq: list[tuple[Frame, str]] = [
        (f, "left") for f in sorted(detect_axis_frames(dark), key=lambda f: -_frame_area(f))
    ]
    if right_axes:
        for f in sorted(detect_right_axis_frames(dark), key=lambda f: -_frame_area(f)):
            if not any(_same_plot_box(f, g) for g, _ in uniq):
                uniq.append((f, "right"))
    return uniq


def calibrate_frames(
    rgb: np.ndarray,
    *,
    superscripts: bool = True,
    right_axes: bool = True,
    split_merged: bool = True,
    max_frames: int = 6,
    v3: bool = True,
    both_y_sides: bool = False,
) -> list[ImageCalibration]:
    """Every plot frame of the image whose two axes calibrate (a pairing
    candidate pool, design pairing-automation.md §12): unlike
    `calibrate_image`, which stops at the first panel, all of them are kept,
    largest first. Frames that do not calibrate are not returned -- they
    cannot project ground truth.

    both_y_sides (design pairing-automation.md 12.8): a plot with a y axis on
    each side holds two quantities; every side that calibrates is returned as
    its own entry (same frame, y_side left / right) instead of only the first."""
    gray = to_gray(rgb)
    dark = gray < DARK
    uniq = _candidate_frames(dark, right_axes)
    out: list[ImageCalibration] = []
    for f, side in uniq[:max_frames]:
        sides = [side] + (["right" if side == "left" else "left"] if right_axes else [])
        for s in sides:
            cal = calibrate_frame(
                gray, dark, f, superscripts=superscripts, y_side=s, split_merged=split_merged,
                v3=v3,
            )
            c = ImageCalibration(f, cal["x_fit"], cal["y_fit"], len(uniq), s)
            if c.ok:
                out.append(c)
                if not both_y_sides:
                    break
    return out


def calibrate_image(
    rgb: np.ndarray,
    *,
    normalize: bool = False,
    max_side: int | None = None,
    min_side: int | None = None,
    **kw,
) -> ImageCalibration | None:
    """_calibrate_unnormalized, after normalising the image when asked
    (design local-model.md「自動校正の改善」): a dark page is inverted (the
    rules expect black axes on white), and a figure whose long side exceeds
    max_side is read shrunk (one smaller than min_side enlarged), its frame
    and fits mapped back to the original pixels."""
    if not normalize:
        return _calibrate_unnormalized(rgb, **kw)
    if dark_background(to_gray(rgb)):
        rgb = 255 - rgb
    h, w = rgb.shape[:2]
    if max_side is not None and max(h, w) > max_side:
        s = max_side / max(h, w)
    elif min_side is not None and max(h, w) < min_side:
        s = min_side / max(h, w)
    else:
        return _calibrate_unnormalized(rgb, **kw)
    small = np.asarray(Image.fromarray(rgb).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS))
    c = _calibrate_unnormalized(small, **kw)
    if c is None:
        return None
    k = 1 / s
    return ImageCalibration(scaled_frame(c.frame, k), scaled_fit(c.x_fit, k),
                            scaled_fit(c.y_fit, k), c.n_frames, c.y_side)


def _calibrate_unnormalized(
    rgb: np.ndarray,
    *,
    superscripts: bool = True,
    right_axes: bool = True,
    split_merged: bool = True,
    max_frames: int = 4,
    v3: bool = False,
) -> ImageCalibration | None:
    """The figure's main plot frame and its two axis fits.

    A scored figure is one panel, so frames are tried largest first and the
    first with both axes calibrated wins, else the largest one is returned.
    A frame's y axis is read on the left, then (right_axes) on the right; a
    frame drawn only as a bottom-right L is found too. None when no frame is
    found (no drawn axis lines)."""
    gray = to_gray(rgb)
    dark = gray < DARK
    uniq = _candidate_frames(dark, right_axes)
    if not uniq:
        return None
    best = None
    for f, side in uniq[:max_frames]:
        sides = [side] + (["right" if side == "left" else "left"] if right_axes else [])
        for s in sides:
            cal = calibrate_frame(
                gray, dark, f, superscripts=superscripts, y_side=s, split_merged=split_merged,
                v3=v3,
            )
            c = ImageCalibration(f, cal["x_fit"], cal["y_fit"], len(uniq), s)
            if c.ok:
                return c
            if best is None:
                best = c
    return best

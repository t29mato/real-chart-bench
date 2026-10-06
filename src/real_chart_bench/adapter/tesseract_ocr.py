"""Tesseract OCR of tick-label strips (docs/design/local-model.md, rule 2:
axis calibration by local OCR, never by Claude/GPT).

Tesseract is Apache-2.0 and runs as the system binary (`tesseract`, 5.x);
nothing is installed into the repo's .venv. A strip is upscaled before OCR
(tick labels are ~8-12 px tall in a 150-200 dpi render, below Tesseract's
comfortable size) and word boxes are mapped back to the strip's own pixels.
"""

from __future__ import annotations

import io
import subprocess

import numpy as np
from PIL import Image

Word = tuple[str, float, float, float, float, float]  # text, x0, y0, x1, y1, conf

_WHITELIST = "0123456789.-−^Ee"


def ocr_words(gray: np.ndarray, *, upscale: int = 3, psm: int = 11,
              whitelist: str | None = _WHITELIST) -> list[Word]:
    """Words in a uint8 grayscale strip, boxes in the strip's pixels.
    `whitelist` None reads any text (legend entries, annotations)."""
    if gray.size == 0 or gray.shape[0] < 4 or gray.shape[1] < 4:
        return []
    img = Image.fromarray(gray).resize(
        (gray.shape[1] * upscale, gray.shape[0] * upscale), Image.LANCZOS
    )
    # a white border: Tesseract misses glyphs that touch the image edge
    pad = 10 * upscale
    canvas = Image.new("L", (img.width + 2 * pad, img.height + 2 * pad), 255)
    canvas.paste(img, (pad, pad))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    wl = ["-c", f"tessedit_char_whitelist={whitelist}"] if whitelist else []
    out = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", str(psm), "-l", "eng", *wl, "tsv"],
        input=buf.getvalue(), capture_output=True, check=False, timeout=60,
    )
    words: list[Word] = []
    for line in out.stdout.decode("utf-8", "replace").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 12 or not cols[11].strip():
            continue
        left, top, w, h = (int(c) for c in cols[6:10])
        conf = float(cols[10])
        x0 = (left - pad) / upscale
        y0 = (top - pad) / upscale
        words.append((cols[11].strip(), x0, y0, x0 + w / upscale, y0 + h / upscale, conf))
    return words

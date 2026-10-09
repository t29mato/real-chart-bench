"""Batch review page for the pairing candidates (design pairing-automation.md
§12.4, scaling-verification.md §4.2).

One question per tile: do the projected Starrydata points sit on the curves
drawn in the figure? (yes / no / unsure). Verdicts are kept in the browser and
exported as JSON keyed by the hash of the picture that was shown (§4.4).

Only figures of the `public` split are embedded (images as data: URIs); the
paper-level CC BY licence is shown on every tile, the per-figure licence check
(design §11) is still open.

    python scripts/eval/generate_pairing_review.py --images-root /abs/data/raw/images [out.html]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import pathlib
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, "src")

from real_chart_bench.adapter.auto_axis_calibration import load_rgb  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
MAX_EDGE = 700
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#f032e6", "#9a6324"]


def crop_box(bbox, shape):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    h_img, w_img = shape[:2]
    return (max(0, int(x0 - 0.3 * w)), max(0, int(y0 - 0.1 * h)),
            min(w_img, int(x1 + 0.1 * w)), min(h_img, int(y1 + 0.25 * h)))


def render(rgb, bbox, curves):
    """(plain JPEG bytes, overlay JPEG bytes) of the frame's surroundings."""
    cx0, cy0, cx1, cy1 = crop_box(bbox, rgb.shape)
    base = Image.fromarray(rgb[cy0:cy1, cx0:cx1])
    scale = min(1.0, MAX_EDGE / max(base.size))
    size = (max(1, int(base.width * scale)), max(1, int(base.height * scale)))
    plain = base.resize(size)
    over = plain.copy()
    d = ImageDraw.Draw(over)
    for i, pts in enumerate(curves):
        col = COLORS[i % len(COLORS)]
        for x, y in pts:
            px, py = (x - cx0) * scale, (y - cy0) * scale
            d.ellipse((px - 4, py - 4, px + 4, py + 4), outline=col, width=2)

    def jpg(im):
        b = io.BytesIO()
        im.convert("RGB").save(b, "JPEG", quality=72)
        return b.getvalue()

    return jpg(plain), jpg(over)


def tile(c, plain, over):
    sha = hashlib.sha256(over).hexdigest()

    def b64(b):
        return base64.b64encode(b).decode()

    ref = html.escape(c["figure_reference"])
    xt, yt = html.escape(c["x_transform"]), html.escape(c["y_transform"])
    meta = (
        f"paper {c['paper_id']} / figure {c['figure_id']} (ref {ref})"
        f" &middot; S={c['S']:.2f} null={c['null']:.2f} M={c['M']:.2f}"
        f" &middot; {len(c['gt_curve_ids'])} GT curves &middot; {xt}/{yt}"
        " &middot; CC BY (paper)"
    )
    buttons = "".join(f'<button data-v="{v}">{v}</button>' for v in ("yes", "no", "unsure"))
    return (
        f'<div class="tile" data-id="{c["candidate_id"]}" data-sha="{sha}"'
        f' data-lane="{c["decision"]}">'
        f'<img src="data:image/jpeg;base64,{b64(over)}"'
        f' data-plain="data:image/jpeg;base64,{b64(plain)}">'
        f'<div class="meta">{meta}</div><div class="btns">{buttons}'
        '<button class="tog">overlay on/off</button></div></div>'
    )


TEMPLATE = pathlib.Path(__file__).with_name("_pairing_review_template.html")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", type=pathlib.Path, default=REPO / "build/pairing_review.html")
    ap.add_argument("--images-root", type=pathlib.Path, default=REPO / "data/raw/images")
    ap.add_argument("--candidates", type=pathlib.Path,
                    default=REPO / "data/manifest/v0/pairing_candidates.json")
    ap.add_argument("--overlays", type=pathlib.Path, default=REPO / "build/pairing_overlays.json")
    args = ap.parse_args()
    cands = json.loads(args.candidates.read_text())["candidates"]
    overlays = json.loads(args.overlays.read_text())
    todo = [c for c in cands if c["decision"].startswith("proposed") and c["split"] == "public"]
    todo.sort(key=lambda c: (c["decision"] != "proposed_high", -c["S"], c["candidate_id"]))
    sections: dict[str, list[str]] = {"proposed_high": [], "proposed_review": []}
    cache: dict = {}
    for c in todo:
        if c["image"] not in cache:
            cache.clear()
            cache[c["image"]] = load_rgb(args.images_root / c["image"])
        rgb = cache[c["image"]]
        plain, over = render(rgb, c["frame_bbox"], overlays[c["candidate_id"]])
        sections[c["decision"]].append(tile(c, plain, over))
    body = ""
    for key, title in (("proposed_high", "High-score lane (still needs your eyes)"),
                       ("proposed_review", "Review lane (lower score or close rival)")):
        tiles = "".join(sections[key])
        body += f"<h2>{title}: {len(sections[key])}</h2><div class=grid>{tiles}</div>"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    page = TEMPLATE.read_text().replace("__N__", str(len(todo)))
    args.out.write_text(page.replace("__BODY__", body))
    print(f"{args.out}: {len(todo)} tiles, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

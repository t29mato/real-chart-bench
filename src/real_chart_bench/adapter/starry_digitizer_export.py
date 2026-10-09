"""An extractor's answer -> a starry-digitizer auto-extraction v1 document,
so a person can start post-editing from it (docs/paper/05-postediting.md).

The format is starry-digitizer's (`starry-digitizer/auto-extraction` v1,
2026-10-09; its description: /var/tmp/starry-digitizer-auto-extraction-schema-v1.md).
Pixel coordinates there are continuous with the origin at the top-left
corner of the top-left pixel. Our labels and detector outputs use the same
convention, so points are written unshifted.

Pure conversion: no I/O.
"""

from __future__ import annotations

FORMAT = "starry-digitizer/auto-extraction"
VERSION = 1


def auto_extraction_doc(
    answer: list[dict],
    *,
    image: str | None,
    width: int | None = None,
    height: int | None = None,
    axes: dict | None = None,
    space: str = "px",
    scores: list[list[float]] | None = None,
    source: dict | None = None,
) -> dict:
    """answer: [{label, x: [...], y: [...]}] in image pixels (space "px") or
    in printed values (space "value", which needs axes). scores, when given,
    has one list per answer series with one score per point. Series with no
    points are dropped; unlabelled ones get starry-digitizer's default name."""
    if space not in ("px", "value"):
        raise ValueError(f"space must be 'px' or 'value', not {space!r}")
    if space == "value" and axes is None:
        raise ValueError("value answers need axes to be placed on the image")
    if scores is not None and len(scores) != len(answer):
        raise ValueError("scores needs one list per series")
    series = []
    for i, s in enumerate(answer):
        pts = [[float(x), float(y)] for x, y in zip(s.get("x") or [], s.get("y") or [],
                                                    strict=True)]
        sc = None
        if scores is not None:
            sc = [float(v) for v in scores[i]]
            if len(sc) != len(pts):
                raise ValueError(f"series {i}: {len(sc)} scores for {len(pts)} points")
        if not pts:
            continue
        series.append({
            "label": s.get("label") or f"series {len(series) + 1}",
            "points_px": pts if space == "px" else None,
            "points_value": pts if space == "value" else None,
            "scores": sc,
        })
    doc: dict = {
        "format": FORMAT,
        "version": VERSION,
        "image": None if image is None else {"url": image, "width": width, "height": height},
        "axes": axes,
        "series": series,
    }
    if source is not None:
        doc["source"] = source
    return doc

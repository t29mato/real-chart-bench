"""Write tests/fixtures/starry_extract_js_cases.json: small synthetic images,
and the points starry-digitizer's own JavaScript (Symbol Extract / Line
Extract) returns for them, so tests/domain/test_starry_extract.py can check
the Python port without node.

The original code is read from the vendored package (unpacked to a scratch
dir); nothing of starry-digitizer is modified or copied into the repo.

  tar xzf ~/starrydata2/frontend/vendor/starry-digitizer-2.0.0-dev-a13c927.tgz -C <tmp>
  python scripts/tools/gen_starry_extract_fixtures.py <tmp>/package/library-build/dist/core.js
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests/fixtures/starry_extract_js_cases.json"
PALETTE = [(255, 255, 255), (0, 0, 0), (220, 30, 30), (225, 35, 28), (30, 30, 220),
           (128, 128, 128), (30, 160, 40)]


def _hit(kind: str, x: int, y: int, cx: int, cy: int, r: int, rng: random.Random) -> bool:
    if kind == "disc":
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    if kind == "rect":
        return abs(x - cx) <= r and abs(y - cy) <= r // 2 + 1
    if kind == "line":
        return abs((y - cy) - (x - cx) * (r - 3) / 3) < 0.8
    return rng.random() < 0.05


def image(rng: random.Random, w: int, h: int) -> list[list[tuple[int, int, int]]]:
    px = [[PALETTE[0] for _ in range(w)] for _ in range(h)]
    for _ in range(rng.randint(3, 9)):
        col = rng.choice(PALETTE[1:])
        kind = rng.choice(["disc", "rect", "line", "noise"])
        cx, cy = rng.randrange(w), rng.randrange(h)
        r = rng.randint(1, 5)
        for y in range(h):
            for x in range(w):
                if _hit(kind, x, y, cx, cy, r, rng):
                    px[y][x] = col
    return px


def main() -> None:
    core = sys.argv[1]
    rng = random.Random(20261006)
    cases = []
    for i in range(32):
        w, h = rng.randint(12, 40), rng.randint(10, 32)
        px = image(rng, w, h)
        strategy = "Symbol Extract" if i % 2 == 0 else "Line Extract"
        case = {
            "strategy": strategy,
            "width": w,
            "height": h,
            "rgba": [v for row in px for p in row for v in (*p, 255)],
            "target": list(rng.choice(PALETTE[1:])),
            "distance_pct": rng.choice([0.5, 1, 3, 10]),
            "params": {"min_diameter_px": rng.choice([1, 3, 5]),
                       "max_diameter_px": rng.choice([4, 8, 100])}
            if strategy == "Symbol Extract"
            else {"dx_px": rng.choice([2, 3, 10]), "dy_px": rng.choice([2, 5, 10])},
        }
        if i % 3 == 0:  # a drawn mask: yellow where examined, transparent elsewhere
            x0, x1 = sorted(rng.sample(range(w), 2))
            y0, y1 = sorted(rng.sample(range(h), 2))
            case["mask_rgba"] = [
                v for y in range(h) for x in range(w)
                for v in ((255, 255, 0, 255) if x0 <= x <= x1 and y0 <= y <= y1 else (0, 0, 0, 0))
            ]
        cases.append(case)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cases, f)
    js = REPO / "scripts/tools/starry_extract_crosscheck.mjs"
    res = subprocess.run(["node", str(js), core, f.name], capture_output=True, text=True,
                         check=True)
    expected = json.loads(res.stdout)
    for c, e in zip(cases, expected, strict=True):
        c["expected"] = e
    OUT.write_text(json.dumps({
        "source": "starry-digitizer 2.0.0-dev (a13c927) library-build/dist/core.js, "
        "run by scripts/tools/starry_extract_crosscheck.mjs",
        "cases": cases,
    }) + "\n")
    print(OUT, len(cases), "cases,", sum(len(e) for e in expected), "points")


if __name__ == "__main__":
    main()

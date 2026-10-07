"""Prepare sealed directories for 方式D with Claude as the orchestrator
(docs/design/local-model.md「方式D: 司令塔 + 道具」): the upper bound of
choosing and configuring the fixed tools. Claude may only run the provided
`tool` command and look at images and overlays (INSTRUCTIONS.md forbids
writing image-processing code and typing answer numbers).

Figures and task ids are the Claude runs' for that condition, scored figures
only, so rows compare figure for figure:

  noaxis -- the v3 noaxis tasks (data/llm_run_v3), answers in values
  pixcal -- the pixcal tasks with the person's calibration
            (data/llm_run_pixcal), answers in image pixels

Each directory is self-contained: images/, tasks.json, INSTRUCTIONS.md, the
`tool` command, and tools/ (a copy of the tool modules -- code only -- and the
detector's raw detections for these images, no labels, no ground truth).
The tools run with the system python3 (numpy, Pillow) and tesseract.

  .venv/bin/python scripts/eval/prepare_orchestrator_claude.py \\
      <work_dir> <model> <condition> parts<k>

Writes <work>/<condition>/<model>/<part>/. Answers are archived to
data/llm_run_orchestrator/<condition>/<model>/<part>.predictions.json and
scored as orch-claude-<condition>.

--v2 (検証とやり直し, docs/design/local-model.md): a new seed and new task
names (key in data/llm_run_orch2/), the single "as printed" rule in
condition 1, and the verify tool, whose verdict `answer` enforces. Archived
with archive_orchestrator_claude.py <work> --v2 and scored as
orch2-claude-<condition>.

  .venv/bin/python scripts/eval/prepare_orchestrator_claude.py \\
      ~/.cache/real-chart-bench/orchestrator/claude-sealed-v2 claude-opus-5-5 noaxis parts2 --v2

--v3 (v3: 検証の誤検知と道具の追加): as --v2 with another seed and names
(key in data/llm_run_orch3/), verify v3 (line pieces, caps, split markers
and minor-tick subdivisions no longer raise a redo), the blob_extract tool
and the v3 tick reading. Archived with archive_orchestrator_claude.py <work>
--v3 and scored as orch3-claude-<condition>.

  .venv/bin/python scripts/eval/prepare_orchestrator_claude.py \\
      ~/.cache/real-chart-bench/orchestrator/claude-sealed-v3 claude-opus-5-5 noaxis parts2 --v3
"""

from __future__ import annotations

import json
import pathlib
import random
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.orchestrator_tools import image_sha256  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

DETS = pathlib.Path.home() / ".cache/real-chart-bench/orchestrator/dets"
PROMPT = REPO / "scripts/eval/orchestrator/claude_prompt.md"
PROMPT_V2 = REPO / "scripts/eval/orchestrator/claude_prompt_v2.md"
PROMPT_V3 = REPO / "scripts/eval/orchestrator/claude_prompt_v3.md"
SOURCES = {
    "noaxis": (REPO / "data/llm_run_v3", "noaxis"),
    "pixcal": (REPO / "data/llm_run_pixcal", "pixcal"),
}
V2_SEED = 20261007
V2_ARCHIVE = REPO / "data/llm_run_orch2"
V3_SEED = 20261008
V3_ARCHIVE = REPO / "data/llm_run_orch3"
AS_PRINTED = (
    "Report numbers on the same scale as the printed tick labels. Do not apply "
    "a multiplier written in the axis title (e.g. '(10^4 S/m)' or 'x10^4'); if "
    "a tick label itself is written like 5x10^4, report 50000."
)
MODULES = [
    "domain/verification.py",
    "domain/marker_blobs.py",
    "domain/starry_extract.py",
    "domain/digitizer_tools.py",
    "domain/orchestration.py",
    "domain/marker_detection.py",
    "domain/axis_frame.py",
    "domain/tick_calibration.py",
    "adapter/orchestrator_tools.py",
    "adapter/auto_axis_calibration.py",
    "adapter/tesseract_ocr.py",
]
CAL_ROW = {
    "noaxis": '`mode`: `"auto"` (frame + tick-label OCR) or `"manual"` with `x` / `y` = '
    '`{"scale": "linear"|"log", "ticks": [[px, value], …]}`. Its result is the '
    "`calibration` of the answer.",
    "pixcal": '`mode`: `"given"` (default): the person\'s calibration and the plot frame, '
    "useful as a mask frame.",
}
ANSWER_NOTE = {
    "noaxis": "The named calibration converts the points to values.",
    "pixcal": "Points stay in image pixels.",
}


def instructions(directory: pathlib.Path, n: int, condition: str, v2: bool = False,
                 v3: bool = False) -> str:
    text = (PROMPT_V3 if v3 else PROMPT_V2 if v2 else PROMPT).read_text()
    body = text.split("\n---\n")[1].strip()
    block = text.split(f"## `{{CONDITION}}`, {condition}\n")[1].split("\n## ")[0].strip()
    return (body.replace("{CONDITION}", block).replace("{CALIBRATION_ROW}", CAL_ROW[condition])
            .replace("{ANSWER_NOTE}", ANSWER_NOTE[condition])
            .replace("{DIR}", str(directory)).replace("{N}", str(n)))


def install_tools(d: pathlib.Path, condition: str, images: list[pathlib.Path]) -> None:
    pkg = d / "tools/real_chart_bench"
    for sub in ("", "domain", "adapter"):
        (pkg / sub).mkdir(parents=True, exist_ok=True)
        (pkg / sub / "__init__.py").write_text("")
    for m in MODULES:
        shutil.copy(REPO / "src/real_chart_bench" / m, pkg / m)
    shutil.copy(REPO / "LICENSE", d / "tools/LICENSE")
    (d / "tools/config.json").write_text(json.dumps({"condition": condition}) + "\n")
    dets = d / "tools/dets"
    dets.mkdir()
    for img in images:
        sha = image_sha256(img)
        src = DETS / f"{sha}.json"
        if not src.exists():
            raise SystemExit(f"no detections for {img} -- run export_detector_cache.py")
        rec = json.loads(src.read_text())
        rec.pop("checkpoint", None)  # no paths outside the directory
        (dets / f"{sha}.json").write_text(json.dumps(rec) + "\n")
    tool = d / "tool"
    shutil.copy(REPO / "scripts/eval/orchestrator/sealed_tool.py", tool)
    tool.chmod(0o755)


def v2_tasks(condition: str, seed: int = V2_SEED,
             archive: pathlib.Path = V2_ARCHIVE) -> tuple[dict, list[dict]]:
    """The v2 run (検証とやり直し): a fresh seed and names, every scored
    figure, and the single "as printed" reporting rule (design 7.82) for
    condition 1 -- the v1 noaxis tasks still carried the old 10^tick / kelvin
    rules. Condition 2 hands over the person's tick calibration, whose values
    are in printed space already (tick_calibration.json). The key and the
    task lists are archived to data/llm_run_orch2/ (never into the sealed
    directories); building twice gives the same files."""
    from PIL import Image

    pairings = select_verified_pairings(load_registry(REPO / "data/verified_pairs/registry.json"))
    cal = {(c["paper_id"], c["figure_id"]): c for c in json.loads(
        (REPO / "data/verified_pairs/tick_calibration.json").read_text())["figures"]}
    random.Random(seed).shuffle(pairings)
    key, noaxis, pixcal = {}, [], []
    for i, p in enumerate(pairings, 1):
        name = f"fig_{i:03d}.png"
        key[name] = {"paper_id": p.paper_id, "figure_id": p.figure_id,
                     "image_path": p.image_path}
        noaxis.append({"id": name, "x_report": AS_PRINTED, "y_report": AS_PRINTED})
        c = cal[(p.paper_id, p.figure_id)]
        w, h = Image.open(REPO / p.image_path).size
        pixcal.append({"id": name, "image_size": [w, h], "x_scale": c["x_scale"],
                       "y_scale": c["y_scale"],
                       "x_ticks": [{"pixel_x": t["px"], "value": t["value"]} for t in c["x"]],
                       "y_ticks": [{"pixel_y": t["px"], "value": t["value"]} for t in c["y"]]})
    files = {archive / "_key.json": key, archive / "noaxis/tasks.json": noaxis,
             archive / "pixpts_px/tasks.json": pixcal}
    for path, obj in files.items():
        text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
        if path.exists() and path.read_text() != text:
            raise SystemExit(f"{path} differs from this build -- refusing to overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return key, noaxis if condition == "noaxis" else pixcal


def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("--v2", "--v3")]
    v3 = "--v3" in sys.argv
    v2 = "--v2" in sys.argv or v3
    work, model, condition, spec = pathlib.Path(args[0]).resolve(), *args[1:4]
    if REPO in work.parents or work == REPO:
        raise SystemExit("work dir must be outside the repository")
    scored = {
        p.figure_id
        for p in select_verified_pairings(load_registry(REPO / "data/verified_pairs/registry.json"))
    }
    if v3:
        key, tasks = v2_tasks(condition, V3_SEED, V3_ARCHIVE)
    elif v2:
        key, tasks = v2_tasks(condition)
    else:
        run, name = SOURCES[condition]
        key = json.loads((run / "_key.json").read_text())
        tasks = json.loads((run / name / "tasks.json").read_text())
    tasks = [t for t in tasks if key[t["id"]]["figure_id"] in scored]
    if not spec.startswith("parts"):
        raise SystemExit("spec must be parts<k>")
    k = int(spec[5:])
    for i in range(k):
        batch = tasks[i::k]
        d = work / condition / model / f"part{i + 1}"
        if d.exists():
            raise SystemExit(f"{d} exists -- refusing to overwrite a run in progress")
        (d / "images").mkdir(parents=True)
        for t in batch:
            shutil.copy(REPO / key[t["id"]]["image_path"], d / "images" / t["id"])
        (d / "tasks.json").write_text(json.dumps(batch, indent=2) + "\n")
        (d / "INSTRUCTIONS.md").write_text(instructions(d, len(batch), condition, v2, v3) + "\n")
        install_tools(d, condition, [d / "images" / t["id"] for t in batch])
        print(d, len(batch))


if __name__ == "__main__":
    main()

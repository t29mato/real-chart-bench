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
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.orchestrator_tools import image_sha256  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

DETS = pathlib.Path.home() / ".cache/real-chart-bench/orchestrator/dets"
PROMPT = REPO / "scripts/eval/orchestrator/claude_prompt.md"
SOURCES = {
    "noaxis": (REPO / "data/llm_run_v3", "noaxis"),
    "pixcal": (REPO / "data/llm_run_pixcal", "pixcal"),
}
MODULES = [
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


def instructions(directory: pathlib.Path, n: int, condition: str) -> str:
    text = PROMPT.read_text()
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


def main() -> None:
    work, model, condition, spec = pathlib.Path(sys.argv[1]).resolve(), *sys.argv[2:5]
    if REPO in work.parents or work == REPO:
        raise SystemExit("work dir must be outside the repository")
    run, name = SOURCES[condition]
    key = json.loads((run / "_key.json").read_text())
    scored = {
        p.figure_id
        for p in select_verified_pairings(load_registry(REPO / "data/verified_pairs/registry.json"))
    }
    tasks = [t for t in json.loads((run / name / "tasks.json").read_text())
             if key[t["id"]]["figure_id"] in scored]
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
        (d / "INSTRUCTIONS.md").write_text(instructions(d, len(batch), condition) + "\n")
        install_tools(d, condition, [d / "images" / t["id"] for t in batch])
        print(d, len(batch))


if __name__ == "__main__":
    main()

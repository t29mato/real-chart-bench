"""方式D, local orchestrator (docs/design/local-model.md「方式D: 司令塔 + 道具」):
a local VLM chooses and configures the fixed tools of
adapter/orchestrator_tools.py, looks at the overlays they draw, and finally
names the tool results that form the answer. No number of the answer is
written by the model.

All figures of a condition run in lockstep: each round, every unfinished
conversation gets one model turn (one batched vLLM call), then its tool calls
run on CPU in a process pool. At most MAX_STEPS turns per figure; a figure
that does not finish (or whose final action is unusable) gets the fixed
fallback of domain/orchestration.fallback_final.

Conditions:
  noaxis -- condition 1: the model calibrates the axes with tick_calibration
            (automatic OCR, or ticks it reads itself), values in printed
            space -> <out>/noaxis.jsonl, scored as local-orch-<run>-noaxis
  pixcal -- condition 2: the person's calibration is given; the answer is
            pixel points, converted by the scorer through the same
            calibration -> <out>/pixpts_px.jsonl, scored as local-orch-<run>-pixcal

Policies:
  vlm      -- Qwen3.5-9B (bf16) through vLLM (run under the GPU lock)
  scripted -- no model: tick_calibration (cond. 1), marker_detector with
              defaults, final with every series. A CPU check of the loop and
              the scorer path; it must reproduce 方式A / 方式C's -posta rows.

  flock /tmp/rcb-gpu.lock ~/.cache/real-chart-bench/vlm-cuda/bin/python \\
      scripts/eval/orchestrator/run_local.py --policy vlm --run qwen3.5-9b-orch
  .venv/bin/python scripts/eval/orchestrator/run_local.py --policy scripted --run orch-scripted

--dev runs the Starrydata validation figures (not the benchmark) instead,
for checking the loop; nothing is chosen on the benchmark.
"""

from __future__ import annotations

import argparse
import base64
import importlib.metadata as md
import io
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from PIL import Image  # noqa: E402

from real_chart_bench.adapter.orchestrator_tools import ToolBox, ToolError  # noqa: E402
from real_chart_bench.domain.digitizer_tools import (  # noqa: E402
    calibration_from_person,
    series_to_values,
)
from real_chart_bench.domain.orchestration import (  # noqa: E402
    MAX_RETRIES,
    MAX_STEPS,
    action_schema,
    answer_key,
    assemble_final,
    fallback_final,
    params_to_image,
    parse_action,
    result_to_view,
    retry_decision,
)
from real_chart_bench.domain.verification import Thresholds  # noqa: E402

DETS = Path.home() / ".cache/real-chart-bench/orchestrator/dets"
CACHE = Path.home() / ".cache/real-chart-bench/orchestrator"
VIEW_LONG_SIDE = 1024
MAX_TOKENS_PER_TURN = 768
MAX_MODEL_LEN = 16384
MODEL = "Qwen/Qwen3.5-9B"
TOOLS_BY_COND = {
    "noaxis": ["marker_detector", "dominant_colors", "symbol_extract", "line_extract", "mask",
               "tick_calibration", "render_overlay"],
    "pixcal": ["marker_detector", "dominant_colors", "symbol_extract", "line_extract", "mask",
               "tick_calibration", "render_overlay"],
}
CAL_TOOL = {
    "noaxis": (
        '- tick_calibration {mode: "auto"}: finds the plot frame and reads the tick labels '
        "(OCR). If ok is false, read the tick labels in the image yourself and call "
        '{mode: "manual", x: {scale: "linear" or "log", ticks: [[px, value], ...]}, '
        "y: {...}}, using tick mark positions from its tick_marks (x ticks: px is the x "
        "pixel; y ticks: px is the y pixel)."
    ),
    "pixcal": (
        '- tick_calibration {mode: "given"}: the axis calibration a person made, with the '
        "plot frame; its id works as a mask frame."
    ),
}
CONDITION_TEXT = {
    "noaxis": (
        "The axes are not calibrated: the final action must name a successful "
        "tick_calibration result as calibration; the answer's values are converted through "
        "it, in the numbers printed on the tick labels. {REPORT}"
    ),
    "pixcal": (
        "A person has calibrated the axes: the answer is converted to values through their "
        'calibration, so the final action\'s calibration can be "".'
    ),
}
# scripted policy with --verify: what a redo tries next (fixed in advance)
SCRIPT_LADDER = [{}, {"long_side": 1280}, {"threshold": 0.25}]
VERIFY_TEXT = (
    "Your final action is checked by verify: are the points on markers, are there "
    "marker-like places with no point, do the detector's peaks agree, and (condition 1) do "
    "the tick labels fit the calibration. If verify says redo, you get its reasons: fix "
    "them with a different tool, setting or mask and give a new final action. After {R} "
    "redos the best-checked final is answered."
)
PLAN = {
    "noaxis": "(1) tick_calibration; (2) marker_detector with defaults; (3) render_overlay "
    "of both; (4) adjust if needed; (5) final.",
    "pixcal": "(1) marker_detector with defaults; (2) render_overlay of it; (3) adjust if "
    "needed; (4) final.",
}


# ------------------------------------------------------------------ tasks


def scored_ids() -> set[str]:
    from real_chart_bench.adapter.verified_pairing_registry import load_registry
    from real_chart_bench.usecase.real_image_gate import select_verified_pairings

    return {p.figure_id for p in select_verified_pairings(
        load_registry(REPO / "data/verified_pairs/registry.json"))}


def bench_tasks(cond: str) -> list[dict]:
    scored = scored_ids()
    if cond == "noaxis":
        run = REPO / "data/llm_run_v3"
        key = json.loads((run / "_key.json").read_text())
        tasks = json.loads((run / "noaxis/tasks.json").read_text())
        extra = {t["id"]: {"report": t.get("y_report", "")} for t in tasks}
        cal = {}
    else:
        run = REPO / "data/llm_run_pixcal"
        key = json.loads((run / "_key.json").read_text())
        tasks = json.loads((run / "pixpts_px/tasks.json").read_text())
        cal = {t["id"]: calibration_from_person(t)
               for t in json.loads((run / "pixcal/tasks.json").read_text())}
        extra = {}
    out = []
    for t in tasks:
        k = key[t["id"]]
        if k["figure_id"] not in scored:
            continue
        out.append({"fig": t["id"], "image": REPO / k["image_path"],
                    "image_path": k["image_path"],
                    "paper_figure": f"{k['paper_id']}-{k['figure_id']}",
                    "calibration": cal.get(t["id"]), **extra.get(t["id"], {})})
    return out


def dev_tasks(cond: str, n: int) -> list[dict]:
    """Starrydata validation figures (papers outside the benchmark): the loop
    is checked here. The person's calibration of condition 2 is stood in for
    by the label's own ticks."""
    root = Path.home() / ".cache/real-chart-bench/train-data/starrydata"
    sys.path.insert(0, str(REPO / "scripts/train/marker_detector"))
    from real_chart_bench.domain.marker_detection import is_validation, split_key

    out = []
    for line in (root / "labels.jsonl").read_text().splitlines():
        lab = json.loads(line)
        if not is_validation(split_key(lab), 0.2):
            continue
        ax = lab["axes"]
        cal = {a: {"scale": ax[a]["scale"], "ticks": [[t["px"], t["value"]] for t in
                                                      ax[a]["ticks"]]} for a in ("x", "y")}
        out.append({"fig": Path(lab["image"]).name, "image": root / lab["image"],
                    "image_path": lab["image"], "paper_figure": f"dev-{lab['paper_id']}",
                    "calibration": {**cal, "frame": None, "source": "label"} if cond == "pixcal"
                    else None, "report": ""})
    return out[:n]


# ------------------------------------------------------------------ tools (CPU pool)


def _exec(job: dict) -> dict:
    t0 = time.time()
    results = job["results"]

    def resolve(ref):
        if ref not in results:
            raise ToolError(f"no result {ref!r}")
        return results[ref]

    tb = ToolBox(Path(job["image"]), dets_dir=DETS, given_calibration=job["calibration"],
                 allow_auto_calibration=job["cond"] == "noaxis", out_dir=Path(job["out_dir"]))
    try:
        res = tb.run(job["tool"], job["params"], resolve)
        err = None
    except (ToolError, ValueError, TypeError, KeyError) as exc:
        res, err = None, f"{type(exc).__name__}: {exc}"
    return {"result": res, "error": err, "seconds": time.time() - t0}


# ------------------------------------------------------------------ conversations


@dataclass
class Conv:
    task: dict
    cond: str
    view_scale: float
    view_size: tuple[int, int]
    turns: list = field(default_factory=list)  # (role, text, image path | None)
    results: dict = field(default_factory=dict)
    order: list = field(default_factory=list)
    log: list = field(default_factory=list)
    steps: int = 0
    done: bool = False
    final: tuple | None = None
    finish: str = ""
    prompt_tokens: int = 0
    generation_tokens: int = 0
    model_seconds: float = 0.0
    tool_seconds: float = 0.0
    verify_on: bool = False
    attempts: list = field(default_factory=list)  # verified finals (検証とやり直し)


def system_prompt(cond: str, conv: Conv, report: str) -> str:
    text = (HERE / "prompt.md").read_text()
    if conv.verify_on:
        text = text.rstrip("\n") + "\n\n" + VERIFY_TEXT.replace("{R}", str(MAX_RETRIES)) + "\n"
    return (text.replace("{VIEW_W}", str(conv.view_size[0]))
            .replace("{VIEW_H}", str(conv.view_size[1]))
            .replace("{CALIBRATION_TOOL}", CAL_TOOL[cond])
            .replace("{CONDITION}", CONDITION_TEXT[cond].replace("{REPORT}", report))
            .replace("{PLAN}", PLAN[cond])
            .replace("{MAX_STEPS}", str(MAX_STEPS)))


def view_image(path: Path, scale: float) -> str:
    im = Image.open(path).convert("RGB")
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def messages_for(conv: Conv) -> list[dict]:
    """The chat so far; only the figure and the latest overlay keep their image."""
    last_img = max((i for i, t in enumerate(conv.turns) if t[2] and i > 0), default=None)
    msgs = []
    for i, (role, text, img) in enumerate(conv.turns):
        if role == "assistant":
            msgs.append({"role": "assistant", "content": text})
            continue
        content = []
        if img and (i == 0 or i == last_img):
            content.append({"type": "image_url",
                            "image_url": {"url": view_image(Path(img), conv.view_scale)}})
        elif img:
            text = text + " (that overlay image is no longer shown)"
        content.append({"type": "text", "text": text})
        msgs.append({"role": "user", "content": content})
    return msgs


def scripted_turn(conv: Conv) -> str:
    """No model: calibrate (cond. 1), detector with defaults, final with all.
    With --verify, a redo runs the next detector setting of SCRIPT_LADDER."""
    base = {"thought": "", "params": {}, "calibration": "", "series": []}
    ids = conv.order
    if conv.cond == "noaxis" and not ids:
        return json.dumps({**base, "action": "call", "tool": "tick_calibration",
                           "params": {"mode": "auto"}})
    pts_ids = [r for r in ids if conv.results[r].get("kind") == "points"]
    redo = conv.verify_on and len(conv.attempts) >= len(pts_ids)
    if not pts_ids or (redo and len(pts_ids) < len(SCRIPT_LADDER)):
        return json.dumps({**base, "action": "call", "tool": "marker_detector",
                           "params": SCRIPT_LADDER[len(pts_ids)]})
    pts = pts_ids[-1]
    cal = ids[0] if conv.cond == "noaxis" else ""
    return json.dumps({**base, "action": "final", "tool": "", "calibration": cal,
                       "series": [{"from": pts, "index": -1, "label": ""}]})


def finish(conv: Conv, how: str, picked) -> None:
    conv.done, conv.finish, conv.final = True, how, picked


def apply_turn(conv: Conv, text: str, tools: list[str]) -> dict | None:
    """Parse one model turn. Returns a tool job, or None (finished, or an
    error message was queued for the model)."""
    conv.steps += 1
    conv.turns.append(("assistant", text, None))
    need_cal = conv.cond == "noaxis"
    try:
        act = parse_action(text, tools)
    except ValueError as exc:
        conv.log.append({"step": conv.steps, "raw": text[:2000], "error": str(exc)})
        conv.turns.append(("user", f"Error: {exc}. Reply with one JSON action.", None))
        return None
    if act.kind == "final":
        try:
            picked = assemble_final(act, conv.results, need_cal)
            conv.log.append({"step": conv.steps, "final": {"calibration": act.calibration,
                                                           "series": list(act.series)},
                             "thought": act.thought})
            if conv.verify_on:  # verify decides: accept, or redo (検証とやり直し)
                cal = act.calibration if need_cal else ""
                return {"rid": None, "tool": "verify", "final": picked,
                        "key": answer_key(act.series, cal),
                        "params": {"series": [{"from": s["from"], "index": s["index"]}
                                              for s in act.series],
                                   **({"calibration": cal} if cal else {})}}
            finish(conv, "final", picked)
        except ValueError as exc:
            conv.log.append({"step": conv.steps, "final_error": str(exc),
                             "series": list(act.series), "calibration": act.calibration})
            conv.turns.append(("user", f"Error in final action: {exc}.", None))
        return None
    params = params_to_image(act.tool, act.params, conv.view_scale)
    rid = f"r{len(conv.order) + 1}"
    conv.log.append({"step": conv.steps, "thought": act.thought, "call": act.tool,
                     "params_view": act.params, "id": rid})
    return {"rid": rid, "tool": act.tool, "params": params}


def receive_verify(conv: Conv, job: dict, out: dict) -> None:
    """A final action came back from verify: the retry rules decide."""
    entry = conv.log[-1]
    if out["error"]:  # verify itself failed: the final stands as before
        entry["verify_error"] = out["error"]
        finish(conv, "final", job["final"])
        return
    v = out["result"]["verdict"]
    conv.attempts.append({"key": job["key"], "accept": v["accept"], "score": v["score"],
                          "picked": job["final"]})
    dec = retry_decision(conv.attempts)
    entry["verify"] = {"accept": v["accept"], "score": v["score"], "reasons": v["reasons"],
                       "hints": v["hints"], "decision": dec["action"]}
    if dec["action"] == "accept":
        finish(conv, "final", conv.attempts[dec["pick"]]["picked"])
    elif dec["action"] == "best":
        finish(conv, "final-best", conv.attempts[dec["pick"]]["picked"])
    else:
        view = result_to_view(out["result"], conv.view_scale)
        conv.turns.append(("user", (
            f"verify: redo. {'; '.join(v['reasons'])}. What looks wrong: "
            f"{'; '.join(v['hints']) or 'nothing specific'}. Places that look like markers but "
            f"have no point (view pixels): {view['missed_places'][:10]}. Fix it with a "
            "different tool, setting or mask, then give a new final action "
            f"({dec['retries_left'] + 1} redos left)."), None))


def best_attempt(conv: Conv):
    i = max(range(len(conv.attempts)), key=lambda k: (conv.attempts[k]["score"], -k))
    return conv.attempts[i]["picked"]


def receive(conv: Conv, job: dict, out: dict) -> None:
    conv.tool_seconds += out["seconds"]
    if job.get("final") is not None:
        receive_verify(conv, job, out)
        return
    rid = job["rid"]
    entry = conv.log[-1]
    if out["error"]:
        entry["error"] = out["error"]
        conv.turns.append(("user", f"Error from {job['tool']}: {out['error']}", None))
        return
    res = out["result"]
    conv.results[rid] = res
    conv.order.append(rid)
    view = result_to_view(res, conv.view_scale)
    entry["result_view"] = view
    text = f"Result {rid} ({job['tool']}): {json.dumps(view)}"[:3000]
    img = res.get("path") if res.get("kind") == "overlay" else None
    conv.turns.append(("user", text, img))


def close_turn(conv: Conv) -> None:
    left = MAX_STEPS - conv.steps
    if conv.done:
        return
    if left <= 0:
        conv.log.append({"step": conv.steps, "cap": True})
        if conv.attempts:  # verified finals exist: the best-scoring one
            finish(conv, "final-best", best_attempt(conv))
            return
        picked = fallback_final(conv.results, conv.order, conv.cond == "noaxis")
        finish(conv, "fallback", picked)
        return
    role, text, img = conv.turns[-1]
    note = (" This is your last turn: give the final action now." if left == 1
            else f" ({left} turns left.)")
    conv.turns[-1] = (role, text + note, img)


# ------------------------------------------------------------------ engine


class VlmEngine:
    def __init__(self) -> None:
        from huggingface_hub import snapshot_download
        from vllm import LLM

        self.path = snapshot_download(MODEL)
        t0 = time.time()
        self.llm = LLM(model=self.path, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
                       gpu_memory_utilization=0.92, limit_mm_per_prompt={"image": 2},
                       seed=0, enforce_eager=True)
        self.load_seconds = time.time() - t0

    def env(self) -> dict:
        import torch

        return {"repo_id": MODEL, "revision": Path(self.path).name, "vllm": md.version("vllm"),
                "torch": torch.__version__, "transformers": md.version("transformers"),
                "gpu": torch.cuda.get_device_name(0), "precision": "bf16",
                "load_seconds": round(self.load_seconds, 1), "max_model_len": MAX_MODEL_LEN,
                "max_tokens_per_turn": MAX_TOKENS_PER_TURN, "temperature": 0.0,
                "enable_thinking": False, "constrained_json_schema": True,
                "enforce_eager": True, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")}

    def generate(self, convs: list[Conv], systems: list[str], schema: dict):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        msgs = [[{"role": "system", "content": s}, *messages_for(c)]
                for c, s in zip(convs, systems, strict=True)]
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS_PER_TURN,
                            structured_outputs=StructuredOutputsParams(json=schema))
        res = self.llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False},
                            use_tqdm=False)
        return [(r.outputs[0].text, len(r.prompt_token_ids or []), len(r.outputs[0].token_ids))
                for r in res]


# ------------------------------------------------------------------ run


def run_condition(cond: str, tasks: list[dict], engine, pool, out_dir: Path, policy: str,
                  verify: bool = False) -> list[dict]:
    tools = TOOLS_BY_COND[cond]
    schema = action_schema(tools)
    convs = []
    for t in tasks:
        w, h = Image.open(t["image"]).size
        s = min(1.0, VIEW_LONG_SIDE / max(w, h))
        c = Conv(task=t, cond=cond, view_scale=s, view_size=(round(w * s), round(h * s)),
                 verify_on=verify)
        c.turns.append(("user", f"Task {t['fig']}. Find the data points of every marker "
                        "series in the main plot.", str(t["image"])))
        convs.append(c)
    systems = {id(c): system_prompt(cond, c, c.task.get("report", "")) for c in convs}
    rnd = 0
    while True:
        active = [c for c in convs if not c.done]
        if not active:
            break
        rnd += 1
        t0 = time.time()
        if policy == "vlm":
            outs = engine.generate(active, [systems[id(c)] for c in active], schema)
        else:
            outs = [(scripted_turn(c), 0, 0) for c in active]
        share = (time.time() - t0) / len(active)
        jobs = []
        for c, (text, n_prompt, n_gen) in zip(active, outs, strict=True):
            c.model_seconds += share
            c.prompt_tokens += n_prompt
            c.generation_tokens += n_gen
            job = apply_turn(c, text, tools)
            if job is not None:
                jobs.append((c, job))
        payloads = [{"image": str(c.task["image"]), "calibration": c.task["calibration"],
                     "cond": cond, "tool": j["tool"], "params": j["params"],
                     "results": c.results,
                     "out_dir": str(CACHE / out_dir.name / "overlays" / cond / c.task["fig"])}
                    for c, j in jobs]
        for (c, j), o in zip(jobs, pool.map(_exec, payloads), strict=True):
            receive(c, j, o)
        for c in active:
            close_turn(c)
        n_done = sum(c.done for c in convs)
        print(f"[{cond}] round {rnd}: {len(active)} active, {len(jobs)} tool calls, "
              f"{time.time() - t0:.0f}s, finished {n_done}/{len(convs)}", flush=True)
    recs = []
    tdir = out_dir / "transcripts" / cond
    tdir.mkdir(parents=True, exist_ok=True)
    for c in convs:
        t = c.task
        rec = {"fig": t["fig"], "condition": "noaxis" if cond == "noaxis" else "pixpts_px",
               "paper_figure": t["paper_figure"], "image_path": t["image_path"], "error": None,
               "parsed": None, "parse_error": None, "finish": c.finish, "n_steps": c.steps,
               "tools_called": [e.get("call") for e in c.log if e.get("call")],
               "prompt_tokens": c.prompt_tokens, "generation_tokens": c.generation_tokens,
               "seconds": round(c.model_seconds + c.tool_seconds, 2),
               "model_seconds": round(c.model_seconds, 2),
               "tool_seconds": round(c.tool_seconds, 2),
               "verify": [{k: a[k] for k in ("key", "accept", "score")} for a in c.attempts]}
        if c.final is None:
            rec["parse_error"] = "no answer: no points result" + (
                " or calibration" if cond == "noaxis" else "")
        else:
            series, cal = c.final
            if cond == "noaxis":
                rec["parsed"] = series_to_values(series, cal)
                rec["calibration"] = cal
            else:
                rec["parsed"] = series
        recs.append(rec)
        (tdir / f"{t['fig']}.json").write_text(json.dumps(
            {"fig": t["fig"], "paper_figure": t["paper_figure"], "finish": c.finish,
             "log": c.log}, indent=1, default=str) + "\n")
    name = "noaxis.jsonl" if cond == "noaxis" else "pixpts_px.jsonl"
    (out_dir / name).write_text("".join(json.dumps(r) + "\n" for r in recs))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["vlm", "scripted"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--conditions", default="noaxis,pixcal")
    ap.add_argument("--dev", type=int, default=0, help="n Starrydata validation figures")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--verify", action="store_true",
                    help="check each final with verify; its verdict decides redo (検証とやり直し)")
    args = ap.parse_args()
    root = CACHE / "dev-runs" if args.dev else REPO / "data/local_orchestrator_runs"
    out_dir = root / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"))
    engine = VlmEngine() if args.policy == "vlm" else None
    env = {
        "kind": "orchestrator",
        "policy": args.policy,
        "display_name": ("Qwen3.5-9B 司令塔 + 道具 (方式D)" if args.policy == "vlm"
                         else "固定手順: 検出器 + 自動校正 (方式D の確認用)"),
        "max_steps": MAX_STEPS,
        "view_long_side": VIEW_LONG_SIDE,
        "tools": TOOLS_BY_COND,
        "detector_cache": "方式A (b) weights, raw detections exported by "
        "scripts/tools/export_detector_cache.py (CPU run)",
        "python": platform.python_version(),
        "cpu_tool_workers": args.workers,
        "prompt": "scripts/eval/orchestrator/prompt.md",
        "verify": ({"thresholds": Thresholds().as_dict(), "max_retries": MAX_RETRIES,
                    "prompt_addition": VERIFY_TEXT} if args.verify else None),
        **(engine.env() if engine else {}),
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    code = 0
    try:
        for cond in args.conditions.split(","):
            tasks = dev_tasks(cond, args.dev) if args.dev else bench_tasks(cond)
            if args.limit:
                tasks = tasks[: args.limit]
            t0 = time.time()
            recs = run_condition(cond, tasks, engine, pool, out_dir, args.policy, args.verify)
            fin = {k: sum(r["finish"] == k for r in recs)
                   for k in ("final", "final-best", "fallback")}
            print(f"[{cond}] {len(recs)} figures in {time.time() - t0:.0f}s, {fin}, "
                  f"answered {sum(r['parsed'] is not None for r in recs)}", flush=True)
        print(f"書き出し: {out_dir}", flush=True)
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        pool.shutdown(cancel_futures=True)
        shutdown_engine(engine)
    os._exit(code)  # vLLM's engine process can keep the interpreter waiting


def shutdown_engine(engine) -> None:
    """Stop vLLM's EngineCore before the hard exit. os._exit alone left the
    engine process orphaned holding the GPU (2026-10-06, 23 GB, after the
    GPU lock was released) -- the next lock holder could not start."""
    if engine is not None:
        try:
            engine.llm.llm_engine.engine_core.shutdown()
        except Exception as exc:  # noqa: BLE001
            print(f"engine shutdown: {exc}", flush=True)
    for p in mp.active_children():
        p.terminate()
        p.join(10)
        if p.is_alive():
            p.kill()
    try:  # anything else this process started (vLLM may use its own launcher)
        import psutil

        for c in psutil.Process().children(recursive=True):
            c.kill()
    except ImportError:
        pass


if __name__ == "__main__":
    main()

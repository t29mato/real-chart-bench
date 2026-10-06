"""QLoRA fine-tuning of Qwen3.5-9B for approach B (docs/design/local-model.md,
方式B): image + v3 single-shot prompt -> the JSON answer the benchmark scores.

- base weights in 4-bit NF4 (bitsandbytes, double quant, bf16 compute); the
  vision tower stays frozen and unquantized, LoRA on the language model only
  (attention q/k/v/o, gated-delta-net in_proj_qkv/in_proj_z/out_proj, MLP),
  module names vLLM 0.30 can serve as a LoRA on the bf16 base
- loss on the answer tokens only (prompt and image masked)
- images capped at MAX_PIXELS (aspect kept); the inference worker applies the
  same cap (worker_ft.py imports cap_image from here)
- resumable and time-sliced: the GPU is shared through flock, so each call
  trains for at most --wall-minutes, saves adapter + optimizer + scheduler +
  step and exits; run_train.sh loops, releasing the lock between slices. A
  fixed seed fixes the data order, so a resumed run continues where it was.

Usage: python train.py --data DIR [DIR ...] --out RUN_DIR [options]
Exit code 0 = finished, 3 = slice over (call again).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAX_PIXELS = 1_600_000
SLICE_EXIT = 3
LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj"
    r"|gate_proj|up_proj|down_proj)"
)


def cap_image(img, max_pixels: int = MAX_PIXELS):
    """RGB image with at most max_pixels pixels, aspect ratio kept."""
    img = img.convert("RGB")
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), resample=3)  # bicubic


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, type=pathlib.Path)
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="sampling weight per --data dir (default: proportional to size)")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--steps", type=int, default=1000, help="optimizer steps")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-target-tokens", type=int, default=3000)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--pixcal-fraction", type=float, default=0.3)
    ap.add_argument("--val-fraction", type=float, default=0.03)
    ap.add_argument("--val-n", type=int, default=48, help="val examples for the loss")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--wall-minutes", type=float, default=60)
    ap.add_argument("--limit-per-dir", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-adapter", type=pathlib.Path, default=None,
                    help="start from this adapter instead of a fresh LoRA")
    return ap.parse_args(argv)


def build_order(examples, dirs, weights, n_needed, seed):
    """Deterministic sampling order of training example indices."""
    rng = random.Random(seed)
    by_dir = {str(d): [i for i, e in enumerate(examples) if e["image"].startswith(str(d) + "/")]
              for d in dirs}
    by_dir = {k: v for k, v in by_dir.items() if v}
    if weights is None:
        weights = [len(v) for v in by_dir.values()]
    pools = {k: [] for k in by_dir}
    order = []
    keys = list(by_dir)
    for _ in range(n_needed):
        k = rng.choices(keys, weights=weights[: len(keys)])[0]
        if not pools[k]:
            pools[k] = by_dir[k][:]
            rng.shuffle(pools[k])
        order.append(pools[k].pop())
    return order


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    import torch
    from examples import load_examples
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model
    from PIL import Image
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"step": 0}
    if state.get("done"):
        print("already finished", flush=True)
        return 0

    examples, skipped = load_examples(args.data, pixcal_fraction=args.pixcal_fraction,
                                      val_fraction=args.val_fraction,
                                      limit_per_dir=args.limit_per_dir)
    train = [e for e in examples if e["split"] == "train"]
    val = [e for e in examples if e["split"] == "val"]
    random.Random(1).shuffle(val)
    val = val[: args.val_n]
    print(f"examples train={len(train)} val={len(val)} skipped={dict(skipped)}", flush=True)
    if not state_path.exists():
        (out / "config.json").write_text(json.dumps(
            {**{k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
             "data": [str(d) for d in args.data], "n_train": len(train), "n_val": len(val),
             "skipped": dict(skipped), "lora_targets": LORA_TARGETS}, indent=2) + "\n")
        (out / "val_keys.json").write_text(json.dumps(
            sorted(e["key"] for e in examples if e["split"] == "val")) + "\n")

    path = snapshot_download(args.model)
    processor = AutoProcessor.from_pretrained(path)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # re.match from the start of the module name (transformers 5)
        llm_int8_skip_modules=["model.visual", "lm_head", "model.language_model.embed_tokens"],
    )
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        path, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    adapter_dir = out / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    elif args.init_adapter is not None:
        # continue from another run's adapter (fresh optimizer and schedule)
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
            target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
        ))
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"loaded in {time.time() - t0:.0f}s; trainable params {n_train_params / 1e6:.1f}M; "
          f"mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
    if (out / "optim.pt").exists():
        ck = torch.load(out / "optim.pt", map_location="cuda", weights_only=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])

    order = build_order(train, args.data, args.weights, args.steps * args.accum, args.seed)

    def encode(ex):
        img = cap_image(Image.open(ex["image"]), args.max_pixels)
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": ex["prompt"]}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True,
                                               enable_thinking=False, tokenize=False)
        full = prompt + ex["target"] + "<|im_end|>\n"
        enc = processor(text=[full], images=[img], return_tensors="pt")
        n_prompt = processor(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        ids = enc["input_ids"]
        if ids.shape[1] - n_prompt > args.max_target_tokens:
            return None
        labels = ids.clone()
        labels[:, :n_prompt] = -100
        labels[:, -1] = -100  # trailing newline after <|im_end|>
        enc["labels"] = labels
        return {k: v.to("cuda") for k, v in enc.items()}

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for ex in val:
                b = encode(ex)
                if b is None:
                    continue
                tot += model(**b).loss.item()
                n += 1
        model.train()
        return tot / max(n, 1)

    def save(step, extra=None):
        model.save_pretrained(str(adapter_dir))
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, out / "optim.pt")
        st = {"step": step, "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
              "train_seconds": round(state.get("train_seconds", 0) + time.time() - t_slice, 1)}
        st.update(extra or {})
        state_path.write_text(json.dumps(st, indent=2) + "\n")
        return st

    model.train()
    step = state["step"]
    log = (out / "log.jsonl").open("a")
    if step == 0 and val:
        vl = val_loss()
        log.write(json.dumps({"step": 0, "val_loss": vl}) + "\n")
        log.flush()
        print(f"step 0 val_loss {vl:.4f}", flush=True)
    t_slice = time.time()
    deadline = t_start + args.wall_minutes * 60
    while step < args.steps:
        ts = time.time()
        loss_sum, n_tok_sum, n_ok = 0.0, 0, 0
        for j in range(args.accum):
            ex = train[order[step * args.accum + j]]
            b = encode(ex)
            if b is None:
                continue
            loss = model(**b).loss / args.accum
            loss.backward()
            loss_sum += loss.item()
            n_tok_sum += int(b["input_ids"].shape[1])
            n_ok += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "loss": loss_sum * args.accum / max(n_ok, 1),
               "lr": sched.get_last_lr()[0],
               "sec": round(time.time() - ts, 1), "tokens": n_tok_sum,
               "mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if step % args.save_every == 0 or step == args.steps:
            if val:
                rec["val_loss"] = val_loss()
            save(step)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if time.time() > deadline and step < args.steps:
            save(step)
            print(f"slice over at step {step}", flush=True)
            return SLICE_EXIT
    save(step, {"done": True})
    print("finished", flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
        os._exit(code)

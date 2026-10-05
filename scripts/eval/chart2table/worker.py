"""Chart-to-table inference worker for chart models trained on synthetic charts
(design §7.68, §7.75 (1)). Runs under its own venv, never this package's.

Loads one model once, then for every image in --manifest (JSONL of
{"figure_id", "image_path", "image_key"}) appends one record with the model's
raw text output to --output. Figures already in --output are skipped, so a run
resumes where it stopped. One progress line per figure on stdout.

Prompts and generation settings are each model's documented ones (model card,
or the authors' demo for TinyChart); they are written into every record so the
result file says exactly what was asked.
"""

import argparse
import json
import sys
import time
import traceback

import torch
from PIL import Image

SPECS = {
    "deplot": {
        "repo": "google/deplot",
        "prompt": "Generate underlying data table of the figure below:",
        "generate": {"max_new_tokens": 512},
        # The HF processor's default is 2048 patches. With 4096 this
        # environment reproduces the paper's PlotQA numbers (RNSS 97.5 / RMS-F1
        # 94.3 on 100 test charts vs 97.1 / 94.2 published); with 2048 it gets
        # 93.4 / 84.0. So 4096 is the setting the paper's numbers come from.
        "processor": {"max_patches": 4096},
        "source": "model card prompt (google/deplot); max_patches 4096 (design 7.75)",
    },
    "unichart": {
        "repo": "ahmed-masry/unichart-base-960",
        "prompt": "<extract_data_table> <s_answer>",
        "generate": {"num_beams": 4, "early_stopping": True},
        "source": "model card task table (ahmed-masry/unichart-base-960)",
    },
    "chartgemma": {
        "repo": "ahmed-masry/chartgemma",
        "prompt": "Generate the underlying data table of the chart.",
        "generate": {"num_beams": 4, "max_new_tokens": 512},
        "source": (
            "generation settings from the model card; the card gives no chart-to-table "
            "prompt, so this one is ours"
        ),
    },
    "granite-vision": {
        "repo": "ibm-granite/granite-vision-4.1-4b",
        "prompt": "<chart2csv>",
        "generate": {"max_new_tokens": 4096, "do_sample": False},
        "source": "model card task tag (ibm-granite/granite-vision-4.1-4b)",
        # the model card's own loading code needs the repository's modeling
        # code; IBM's official repo, run in an isolated venv
        "trust_remote_code": True,
    },
    "tinychart": {
        "repo": "mPLUG/TinyChart-3B-768",
        "prompt": "Generate underlying data table for the chart.",
        "generate": {"max_new_tokens": 1024, "temperature": 0, "conv_mode": "phi"},
        "source": "authors' demo app (X-PLUG/mPLUG-DocOwl TinyChart/app.py)",
    },
}


def _fmt(secs: float) -> str:
    secs = int(round(secs))
    return f"{secs // 60}m{secs % 60:02d}s"


class Runner:
    def __init__(self, name: str, spec: dict, device: str):
        self.name, self.spec, self.device = name, spec, device
        repo = spec["repo"]
        if name == "deplot":
            from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

            self.processor = Pix2StructProcessor.from_pretrained(repo)
            self.model = Pix2StructForConditionalGeneration.from_pretrained(repo).to(device)
        elif name == "unichart":
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            self.processor = DonutProcessor.from_pretrained(repo)
            self.model = VisionEncoderDecoderModel.from_pretrained(repo).to(device)
        elif name == "chartgemma":
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(repo)
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                repo, torch_dtype=torch.float16
            ).to(device)
        elif name == "granite-vision":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                repo, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
        elif name == "tinychart":
            # The authors' README: the vision tower comes from the
            # TinyChart-3B-768-siglip repo, so a local copy of the model with
            # config.json's mm_vision_tower pointed at it (weights linked, not
            # changed) is passed in TINYCHART_PATH.
            import os

            from tinychart.mm_utils import get_model_name_from_path
            from tinychart.model.builder import load_pretrained_model

            repo = os.environ.get("TINYCHART_PATH", repo)

            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    repo, model_base=None, model_name=get_model_name_from_path(repo), device=device
                )
            )
        self.model.eval()

    @torch.inference_mode()
    def run(self, image_path: str, prompt: str | None = None) -> str:
        """`prompt` overrides the model's standard prompt -- for reproducing a
        published number on test items that carry their own (TinyChart)."""
        prompt, gen = prompt or self.spec["prompt"], self.spec["generate"]
        if self.name == "tinychart":
            from tinychart.eval.run_tiny_chart import inference_model

            return inference_model(
                [image_path],
                prompt,
                self.model,
                self.tokenizer,
                self.image_processor,
                self.context_len,
                conv_mode=gen["conv_mode"],
                temperature=gen["temperature"],
                max_new_tokens=gen["max_new_tokens"],
            )
        image = Image.open(image_path).convert("RGB")
        if self.name == "deplot":
            inputs = self.processor(
                images=image, text=prompt, return_tensors="pt", **self.spec.get("processor", {})
            ).to(self.device)
            out = self.model.generate(**inputs, **gen)
            return self.processor.decode(out[0], skip_special_tokens=True)
        if self.name == "unichart":
            tok = self.processor.tokenizer
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
            pixels = self.processor(image, return_tensors="pt").pixel_values
            out = self.model.generate(
                pixels.to(self.device),
                decoder_input_ids=ids.to(self.device),
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
                bad_words_ids=[[tok.unk_token_id]],
                **gen,
            )
            seq = self.processor.batch_decode(out)[0]
            seq = seq.replace(tok.eos_token, "").replace(tok.pad_token, "")
            return seq.split("<s_answer>")[-1].strip()
        if self.name == "chartgemma":
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            n = inputs["input_ids"].shape[1]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model.generate(**inputs, **gen)
            return self.processor.batch_decode(
                out[:, n:], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        if self.name == "granite-vision":
            conv = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ]
            text = self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
                self.model.device
            )
            out = self.model.generate(**inputs, use_cache=True, **gen)
            n = inputs["input_ids"].shape[1]
            return self.processor.batch_decode(out[:, n:], skip_special_tokens=True)[0]
        raise ValueError(self.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(SPECS))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    spec = SPECS[args.model]

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    done = set()
    try:
        with open(args.output) as f:
            done = {json.loads(line)["figure_id"] for line in f if line.strip()}
    except FileNotFoundError:
        pass
    todo = [m for m in manifest if m["figure_id"] not in done]
    print(
        f"[start] {args.model}: {len(manifest)} figures, {len(done)} done, {len(todo)} to run; "
        f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}",
        flush=True,
    )
    t_load = time.time()
    runner = Runner(args.model, spec, args.device)
    print(f"[model] loaded in {time.time() - t_load:.1f}s", flush=True)

    import transformers

    t_run = time.time()
    with open(args.output, "a") as out:
        for i, item in enumerate(todo, 1):
            t0 = time.time()
            record = {
                "figure_id": item["figure_id"],
                "image_key": item["image_key"],
                "model": args.model,
                "repo": spec["repo"],
                "prompt": spec["prompt"],
                "generate": spec["generate"],
                "processor": spec.get("processor", {}),
                "prompt_source": spec["source"],
                "trust_remote_code": spec.get("trust_remote_code", False),
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "raw_text": None,
                "error": None,
            }
            try:
                record["raw_text"] = runner.run(item["image_path"])
            except Exception:  # noqa: BLE001 -- one figure must not kill the run
                record["error"] = traceback.format_exc()
            record["seconds"] = round(time.time() - t0, 3)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            eta = (time.time() - t_run) / i * (len(todo) - i)
            status = "ok" if record["error"] is None else "ERROR"
            n_chars = len(record["raw_text"] or "")
            print(
                f"[{len(done) + i}/{len(manifest)}] {item['figure_id']} {status} "
                f"{n_chars} chars {record['seconds']:.1f}s eta {_fmt(eta)}",
                flush=True,
            )
            if record["error"]:
                print(record["error"].rstrip().splitlines()[-1], flush=True)
    print(f"[done] {len(todo)} figures in {_fmt(time.time() - t_run)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

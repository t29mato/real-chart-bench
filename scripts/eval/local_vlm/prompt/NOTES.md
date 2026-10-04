# Single-shot prompt vs scripts/eval/llm_run_v2_prompt.md

Kept verbatim: the extraction instructions (series definition, insets ignored), the three per-series
rules (markers / lines / joining lines + error bars), the {CONDITION} texts, the per-series JSON shape
{"label", "x", "y"}, "plain numbers, x and y of equal length", "still give your best attempt".

Removed (agent-only): work directory / "do not read outside {DIR}" / tasks.json & images/ directory,
"You may use any method ... write and run Python", "Write {DIR}/predictions.json", "Rewrite the file
every few figures", "check the file parses and has all {N} keys", final-reply MODEL line,
noaxis {AXIS_NOTES} (separate axis_notes.json file; it was not scored).

Changed: one image per call; the task entry (the same object that was in the agent's tasks.json:
id + x_range/y_range/x_scale/y_scale, or id + x_report/y_report) is pasted inline after "Task:";
"Each task gives" -> "The task gives"; output = one JSON object with the single key = task id
("Answer with this JSON and nothing else"). "looking for it would invalidate the run" dropped
(no tools).

Decoding: JSON-schema-constrained (mlx_vlm.structured.build_json_schema_logits_processor, llguidance)
with schema {"<id>": [{"label": str, "x": [number], "y": [number]}]}; lenient parse still applied
(first JSON object) and parse failures recorded.

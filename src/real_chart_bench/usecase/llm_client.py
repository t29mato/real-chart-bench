"""Port for a multimodal LLM completion call (design §7.16: LLM baseline
scaffold, 司令塔加速指示 — implementation only, no execution without
オーナー approval since it costs real money).

Deliberately abstract over the specific vendor SDK (Anthropic/OpenAI/
Google) so LlmModelRunner (adapter) doesn't hardcode one — a concrete
LlmClientPort implementation per vendor is added when execution is
approved, not before.
"""

from __future__ import annotations

from typing import Protocol


class LlmClientPort(Protocol):
    def complete(self, *, model: str, image_bytes: bytes, prompt: str) -> str:
        """Sends one multimodal request, returns the raw text response."""
        ...

"""LLM and embedding interfaces. Modules depend on these protocols, never on a vendor SDK.

Rules for using the LLM in this project:
- The LLM reads, extracts, judges and writes text.
- Code does arithmetic, comparison, filtering, ranking and de-duplication decisions.
- Every LLM output that feeds a decision is a validated pydantic object.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-sonnet-5-5"
CHEAP_MODEL = "claude-haiku-5-5"


class LLM(Protocol):
    def complete(self, system: str, prompt: str, *, model: str | None = None) -> str: ...

    def structured(self, schema: type[T], system: str, prompt: str, *, model: str | None = None) -> T: ...


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalized vectors."""
        ...


class AnthropicLLM:
    def __init__(self, api_key: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def complete(self, system: str, prompt: str, *, model: str | None = None) -> str:
        resp = self._client.messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def structured(self, schema: type[T], system: str, prompt: str, *, model: str | None = None) -> T:
        tool = {
            "name": "return_result",
            "description": "Return the result.",
            "input_schema": schema.model_json_schema(),
        }
        resp = self._client.messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=4096,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "return_result"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return schema.model_validate(block.input)
        raise ValueError("model returned no structured result")


class FakeLLM:
    """Deterministic stand-in for tests. Pass canned responses keyed by schema name."""

    def __init__(self, structured: dict[str, BaseModel | dict] | None = None, text: str = "ok") -> None:
        self._structured = structured or {}
        self._text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str, *, model: str | None = None) -> str:
        self.calls.append((system, prompt))
        return self._text

    def structured(self, schema: type[T], system: str, prompt: str, *, model: str | None = None) -> T:
        self.calls.append((system, prompt))
        canned = self._structured[schema.__name__]
        if isinstance(canned, BaseModel):
            return schema.model_validate(canned.model_dump())
        return schema.model_validate(canned)


class HashEmbedder:
    """Bag-of-words hashing embedder. Deterministic, offline, good enough for tests."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) % 2 == 0 else -1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


def dumps(obj: BaseModel) -> str:
    return json.dumps(obj.model_dump(mode="json"), sort_keys=True)

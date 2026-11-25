"""A fake OpenRouter served through httpx.MockTransport. No network, no key."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest

from divergent_mcp.openrouter import OpenRouterClient

FAKE_KEY = "test-key-not-real"

CATALOGUE = [
    {
        "id": mid,
        "created": 1_700_000_000 + i,
        "architecture": {"output_modalities": ["text"]},
        "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
        "supported_parameters": params,
        "reasoning": reasoning,
    }
    for i, (mid, params, reasoning) in enumerate(
        [
            ("anthropic/claude-haiku-5.5", ["reasoning", "max_tokens"], {"supported_efforts": ["low", "none"]}),
            ("google/gemini-3.8-flash", ["temperature", "top_p", "reasoning", "max_tokens"], {"mandatory": True, "supported_efforts": ["high", "low"]}),
            ("openai/gpt-6.1-terra", ["reasoning", "max_tokens"], {"supported_efforts": ["medium", "none"]}),
            ("deepseek/deepseek-v4.1-flash", ["temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"], None),
            ("qwen/qwen3.8-flash", ["temperature", "top_p", "presence_penalty", "reasoning", "max_tokens"], {"mandatory": False}),
            ("mistralai/mistral-small-2603", ["temperature", "top_p", "max_tokens"], None),
            ("meta-llama/llama-4-maverick", ["temperature", "top_p", "max_tokens"], None),
            ("z-ai/glm-5.3-flash", ["temperature", "max_tokens"], None),
            ("cohere/command-a-plus", ["temperature", "max_tokens"], None),
        ]
    )
]

# Each model contributes the shared obvious idea plus ideas of its own.
OBVIOUS = {
    "title": "Longer opening hours",
    "idea": "Keep the library open late on weekday evenings so working people can visit.",
    "twist": "None really.",
    "probability": 0.6,
}
THEMES = [
    ("Tool lending wall", "Lend drills, ladders and sewing machines alongside books from a pegboard wall."),
    ("Silent disco story hour", "Children wear wireless headphones and follow a narrated treasure hunt between shelves."),
    ("Seed vault drawers", "Old card catalogue drawers hold seed packets that patrons borrow and return after harvest."),
    ("Midnight repair cafe", "Volunteers fix bicycles and radios in the reading room once a month after closing."),
    ("Recipe swap freezer", "A community freezer in the lobby holds labelled home cooked meals traded for handwritten recipes."),
    ("Mystery map quest", "Hidden envelopes in atlases lead patrons across town landmarks to collect stamps."),
    ("Podcast confession booth", "A soundproof phone box records neighbours telling stories for a local audio archive."),
    ("Telescope checkout night", "Borrow telescopes on clear evenings with an astronomer guiding rooftop stargazing."),
    ("Language exchange speed dating", "Rotating five minute conversations pair immigrants with learners of their native tongue."),
]


def ideas_for(model: str, count: int) -> list[dict[str, Any]]:
    start = sum(map(ord, model)) % len(THEMES)
    own = [
        {"title": t, "idea": d, "twist": f"Unusual, from {model}.", "probability": 0.05}
        for t, d in (THEMES[(start + k) % len(THEMES)] for k in range(count - 1))
    ]
    return [OBVIOUS, *own]


class FakeOpenRouter:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls = 0
        self.catalogue_calls = 0
        self.fail_models: set[str] = set()
        self.garbage_models: set[str] = set()
        self.fail_embeddings = False
        self.fail_catalogue = False
        self.fail_judge = False
        self.auth_headers: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            self.catalogue_calls += 1
            if self.fail_catalogue:
                return httpx.Response(503, json={"error": {"message": "down"}})
            return httpx.Response(200, json={"data": CATALOGUE})
        self.auth_headers.append(request.headers.get("authorization", ""))
        body = json.loads(request.content)
        if path.endswith("/embeddings"):
            self.embed_calls += 1
            if self.fail_embeddings:
                return httpx.Response(404, json={"error": {"message": "no such model"}})
            data = [
                {"index": i, "embedding": self._embed(text)} for i, text in enumerate(body["input"])
            ]
            return httpx.Response(200, json={"data": data, "usage": {"cost": 0.00001}})
        self.chat_calls.append(body)
        model = body["model"]
        user = body["messages"][-1]["content"]
        if "Score every idea" in user:
            if self.fail_judge:
                return httpx.Response(500, json={"error": {"message": "judge down"}})
            scores = []
            for cid, title in re.findall(r"^\[(\d+)\] (.*?):", user, flags=re.MULTILINE):
                obvious = title == OBVIOUS["title"]
                scores.append(
                    {"id": int(cid), "novelty": 1 if obvious else 8, "utility": 9 if obvious else 6}
                )
            text = "```json\n" + json.dumps({"scores": scores}) + "\n```"
        elif model in self.fail_models:
            return httpx.Response(400, json={"error": {"message": "Provider returned error"}})
        elif model in self.garbage_models:
            text = "I'd rather not use JSON today."
        else:
            count = int(re.search(r"Give exactly (\d+) ideas", user).group(1))
            text = "Here you go:\n" + json.dumps({"ideas": ideas_for(model, count)})
        return httpx.Response(
            200,
            json={
                "model": model,
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 200, "cost": 0.0002},
            },
        )

    @staticmethod
    def _embed(text: str) -> list[float]:
        """A tiny bag-of-words hash embedding: shared words give high cosine."""
        vec = [0.0] * 64
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[sum(map(ord, word)) % 64] += 1.0
        return vec


@pytest.fixture
def fake() -> FakeOpenRouter:
    return FakeOpenRouter()


@pytest.fixture
async def client(fake: FakeOpenRouter):
    c = OpenRouterClient(FAKE_KEY, transport=httpx.MockTransport(fake.handler), max_retries=0)
    yield c
    await c.aclose()


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Unit tests must not pick up the developer's real key or model overrides."""
    if "live" in request.keywords:
        return
    for var in ("OPENROUTER_API_KEY", "DIVERGENT_MODELS", "DIVERGENT_EMBEDDING_MODEL", "DIVERGENT_DEADLINE_S", "DIVERGENT_DEDUP_THRESHOLD", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)

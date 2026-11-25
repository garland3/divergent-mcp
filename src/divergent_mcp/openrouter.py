"""Minimal async OpenRouter client: chat completions, embeddings, model catalogue."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
CATALOGUE_TTL_S = 3600.0


class OpenRouterError(RuntimeError):
    """A failed OpenRouter call. The message never contains the API key."""


class MissingKeyError(OpenRouterError):
    pass


@dataclass
class ChatResult:
    text: str
    model: str
    cost: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _error_text(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        msg = err.get("message") if isinstance(err, dict) else str(err)
    except ValueError:
        msg = resp.text
    return f"HTTP {resp.status_code}: {str(msg)[:300]}"


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.max_retries = max_retries
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport)
        self._catalogue: tuple[float, list[dict[str, Any]]] | None = None

    def __repr__(self) -> str:  # never show the key
        return f"OpenRouterClient(base_url={self.base_url!r}, key={'set' if self._api_key else 'missing'})"

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> OpenRouterClient:  # noqa: PYI034
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise MissingKeyError(
                "OPENROUTER_API_KEY is not set. Export it in the environment "
                "(or pass it through the MCP server's env block)."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "divergent-mcp",
        }

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = self._headers()
        last = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._http.post(self.base_url + path, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
            else:
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                    except ValueError:
                        last = "non-JSON response body"
                    else:
                        # OpenRouter can report a provider failure inside a 200.
                        failed = (
                            isinstance(body, dict)
                            and "error" in body
                            and "choices" not in body
                            and "data" not in body
                        )
                        if not failed:
                            return body
                        last = _error_text(resp)
                else:
                    last = _error_text(resp)
                    if resp.status_code not in RETRY_STATUS:
                        break
            if attempt < self.max_retries:
                await asyncio.sleep(0.6 * (attempt + 1))
        raise OpenRouterError(last)

    async def chat(self, model: str, messages: list[dict[str, str]], **params: Any) -> ChatResult:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        payload.update({k: v for k, v in params.items() if v is not None})
        body = await self._post("/chat/completions", payload)
        try:
            choice = body["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise OpenRouterError(f"unexpected response shape from {model}") from exc
        if isinstance(text, list):  # content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        usage = body.get("usage") or {}
        cost = usage.get("cost")
        return ChatResult(
            text=text,
            model=body.get("model", model),
            cost=float(cost) if isinstance(cost, (int, float)) else None,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason"),
            raw=body,
        )

    async def embed(self, model: str, inputs: list[str]) -> tuple[list[list[float]], float | None]:
        body = await self._post("/embeddings", {"model": model, "input": inputs})
        try:
            rows = sorted(body["data"], key=lambda r: r.get("index", 0))
            vectors = [list(map(float, r["embedding"])) for r in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenRouterError(f"unexpected embeddings response from {model}") from exc
        if len(vectors) != len(inputs):
            raise OpenRouterError("embeddings response has the wrong number of vectors")
        cost = (body.get("usage") or {}).get("cost")
        return vectors, float(cost) if isinstance(cost, (int, float)) else None

    async def catalogue(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """The public model list. No key needed; cached in-process for an hour."""
        now = time.monotonic()
        if self._catalogue and not refresh and now - self._catalogue[0] < CATALOGUE_TTL_S:
            return self._catalogue[1]
        try:
            resp = await self._http.get(self.base_url + "/models", timeout=15.0)
            resp.raise_for_status()
            data = resp.json()["data"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise OpenRouterError(f"could not fetch model catalogue: {type(exc).__name__}") from exc
        self._catalogue = (now, data)
        return data

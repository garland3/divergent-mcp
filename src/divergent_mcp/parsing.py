"""Forgiving extraction of JSON from model output."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _balanced(text: str, start: int) -> str | None:
    """The balanced {...} or [...] starting at `start`, respecting strings."""
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Any:
    """First JSON object or array found in `text`, or None."""
    if not text:
        return None
    text = _THINK.sub("", text).strip()
    candidates = [text] + [m.group(1).strip() for m in _FENCE.finditer(text)]
    for cand in candidates:
        try:
            return json.loads(cand)
        except ValueError:
            pass
    for i, ch in enumerate(text):
        if ch in "{[":
            chunk = _balanced(text, i)
            if chunk:
                try:
                    return json.loads(chunk)
                except ValueError:
                    continue
    return None


def extract_objects(text: str) -> list[dict[str, Any]]:
    """Every parsable, non-overlapping JSON object in `text`, outermost first.

    Used to salvage ideas when the reply as a whole is not valid JSON (truncated,
    or one item has an unescaped quote): the well-formed items are still usable.
    """
    text = _THINK.sub("", text or "")
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            chunk = _balanced(text, i)
            if chunk:
                try:
                    obj = json.loads(chunk)
                except ValueError:
                    obj = None
                if isinstance(obj, dict):
                    out.append(obj)
                    i += len(chunk)
                    continue
        i += 1
    return out


def _clean(value: Any, limit: int) -> str:
    if value is None:
        return ""
    s = " ".join(str(value).split())
    return s[:limit].rstrip()


def _as_probability(value: Any) -> float | None:
    try:
        p = float(str(value).strip().rstrip("%")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, str) and value.strip().endswith("%"):
        p /= 100.0
    return p if 0.0 <= p <= 1.0 else None


def parse_ideas(text: str) -> list[dict[str, Any]]:
    """Normalised ideas from a generation reply: title, idea, twist, probability."""
    data = extract_json(text)
    items: list[Any] | None = None
    if isinstance(data, dict):
        items = next(
            (data[k] for k in ("ideas", "responses", "items", "results") if isinstance(data.get(k), list)),
            None,
        )
    elif isinstance(data, list):
        items = data
    if items is None:
        # Not valid as a whole: keep whichever individual idea objects do parse.
        items = [o for o in extract_objects(text) if "idea" in o or "title" in o]
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            item = {"idea": item}
        if not isinstance(item, dict):
            continue
        idea = _clean(
            item.get("idea") or item.get("description") or item.get("text") or item.get("response"),
            700,
        )
        title = _clean(item.get("title") or item.get("name"), 120)
        if not idea and not title:
            continue
        if not idea:
            idea = title
        if not title:
            title = " ".join(idea.split()[:8])
        out.append(
            {
                "title": title,
                "idea": idea,
                "twist": _clean(item.get("twist") or item.get("why_novel"), 300),
                "probability": _as_probability(item.get("probability")),
            }
        )
    return out


def parse_scores(text: str) -> dict[int, tuple[float, float]]:
    """Judge reply -> {candidate id: (novelty, utility)}, both clamped to 1..10."""
    data = extract_json(text)
    if isinstance(data, dict):
        items = next((v for v in data.values() if isinstance(v, list)), [])
    elif isinstance(data, list):
        items = data
    else:
        items = []
    out: dict[int, tuple[float, float]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item["id"])
            nov = float(item.get("novelty", item.get("n")))
            util = float(item.get("utility", item.get("u")))
        except (KeyError, TypeError, ValueError):
            continue
        out[cid] = (max(1.0, min(10.0, nov)), max(1.0, min(10.0, util)))
    return out

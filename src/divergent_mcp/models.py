"""Model pool: one or two picks per model family, resolved against the live catalogue.

Model ids on OpenRouter churn. The preference lists below are tried first; when a
preferred id is gone the newest general-purpose model of the same vendor within the
tier's price cap is used instead, so the pool keeps its family diversity without a
code change. `DIVERGENT_MODELS` (comma-separated ids) replaces the pool outright.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

BUDGETS = ("low", "medium", "high")

# vendor prefix -> (cheap preferences, premium preferences), verified 2026-11-25.
FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "anthropic": (("anthropic/claude-haiku-5.5",), ("anthropic/claude-sonnet-5.5",)),
    "google": (("google/gemini-3.8-flash",), ("google/gemini-4-hydrogen",)),
    "openai": (("openai/gpt-6.1-terra",), ("openai/gpt-6.2-sol",)),
    "deepseek": (("deepseek/deepseek-v4.1-flash",), ("deepseek/deepseek-v4-pro",)),
    "qwen": (("qwen/qwen3.8-flash",), ("qwen/qwen4-max",)),
    "mistralai": (("mistralai/mistral-small-2603",), ("mistralai/mistral-large-4-0",)),
    "meta-llama": (("meta-llama/llama-4-maverick",), ()),
    "z-ai": (("z-ai/glm-5.3-flash",), ()),
    "cohere": (("cohere/command-a-plus",), ()),
    "x-ai": ((), ("x-ai/grok-4.7",)),
}

# Output price caps in USD per million tokens, used only for catalogue fallbacks.
CHEAP_CAP, PREMIUM_CAP = 2.5, 15.0
# Families that get their premium model at budget "medium".
MEDIUM_PREMIUM = ("anthropic", "google", "openai")

_SPECIALISED = re.compile(
    r"guard|embed|ocr|tts|audio|voice|image|vision|code|coder|legal|multi-agent|"
    r"realtime|search|research|safeguard|:free|:batch|-exp\b|preview",
    re.IGNORECASE,
)
_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    tier: str = "cheap"  # cheap | premium | custom
    # None means "catalogue unavailable": send everything and let OpenRouter drop extras.
    supported: frozenset[str] | None = None
    reasoning: dict[str, Any] | None = field(default=None, hash=False, compare=False)
    price_in: float | None = None  # USD per million tokens
    price_out: float | None = None
    verified: bool = False  # present in the live catalogue


def family_of(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else model_id


def _price(entry: dict[str, Any], key: str) -> float | None:
    try:
        return float(entry["pricing"][key]) * 1e6
    except (KeyError, TypeError, ValueError):
        return None


def _is_text_model(entry: dict[str, Any]) -> bool:
    arch = entry.get("architecture") or {}
    return "text" in (arch.get("output_modalities") or ["text"])


def spec_from_entry(entry: dict[str, Any], tier: str) -> ModelSpec:
    return ModelSpec(
        id=entry["id"],
        family=family_of(entry["id"]),
        tier=tier,
        supported=frozenset(entry.get("supported_parameters") or ()),
        reasoning=entry.get("reasoning") if isinstance(entry.get("reasoning"), dict) else None,
        price_in=_price(entry, "prompt"),
        price_out=_price(entry, "completion"),
        verified=True,
    )


def _fallback(index: dict[str, dict[str, Any]], family: str, cap: float, floor: float) -> str | None:
    """Newest general-purpose text model of a vendor priced within (floor, cap]."""
    best: tuple[int, str] | None = None
    for mid, entry in index.items():
        if family_of(mid) != family or _SPECIALISED.search(mid) or not _is_text_model(entry):
            continue
        if entry.get("expiration_date"):
            continue
        out = _price(entry, "completion")
        if out is None or not (floor < out <= cap):
            continue
        created = int(entry.get("created") or 0)
        if best is None or created > best[0]:
            best = (created, mid)
    return best[1] if best else None


def _pick(
    index: dict[str, dict[str, Any]] | None, family: str, prefs: tuple[str, ...], tier: str
) -> ModelSpec | None:
    if index is None:  # no catalogue: trust the first preference
        return ModelSpec(id=prefs[0], family=family, tier=tier) if prefs else None
    for mid in prefs:
        if mid in index:
            return spec_from_entry(index[mid], tier)
    if not prefs:
        return None
    cap, floor = (CHEAP_CAP, 0.0) if tier == "cheap" else (PREMIUM_CAP, CHEAP_CAP)
    mid = _fallback(index, family, cap, floor)
    return spec_from_entry(index[mid], tier) if mid else None


def env_models() -> list[str]:
    raw = os.environ.get("DIVERGENT_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def resolve_pool(
    catalogue: list[dict[str, Any]] | None,
    budget: str = "low",
    override: list[str] | None = None,
) -> list[ModelSpec]:
    """The generator pool for a budget. `override` (or DIVERGENT_MODELS) wins outright."""
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {', '.join(BUDGETS)}")
    index = {e["id"]: e for e in catalogue if "id" in e} if catalogue else None
    custom = [m for m in (override or []) if m and m.strip()] or env_models()
    if custom:
        pool = []
        for mid in dict.fromkeys(m.strip() for m in custom):
            if index and mid in index:
                pool.append(spec_from_entry(index[mid], "custom"))
            else:
                pool.append(ModelSpec(id=mid, family=family_of(mid), tier="custom"))
        return pool

    pool: list[ModelSpec] = []
    for family, (cheap, premium) in FAMILIES.items():
        want_premium = budget == "high" or (budget == "medium" and family in MEDIUM_PREMIUM)
        spec = None
        if want_premium:
            spec = _pick(index, family, premium, "premium")
        if spec is None and (cheap or budget != "low"):
            spec = _pick(index, family, cheap, "cheap")
        if spec is None and budget != "low":
            spec = _pick(index, family, premium, "premium")
        if spec is not None:
            pool.append(spec)
    return pool


def judge_pool(pool: list[ModelSpec], count: int) -> list[ModelSpec]:
    """Judges from different families, preferring the ones that follow JSON formats best."""
    preferred = ("google", "openai", "anthropic", "mistralai", "deepseek", "qwen")
    ranked = sorted(
        pool,
        key=lambda m: (
            preferred.index(m.family) if m.family in preferred else len(preferred),
            m.tier != "premium",
        ),
    )
    out: list[ModelSpec] = []
    for m in ranked:
        if m.family not in {j.family for j in out}:
            out.append(m)
        if len(out) >= count:
            break
    return out


def lowest_reasoning(spec: ModelSpec) -> dict[str, Any] | None:
    """The cheapest reasoning setting a model accepts, or None to leave it alone.

    Left at their defaults, some small models spend the whole token budget thinking
    and return nothing, so reasoning is turned down as far as each model allows.
    """
    if spec.supported is None or "reasoning" not in spec.supported or not spec.reasoning:
        return None
    efforts = spec.reasoning.get("supported_efforts") or ()
    for effort in _EFFORT_ORDER:
        if effort in efforts:
            return {"effort": effort}
    if not spec.reasoning.get("mandatory"):
        return {"enabled": False}
    if spec.reasoning.get("supports_max_tokens"):
        return {"max_tokens": 1024}
    return None


def fit_params(spec: ModelSpec, params: dict[str, Any]) -> dict[str, Any]:
    """Drop sampling parameters a model does not accept and clamp the rest."""
    out = dict(params)
    if spec.family == "anthropic" and "temperature" in out:
        out["temperature"] = min(out["temperature"], 1.0)
    if spec.supported is not None:
        out = {k: v for k, v in out.items() if k in spec.supported}
    return out


def describe_pool(pool: list[ModelSpec]) -> list[dict[str, Any]]:
    return [
        {
            "id": m.id,
            "family": m.family,
            "tier": m.tier,
            "in_catalogue": m.verified,
            "usd_per_mtok_in": m.price_in,
            "usd_per_mtok_out": m.price_out,
        }
        for m in pool
    ]

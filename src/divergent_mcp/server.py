"""FastMCP server: divergent ideation tools backed by OpenRouter."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from . import strategies
from .models import BUDGETS, describe_pool, resolve_pool
from .openrouter import OpenRouterClient, OpenRouterError
from .pipeline import MAX_IDEAS, PROFILES, IdeationError, IdeationRequest, ideate, plan_only

mcp = FastMCP(
    "divergent",
    instructions=(
        "Divergent ideation. generate_ideas fans a topic out to many different LLM families "
        "through OpenRouter, each pushed by a different creative lens, then removes "
        "duplicates and selects a set that balances novelty and usefulness while keeping "
        "some high-novelty wildcards. Use it when the user wants many genuinely different "
        "ideas, names, angles or options, not for factual questions. It takes 20-90 seconds "
        "and spends OpenRouter credit; the topic text is sent to external model providers. "
        "plan_ideation previews a run for free."
    ),
)

# One client per server process, so the model catalogue is fetched once an hour.
_client: OpenRouterClient | None = None


def _get_client() -> OpenRouterClient:
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client


Topic = Annotated[
    str,
    Field(description="What to brainstorm about, stated as a question or a brief. Be specific."),
]
N = Annotated[int, Field(ge=1, le=MAX_IDEAS, description="How many ideas to return.")]
Spice = Annotated[
    int,
    Field(
        ge=0,
        le=3,
        description=(
            "How far from the obvious to push: 0 conventional, 1 mild, 2 adventurous "
            "(default), 3 wild. Raises temperature, shifts the lens mix toward inversions "
            "and forced analogies, and reserves more slots for high-novelty wildcards."
        ),
    ),
]
Budget = Annotated[
    Literal["low", "medium", "high"],
    Field(
        description=(
            "Spend level. low: 5-10 calls to cheap models, one judge. medium: up to 14 calls, "
            "premium models for three families, two judges. high: up to 24 calls, premium "
            "models throughout."
        )
    ),
]
Constraints = Annotated[
    list[str] | None,
    Field(description="Requirements every idea must satisfy, one per item. Optional."),
]
Context = Annotated[
    str | None,
    Field(description="Optional background the models should know (audience, what was tried)."),
]
Models = Annotated[
    list[str] | None,
    Field(description="Optional OpenRouter model ids to use instead of the default pool."),
]
Lenses = Annotated[
    list[str] | None,
    Field(description="Optional subset of lens names from list_strategies. Default: all."),
]
Seed = Annotated[
    int | None,
    Field(description="Optional seed for the plan (model/lens pairing and sampling jitter)."),
]


def _request(
    topic: str,
    n: int,
    spice: int,
    budget: str,
    constraints: list[str] | None,
    context: str | None,
    models: list[str] | None,
    lenses: list[str] | None,
    seed: int | None,
) -> IdeationRequest:
    req = IdeationRequest(
        topic=topic,
        n=n,
        spice=spice,
        budget=budget,
        constraints=list(constraints or []),
        context=context,
        models=models,
        strategies=lenses,
        seed=seed,
    )
    try:
        req.validate()
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    return req


@mcp.tool
async def generate_ideas(
    topic: Topic,
    n: N = 8,
    spice: Spice = 2,
    budget: Budget = "low",
    constraints: Constraints = None,
    context: Context = None,
    models: Models = None,
    lenses: Lenses = None,
    seed: Seed = None,
    include_trace: Annotated[
        bool, Field(description="Also return one record per model call (model, lens, outcome).")
    ] = False,
) -> dict[str, Any]:
    """Brainstorm n distinct ideas on a topic using many models and creative lenses in parallel.

    Returns ranked ideas, each with a title, the idea, its twist, a kind (wildcard, balanced
    or solid), novelty and utility scores from 0 to 1, and which models and lenses produced
    it, plus run statistics including cost. Show wildcards to the user as such: they are
    kept for their novelty, not their safety.
    """
    req = _request(topic, n, spice, budget, constraints, context, models, lenses, seed)
    try:
        return await ideate(req, _get_client(), include_trace=include_trace)
    except (IdeationError, OpenRouterError, ValueError) as exc:
        raise ToolError(str(exc)) from None


@mcp.tool
async def plan_ideation(
    topic: Topic,
    n: N = 8,
    spice: Spice = 2,
    budget: Budget = "low",
    constraints: Constraints = None,
    models: Models = None,
    lenses: Lenses = None,
    seed: Seed = None,
) -> dict[str, Any]:
    """Preview what generate_ideas would do, without calling any model or spending credit.

    Returns the planned calls (model, lens, lens detail, sampling settings), the judges and
    the resolved model pool with prices.
    """
    req = _request(topic, n, spice, budget, constraints, None, models, lenses, seed)
    try:
        return await plan_only(req, _get_client())
    except (IdeationError, OpenRouterError, ValueError) as exc:
        raise ToolError(str(exc)) from None


@mcp.tool
def list_strategies() -> dict[str, Any]:
    """List the creative lenses, how each is weighted at every spice level, and the budgets."""
    return {
        "strategies": strategies.describe(),
        "spice_levels": {
            "0": "conventional: mostly plain brainstorming, low temperature",
            "1": "mild: all lenses, moderate temperature",
            "2": "adventurous (default)",
            "3": "wild: no plain brainstorming, half the slots go to wildcards",
        },
        "budgets": {name: PROFILES[name] for name in BUDGETS},
    }


@mcp.tool
async def list_models(
    budget: Budget = "low",
    refresh: Annotated[bool, Field(description="Re-fetch the OpenRouter catalogue.")] = False,
) -> dict[str, Any]:
    """Show the model pool a budget resolves to right now, one model per family, with prices."""
    client = _get_client()
    warnings: list[str] = []
    try:
        catalogue = await client.catalogue(refresh=refresh)
    except OpenRouterError:
        catalogue = None
        warnings.append("model catalogue unavailable; showing the built-in list unchecked")
    pool = resolve_pool(catalogue, budget)
    return {
        "budget": budget,
        "api_key_configured": client.has_key,
        "models": describe_pool(pool),
        "warnings": warnings,
    }


async def tool_names() -> list[str]:
    return [tool.name for tool in await mcp.list_tools()]

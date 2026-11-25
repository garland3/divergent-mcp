"""Opt-in smoke test against the real OpenRouter API. Costs well under one cent.

    DIVERGENT_LIVE=1 OPENROUTER_API_KEY=... uv run pytest -m live -q
"""

import os

import pytest

from divergent_mcp.pipeline import IdeationRequest, ideate

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.environ.get("DIVERGENT_LIVE") == "1" and os.environ.get("OPENROUTER_API_KEY")),
        reason="set DIVERGENT_LIVE=1 and OPENROUTER_API_KEY to run the live smoke test",
    ),
]


async def test_live_ideation_round_trip():
    key = os.environ["OPENROUTER_API_KEY"]
    req = IdeationRequest("unusual uses for a retired city bus", n=5, spice=2, budget="low")
    result = await ideate(req, include_trace=True)
    assert result["returned"] == 5
    stats = result["stats"]
    assert stats["calls_ok"] >= 3 and len(stats["families"]) >= 3
    assert stats["judges"], result["warnings"]
    assert stats["cost_usd"] is not None and stats["cost_usd"] < 0.10
    assert any(i["kind"] == "wildcard" for i in result["ideas"])
    assert all(i["utility"] is not None for i in result["ideas"])
    assert key not in repr(result)

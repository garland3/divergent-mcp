"""End-to-end pipeline, MCP server and CLI against the fake OpenRouter."""

import asyncio
import json

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from divergent_mcp import cli, server
from divergent_mcp.openrouter import MissingKeyError, OpenRouterClient, OpenRouterError
from divergent_mcp.pipeline import IdeationError, IdeationRequest, ideate, plan_only

from .conftest import FAKE_KEY, OBVIOUS


async def test_full_run_dedups_scores_and_selects(fake, client):
    result = await ideate(IdeationRequest("busier library", n=6, seed=3), client, include_trace=True)
    stats = result["stats"]
    assert result["returned"] == 6 and [i["rank"] for i in result["ideas"]] == [1, 2, 3, 4, 5, 6]
    assert stats["calls_ok"] == stats["calls_planned"] == 5
    assert len(stats["families"]) == 5
    assert stats["similarity"] == "embeddings+lexical" and fake.embed_calls == 1
    assert stats["judges"] == ["google/gemini-3.8-flash"]
    # 5 generation calls + 1 judge call + 1 embeddings call, at the fake's prices.
    assert stats["cost_usd"] == pytest.approx(6 * 0.0002 + 0.00001)

    # Every model returned the same obvious idea: it must collapse to one cluster.
    assert stats["duplicates_merged"] >= 4
    titles = [i["title"] for i in result["ideas"]]
    assert len(titles) == len(set(titles))
    obvious = [i for i in result["ideas"] if i["title"] == OBVIOUS["title"]]
    for idea in obvious:
        assert idea["support"] == 5 and len(idea["sources"]) == 5
        assert idea["kind"] == "solid" and idea["novelty"] < 0.3

    kinds = [i["kind"] for i in result["ideas"]]
    assert kinds.count("wildcard") == stats["wildcards"] == 2
    for idea in result["ideas"]:
        assert 0 <= idea["novelty"] <= 1 and 0 <= idea["utility"] <= 1
        assert idea["sources"][0]["model"] and idea["sources"][0]["strategy"]
    assert len(result["trace"]) == 5 and all(t["ok"] for t in result["trace"])


async def test_requests_carry_the_key_only_in_the_header_and_fitted_params(fake, client):
    result = await ideate(IdeationRequest("busier library", n=4, seed=1), client)
    assert fake.auth_headers and set(fake.auth_headers) == {f"Bearer {FAKE_KEY}"}
    assert FAKE_KEY not in json.dumps(result) and FAKE_KEY not in repr(client)
    for call in fake.chat_calls:
        if call["model"].startswith(("anthropic/", "openai/")):
            assert "temperature" not in call and call["reasoning"] == {"effort": "none"}
        assert call["max_tokens"] >= 2500


async def test_failed_and_unparsable_calls_are_tolerated(fake, client):
    req = IdeationRequest("busier library", n=5, seed=3)
    planned = [c["model"] for c in (await plan_only(req, client))["calls"]]
    fake.fail_models = {planned[0]}
    fake.garbage_models = {planned[1]}
    result = await ideate(req, client, include_trace=True)
    assert result["stats"]["calls_ok"] == 3
    assert result["returned"] == 5
    assert any("2 of 5 generation calls failed" in w for w in result["warnings"])
    errors = [t.get("error", "") for t in result["trace"]]
    assert any("HTTP 400" in e for e in errors) and any("no parsable ideas" in e for e in errors)


def _one_idea_handler(bodies, reject):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            entry = {"id": "x/y", "supported_parameters": ["temperature", "reasoning"], "reasoning": {"supported_efforts": ["high", "none"]}}
            return httpx.Response(200, json={"data": [entry]})
        body = json.loads(request.content)
        bodies.append(body)
        if reject(body):
            return httpx.Response(400, json={"error": {"message": "Provider returned error"}})
        text = json.dumps({"ideas": [{"title": "One", "idea": "Only idea."}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    return handler


async def test_a_rejected_reasoning_effort_is_retried_at_low_effort():
    # The catalogue says "none" is allowed; the provider disagrees.
    bodies = []
    handler = _one_idea_handler(bodies, lambda b: b.get("reasoning") == {"effort": "none"})
    async with OpenRouterClient(FAKE_KEY, transport=httpx.MockTransport(handler), max_retries=0) as c:
        req = IdeationRequest("t", n=1, models=["x/y"], max_calls=1, judge=False, embeddings=False)
        result = await ideate(req, c)
    assert result["stats"]["calls_ok"] == 1 and len(bodies) == 2
    assert bodies[0]["reasoning"] == {"effort": "none"} and bodies[1]["reasoning"] == {"effort": "low"}
    assert bodies[1]["temperature"] == bodies[0]["temperature"]  # sampling jitter is kept


async def test_a_request_rejected_twice_is_retried_plain():
    bodies = []
    handler = _one_idea_handler(bodies, lambda b: "reasoning" in b or "temperature" in b)
    async with OpenRouterClient(FAKE_KEY, transport=httpx.MockTransport(handler), max_retries=0) as c:
        req = IdeationRequest("t", n=1, models=["x/y"], max_calls=1, judge=False, embeddings=False)
        result = await ideate(req, c)
    assert result["returned"] == 1 and len(bodies) == 3
    assert set(bodies[-1]) == {"model", "messages", "max_tokens"}


async def test_all_calls_failing_is_a_clear_error(fake, client):
    req = IdeationRequest("busier library", n=4, seed=3)
    fake.fail_models = {c["model"] for c in (await plan_only(req, client))["calls"]}
    with pytest.raises(IdeationError, match="every generation call failed: HTTP 400"):
        await ideate(req, client)


async def test_degrades_without_embeddings_judge_or_catalogue(fake, client):
    fake.fail_embeddings = fake.fail_judge = fake.fail_catalogue = True
    result = await ideate(IdeationRequest("busier library", n=4, seed=3), client)
    assert result["returned"] == 4
    assert result["stats"]["similarity"] == "lexical" and result["stats"]["judges"] == []
    assert all(i["utility"] is None for i in result["ideas"])
    text = " ".join(result["warnings"])
    assert "catalogue unavailable" in text and "embeddings unavailable" in text
    assert "judging failed" in text
    # Lexical dedup alone still merges the idea every model repeated.
    assert [i["title"] for i in result["ideas"]].count(OBVIOUS["title"]) <= 1


async def test_judge_and_embeddings_can_be_switched_off(fake, client):
    req = IdeationRequest("busier library", n=4, seed=3, judge=False, embeddings=False)
    result = await ideate(req, client)
    assert fake.embed_calls == 0 and len(fake.chat_calls) == result["stats"]["calls_planned"]
    assert result["warnings"] == []


async def test_constraints_and_context_reach_generators_and_judge(fake, client):
    req = IdeationRequest("busier library", n=4, seed=3, constraints=["costs under $50"], context="Rural branch.")
    await ideate(req, client)
    users = [c["messages"][-1]["content"] for c in fake.chat_calls]
    assert all("costs under $50" in u for u in users)
    assert sum("Rural branch." in u for u in users) == len(users) - 1  # judge needs no background


async def test_slow_models_are_cut_at_the_deadline():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(503)
        await asyncio.sleep(30)
        return httpx.Response(500)

    async with OpenRouterClient(FAKE_KEY, transport=httpx.MockTransport(handler)) as c:
        req = IdeationRequest("t", n=3, deadline_s=10, judge=False)
        req.deadline_s = 0.5
        with pytest.raises(IdeationError, match="too slow; cancelled"):
            await ideate(req, c)


async def test_missing_key_fails_before_any_request(fake):
    async with OpenRouterClient("", transport=httpx.MockTransport(fake.handler)) as c:
        with pytest.raises(MissingKeyError, match="OPENROUTER_API_KEY is not set"):
            await ideate(IdeationRequest("t"), c)
        assert fake.chat_calls == [] and fake.catalogue_calls == 0
        plan = await plan_only(IdeationRequest("t"), c)  # planning needs no key
        assert plan["calls"]


async def test_client_retries_transient_errors_and_reports_provider_errors():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        if len(seen) == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        if len(seen) == 2:
            return httpx.Response(200, json={"error": {"message": "upstream exploded", "code": 502}})
        return httpx.Response(200, json={"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]}}]})

    async with OpenRouterClient(FAKE_KEY, transport=httpx.MockTransport(handler), max_retries=2) as c:
        result = await c.chat("a/b", [{"role": "user", "content": "x"}])
        assert result.text == "hi" and result.cost is None and len(seen) == 3
    async with OpenRouterClient(FAKE_KEY, transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "No auth"}})), max_retries=2) as c:
        with pytest.raises(OpenRouterError, match="HTTP 401: No auth") as info:
            await c.chat("a/b", [])
        assert FAKE_KEY not in str(info.value)


# ---------------------------------------------------------------- MCP server


@pytest.fixture
def mcp_client(client, monkeypatch):
    monkeypatch.setattr(server, "_client", client)
    return Client(server.mcp)


async def test_mcp_exposes_four_tools_with_plain_schemas(mcp_client):
    async with mcp_client as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert list(tools) == ["generate_ideas", "plan_ideation", "list_strategies", "list_models"]
    schema = tools["generate_ideas"].inputSchema
    assert schema["required"] == ["topic"]
    assert schema["properties"]["spice"]["maximum"] == 3
    assert "$defs" not in schema and "$ref" not in json.dumps(schema)  # nothing recursive
    assert all(t.description for t in tools.values())


async def test_mcp_generate_plan_and_discovery_tools(fake, mcp_client):
    async with mcp_client as c:
        run = (await c.call_tool("generate_ideas", {"topic": "busier library", "n": 4, "seed": 3, "constraints": ["free"]})).data
        assert run["returned"] == 4 and run["constraints"] == ["free"] and "trace" not in run
        before = len(fake.chat_calls)
        plan = (await c.call_tool("plan_ideation", {"topic": "busier library", "n": 4, "seed": 3, "budget": "medium"})).data
        assert len(fake.chat_calls) == before  # planning spends nothing
        assert len(plan["calls"]) == 6 and len(plan["judges"]) == 2
        lenses = (await c.call_tool("list_strategies", {})).data
        assert len(lenses["strategies"]) == 8 and set(lenses["budgets"]) == {"low", "medium", "high"}
        models = (await c.call_tool("list_models", {"budget": "low"})).data
        assert models["api_key_configured"] is True and len(models["models"]) == 9
        assert fake.catalogue_calls == 1  # cached across tool calls


async def test_mcp_reports_bad_input_and_upstream_failure_as_tool_errors(fake, mcp_client):
    async with mcp_client as c:
        with pytest.raises(ToolError):
            await c.call_tool("generate_ideas", {"topic": "x", "n": 99})
        with pytest.raises(ToolError, match="unknown strategy"):
            await c.call_tool("generate_ideas", {"topic": "x", "lenses": ["nope"]})
        with pytest.raises(ToolError, match="topic must not be empty"):
            await c.call_tool("plan_ideation", {"topic": "  "})


async def test_mcp_without_a_key_explains_itself(fake, monkeypatch):
    keyless = OpenRouterClient("", transport=httpx.MockTransport(fake.handler))
    monkeypatch.setattr(server, "_client", keyless)
    async with Client(server.mcp) as c:
        with pytest.raises(ToolError, match="OPENROUTER_API_KEY is not set"):
            await c.call_tool("generate_ideas", {"topic": "x"})
        assert (await c.call_tool("list_models", {})).data["api_key_configured"] is False
    await keyless.aclose()


# ---------------------------------------------------------------- CLI


@pytest.fixture
def cli_fake(fake, monkeypatch):
    """Route the CLI's own client through the fake transport."""
    real = OpenRouterClient

    def factory(*args, **kwargs):
        return real(FAKE_KEY, transport=httpx.MockTransport(fake.handler), max_retries=0)

    monkeypatch.setattr("divergent_mcp.pipeline.OpenRouterClient", factory)
    monkeypatch.setattr("divergent_mcp.cli.OpenRouterClient", factory)
    return fake


def test_cli_ideas_text_and_json(cli_fake, capsys):
    assert cli.main(["ideas", "busier library", "-n", "4", "--seed", "3", "-s", "3", "-c", "free"]) == 0
    out = capsys.readouterr().out
    assert "4 ideas on: busier library" in out and "[wildcard, novelty" in out
    assert "model families" in out and "cost $" in out and FAKE_KEY not in out

    assert cli.main(["ideas", "busier library", "-n", "3", "--seed", "3", "--json", "--trace", "--no-judge", "-l", "persona", "-m", "acme/one"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["returned"] == 3 and data["stats"]["judges"] == []
    assert {t["model"] for t in data["trace"]} == {"acme/one"}
    assert {t["strategy"] for t in data["trace"]} == {"persona"}


def test_cli_plan_models_and_strategies(cli_fake, capsys):
    assert cli.main(["plan", "busier library", "-n", "12", "-b", "high", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "12 calls" in out and "judges: google/" in out
    assert cli_fake.chat_calls == []
    assert cli.main(["models", "-b", "low"]) == 0
    assert "anthropic/claude-haiku-5.5" in capsys.readouterr().out
    assert cli.main(["strategies"]) == 0
    assert "verbalized_sampling" in capsys.readouterr().out


def test_cli_errors_exit_nonzero_without_a_traceback(fake, monkeypatch, capsys):
    real = OpenRouterClient
    monkeypatch.setattr(
        "divergent_mcp.pipeline.OpenRouterClient",
        lambda *a, **k: real("", transport=httpx.MockTransport(fake.handler)),
    )
    assert cli.main(["ideas", "x"]) == 1
    assert "error: OPENROUTER_API_KEY is not set" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["ideas", "x", "--spice", "9"])

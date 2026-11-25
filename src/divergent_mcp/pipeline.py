"""The ideation pipeline: plan -> fan out -> dedup -> judge -> score -> select."""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from . import similarity, strategies
from .models import (
    BUDGETS,
    ModelSpec,
    describe_pool,
    fit_params,
    judge_pool,
    lowest_reasoning,
    resolve_pool,
)
from .openrouter import MissingKeyError, OpenRouterClient, OpenRouterError
from .parsing import parse_ideas, parse_scores

MAX_IDEAS = 30
DEFAULT_DEADLINE_S = 90.0
DEFAULT_EMBEDDING_MODEL = "perplexity/pplx-embed-v1-0.6b"
CONCURRENCY = 16
# After this many seconds, stop waiting once this share of generation calls is back.
SOFT_GENERATION_S = {"low": 25.0, "medium": 35.0, "high": 45.0}  # premium models are slower
GENERATION_QUORUM = 0.6

# Duplicate thresholds, tuned on real runs with the default embedding model: paraphrases
# of one idea from different models score 0.41-0.58 on mean-centred embeddings, while
# related-but-distinct ideas stay below about 0.35. Wording overlap between paraphrases
# is low (TF-IDF cosine 0.13-0.18 against under 0.10 for distinct ideas), so the lexical
# threshold is loose when it is the only signal and strict when embeddings are present.
DENSE_THRESHOLD = 0.40
LEX_THRESHOLD_ALONE = 0.15
LEX_THRESHOLD_WITH_DENSE = 0.50

# Per budget: ideas requested per idea wanted, call bounds, batch size, judge count.
PROFILES: dict[str, dict[str, float]] = {
    "low": {"oversample": 3.0, "min_calls": 5, "max_calls": 10, "batch": 4, "judges": 1},
    "medium": {"oversample": 3.5, "min_calls": 6, "max_calls": 14, "batch": 4, "judges": 2},
    "high": {"oversample": 5.0, "min_calls": 8, "max_calls": 24, "batch": 5, "judges": 2},
}
MAX_BATCH = 8

# Per spice level 0..3.
BASE_TEMPERATURE = (0.7, 0.9, 1.05, 1.2)
MAX_PRESENCE = (0.2, 0.4, 0.6, 0.9)
WILDCARD_FRACTION = (0.10, 0.20, 0.35, 0.50)
NOVELTY_WEIGHT = (0.30, 0.40, 0.50, 0.60)

# A wildcard may be impractical, but it has to be a coherent answer to the topic.
UTILITY_FLOOR = 0.3
DIVERSITY_PENALTY = 0.25
SOLID_MARGIN = 0.15


class IdeationError(RuntimeError):
    pass


@dataclass
class IdeationRequest:
    topic: str
    n: int = 8
    spice: int = 2
    budget: str = "low"
    constraints: list[str] = field(default_factory=list)
    context: str | None = None
    models: list[str] | None = None
    strategies: list[str] | None = None
    seed: int | None = None
    judge: bool = True
    embeddings: bool = True
    deadline_s: float | None = None
    max_calls: int | None = None

    def validate(self) -> None:
        if not self.topic or not self.topic.strip():
            raise ValueError("topic must not be empty")
        if not 1 <= self.n <= MAX_IDEAS:
            raise ValueError(f"n must be between 1 and {MAX_IDEAS}")
        if not strategies.SPICE_MIN <= self.spice <= strategies.SPICE_MAX:
            raise ValueError("spice must be 0 (conventional) to 3 (wild)")
        if self.budget not in BUDGETS:
            raise ValueError(f"budget must be one of {', '.join(BUDGETS)}")
        for name in self.strategies or []:
            if name not in strategies.STRATEGIES:
                raise ValueError(
                    f"unknown strategy {name!r}; known: {', '.join(strategies.STRATEGIES)}"
                )
        if self.max_calls is not None and self.max_calls < 1:
            raise ValueError("max_calls must be at least 1")


@dataclass
class PlannedCall:
    index: int
    model: ModelSpec
    strategy: str
    detail: str
    count: int
    messages: list[dict[str, str]]
    params: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "call": self.index,
            "model": self.model.id,
            "strategy": self.strategy,
            "detail": self.detail,
            "ideas": self.count,
            "sampling": {k: v for k, v in self.params.items() if k not in ("max_tokens", "reasoning")},
        }


def default_deadline() -> float:
    try:
        return max(10.0, float(os.environ.get("DIVERGENT_DEADLINE_S", DEFAULT_DEADLINE_S)))
    except ValueError:
        return DEFAULT_DEADLINE_S


def dense_threshold() -> float:
    """Embedding duplicate threshold; override it when you change the embedding model."""
    try:
        return max(0.05, float(os.environ.get("DIVERGENT_DEDUP_THRESHOLD", DENSE_THRESHOLD)))
    except ValueError:
        return DENSE_THRESHOLD


def build_plan(req: IdeationRequest, pool: list[ModelSpec]) -> list[PlannedCall]:
    """Pair models with lenses and jittered sampling settings. Pure and seedable."""
    req.validate()
    if not pool:
        raise IdeationError("no models available for this budget")
    prof = PROFILES[req.budget]
    rng = random.Random(req.seed)
    batch = int(prof["batch"])
    wanted = req.n * prof["oversample"]
    calls = max(int(prof["min_calls"]), min(int(prof["max_calls"]), math.ceil(wanted / batch)))
    if req.max_calls is not None:
        calls = min(calls, req.max_calls)
    batch = min(MAX_BATCH, max(batch, math.ceil(wanted / calls)))

    order = list(pool)
    rng.shuffle(order)
    lenses = strategies.allocate(calls, req.spice, rng, req.strategies)
    plan: list[PlannedCall] = []
    for i, lens in enumerate(lenses):
        model = order[i % len(order)]
        messages, detail = strategies.build_messages(
            req.topic,
            strategy=lens,
            count=batch,
            spice=req.spice,
            rng=rng,
            constraints=req.constraints,
            context=req.context,
        )
        sampling = {
            "temperature": round(
                max(0.3, min(1.5, BASE_TEMPERATURE[req.spice] + rng.uniform(-0.15, 0.25))), 2
            ),
            "top_p": round(rng.uniform(0.9, 1.0), 2),
            "presence_penalty": round(rng.uniform(0.0, MAX_PRESENCE[req.spice]), 2),
            "frequency_penalty": round(rng.uniform(0.0, 0.3), 2),
        }
        params = fit_params(model, sampling)
        params["max_tokens"] = max(2500, 500 * batch)
        reasoning = lowest_reasoning(model)
        if reasoning:
            params["reasoning"] = reasoning
        plan.append(PlannedCall(i + 1, model, lens, detail, batch, messages, params))
    return plan


def select(
    novelty: list[float],
    utility: list[float],
    sim: similarity.Matrix,
    n: int,
    spice: int,
) -> list[tuple[int, str]]:
    """Pick n ideas as (index, kind). Wildcards are chosen first, on novelty alone.

    A single blended score lets safe, well-rounded ideas crowd out strange ones, so a
    spice-dependent share of the slots is reserved for the most novel ideas that clear
    a low utility floor. The rest are filled by a novelty/utility geometric blend.
    Both passes use a diversity penalty against what is already picked.
    """
    total = len(novelty)
    n = min(n, total)
    if n <= 0:
        return []
    spice = strategies.clamp_spice(spice)
    n_wild = round(n * WILDCARD_FRACTION[spice])
    if spice >= 1 and n >= 3:
        n_wild = max(1, n_wild)
    w = NOVELTY_WEIGHT[spice]
    blend = [(novelty[i] ** w) * (max(utility[i], 1e-6) ** (1 - w)) for i in range(total)]

    chosen: list[tuple[int, str]] = []
    taken: set[int] = set()

    def pick(score: list[float], eligible: list[int]) -> int | None:
        best, best_v = None, -math.inf
        for i in eligible:
            if i in taken:
                continue
            crowd = max((min(1.0, sim[i][j]) for j in taken), default=0.0)
            v = score[i] - DIVERSITY_PENALTY * crowd
            if v > best_v:
                best, best_v = i, v
        return best

    wild_pool = [i for i in range(total) if utility[i] >= UTILITY_FLOOR]
    for _ in range(n_wild):
        i = pick(novelty, wild_pool)
        if i is None:
            break
        taken.add(i)
        chosen.append((i, "wildcard"))
    everyone = list(range(total))
    while len(chosen) < n:
        i = pick(blend, everyone)
        if i is None:
            break
        taken.add(i)
        chosen.append((i, "solid" if utility[i] - novelty[i] > SOLID_MARGIN else "balanced"))
    chosen.sort(key=lambda c: blend[c[0]], reverse=True)
    return chosen


def pareto_front(novelty: list[float], utility: list[float]) -> set[int]:
    front = set()
    for i in range(len(novelty)):
        dominated = any(
            novelty[j] >= novelty[i]
            and utility[j] >= utility[i]
            and (novelty[j] > novelty[i] or utility[j] > utility[i])
            for j in range(len(novelty))
            if j != i
        )
        if not dominated:
            front.add(i)
    return front


JUDGE_SYSTEM = "You are a strict, calibrated evaluator of brainstormed ideas. Reply with JSON only."
JUDGE_CHUNK = 40


def judge_messages(req: IdeationRequest, items: list[tuple[int, str, str]]) -> list[dict[str, str]]:
    parts = [f"Topic: {req.topic.strip()}"]
    reqs = [c.strip() for c in req.constraints if c and c.strip()]
    if reqs:
        parts.append("Requirements every idea must satisfy:\n" + "\n".join(f"- {c}" for c in reqs))
    parts.append("Ideas:\n" + "\n".join(f"[{cid}] {title}: {idea}" for cid, title, idea in items))
    parts.append(
        "Score every idea on two independent 1-10 scales.\n"
        "novelty: how unexpected it would be to someone who knows this area well. "
        "1 = the first thing anyone says, 5 = a fresh variation on a known approach, "
        "10 = you have never seen anything like it.\n"
        "utility: how much real value it would deliver if carried out competently, counting "
        "feasibility and whether it answers the topic and meets the requirements. "
        "1 = off-topic, incoherent or breaks a requirement, 5 = works but marginal, "
        "10 = clearly valuable and doable.\n"
        "Do not let one scale leak into the other: a strange idea can be useful and a "
        "sensible idea can be stale. Use the whole range.\n"
        'Reply with this JSON object and nothing else: '
        '{"scores": [{"id": <id>, "novelty": <1-10>, "utility": <1-10>}]}'
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


async def _gather_until(
    tasks: list[asyncio.Task], timeout: float, *, soft: float | None = None, quorum: float = 1.0
) -> bool:
    """Wait for tasks up to `timeout`; cancel stragglers. True if any were cancelled.

    With `soft`, stop early once that many seconds have passed and at least `quorum`
    of the tasks are done, so one slow model does not hold the whole run.
    """
    if not tasks:
        return False
    start = time.monotonic()
    pending = set(tasks)
    while pending:
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            break
        done = len(tasks) - len(pending)
        if soft is not None and elapsed >= soft and done >= quorum * len(tasks):
            break
        limit = timeout - elapsed
        if soft is not None:
            limit = min(limit, soft - elapsed) if elapsed < soft else min(limit, 1.0)
        _, pending = await asyncio.wait(
            pending, timeout=max(0.05, limit), return_when=asyncio.FIRST_COMPLETED
        )
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return bool(pending)


LOW_EFFORT = {"effort": "low"}


async def _chat(
    client: OpenRouterClient, model: ModelSpec, messages: list[dict[str, str]], params: dict[str, Any]
):
    """One chat call, with fallbacks when the provider rejects the request (HTTP 400).

    The catalogue's capability data is occasionally wrong: a listed reasoning effort
    or sampling parameter that the provider then refuses. First retry with low
    reasoning effort, which nearly every reasoning model accepts and which keeps the
    call fast; then with nothing but max_tokens. A plain request is better than
    losing the model.
    """
    attempts = [params]
    if "reasoning" in params and params["reasoning"] != LOW_EFFORT:
        attempts.append({**params, "reasoning": LOW_EFFORT})
    bare = {k: v for k, v in params.items() if k == "max_tokens"}
    if bare != params:
        attempts.append(bare)
    for i, attempt in enumerate(attempts):
        try:
            return await client.chat(model.id, messages, **attempt)
        except OpenRouterError as exc:
            if "HTTP 400" not in str(exc) or i == len(attempts) - 1:
                raise
    raise AssertionError("unreachable")


async def _pool_for(
    client: OpenRouterClient, req: IdeationRequest, warnings: list[str]
) -> list[ModelSpec]:
    try:
        catalogue = await client.catalogue()
    except OpenRouterError:
        catalogue = None
        warnings.append("model catalogue unavailable; using the built-in model list unchecked")
    return resolve_pool(catalogue, req.budget, req.models)


async def plan_only(req: IdeationRequest, client: OpenRouterClient | None = None) -> dict[str, Any]:
    """What a run would do, without spending anything."""
    req.validate()
    own = client is None
    client = client or OpenRouterClient()
    try:
        warnings: list[str] = []
        pool = await _pool_for(client, req, warnings)
        plan = build_plan(req, pool)
        judges = judge_pool(pool, int(PROFILES[req.budget]["judges"])) if req.judge else []
        return {
            "topic": req.topic,
            "requested": req.n,
            "spice": req.spice,
            "budget": req.budget,
            "calls": [c.summary() for c in plan],
            "raw_ideas_requested": sum(c.count for c in plan),
            "families": sorted({c.model.family for c in plan}),
            "judges": [j.id for j in judges],
            "pool": describe_pool(pool),
            "warnings": warnings,
        }
    finally:
        if own:
            await client.aclose()


async def ideate(
    req: IdeationRequest,
    client: OpenRouterClient | None = None,
    *,
    include_trace: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline and return the selected ideas with scores and run stats."""
    req.validate()
    started = time.monotonic()
    deadline = started + (req.deadline_s or default_deadline())
    own = client is None
    client = client or OpenRouterClient()
    try:
        if not client.has_key:
            raise MissingKeyError(
                "OPENROUTER_API_KEY is not set. Export it in the environment "
                "(or pass it through the MCP server's env block)."
            )
        return await _run(req, client, started, deadline, include_trace)
    finally:
        if own:
            await client.aclose()


async def _run(
    req: IdeationRequest,
    client: OpenRouterClient,
    started: float,
    deadline: float,
    include_trace: bool,
) -> dict[str, Any]:
    warnings: list[str] = []
    cost = 0.0
    cost_known = False
    pool = await _pool_for(client, req, warnings)
    plan = build_plan(req, pool)
    sem = asyncio.Semaphore(CONCURRENCY)
    trace: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    def spend(value: float | None) -> None:
        nonlocal cost, cost_known
        if value is not None:
            cost += value
            cost_known = True

    async def generate(call: PlannedCall) -> None:
        entry = call.summary()
        trace.append(entry)
        t0 = time.monotonic()
        try:
            async with sem:
                result = await _chat(client, call.model, call.messages, call.params)
        except OpenRouterError as exc:
            entry.update(ok=False, error=str(exc), seconds=round(time.monotonic() - t0, 1))
            return
        spend(result.cost)
        ideas = parse_ideas(result.text)[: call.count + 2]
        entry.update(ok=bool(ideas), parsed=len(ideas), seconds=round(time.monotonic() - t0, 1))
        if not ideas:
            entry["error"] = f"no parsable ideas (finish_reason={result.finish_reason})"
        for idea in ideas:
            idea.update(
                model=call.model.id,
                family=call.model.family,
                strategy=call.strategy,
                detail=call.detail,
                call=call.index,
            )
            candidates.append(idea)

    # Phase 1: fan out. Leave time for dedup and judging.
    total = deadline - started
    gen_budget = total * (0.6 if req.judge else 0.85)
    tasks = [asyncio.create_task(generate(c)) for c in plan]
    deadline_hit = await _gather_until(
        tasks, gen_budget, soft=min(SOFT_GENERATION_S[req.budget], gen_budget), quorum=GENERATION_QUORUM
    )
    for entry in trace:
        if "ok" not in entry:
            entry.update(ok=False, error="too slow; cancelled")
    failed = [e for e in trace if not e["ok"]]
    if not candidates:
        reasons = Counter(e.get("error", "unknown") for e in failed).most_common(3)
        raise IdeationError(
            "every generation call failed: " + "; ".join(f"{r} (x{c})" for r, c in reasons)
        )
    if failed:
        warnings.append(f"{len(failed)} of {len(plan)} generation calls failed or timed out")
    candidates.sort(key=lambda c: c["call"])  # stable order regardless of arrival

    # Phase 2: dedup.
    texts = [f"{c['title']}. {c['idea']}" for c in candidates]
    lexical = similarity.lexical_matrix(texts)
    dense = None
    if req.embeddings and len(texts) > 2:
        model = os.environ.get("DIVERGENT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        try:
            vectors, embed_cost = await asyncio.wait_for(
                client.embed(model, texts), timeout=max(1.0, min(15.0, deadline - time.monotonic() - 5))
            )
            spend(embed_cost)
            dense = similarity.dense_matrix(vectors)
        except (OpenRouterError, asyncio.TimeoutError) as exc:
            warnings.append(f"embeddings unavailable, lexical dedup only ({str(exc)[:80] or 'timeout'})")
    sim = similarity.combine(
        lexical,
        dense,
        LEX_THRESHOLD_ALONE if dense is None else LEX_THRESHOLD_WITH_DENSE,
        dense_threshold(),
    )
    clusters = similarity.cluster(sim)
    reps = [c[0] for c in clusters]
    m = len(reps)
    rep_sim = [[sim[a][b] for b in reps] for a in reps]

    # Phase 3: judge the unique ideas.
    judged: dict[int, list[tuple[float, float]]] = {}
    judges_used: list[str] = []
    if req.judge:
        judges = judge_pool(pool, int(PROFILES[req.budget]["judges"]))
        items = [(k + 1, candidates[r]["title"], candidates[r]["idea"]) for k, r in enumerate(reps)]

        async def run_judge(judge: ModelSpec, chunk: list[tuple[int, str, str]]) -> None:
            params = fit_params(judge, {"temperature": 0.0})
            params["max_tokens"] = max(2500, 60 * len(chunk) + 1500)
            reasoning = lowest_reasoning(judge)
            if reasoning:
                params["reasoning"] = reasoning
            try:
                async with sem:
                    result = await _chat(client, judge, judge_messages(req, chunk), params)
            except OpenRouterError:
                return
            spend(result.cost)
            scores = parse_scores(result.text)
            valid = {cid for cid, _, _ in chunk}
            got = False
            for cid, pair in scores.items():
                if cid in valid:
                    judged.setdefault(cid - 1, []).append(pair)
                    got = True
            if got and judge.id not in judges_used:
                judges_used.append(judge.id)

        jtasks = [
            asyncio.create_task(run_judge(j, items[s : s + JUDGE_CHUNK]))
            for j in judges
            for s in range(0, len(items), JUDGE_CHUNK)
        ]
        if await _gather_until(jtasks, deadline - time.monotonic() - 2):
            deadline_hit = True
        if not judged:
            warnings.append("judging failed; novelty is structural only and utility is unknown")
        elif len(judged) < m:
            warnings.append(f"{m - len(judged)} ideas were not scored by a judge")

    # Phase 4: score.
    iso = similarity.isolation(rep_sim, list(range(m)))
    novelty: list[float] = []
    utility: list[float] = []
    for k, members in enumerate(clusters):
        rarity = 1.0 / math.sqrt(len({candidates[i]["call"] for i in members}))
        structural = 0.6 * iso[k] + 0.4 * rarity
        if k in judged:
            jn = sum(p[0] for p in judged[k]) / len(judged[k])
            ju = sum(p[1] for p in judged[k]) / len(judged[k])
            novelty.append(0.55 * (jn - 1) / 9 + 0.45 * structural)
            utility.append((ju - 1) / 9)
        else:
            novelty.append(structural)
            utility.append(0.5)

    # Phase 5: select.
    picks = select(novelty, utility, rep_sim, req.n, req.spice)
    front = pareto_front(novelty, utility)
    w = NOVELTY_WEIGHT[req.spice]
    ideas_out = []
    for rank, (k, kind) in enumerate(picks, start=1):
        members = clusters[k]
        rep = candidates[members[0]]
        sources = []
        for i in members:
            src = {"model": candidates[i]["model"], "strategy": candidates[i]["strategy"]}
            if candidates[i]["detail"]:
                src["detail"] = candidates[i]["detail"]
            if src not in sources:
                sources.append(src)
        ideas_out.append(
            {
                "rank": rank,
                "title": rep["title"],
                "idea": rep["idea"],
                "twist": rep["twist"],
                "kind": kind,
                "novelty": round(novelty[k], 3),
                "utility": round(utility[k], 3) if k in judged else None,
                "score": round((novelty[k] ** w) * (max(utility[k], 1e-6) ** (1 - w)), 3),
                "pareto": k in front,
                "support": len(members),
                "sources": sources,
            }
        )

    ok_calls = [e for e in trace if e["ok"]]
    result: dict[str, Any] = {
        "topic": req.topic,
        "requested": req.n,
        "returned": len(ideas_out),
        "spice": req.spice,
        "budget": req.budget,
        "constraints": [c for c in req.constraints if c and c.strip()],
        "ideas": ideas_out,
        "stats": {
            "calls_planned": len(plan),
            "calls_ok": len(ok_calls),
            "raw_ideas": len(candidates),
            "unique_ideas": m,
            "duplicates_merged": len(candidates) - m,
            "wildcards": sum(1 for _, kind in picks if kind == "wildcard"),
            "families": sorted({c["family"] for c in candidates}),
            "models": sorted({c["model"] for c in candidates}),
            "strategies": dict(Counter(e["strategy"] for e in ok_calls)),
            "judges": judges_used,
            "similarity": "embeddings+lexical" if dense is not None else "lexical",
            "cost_usd": round(cost, 5) if cost_known else None,
            "elapsed_s": round(time.monotonic() - started, 1),
            "deadline_hit": deadline_hit,
        },
        "warnings": warnings,
    }
    if len(ideas_out) < req.n:
        result["warnings"].append(
            f"only {len(ideas_out)} unique ideas survived; raise the budget or lower n"
        )
    if include_trace:
        result["trace"] = sorted(trace, key=lambda e: e["call"])
    return result

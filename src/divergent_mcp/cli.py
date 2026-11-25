"""`divergent` command line: ideas, plan, models, strategies."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from typing import Any

from . import __version__, strategies
from .models import BUDGETS, describe_pool, resolve_pool
from .openrouter import OpenRouterClient, OpenRouterError
from .pipeline import MAX_IDEAS, IdeationError, IdeationRequest, ideate, plan_only


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("topic", help="What to brainstorm about")
    p.add_argument("-n", "--count", type=int, default=8, help=f"Ideas to return, 1-{MAX_IDEAS} (default 8)")
    p.add_argument(
        "-s", "--spice", type=int, default=2, choices=range(4),
        help="0 conventional, 1 mild, 2 adventurous (default), 3 wild",
    )
    p.add_argument("-b", "--budget", default="low", choices=BUDGETS, help="Spend level (default low)")
    p.add_argument(
        "-c", "--constraint", action="append", default=[], metavar="TEXT",
        help="A requirement every idea must satisfy; repeatable",
    )
    p.add_argument("--context", help="Background text for the models")
    p.add_argument("-m", "--model", action="append", metavar="ID", help="Use these OpenRouter model ids; repeatable")
    p.add_argument(
        "-l", "--lens", action="append", metavar="NAME", choices=list(strategies.STRATEGIES),
        help="Restrict to these lenses; repeatable (see `divergent strategies`)",
    )
    p.add_argument("--seed", type=int, help="Seed for the plan")
    p.add_argument("--max-calls", type=int, help="Hard cap on generation calls")
    p.add_argument("--json", action="store_true", help="Print the full result as JSON")


def _request(args: argparse.Namespace) -> IdeationRequest:
    return IdeationRequest(
        topic=args.topic,
        n=args.count,
        spice=args.spice,
        budget=args.budget,
        constraints=args.constraint,
        context=args.context,
        models=args.model,
        strategies=args.lens,
        seed=args.seed,
        max_calls=args.max_calls,
        judge=not getattr(args, "no_judge", False),
        embeddings=not getattr(args, "no_embeddings", False),
        deadline_s=getattr(args, "deadline", None),
    )


def _wrap(text: str, indent: str = "   ") -> str:
    return textwrap.fill(text, width=92, initial_indent=indent, subsequent_indent=indent)


def render(result: dict[str, Any]) -> str:
    lines = [f"{result['returned']} ideas on: {result['topic']}", ""]
    marks = {"wildcard": "*", "solid": "+", "balanced": " "}
    for idea in result["ideas"]:
        util = "  n/a" if idea["utility"] is None else f"{idea['utility']:.2f}"
        lines.append(
            f"{idea['rank']:>2}.{marks[idea['kind']]}{idea['title']}  "
            f"[{idea['kind']}, novelty {idea['novelty']:.2f}, utility {util}]"
        )
        lines.append(_wrap(idea["idea"]))
        if idea["twist"]:
            lines.append(_wrap("Twist: " + idea["twist"]))
        src = idea["sources"][0]
        more = f" +{idea['support'] - 1} similar" if idea["support"] > 1 else ""
        detail = f" ({src['detail']})" if src.get("detail") else ""
        lines.append(_wrap(f"via {src['model']} / {src['strategy']}{detail}{more}"))
        lines.append("")
    s = result["stats"]
    cost = "unknown" if s["cost_usd"] is None else f"${s['cost_usd']:.4f}"
    lines.append(
        f"{s['calls_ok']}/{s['calls_planned']} calls, {len(s['families'])} model families, "
        f"{s['raw_ideas']} raw -> {s['unique_ideas']} unique, {s['wildcards']} wildcards (*), "
        f"{s['similarity']} dedup, cost {cost}, {s['elapsed_s']}s"
    )
    lines.extend(f"warning: {w}" for w in result["warnings"])
    return "\n".join(lines)


def render_plan(plan: dict[str, Any]) -> str:
    lines = [
        (
            f"Plan for {plan['requested']} ideas (spice {plan['spice']}, budget "
            f"{plan['budget']}): {len(plan['calls'])} calls, {plan['raw_ideas_requested']} raw "
            f"ideas, {len(plan['families'])} families"
        ),
        "",
    ]
    for c in plan["calls"]:
        detail = f" ({c['detail']})" if c["detail"] else ""
        sampling = " ".join(f"{k}={v}" for k, v in c["sampling"].items())
        lines.append(f"{c['call']:>2}. {c['model']:<36} {c['strategy']}{detail}")
        lines.append(f"      {sampling or 'provider default sampling'}")
    lines.append("")
    lines.append("judges: " + (", ".join(plan["judges"]) or "none"))
    lines.extend(f"warning: {w}" for w in plan["warnings"])
    return "\n".join(lines)


async def _models(budget: str) -> dict[str, Any]:
    async with OpenRouterClient() as client:
        try:
            catalogue = await client.catalogue()
        except OpenRouterError:
            catalogue = None
        return {
            "budget": budget,
            "catalogue": catalogue is not None,
            "models": describe_pool(resolve_pool(catalogue, budget)),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="divergent",
        description=(
            "Divergent ideation over OpenRouter: many model families, many creative lenses, "
            "deduplicated and selected for novelty and utility. Reads OPENROUTER_API_KEY."
        ),
    )
    parser.add_argument("--version", action="version", version=f"divergent-mcp {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ideas = sub.add_parser("ideas", help="Generate ideas on a topic")
    _add_run_args(p_ideas)
    p_ideas.add_argument("--no-judge", action="store_true", help="Skip LLM judging (structural novelty only)")
    p_ideas.add_argument("--no-embeddings", action="store_true", help="Lexical dedup only")
    p_ideas.add_argument("--deadline", type=float, help="Overall time limit in seconds (default 90)")
    p_ideas.add_argument("--trace", action="store_true", help="Include per-call records in --json output")

    p_plan = sub.add_parser("plan", help="Show what a run would do, without spending anything")
    _add_run_args(p_plan)

    p_models = sub.add_parser("models", help="Show the model pool for a budget")
    p_models.add_argument("-b", "--budget", default="low", choices=BUDGETS)
    p_models.add_argument("--json", action="store_true")

    p_strat = sub.add_parser("strategies", help="List the creative lenses")
    p_strat.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "ideas":
            result = asyncio.run(ideate(_request(args), include_trace=args.trace))
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else render(result))
        elif args.command == "plan":
            plan = asyncio.run(plan_only(_request(args)))
            print(json.dumps(plan, indent=2, ensure_ascii=False) if args.json else render_plan(plan))
        elif args.command == "models":
            info = asyncio.run(_models(args.budget))
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                if not info["catalogue"]:
                    print("warning: catalogue unavailable; built-in list shown unchecked")
                for m in info["models"]:
                    price = (
                        f"${m['usd_per_mtok_in']:.2f}/${m['usd_per_mtok_out']:.2f} per Mtok"
                        if m["usd_per_mtok_out"] is not None
                        else "price unknown"
                    )
                    print(f"{m['id']:<38} {m['tier']:<8} {price}")
        else:
            info = strategies.describe()
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                for s in info:
                    weights = "/".join(str(w) for w in s["weight_by_spice"])
                    print(f"{s['name']:<20} weights by spice {weights}")
                    print(_wrap(s["summary"]))
    except (IdeationError, OpenRouterError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

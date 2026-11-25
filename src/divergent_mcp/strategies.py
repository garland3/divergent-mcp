"""Prompt lenses. Each one pushes a model off its most likely answer in a different way.

A lens is a short instruction block that is added to the same shared task and
output contract, so the only thing that varies between calls is the push.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SPICE_MIN, SPICE_MAX = 0, 3

# Upper bound on the probability a model is told to sample below, per spice level.
TAIL_PROBABILITY = (0.30, 0.15, 0.10, 0.05)

HARD_CONSTRAINTS = (
    "It must work with no electricity and no internet.",
    "It must cost nothing: no budget, no purchases, only what is already lying around.",
    "It must be finished, start to end, in under ten minutes.",
    "The most obvious component of any normal solution is banned. Name it, then do without it.",
    "It must be usable by someone who cannot see.",
    "It must still make sense 200 years from now.",
    "It has to fit in a coat pocket.",
    "It must be run entirely by children under ten.",
    "It must get better, not worse, when a thousand people use it at once.",
    "Nobody involved may speak or write a word.",
    "It must be illegal to keep it for more than a day: it has to disappear after use.",
    "It must use only things found in a hardware store.",
    "It has to be delightful for the person who would normally hate it most.",
    "It must work during a week-long storm with everyone stuck indoors.",
)

PERSONAS = (
    "a mycologist who thinks in networks of slow underground exchange",
    "a submarine reactor engineer obsessed with failure modes and sealed systems",
    "a medieval guild master who thinks in apprenticeships, marks and monopolies",
    "a stage magician who designs around attention and misdirection",
    "a beekeeper who thinks in swarms, seasons and division of labour",
    "an air traffic controller who thinks in separation, handoffs and holding patterns",
    "a perfumer who composes in top, heart and base notes over time",
    "a street food vendor with a cart, a queue and forty seconds per customer",
    "a cartographer of places that do not exist yet",
    "a hospice nurse who has heard what people regret",
    "a speedrunner who looks for the glitch that skips the whole level",
    "a restorer of antique clocks who trusts mechanisms over software",
    "an insurance actuary who prices everything by its tail risk",
    "a puppeteer who makes simple objects seem alive",
    "a deep-sea biologist who studies life with no light and immense pressure",
    "a tax lawyer who reads rules for what they forgot to forbid",
)

FORMATS = (
    "a museum placard from the year 2090 describing it as a historic artifact",
    "the first claim of a patent application",
    "a product recall notice explaining what went wrong when it got too popular",
    "a classified ad of at most thirty words",
    "a bedtime story a grandparent tells about it",
    "an error message the user sees when it fails",
    "a recipe with ingredients and steps",
    "the rules card of a board game",
    "a one-star review from someone who misunderstood it",
    "a field guide entry, as if it were a species",
)

STIMULI = (
    "a lighthouse", "sourdough starter", "a zipper", "tidal pools", "a pawn shop",
    "jazz improvisation", "a beaver dam", "the postal system", "origami", "a traffic roundabout",
    "an immune system", "a vending machine", "glaciers", "a library card catalogue",
    "karaoke", "a compost heap", "a ski lift", "braille", "a flea market", "echolocation",
)

SCALE_SHIFTS = (
    "for exactly one person, built around their single strangest habit",
    "for a billion people at once, where even rare cases happen daily",
    "a thousand times smaller than anyone would expect",
    "a thousand times slower: it plays out over decades",
    "a hundred times faster than seems reasonable",
    "with a budget a thousand times larger than is sensible",
    "for a town of two hundred people who all know each other",
)


@dataclass(frozen=True)
class Strategy:
    name: str
    summary: str
    # Relative weight at spice 0, 1, 2, 3.
    weights: tuple[int, int, int, int]


STRATEGIES: dict[str, Strategy] = {
    s.name: s
    for s in (
        Strategy("direct", "Plain brainstorm: the model's own best ideas, no push.", (5, 2, 1, 0)),
        Strategy(
            "verbalized_sampling",
            "Verbalized sampling: ask for low-probability answers from the tails of the "
            "model's own distribution, each with a stated probability.",
            (2, 3, 3, 3),
        ),
        Strategy(
            "hard_constraint",
            "A random hard constraint that rules out the usual solutions.",
            (1, 2, 2, 3),
        ),
        Strategy(
            "inversion",
            "Inversion: name the assumption everyone makes, then build on its opposite; or "
            "design the worst idea and flip it.",
            (0, 1, 2, 3),
        ),
        Strategy(
            "persona",
            "Cross-domain persona: answer as an expert from an unrelated field, using "
            "that field's mechanisms.",
            (1, 2, 2, 3),
        ),
        Strategy(
            "format_shift",
            "Format shift: think about the idea as an unusual artifact first, then "
            "distil the idea back out.",
            (0, 1, 2, 2),
        ),
        Strategy(
            "forced_analogy",
            "Forced analogy: map the structure of an unrelated thing onto the topic.",
            (0, 1, 2, 3),
        ),
        Strategy(
            "scale_shift",
            "Scale shift: change the size, speed, audience or budget by orders of magnitude.",
            (1, 1, 2, 2),
        ),
    )
}

SYSTEM_PROMPT = (
    "You are one voice in a large brainstorming ensemble. Other voices already cover the "
    "obvious, sensible ideas, so repeating those is wasted effort. Your job is the ideas only "
    "you would produce from the angle you are given. Every idea must still be a real answer to "
    "the topic: concrete, specific and possible to act on, not a slogan. "
    "Reply with JSON only, no prose and no code fences."
)


def clamp_spice(spice: int) -> int:
    return max(SPICE_MIN, min(SPICE_MAX, int(spice)))


def lens_text(name: str, spice: int, rng: random.Random) -> tuple[str, str]:
    """The instruction block for one call, and a short label of what was drawn."""
    spice = clamp_spice(spice)
    detail = ""
    if name == "direct":
        text = "Give your best, clearly distinct ideas."
    elif name == "verbalized_sampling":
        p = TAIL_PROBABILITY[spice]
        detail = f"p<{p:.2f}"
        text = (
            "Imagine the full distribution of answers you could give to this topic. Do not "
            "give the likely ones. Sample from the tails: each idea must be one you would "
            f"give with probability below {p:.2f}. Report that probability honestly for each "
            "idea in the `probability` field."
        )
    elif name == "hard_constraint":
        detail = rng.choice(HARD_CONSTRAINTS)
        text = f"Extra hard constraint for this round, no exceptions: {detail}"
    elif name == "inversion":
        if rng.random() < 0.5:
            detail = "reverse the core assumption"
            text = (
                "First name, silently, the assumption nearly everyone makes about this topic. "
                "Every idea must start from that assumption being false or reversed."
            )
        else:
            detail = "worst idea, flipped"
            text = (
                "First imagine the worst possible ideas for this topic: harmful, embarrassing, "
                "absurd. Then flip each one into an idea that is good because of what made the "
                "bad version bad. Return only the flipped ideas."
            )
    elif name == "persona":
        detail = rng.choice(PERSONAS)
        text = (
            f"Answer as {detail}. You know little about the usual approaches to this topic, "
            "so bring the mechanisms, habits and vocabulary of your own field and apply them "
            "literally."
        )
    elif name == "format_shift":
        detail = rng.choice(FORMATS)
        text = (
            f"For each idea, first imagine it written as {detail}. Let that format force out "
            "details a normal brainstorm would skip. Then state the underlying idea plainly "
            "in the `idea` field."
        )
    elif name == "forced_analogy":
        detail = rng.choice(STIMULI)
        text = (
            f"Forced connection: {detail}. Work out how it actually functions (its parts, "
            "flows and rules), then carry that structure over to the topic. Each idea must "
            "borrow a different working principle from it, not just its imagery."
        )
    elif name == "scale_shift":
        detail = rng.choice(SCALE_SHIFTS)
        text = f"Design it {detail}. Follow where that leads rather than softening it."
    else:
        raise KeyError(f"unknown strategy {name!r}")
    return text, detail


def build_messages(
    topic: str,
    *,
    strategy: str,
    count: int,
    spice: int,
    rng: random.Random,
    constraints: list[str] | None = None,
    context: str | None = None,
) -> tuple[list[dict[str, str]], str]:
    """Chat messages for one generation call, and the label of the drawn lens detail."""
    lens, detail = lens_text(strategy, spice, rng)
    parts = [f"Topic: {topic.strip()}"]
    if context and context.strip():
        parts.append(f"Background:\n{context.strip()}")
    if constraints:
        listed = "\n".join(f"- {c.strip()}" for c in constraints if c and c.strip())
        if listed:
            parts.append(f"Requirements every idea must satisfy:\n{listed}")
    parts.append(f"Your angle: {lens}")
    parts.append(
        f"Give exactly {count} ideas that differ from each other in kind, not in detail.\n"
        'Reply with this JSON object and nothing else:\n'
        '{"ideas": [{"title": "<at most 8 words>", '
        '"idea": "<2-3 concrete sentences: what it is and how it works>", '
        '"twist": "<one sentence: what makes it non-obvious>", '
        '"probability": <number between 0 and 1: how likely a typical answer is to contain '
        "this idea>}]}"
    )
    return (
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        detail,
    )


def allocate(total: int, spice: int, rng: random.Random, only: list[str] | None = None) -> list[str]:
    """Split `total` calls across strategies in proportion to their spice weights.

    Largest-remainder apportionment with random tie-breaks, then shuffled, so a small
    budget still spreads over the highest-weight lenses instead of repeating one.
    """
    spice = clamp_spice(spice)
    names = list(only) if only else list(STRATEGIES)
    for n in names:
        if n not in STRATEGIES:
            raise KeyError(f"unknown strategy {n!r}; known: {', '.join(STRATEGIES)}")
    weights = {n: STRATEGIES[n].weights[spice] for n in names}
    if only and sum(weights.values()) == 0:
        weights = {n: 1 for n in names}  # explicitly requested lenses always run
    weights = {n: w for n, w in weights.items() if w > 0}
    wsum = sum(weights.values())
    quotas = {n: total * w / wsum for n, w in weights.items()}
    counts = {n: int(q) for n, q in quotas.items()}
    leftover = total - sum(counts.values())
    order = sorted(weights, key=lambda n: (quotas[n] - counts[n], rng.random()), reverse=True)
    for n in order[:leftover]:
        counts[n] += 1
    out = [n for n, c in counts.items() for _ in range(c)]
    rng.shuffle(out)
    return out


def describe() -> list[dict[str, object]]:
    return [
        {"name": s.name, "summary": s.summary, "weight_by_spice": list(s.weights)}
        for s in STRATEGIES.values()
    ]

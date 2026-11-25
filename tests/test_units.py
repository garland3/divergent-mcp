import random

import pytest

from divergent_mcp import similarity, strategies
from divergent_mcp.models import (
    ModelSpec,
    fit_params,
    judge_pool,
    lowest_reasoning,
    resolve_pool,
)
from divergent_mcp.parsing import extract_json, parse_ideas, parse_scores
from divergent_mcp.pipeline import (
    UTILITY_FLOOR,
    IdeationRequest,
    build_plan,
    pareto_front,
    select,
)

from .conftest import CATALOGUE

# ---------------------------------------------------------------- parsing


def test_extract_json_handles_fences_prose_and_think_blocks():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```\nDone.') == {"a": [1, 2]}
    assert extract_json('<think>{"no": 1}</think> answer: {"a": "x } y"} trailing') == {"a": "x } y"}
    assert extract_json("no json here") is None
    assert extract_json("") is None


def test_parse_ideas_normalises_shapes():
    text = '{"ideas": [{"title": "T", "idea": "An  idea.\\n", "twist": "w", "probability": "5%"}, "bare string idea", {"name": "N", "description": "D", "probability": 7}, {"nothing": 1}]}'
    ideas = parse_ideas(text)
    assert [i["title"] for i in ideas] == ["T", "bare string idea", "N"]
    assert ideas[0]["idea"] == "An idea."
    assert ideas[0]["probability"] == 0.05
    assert ideas[2]["probability"] is None  # 7 is not a probability
    assert parse_ideas('[{"idea": "top-level list works"}]')[0]["idea"] == "top-level list works"
    assert parse_ideas("garbage") == []


def test_parse_scores_clamps_and_skips_bad_rows():
    text = '{"scores": [{"id": 1, "novelty": 14, "utility": 0}, {"id": "2", "novelty": "7", "utility": 3.5}, {"id": 3}]}'
    assert parse_scores(text) == {1: (10.0, 1.0), 2: (7.0, 3.5)}


def test_parse_ideas_salvages_the_valid_items_from_broken_json():
    good = '{"title": "A", "idea": "First."}'
    broken = '{"title": "B", "idea": "has an "unescaped" quote"}'
    text = '{"ideas": [' + good + ", " + broken + ', {"title": "C", "idea": "Third."}, {"title": "D", "idea": "cut off'
    assert [i["title"] for i in parse_ideas(text)] == ["A", "C"]


# ---------------------------------------------------------------- strategies


def test_allocation_follows_spice_weights():
    wild = strategies.allocate(16, 3, random.Random(0))
    assert len(wild) == 16
    assert "direct" not in wild  # weight 0 at spice 3
    assert len(set(wild)) == 7
    tame = strategies.allocate(9, 0, random.Random(0))
    assert tame.count("direct") >= 4
    assert "inversion" not in tame and "forced_analogy" not in tame


def test_allocation_respects_an_explicit_subset_even_at_zero_weight():
    assert set(strategies.allocate(4, 0, random.Random(1), ["inversion"])) == {"inversion"}
    with pytest.raises(KeyError):
        strategies.allocate(4, 2, random.Random(1), ["nope"])


@pytest.mark.parametrize("name", list(strategies.STRATEGIES))
def test_every_lens_builds_a_prompt_with_the_shared_contract(name):
    messages, _ = strategies.build_messages(
        "better bus stops",
        strategy=name,
        count=5,
        spice=3,
        rng=random.Random(3),
        constraints=["under $100", "  "],
        context="A town of 9,000.",
    )
    user = messages[1]["content"]
    assert messages[0]["role"] == "system"
    assert "Topic: better bus stops" in user
    assert "- under $100" in user and "A town of 9,000." in user
    assert "Give exactly 5 ideas" in user and '"probability"' in user


def test_verbalized_sampling_tightens_the_tail_with_spice():
    low, _ = strategies.lens_text("verbalized_sampling", 0, random.Random(0))
    high, detail = strategies.lens_text("verbalized_sampling", 3, random.Random(0))
    assert "below 0.30" in low and "below 0.05" in high and detail == "p<0.05"


# ---------------------------------------------------------------- models


def test_pool_has_one_model_per_family_and_reads_catalogue_capabilities():
    pool = resolve_pool(CATALOGUE, "low")
    families = [m.family for m in pool]
    assert len(families) == len(set(families)) == 9
    assert all(m.verified for m in pool)
    haiku = next(m for m in pool if m.family == "anthropic")
    assert "temperature" not in haiku.supported


def test_missing_preferred_model_falls_back_to_newest_of_the_same_vendor():
    cat = [e for e in CATALOGUE if e["id"] != "qwen/qwen3.8-flash"]
    cat += [
        {"id": "qwen/qwen9-flash", "created": 2_000_000_000, "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}, "supported_parameters": ["temperature"]},
        {"id": "qwen/qwen9-coder", "created": 2_100_000_000, "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}},
        {"id": "qwen/qwen9-max", "created": 2_200_000_000, "pricing": {"prompt": "0.000004", "completion": "0.00002"}},
    ]
    qwen = next(m for m in resolve_pool(cat, "low") if m.family == "qwen")
    assert qwen.id == "qwen/qwen9-flash"  # not the coder variant, not the one over the price cap


def test_pool_without_a_catalogue_uses_the_builtin_list_unverified():
    pool = resolve_pool(None, "medium")
    assert pool and not any(m.verified for m in pool)
    assert all(m.supported is None for m in pool)
    assert any(m.tier == "premium" for m in pool)


def test_explicit_models_and_env_override_replace_the_pool(monkeypatch):
    pool = resolve_pool(CATALOGUE, "low", ["acme/new-model", "openai/gpt-6.1-terra"])
    assert [(m.id, m.verified) for m in pool] == [("acme/new-model", False), ("openai/gpt-6.1-terra", True)]
    monkeypatch.setenv("DIVERGENT_MODELS", "a/one, b/two")
    assert [m.id for m in resolve_pool(CATALOGUE, "low")] == ["a/one", "b/two"]
    with pytest.raises(ValueError):
        resolve_pool(CATALOGUE, "enormous")


def test_params_are_fitted_to_what_each_model_accepts():
    pool = {m.family: m for m in resolve_pool(CATALOGUE, "low")}
    sampling = {"temperature": 1.4, "top_p": 0.9, "presence_penalty": 0.5}
    assert fit_params(pool["anthropic"], sampling) == {}
    assert fit_params(pool["google"], sampling) == {"temperature": 1.4, "top_p": 0.9}
    unknown = ModelSpec(id="anthropic/x", family="anthropic")
    assert fit_params(unknown, sampling)["temperature"] == 1.0  # clamped, nothing dropped
    assert len(fit_params(unknown, sampling)) == 3


def test_reasoning_is_turned_down_as_far_as_each_model_allows():
    pool = {m.family: m for m in resolve_pool(CATALOGUE, "low")}
    assert lowest_reasoning(pool["anthropic"]) == {"effort": "none"}
    assert lowest_reasoning(pool["google"]) == {"effort": "low"}  # mandatory: cannot be off
    assert lowest_reasoning(pool["qwen"]) == {"enabled": False}  # no effort levels listed
    assert lowest_reasoning(pool["deepseek"]) is None  # no reasoning support
    assert lowest_reasoning(ModelSpec(id="x/y", family="x")) is None


def test_judges_come_from_different_families():
    judges = judge_pool(resolve_pool(CATALOGUE, "low"), 2)
    assert [j.family for j in judges] == ["google", "openai"]


# ---------------------------------------------------------------- similarity


def test_lexical_similarity_separates_paraphrase_from_unrelated():
    texts = [
        "Lend power tools and ladders from the library alongside books.",
        "The library lends tools like drills and ladders, not only books.",
        "Host a midnight astronomy night with telescopes on the roof.",
    ]
    sim = similarity.lexical_matrix(texts)
    assert sim[0][1] > 0.25 > sim[0][2]
    assert sim[1][0] == sim[0][1] and sim[2][2] == 1.0


def test_dense_similarity_is_mean_centred():
    # All vectors share a large common component; centring must remove it.
    vectors = [[10, 1, 0], [10, 1.1, 0], [10, 0, 1], [10, -1, -1]]
    sim = similarity.dense_matrix(vectors)
    assert sim[0][1] > 0.9
    assert sim[0][3] < 0


def test_combine_cluster_and_isolation():
    lexical = [[1, 0.5, 0.0], [0.5, 1, 0.1], [0.0, 0.1, 1]]
    dense = [[1, 0.1, 0.7], [0.1, 1, 0.0], [0.7, 0.0, 1]]
    lex_only = similarity.combine(lexical, None, 0.45, 0.62)
    assert similarity.cluster(lex_only) == [[0, 1], [2]]
    both = similarity.combine(lexical, dense, 0.45, 0.62)
    assert similarity.cluster(both) == [[0, 1, 2]]  # either signal is enough
    iso = similarity.isolation(lex_only, [0, 1, 2])
    assert iso[2] == 1.0 and min(iso.values()) == 0.0


# ---------------------------------------------------------------- planning


def test_plan_is_deterministic_and_spreads_over_families():
    pool = resolve_pool(CATALOGUE, "low")
    req = IdeationRequest("topic", n=10, spice=2, seed=7)
    a = [c.summary() for c in build_plan(req, pool)]
    b = [c.summary() for c in build_plan(req, pool)]
    assert a == b
    assert len(a) == 8 and len({c["model"] for c in a}) == 8
    assert sum(c["ideas"] for c in a) >= 30  # 3x oversampling at budget low
    assert len({c["strategy"] for c in a}) >= 5
    other = [c.summary() for c in build_plan(IdeationRequest("topic", n=10, seed=8), pool)]
    assert other != a


def test_plan_jitters_sampling_and_respects_model_capabilities():
    pool = resolve_pool(CATALOGUE, "high")
    plan = build_plan(IdeationRequest("topic", n=20, spice=3, budget="high", seed=1), pool)
    assert len(plan) == 20
    temps = {c.params["temperature"] for c in plan if "temperature" in c.params}
    assert len(temps) > 5 and all(0.3 <= t <= 1.5 for t in temps)
    for call in plan:
        assert call.params["max_tokens"] >= 2500
        if call.model.family in ("anthropic", "openai"):
            assert "temperature" not in call.params
            assert call.params["reasoning"] == {"effort": "none"}
    calm = build_plan(IdeationRequest("topic", n=20, spice=0, budget="high", seed=1), pool)
    assert max(c.params.get("temperature", 0) for c in calm) < min(temps) + 0.5


def test_plan_caps_and_validation():
    pool = resolve_pool(CATALOGUE, "low")
    assert len(build_plan(IdeationRequest("t", n=30, max_calls=3), pool)) == 3
    assert build_plan(IdeationRequest("t", n=30), pool)[0].count <= 8
    for bad in (
        IdeationRequest(" "),
        IdeationRequest("t", n=0),
        IdeationRequest("t", n=31),
        IdeationRequest("t", spice=4),
        IdeationRequest("t", budget="huge"),
        IdeationRequest("t", strategies=["nope"]),
    ):
        with pytest.raises(ValueError):
            bad.validate()


# ---------------------------------------------------------------- selection


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def test_selection_keeps_high_novelty_wildcards_alive():
    # 0-3: safe and useful. 4: strange but workable. 5: strange and useless.
    novelty = [0.30, 0.32, 0.28, 0.35, 0.95, 0.99]
    utility = [0.90, 0.88, 0.92, 0.85, 0.40, 0.05]
    picks = dict(select(novelty, utility, _identity(6), 3, spice=2))
    assert picks[4] == "wildcard"  # would lose on any averaged score
    assert 5 not in picks  # below the utility floor: novelty alone is not enough
    assert utility[5] < UTILITY_FLOOR
    assert sum(1 for k in picks.values() if k == "solid") == 2


def test_spice_controls_the_wildcard_share():
    novelty = [i / 10 for i in range(10)]
    utility = [1 - i / 12 for i in range(10)]
    count = lambda spice: sum(1 for _, k in select(novelty, utility, _identity(10), 8, spice) if k == "wildcard")
    assert [count(s) for s in range(4)] == [1, 2, 3, 4]


def test_selection_penalises_near_duplicates_of_what_is_already_picked():
    novelty = [0.9, 0.89, 0.6]
    utility = [0.8, 0.8, 0.7]
    sim = _identity(3)
    sim[0][1] = sim[1][0] = 0.95  # 1 is nearly the same idea as 0
    assert {i for i, _ in select(novelty, utility, sim, 2, spice=0)} == {0, 2}


def test_selection_handles_small_pools():
    assert select([], [], [], 5, 2) == []
    assert len(select([0.5], [0.5], [[1.0]], 5, 2)) == 1


def test_pareto_front():
    assert pareto_front([0.9, 0.5, 0.2, 0.4], [0.2, 0.5, 0.9, 0.4]) == {0, 1, 2}

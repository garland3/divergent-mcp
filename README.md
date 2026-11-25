# divergent-mcp

Divergent ideation over [OpenRouter](https://openrouter.ai): ask for N ideas on a topic
and get a set drawn from many model families, each pushed off its default answer by a
different creative lens, then deduplicated and selected for both novelty and usefulness.

It ships as a CLI (`divergent`) and a stdio MCP server (`divergent-mcp`).

## Why

One prompt to one model returns that model's most likely ideas. Asking again, or asking
for more, mostly returns variations on them. This tool widens the search in three ways
and then narrows it without averaging the interesting ideas away:

1. **Many models.** Calls fan out in parallel across families (Anthropic, Google,
   OpenAI, DeepSeek, Qwen, Mistral, Llama, GLM, Cohere, Grok). Different training data
   gives different defaults.
2. **Many lenses.** Each call gets a different prompt strategy and jittered sampling
   settings (temperature, top_p, presence and frequency penalties), in small batches.
3. **Selection that protects wildcards.** Ideas are clustered, scored for novelty and
   utility separately, and a share of the output is reserved for the most novel ideas
   that clear a low utility bar.

## Install

Needs Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/garland3/divergent-mcp.git
cd divergent-mcp
uv sync
export OPENROUTER_API_KEY=...   # never put the key in a file in this repo
```

## CLI

```bash
uv run divergent ideas "ways to make a neighborhood library more used"
uv run divergent ideas "names for a sourdough bakery" -n 12 --spice 3 --budget medium
uv run divergent ideas "onboarding for a CLI tool" -c "no network access" -c "under 5 minutes"
uv run divergent ideas "reduce meeting load" --json --trace > run.json

uv run divergent plan "reduce meeting load" -n 10 -b high   # free: shows the planned calls
uv run divergent models -b medium                           # the model pool right now
uv run divergent strategies                                 # the lenses
```

| Option | Meaning |
|---|---|
| `-n, --count` | Ideas to return, 1-30 (default 8) |
| `-s, --spice` | 0 conventional, 1 mild, 2 adventurous (default), 3 wild |
| `-b, --budget` | `low` (default), `medium`, `high` |
| `-c, --constraint` | A requirement every idea must satisfy; repeatable |
| `--context` | Background text for the models |
| `-m, --model` | Use these OpenRouter model ids instead of the pool; repeatable |
| `-l, --lens` | Restrict to these lenses; repeatable |
| `--seed` | Seed for the plan (model/lens pairing, sampling jitter) |
| `--max-calls` | Hard cap on generation calls |
| `--no-judge` | Skip LLM judging; novelty is structural only, utility unknown |
| `--no-embeddings` | Lexical dedup only |
| `--deadline` | Overall time limit in seconds (default 90) |
| `--json`, `--trace` | Full result as JSON; add one record per model call |

Output marks each idea `*` wildcard, `+` solid, or balanced, with novelty and utility
from 0 to 1 and the model and lens that produced it. The last line reports calls, model
families, raw and unique idea counts, cost and time.

A `low` run for 8 ideas makes 6 generation calls, one embeddings call and one judge
call, takes roughly 15-25 seconds and costs about one cent. `high` uses premium models
and costs 10-20 times that.

## MCP server

```bash
uv run divergent-mcp --stdio        # serve (stdio is the default)
uv run divergent-mcp --list-tools
```

| Tool | What it does |
|---|---|
| `generate_ideas` | Run the pipeline. Arguments: `topic`, `n`, `spice`, `budget`, `constraints`, `context`, `models`, `lenses`, `seed`, `include_trace` |
| `plan_ideation` | Preview the calls a run would make. Spends nothing, needs no key |
| `list_strategies` | The lenses, their weight at each spice level, and the budget profiles |
| `list_models` | The model pool a budget resolves to right now, with prices |

Client configuration (the key comes from the client's environment):

```json
{
  "mcpServers": {
    "divergent": {
      "command": "/path/to/divergent-mcp/.venv/bin/divergent-mcp",
      "args": ["--stdio"],
      "env": { "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}" }
    }
  }
}
```

The topic, context and constraints are sent to external model providers through
OpenRouter. Do not use it for text that must stay private.

## How it works

```
plan -> fan out -> dedup -> judge -> score -> select
```

**Plan.** The budget sets the number of calls and the batch size, oversampling the
request 3-5x. Calls are spread over a shuffled pool with one model per family, and over
lenses in proportion to their weight at the chosen spice level.

**Lenses.**

| Lens | The push |
|---|---|
| `direct` | None: the model's own best ideas. Dominant at spice 0, absent at spice 3 |
| `verbalized_sampling` | Sample from the tails: each idea must have a stated probability below a threshold (0.30 down to 0.05 with spice) |
| `hard_constraint` | A random constraint that rules out the usual solutions |
| `inversion` | Reverse the assumption everyone makes, or design the worst idea and flip it |
| `persona` | Answer as an expert from an unrelated field, using that field's mechanisms |
| `format_shift` | Imagine the idea as a patent claim, a recall notice, a recipe, then distil it |
| `forced_analogy` | Map the working structure of an unrelated thing onto the topic |
| `scale_shift` | Change size, speed, audience or budget by orders of magnitude |

**Fan out.** All calls run concurrently. Sampling parameters are fitted to what each
model accepts according to the OpenRouter catalogue, and reasoning is turned down as far
as each model allows (left at default, some small models spend the whole token budget
thinking). A call rejected with HTTP 400 is retried once as a plain request. After 25
seconds the run stops waiting once 60% of calls are back, so one slow model does not hold
it. Replies are parsed forgivingly: code fences, surrounding prose and partly broken JSON
are all handled.

**Dedup.** Ideas are embedded through OpenRouter and compared by cosine after
subtracting the pool mean, which removes what all ideas share because they are about the
same topic. Ideas above the threshold join one cluster. If embeddings fail, a built-in
TF-IDF comparison is used instead. How many separate calls produced a cluster is kept as
`support`.

**Judge.** One or two judge models from different families rate each unique idea for
novelty and utility on separate 1-10 scales, with instructions not to let one leak into
the other.

**Score.** Novelty blends the judge rating (55%) with two structural signals: how
isolated the idea is from its nearest neighbours in the pool, and how few independent
calls arrived at it. Utility is the judge rating.

**Select.** A spice-dependent share of the slots (10% to 50%) goes to the most novel
ideas whose utility clears a low floor: these are the wildcards. The remaining slots are
filled by a geometric blend of novelty and utility. Both passes penalise similarity to
ideas already chosen. A single blended score would let well-rounded ideas crowd out
strange ones; reserving slots is what keeps them.

Every stage degrades rather than fails: no catalogue means the built-in model list is
used unchecked, no embeddings means lexical dedup, no judge means structural novelty with
utility reported as unknown. The result's `warnings` say which of these happened.

## Configuration

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Required to generate. Read from the environment only |
| `DIVERGENT_MODELS` | Comma-separated model ids that replace the default pool |
| `DIVERGENT_EMBEDDING_MODEL` | Embedding model (default `perplexity/pplx-embed-v1-0.6b`) |
| `DIVERGENT_DEDUP_THRESHOLD` | Duplicate threshold on centred embedding cosine (default 0.40); retune if you change the embedding model |
| `DIVERGENT_DEADLINE_S` | Overall time limit per run in seconds (default 90) |
| `OPENROUTER_BASE_URL` | API base (default `https://openrouter.ai/api/v1`) |

Model ids on OpenRouter change often. The preferred ids live in
`src/divergent_mcp/models.py`; when one disappears from the catalogue, the newest
general-purpose model from the same vendor within the tier's price cap is used instead.

## Development

```bash
uv sync --extra dev
uv run pytest -q            # 51 tests against a fake OpenRouter; no network, no key
uv run ruff check .

# Optional live smoke test, well under one cent:
DIVERGENT_LIVE=1 OPENROUTER_API_KEY=... uv run pytest -m live -q
```

## License

MIT

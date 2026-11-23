# jev-tau2-harness

Replication code and saved runs for the preprint

> **System-1 Orchestration for Enterprise Conversational Agents: Typed Decisions Beside the LLM on τ²-Bench Airline**
> Sai Vamshi Atukuri, 2026. Preprint (Preprints.org). DOI to be added.
> Full text (HTML) and PDF: https://vamshibobby.github.io/papers/system1-orchestration-tau2-airline/

Tool-calling chat agents keep the customer waiting through one LLM call after another, and many of those calls make
decisions that need no general reasoning. The paper puts a **decision model** *beside* the agent's LLM, not in front
of it. A decision model answers yes/no and multiple-choice questions with probabilities; the experiments use Jev, a
hosted typed-decision model. One Jev call per customer turn runs in parallel with a speculative model call. It decides
whether the turn needs a tool, which of the customer's reservations the turn is about, whether flight statuses matter,
whether the customer confirmed a write, and the route and date of a flight search. Code then issues those tool calls
in one batch and checks every write against the airline policy. On τ²-Bench Airline (50 tasks × 4 trials, Gemini 2.5
Flash as the simulated customer) the paper compares three agent models, alone and inside the harness: code issued
77–84% of tool calls, pass^4 rose by 10.7 points on average, and conversations with a policy-forbidden write fell from
43 to 10. Four smaller open models (50 tasks × 1 trial) are compared as well.

This repository holds the harness, the baseline agent, the run and evaluation scripts, and all 16 runs reported in the
paper (τ² transcripts and per-call event logs). Every table and figure can be regenerated offline from them.

## Contents

| Path | What it is |
|---|---|
| `agents.py` | The agents: `FlashAgent` (`--agent flash`, the model alone) and `JevV5Agent` (`--agent jev`, the harness in the paper). `JevAgent` (`--agent jev_v4`) is the earlier harness iteration that v5 extends. |
| `jev.py` | Jev client and every question the harness asks (Table 1) |
| `policy.py` | The write gate's airline-policy rules, and the itinerary totals appended to search results |
| `gemini.py` | Minimal Gemini REST client (Vertex AI or Gemini API) that records time to first token |
| `openai_chat.py` | Streaming client for OpenAI-compatible endpoints: Vertex AI managed models (Gemma) and OpenRouter |
| `events.py` | The per-call event log (`results/<run>/events.jsonl`) |
| `run_tau2.py` | Runs τ²-Bench airline with one agent and one model |
| `eval_tau2.py` | All metrics; `--paired` / `--pooled` add bootstrap statistics; `--review-failures` runs τ²'s LLM reviewer |
| `check_policy.py` | Offline check that the policy rules never block a correct (gold) write |
| `reproduce_tables.py` | Regenerates Tables 3–6, Figures 2–3 and the numbers quoted in the text, and compares them with the paper |
| `docs/HOW_IT_WORKS.md` | One turn of the harness in detail, every Jev question and threshold, the event log and the metrics |
| `results/` | The 16 runs of the paper |

## Setup

Python 3.12 or 3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # includes tau2-bench pinned to commit 5bfa7e3 (v1.0.1)
```

That is enough for `reproduce_tables.py` and `eval_tau2.py`. New runs and `check_policy.py` also need τ²'s airline
database, which lives in the tau2-bench repository rather than in the package. Fetch just that folder at the same
commit and point `TAU2_DATA_DIR` at it:

```bash
git clone --filter=blob:none --no-checkout https://github.com/sierra-research/tau2-bench.git vendor/tau2-bench
git -C vendor/tau2-bench sparse-checkout set data/tau2/domains/airline
git -C vendor/tau2-bench checkout 5bfa7e37b36656b37dc6d022156be6563c1007f3
export TAU2_DATA_DIR="$PWD/vendor/tau2-bench/data"
```

### Credentials (new runs only)

Copy `.env.example` to `.env` and fill in what your runs need:

* **Jev** (the decision model), through its provider's API: `OPENROUTER_API_KEY`. Jev was called as
  `~typesafe/jev-latest` between 30 September and 6 October 2026.
* **Agent models and the simulated customer**: the paper served Gemini 3.8 Flash, Gemini 3.5 Flash-Lite, Gemma 4 26B
  A4B and the simulated customer (Gemini 2.5 Flash) from Vertex AI. Set `GOOGLE_APPLICATION_CREDENTIALS` (a
  service-account key with the Vertex AI User role) and `VERTEX_PROJECT`. You can use the Gemini API instead
  (`GEMINI_API_KEY`, plus `--user-llm gemini/gemini-2.5-flash`). The simulated customer takes any litellm model id.
* **Smaller open models** (Table 5): an OpenAI-compatible endpoint. The paper used OpenRouter with one pinned provider
  per model (`--model openrouter/<id> --provider <slug>`).

## Regenerate the paper's tables and figures (offline, free)

```bash
python reproduce_tables.py
```

It prints Tables 3, 4, 5 and 6, the data behind Figures 2 and 3, and the numbers quoted in the text. Each one is
compared with the paper, and any difference is listed at the end. It writes `reproduced/tables.txt`, the figure data
as CSV, and the two figures as PNG files. `check_policy.py` runs as part of it when `TAU2_DATA_DIR` is set. The
script needs about a minute and no API keys.

Current result: every table, figure and quoted number in the paper matches.

The individual scripts:

```bash
python eval_tau2.py                                          # Table 3 and Table 6 for the six 4-trial runs
python eval_tau2.py latency_flash_alone latency_flash_harness
python eval_tau2.py small_nemotron35_alone small_nemotron35_harness   # and the other small_* pairs
python eval_tau2.py --paired flash_alone flash_harness       # bootstrap intervals and sign test for one pair
python eval_tau2.py --pooled                                 # effect averaged over the three main models
python check_policy.py                                       # the gate blocks 0 of 31 gold writes (needs TAU2_DATA_DIR)
```

## Rerun the 16 runs

Every run writes `results/<name>/results.json` (τ²'s format) and `results/<name>/events.jsonl`. Defaults are the
paper's settings: τ²-Bench airline, all 50 tasks, seed 300, 4 trials, Gemini 2.5 Flash as the simulated customer,
τ²'s original system prompt, thinking level low for Gemini. Use a new `--name` to keep the shipped runs.

**Table 3 (and Tables 4 and 6): three models, 50 tasks × 4 trials.** Flash-Lite's two runs were made side by side
(at the same time, in two terminals), because Table 4 uses their latencies.

```bash
python run_tau2.py --agent flash --name flash_alone
python run_tau2.py --agent jev   --name flash_harness
python run_tau2.py --agent flash --name flashlite_alone   --model gemini-3.5-flash-lite
python run_tau2.py --agent jev   --name flashlite_harness --model gemini-3.5-flash-lite
python run_tau2.py --agent flash --name gemma_alone       --model google/gemma-4-26b-a4b-it-maas
python run_tau2.py --agent jev   --name gemma_harness     --model google/gemma-4-26b-a4b-it-maas
```

**Table 4 and Figures 2–3, Gemini 3.8 Flash latency: 50 tasks × 1 trial, both arms side by side.**

```bash
python run_tau2.py --agent flash --name latency_flash_alone   --trials 1
python run_tau2.py --agent jev   --name latency_flash_harness --trials 1
```

**Table 5: smaller open models through OpenRouter, 50 tasks × 1 trial, each pair side by side on one pinned
provider.** The paper's providers were DeepInfra (FP8) for Qwen, DeepInfra (BF16) for Nemotron, Friendli for Gemma 4
31B and Upstage for Solar Mini 4. Pass them as OpenRouter provider slugs to `--provider`; check the slug on the
model's OpenRouter page. `--thinking low` sets reasoning effort low, as in the paper; the other two models ran at
their defaults.

```bash
python run_tau2.py --agent flash --name small_qwen35a3b_alone    --trials 1 --model openrouter/qwen/qwen3.6-35b-a3b --provider deepinfra/fp8 --thinking low
python run_tau2.py --agent jev   --name small_qwen35a3b_harness  --trials 1 --model openrouter/qwen/qwen3.6-35b-a3b --provider deepinfra/fp8 --thinking low
python run_tau2.py --agent flash --name small_nemotron35_alone   --trials 1 --model openrouter/nvidia/nemotron-3.5-lightning --provider deepinfra/bf16 --thinking low
python run_tau2.py --agent jev   --name small_nemotron35_harness --trials 1 --model openrouter/nvidia/nemotron-3.5-lightning --provider deepinfra/bf16 --thinking low
python run_tau2.py --agent flash --name small_gemma31b_alone     --trials 1 --model openrouter/google/gemma-4-31b-it --provider friendli
python run_tau2.py --agent jev   --name small_gemma31b_harness   --trials 1 --model openrouter/google/gemma-4-31b-it --provider friendli
python run_tau2.py --agent flash --name small_solarmini4_alone   --trials 1 --model openrouter/upstage/solar-mini4 --provider upstage
python run_tau2.py --agent jev   --name small_solarmini4_harness --trials 1 --model openrouter/upstage/solar-mini4 --provider upstage
```

**Table 6, lower rows:** τ²'s LLM reviewer (Gemini 2.5 Pro as judge, on Vertex AI) on the failures of the two
4-trial Gemini 3.8 Flash runs:

```bash
python eval_tau2.py --review-failures flash_alone flash_harness   # writes results/<run>/failure_review.json
```

Other flags: `--task-ids 1 24` (a subset; a 2-task, 1-trial smoke test costs about $0.15), `--concurrency` (default
3 conversations at a time), `--resume` (repeat an interrupted run, or one where some conversations ended in an endpoint
error), `--spend-stop <dollars>` (caps the agent model's spend). Point `reproduce_tables.py` at new runs by editing the
run names at the top of the file, or use `eval_tau2.py <run> <run>`.

### Cost and time

Measured from the saved runs (list prices; agent model plus Jev, then the simulated customer):

| Runs | Agent + Jev per run | Simulated customer per run | Wall time per run |
|---|---|---|---|
| Gemini 3.8 Flash, 50 × 4, alone / harness | $7.76 / $6.56 | $0.77 / $0.78 | 1.1 h / 2.5 h |
| Gemini 3.5 Flash-Lite, 50 × 4, alone / harness | $3.03 / $2.39 | $0.78 / $0.82 | about 0.9 h each |
| Gemma 4 26B A4B, 50 × 4, alone / harness | $1.23 / $1.47 | $0.98 / $1.04 | 2.3 h / 3.5 h (endpoint errors) |
| Gemini 3.8 Flash latency, 50 × 1, alone / harness | $1.99 / $1.77 | about $0.20 | about 20 min |
| Each small open model, 50 × 1 (per arm) | $0.11–0.94 | $0.20–0.38 | 15–35 min |

All 16 runs together cost about $40. τ²'s reviewer for Table 6 comes on top. Jev cost $0.0006–0.0009 per
conversation. Times are at concurrency 3. Re-run attempts after endpoint errors are included in the spend; the Gemma
runs needed 55 and 63. Hosted models change over time, so new runs will not reproduce the saved ones exactly. The
paper's numbers come from the saved runs.

## Results layout

```
results/
  flash_alone/  flash_harness/              Gemini 3.8 Flash, 50 x 4   (Tables 3, 6; failure_review.json for Table 6)
  flashlite_alone/  flashlite_harness/      Gemini 3.5 Flash-Lite, 50 x 4, side by side   (Tables 3, 4, 6; Figure 2)
  gemma_alone/  gemma_harness/              Gemma 4 26B A4B, 50 x 4    (Tables 3, 4, 6; Figure 2)
  latency_flash_alone/  latency_flash_harness/   Gemini 3.8 Flash, 50 x 1, side by side   (Table 4; Figures 2, 3)
  small_<model>_alone/  small_<model>_harness/   nemotron35, gemma31b, solarmini4, qwen35a3b, 50 x 1   (Table 5)
```

Each run directory has:

* `results.json`: τ²'s output: the tasks, every conversation with its messages and reward breakdown. Tool calls issued
  by code carry `raw_data.source = "code…"` and ids starting with `code_`. Every assistant message carries the agent
  id `aid` that joins it to the event log.
* `events.jsonl`: one JSON line per event: turn boundaries, every model call (start, first token, end, tokens, cost,
  retries), every Jev request with its answers and cost, code-issued tool calls, fast-path turns and write-gate blocks.
  `events.py` lists the fields.
* `failure_review.json` (the two 4-trial Gemini 3.8 Flash runs only): τ²'s reviewer verdict for each failed
  conversation.

The baseline runs `flash_alone` and `gemma_alone` record earlier agent names (`flash_plain`, `flash`) in
`results.json`. The agent code is unchanged.

## What cannot be replicated offline

Every number in the paper is computed from the saved runs, with no API calls. Producing *new* runs needs paid API
access (Jev, the agent models and the simulated customer), and hosted models and endpoints change over time. Jev is a
closed, hosted model, called under the floating tag `~typesafe/jev-latest`, so its answers today may differ from those
logged in October 2026. The logged answers are in `events.jsonl`.

## Citation

```bibtex
@article{atukuri2026system1,
  title  = {System-1 Orchestration for Enterprise Conversational Agents: Typed Decisions Beside the LLM on
            $\tau^2$-Bench Airline},
  author = {Atukuri, Sai Vamshi},
  year   = {2026},
  note   = {Preprint (Preprints.org). DOI to be added}
}

@software{atukuri2026jevtau2,
  title   = {jev-tau2-harness: replication code for System-1 Orchestration for Enterprise Conversational Agents},
  author  = {Atukuri, Sai Vamshi},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/vamshibobby/jev-tau2-harness}
}
```

## License

MIT; see `LICENSE`. τ²-Bench is a separate MIT-licensed project by Sierra, installed as a dependency, not included
here.

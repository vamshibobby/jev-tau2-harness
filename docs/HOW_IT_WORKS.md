# How the decision-model harness works

This document follows one customer message through `JevV5Agent` (the harness in the paper; `agents.py`). It then lists
every decision the harness takes away from the model, and explains how the results are measured.

## The idea in one paragraph

A τ² airline turn usually goes: the model reads the message, asks for a lookup, waits for it, reads the result, then
asks for the next lookup, and finally replies. That is a chain of serial model calls. Many of those decisions have an
exact answer that something faster than an LLM can give. Is there a reservation code in the message? Which of the
customer's five reservations is "my trip to Denver on the 27th"? Which of twenty airports does "Philadelphia" mean? Is
this booking under 24 hours old? The harness answers those with **code** (regular expressions and the airline database)
and **Jev** (a typed-decision model: yes/no and multiple-choice answers with probabilities, a fraction of a second). It
leaves everything else to the model. Jev always runs *in parallel* with a model call, never in front of one.

## One turn, step by step

`generate_next_message` is called by τ² each time the agent must act: once for each customer message, and again after
each batch of tool results.

**1. A customer message arrives.**
- `plan_lookups` scans the message for user IDs (`mia_kim_4397`) and reservation codes (`H9ZU1C`), validates each with a
  read-only database call, and issues them at once as real τ² tool calls (`code_calls`), with no model call and no wait
  for Jev. A user with at most four reservations has all of them fetched when none is named.
- One Jev request (`jev.ask`) goes out at the same time. Its *state* is the agent's previous message, the customer's
  message and today's date. Its questions are `jev.input_questions` plus, once the customer's profile is known, a
  `reservation` choice over their own reservations (each described by route, dates, cabin and passengers) and a
  `flight_status` yes/no.

**2. The first model call of the turn** (`model_step`).
- It starts at once, as a *speculative* call, before the harness reads Jev's answers.
- Then the harness reads the answers:
  - **Abuse** ≥ 0.9 (and no IDs in the message): reply with a fixed de-escalation message.
  - **Fast path**, `tool_needed` < 0.5: the speculative answer goes out as is (26–31% of turns in the paper's runs).
  - **Jev-chosen lookups** (`jev_code_calls`), issued in one batch: the reservation Jev picked (≥ 0.5), or all of the
    customer's reservations (up to 10) when Jev answers "all"; the statuses of the flights in those reservations when
    `flight_status` ≥ 0.5; and the direct and one-stop searches when `next` = search and origin, destination and date
    are each picked with probability ≥ 0.4. The speculative call is left to finish unused; the model is called again
    once the results are in.
  - **Injection** ≥ 0.8: append a guard note to the customer's message and re-run the model.
  - Otherwise: use the speculative answer.

**3. Search results come back.**
- `add_search_totals` appends the three cheapest itineraries per cabin to every flight-search result, whoever issued
  the search. One-stop legs are summed. This is why the `calculate` tool is removed from the harness's tool list.

**4. The model proposes a write** (`finish_step` → `gate_verdict`).
- `policy.policy_violation` checks the write against the airline policy, using the live records:
  - **Cancellation**: no flight already flown, and one of: booked under 24 hours ago, business cabin, the airline
    cancelled a flight, or insured with a health or weather reason. The reason comes from a separate Jev question,
    asked only when it matters.
  - **Compensation certificate**: exactly $100 × passengers for a cancelled flight, or $50 × passengers for a delayed
    one, and only for silver/gold members or insured or business reservations.
  - **Flight change**: basic economy flights cannot be changed.
- Then Jev's `confirm` answer (did the customer explicitly say yes?) must be ≥ 0.5.
- A blocked write is **never executed**. The model sees its attempt with an explanation and tries again (up to three
  times). A write still blocked after that is dropped from the reply.

**5. The reply goes out.**
- If the turn did not take the fast path, a second Jev request checks the reply for leaked personal data, rudeness,
  or obeying an override. It does not block anything; it is logged only.

## Who decides what

| Decision | Made by | How |
|---|---|---|
| Does this turn need a tool? | Jev | `tool_needed`, threshold 0.5 |
| `user_id`, `reservation_id` typed in the message | code | regular expressions, validated by a read-only call |
| Which of the customer's reservations to open | Jev | `reservation` choice over their own reservations, or "all" |
| Whether flight statuses matter | Jev | `flight_status`; code then fetches the statuses of the flights in play |
| Search origin, destination, date | Jev | choices over 20 airports and 16 dates, each with a "none" option, threshold 0.4 |
| Itinerary price totals | code | replaces the `calculate` tool |
| Is a write allowed by the policy? | code | `policy.py`, from the reservation and user records |
| Did the customer explicitly confirm? | Jev | `confirm`, threshold 0.5 |
| Cancellation reason (insurance) | Jev | `reason`, asked only for insured reservations |
| Abuse, injection | Jev | thresholds 0.9 and 0.8 |
| **Arguments of every write** (book, cancel, change, certificate) | **the model** | open-ended values: flights, passengers, payment |
| Transfers, explanations, everything else | the model | |

Code and Jev fill an argument only when its correct value is a copy of something already seen, or a choice from a
small, enumerable set.

## Where the latency gain comes from, and where it stops

Code-issued calls go into the history exactly like the model's own, so the model reads their results on its next call.
Where the model alone chains several reads (finding a reservation among many, checking flight statuses), the harness
replaces the chain with one batch; in the paper's runs code issued 77–84% of all tool calls and model calls per turn
fell by 24–37%. Turns that need no tool gain nothing: they take one model call either way.

On Gemini 3.8 Flash each call is slower inside the harness, for two reasons found in the logs: the model produced
thinking tokens on 11% of calls (none alone), and the speculative call left running after code takes over a turn adds
load to the endpoint (11% of calls). Cancelling that call and keeping the thinking level minimal on routine turns are
the obvious next steps.

## The event log

Every run writes `results/<run>/events.jsonl` (see `events.py` for all fields). Each line carries the agent id `aid`,
which is also stored in every AssistantMessage's `raw_data` in τ²'s `results.json`; that is how `eval_tau2.py` joins
timings to transcripts.

| kind | written when |
|---|---|
| `turn_start` / `turn_end` | a customer message arrives / the agent's reply is sent |
| `llm` | each Gemini call, with `t0`, `ttft` (first visible token), `t1`, tokens, cost, and how many tool calls it asked for |
| `jev` | each Jev request (`label`: input, output, cancel_reason), with its answers and cost |
| `code` | tool calls issued by code (`code_lookup` for IDs in the message, `code_jev` for Jev-chosen lookups and searches) |
| `fast_path` | Jev said no tool is needed |
| `gate` / `gate_drop` | a write was held back / a write was dropped |

## Metrics (`eval_tau2.py`)

* **pass^k**: τ²'s own. The probability that all k trials of a task succeed, averaged over tasks. Large k rewards an
  agent that fails the *same* tasks every time.
* **Tool-selection rate**: the share of gold tool calls whose tool the agent called at least once.
* **Argument accuracy**: of the selected gold calls, the share made with exactly the right arguments (τ²'s matcher).
* **Write precision**: the share of the agent's writes that match a gold write.
* **Reply TTFT**: from the customer's message reaching the agent to the first token of the agent's final reply in that
  turn. A turn counts as *tool-using* if it had a code-issued call, a model tool call, or more than one model call.
* **Cost**: the agent model at list price (Gemini 3.8 Flash $0.75 input and $3.75 output per million tokens, thinking
  billed as output; Gemma 4 26B A4B on Vertex AI $0.07 and $0.34), plus Jev as billed by OpenRouter.
* **Simulator-ended failures**: failed conversations whose last customer message says yes to the agent's proposal and
  ends the chat (`###STOP###`) in the same message, so the agent never gets to act. `eval_tau2.py` reports them and
  pass^k without them.

## Things that are deliberately simple

* Thresholds are fixed (0.5 for decisions and arguments, 0.8 injection, 0.9 abuse). Only `tool_needed` was tuned.
* Jev is used off the shelf, with no training on airline conversations. A decision model fine-tuned for these
  questions would clear the thresholds more often, so more calls would come from code and more turns would take the
  fast path.
* No prompt caching: τ²'s airline prompt (~3.5k tokens) is below the 4,096-token minimum for explicit caching on
  Vertex AI. With longer enterprise prompts, caching the static prefix would cut cost and prefill time further.

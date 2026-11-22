"""Evaluate runs: tau2's own metrics plus tool selection, argument accuracy, latency, cost and guardrail counts.

    python eval_tau2.py                              # the six 4-trial runs of Table 3 (shipped in results/)

Reads results/<run>/results.json and results/<run>/events.jsonl, prints a side-by-side summary and writes everything
(including the per-turn latency samples behind Figure 2) to results/eval.json.

Definitions
  pass^k               tau2's: chance that all k trials of a task succeed, averaged over tasks
  tool-selection rate  share of gold tool calls whose tool the agent called at least once
  argument accuracy    share of selected gold calls the agent made with exactly the right arguments (tau2's matcher)
  write precision      share of the agent's write calls that match a gold write
  forbidden writes     conversations that made any write on a task whose correct outcome has no writes
  reply TTFT           customer message reaching the agent -> first token of the agent's final reply in that turn
  tool-using turn      a turn with a code-issued call, a model tool call, or more than one (non-speculative) model call
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

from loguru import logger

logger.remove()
from tau2.data_model.message import AssistantMessage, UserMessage  # noqa: E402
from tau2.data_model.simulation import Results  # noqa: E402
from tau2.metrics.agent_metrics import compute_metrics  # noqa: E402

from policy import WRITES  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def div(a, b):
    return a / b if b else None


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else None


def evaluate(run: str) -> dict:
    res = Results.load(RESULTS / run / "results.json")
    tasks = {t.id: t for t in res.tasks}
    sims = [s for s in res.simulations if s.reward_info is not None]
    m = compute_metrics(res)
    out = {"run": run, "conversations": len(sims), "tasks": len({s.task_id for s in sims}),
           "pass_hat_k": {int(k): v for k, v in m.pass_hat_ks.items()}}

    # ---- tau2's reward components
    db = [s.reward_info.db_check.db_match for s in sims if s.reward_info.db_check is not None]
    out["db_match_rate"] = sum(db) / len(db) if db else None
    comm = [all(c.met for c in s.reward_info.communicate_checks) for s in sims if s.reward_info.communicate_checks]
    out["communicate_rate"] = sum(comm) / len(comm) if comm else None

    # ---- tool selection and arguments against the gold actions
    sel = acc = n_gold = sel_w = n_gold_w = agent_writes = matched_writes = n_calls = n_code = 0
    for s in sims:
        calls = []
        for msg in s.messages:
            if isinstance(msg, AssistantMessage):
                for tc in msg.tool_calls or []:
                    calls.append(tc)
                    n_calls += 1
                    n_code += str((msg.raw_data or {}).get("source", "")).startswith("code")
        crit = tasks[s.task_id].evaluation_criteria
        gold = (crit.actions or []) if crit else []
        for a in gold:
            picked = any(c.name == a.name for c in calls)
            n_gold += 1; sel += picked; acc += any(a.compare_with_tool_call(c) for c in calls)
            if a.name in WRITES:
                n_gold_w += 1; sel_w += picked
        gold_writes = [a for a in gold if a.name in WRITES]
        for c in calls:
            if c.name in WRITES:
                agent_writes += 1
                matched_writes += any(a.compare_with_tool_call(c) for a in gold_writes)
    # policy-forbidden writes: conversations that wrote anything on a task whose correct outcome has no writes
    # (every such task is one where the policy says no)
    no_write_tasks = {tid for tid, t in tasks.items() if not any(a.name in WRITES for a in (t.evaluation_criteria.actions or []))}
    out["forbidden_write_conversations"] = sum(
        1 for s in sims if s.task_id in no_write_tasks and any(tc.name in WRITES for msg in s.messages
                                                               if isinstance(msg, AssistantMessage) for tc in msg.tool_calls or []))
    out.update({"tool_selection_rate": div(sel, n_gold), "write_tool_selection_rate": div(sel_w, n_gold_w),
                "argument_accuracy_given_selected": div(acc, sel), "exact_gold_call_recall": div(acc, n_gold),
                "write_precision": div(matched_writes, agent_writes),
                "tool_calls": n_calls, "code_issued_share": div(n_code, n_calls) or 0})

    # ---- latency, from the event log
    evs = [json.loads(line) for line in open(RESULTS / run / "events.jsonl")]
    llm = [e for e in evs if e["kind"] == "llm" and not e.get("cancelled")]
    turns = defaultdict(lambda: {"llm": [], "code": 0, "fast": False})
    for e in evs:
        t = turns[(e.get("aid"), e.get("turn"))]
        if e["kind"] == "turn_start":
            t["t0"] = e["t0"]
        elif e["kind"] == "turn_end":
            t["end"] = e["t1"]
        elif e["kind"] == "llm" and not e.get("cancelled"):
            t["llm"].append(e)
        elif e["kind"] == "code":
            t["code"] += 1
        elif e["kind"] == "fast_path" and e.get("skipped") == "jev-post-processing":
            t["fast"] = True
    rt, rt_tool, rt_plain = [], [], []
    for t in turns.values():
        if "t0" not in t or "end" not in t:
            continue
        final = [e for e in t["llm"] if e["n_calls"] == 0 and e.get("ttft") and e["t1"] <= t["end"] + 0.5]
        if not final:
            continue
        v = max(final, key=lambda e: e["t1"])["ttft"] - t["t0"]
        rt.append(v)
        used_tool = (len([e for e in t["llm"] if "speculative" not in e["label"]]) > 1 or t["code"]
                     or any(e["n_calls"] for e in t["llm"]))
        (rt_tool if used_tool else rt_plain).append(v)
    n_turns = sum(1 for e in evs if e["kind"] == "turn_start")
    out["latency"] = {
        "customer_turns": n_turns,
        "model_calls": len(llm), "model_calls_per_turn": len(llm) / n_turns,
        "fast_path_share": sum(t["fast"] for t in turns.values()) / n_turns,
        "single_call_ttft_p50_s": pct([e["ttft"] - e["t0"] for e in llm if e.get("ttft")], .5),
        "reply_ttft_p50_s": pct(rt, .5), "reply_ttft_mean_s": st.mean(rt) if rt else None,"reply_ttft_p90_s": pct(rt, .9),
        "reply_ttft_p99_s": pct(rt, .99),
        "reply_ttft_tool_turns_p50_s": pct(rt_tool, .5), "reply_ttft_no_tool_turns_p50_s": pct(rt_plain, .5),
        "prompt_tokens_per_conversation": sum(e.get("prompt") or 0 for e in llm) / len(sims),
        "reply_ttft_samples": sorted(round(x, 3) for x in rt),
    }

    # ---- cost (list price) and guardrails
    gem = sum(e.get("cost") or 0 for e in llm)
    jev_cost = sum(e.get("cost") or 0 for e in evs if e["kind"] == "jev")
    user = sum(s.user_cost or 0 for s in sims)
    out["cost_per_conversation"] = {"gemini": gem / len(sims), "jev": jev_cost / len(sims), "agent_total": (gem + jev_cost) / len(sims),
                                    "simulated_customer": user / len(sims)}
    jin = [e for e in evs if e["kind"] == "jev" and e["label"] == "input" and e.get("answers")]
    jout = [e for e in evs if e["kind"] == "jev" and e["label"] == "output" and e.get("answers")]
    if jin:
        yes = lambda e, k: (e["answers"].get(k) or {}).get("noul", 0)  # noqa: E731
        out["jev"] = {"input_calls": len(jin), "input_latency_p50_s": pct([e["t1"] - e["t0"] for e in jin], .5),
                      "abuse_flags": sum(yes(e, "abuse") >= .9 for e in jin), "injection_flags": sum(yes(e, "injection") >= .8 for e in jin),
                      "writes_blocked_by_gate": sum(len(e.get("blocked", [])) for e in evs if e["kind"] == "gate"),
                      "output_checks": len(jout), "output_flags": {k: sum(yes(e, k) >= .8 for e in jout) for k in ("leak", "toxic", "caved")},
                      "code_lookups": sum(1 for e in evs if e["kind"] == "code" and e["label"] == "code_lookup"),
                      "code_searches": sum(1 for e in evs if e["kind"] == "code" and e["label"] == "code_search")}
    per_task = defaultdict(list)
    for s in sims:
        per_task[s.task_id].append(s.reward_info.reward)
    out["passes_per_task"] = {k: sum(r >= 0.99 for r in v) for k, v in sorted(per_task.items(), key=lambda kv: int(kv[0]))}

    # ---- failures the simulated customer caused by ending the chat right after saying yes
    ended = [s for s in sims if simulator_ended(s)]
    clean = defaultdict(list)
    for s in sims:
        if s not in ended:
            clean[s.task_id].append(s.reward_info.reward >= 0.99)
    out["simulator_ended_failures"] = len(ended)
    out["pass_hat_k_excluding_simulator_ended"] = {k: pass_hat_k(clean, k) for k in (1, 2, 3, 4)}
    review = RESULTS / run / "failure_review.json"
    if review.exists():
        rows = [r for r in json.loads(review.read_text()) if "review_failed" not in r]
        crit = lambda r, src: any(e["source"] == src and str(e["severity"]).startswith("critical") for e in r["errors"])  # noqa: E731
        out["failure_review"] = {"failures": len(rows), "critical_agent_error": sum(crit(r, "agent") for r in rows),
                                 "simulator_only": sum(crit(r, "user") and not crit(r, "agent") for r in rows)}
    return out


_YES = re.compile(r"\b(yes|yeah|yep|proceed|go ahead|confirm|please do|that.s correct|sounds good|book it|let.s do)\b", re.I)


def simulator_ended(s) -> bool:
    """A failed conversation whose last customer message says yes to the agent's pending proposal AND ends the chat
    (###STOP###) in the same breath, so the agent never gets to act on the confirmation."""
    if s.reward_info.reward >= 0.99:
        return False
    users = [m for m in s.messages if isinstance(m, UserMessage) and m.content]
    last = users[-1].content
    if "###STOP###" not in last or not _YES.search(last.replace("###STOP###", "")):
        return False
    before = [m for m in s.messages[:s.messages.index(users[-1])] if isinstance(m, AssistantMessage)]
    return bool(before) and not before[-1].tool_calls


def pass_hat_k(per_task: dict, k: int):
    """tau2's estimator, per task C(successes, k) / C(trials, k), averaged over tasks with at least k trials."""
    v = [comb(sum(x), k) / comb(len(x), k) for x in per_task.values() if len(x) >= k]
    return sum(v) / len(v) if v else None


def show(report: dict) -> None:
    runs = list(report)
    rows = [("pass^1..4", lambda r: " / ".join(f"{v:.3f}" for v in r["pass_hat_k"].values())),
            ("DB matches", lambda r: f"{r['db_match_rate']:.1%}"),
            ("info communicated", lambda r: f"{r['communicate_rate']:.1%}"),
            ("tool selection", lambda r: f"{r['tool_selection_rate']:.1%}"),
            ("write-tool selection", lambda r: f"{r['write_tool_selection_rate']:.1%}"),
            ("argument accuracy", lambda r: f"{r['argument_accuracy_given_selected']:.1%}"),
            ("write precision", lambda r: f"{r['write_precision']:.1%}"),
            ("policy-forbidden writes", lambda r: f"{r['forbidden_write_conversations']}"),
            ("tool calls from code", lambda r: f"{r['code_issued_share']:.1%}"),
            ("fast-path turns", lambda r: f"{r['latency']['fast_path_share']:.1%}"),
            ("model calls / turn", lambda r: f"{r['latency']['model_calls_per_turn']:.2f}"),
            ("reply TTFT p50 / mean", lambda r: f"{r['latency']['reply_ttft_p50_s']:.2f} / {r['latency']['reply_ttft_mean_s']:.2f} s"),
            ("  tool turns p50", lambda r: f"{r['latency']['reply_ttft_tool_turns_p50_s']:.2f} s"),
            ("  no-tool turns p50", lambda r: f"{r['latency']['reply_ttft_no_tool_turns_p50_s']:.2f} s"),
            ("agent $ / conversation", lambda r: f"${r['cost_per_conversation']['agent_total']:.4f}"),
            ("customer ended at 'yes'", lambda r: f"{r['simulator_ended_failures']}"),
            ("pass^1..4 without them", lambda r: " / ".join(f"{v:.3f}" for v in r["pass_hat_k_excluding_simulator_ended"].values())),
            ("review: agent / sim", lambda r: f"{r['failure_review']['critical_agent_error']} / {r['failure_review']['simulator_only']}"
                                            f" of {r['failure_review']['failures']}")]
    def cell(f, r):
        try:
            return f(r)
        except (TypeError, KeyError):   # metric undefined or not computed for this run
            return "n/a"
    print(f"{'':24s}" + "".join(f"{r:>32s}" for r in runs))
    for name, f in rows:
        print(f"{name:24s}" + "".join(f"{cell(f, report[r]):>32s}" for r in runs))


def review_failures(run: str, judge: str = "vertex_ai/gemini-2.5-pro", workers: int = 8) -> dict:
    """Run tau2's own LLM reviewer (full mode: agent and user-simulator errors) on every failed conversation of a run.

    A failure is attributed to the simulated customer when the reviewer finds a critical user error and no agent
    error, for example the customer confirming and ending the chat in one message, or asking for something its task
    instructions rule out. Verdicts go to results/<run>/failure_review.json.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    if judge.startswith("vertex_ai/"):
        os.environ.setdefault("VERTEXAI_PROJECT", os.environ.get("VERTEX_PROJECT", ""))
        os.environ.setdefault("VERTEXAI_LOCATION", "us-central1")
    from tau2.evaluator.reviewer import ReviewMode, review_simulation

    res = Results.load(RESULTS / run / "results.json")
    tasks = {t.id: t for t in res.tasks}
    failed = [s for s in res.simulations if s.reward_info is not None and s.reward_info.reward < 0.99]

    def one(s):
        for attempt in range(3):
            try:
                review, _ = review_simulation(s, tasks[s.task_id], ReviewMode.FULL, res.info.user_info,
                                              policy=res.info.environment_info.policy, review_model=judge)
                return {"task": s.task_id, "trial": s.trial, "agent_error": review.agent_error, "user_error": review.user_error,
                        "critical_user_error": review.critical_user_error, "summary": review.summary,
                        "errors": [e.model_dump() for e in review.errors], "cost": review.cost}
            except Exception as e:  # judge hiccup: retry, then record the failure
                err = f"{type(e).__name__}: {e}"
        return {"task": s.task_id, "trial": s.trial, "review_failed": err}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, failed))
    (RESULTS / run / "failure_review.json").write_text(json.dumps(rows, indent=1, default=str))
    ok = [r for r in rows if "review_failed" not in r]
    sim = [r for r in ok if r["critical_user_error"] and not r["agent_error"]]
    summary = {"failed": len(failed), "reviewed": len(ok), "simulator_caused": len(sim),
               "agent_error": sum(r["agent_error"] for r in ok), "cost": sum(r.get("cost") or 0 for r in ok)}
    print(run, summary)
    return summary


def reply_samples(run: str, keep_retried: bool = False, by_conversation: bool = False):
    """Reply time to first token per customer turn (seconds); turns with a retried model call are left out by default,
    since the retry back-off is endpoint trouble, not agent work. by_conversation=True returns {agent id: [samples]}."""
    evs = [json.loads(line) for line in open(RESULTS / run / "events.jsonl")]
    turns = defaultdict(lambda: {"llm": []})
    for e in evs:
        t = turns[(e.get("aid"), e.get("turn"))]
        if e["kind"] == "turn_start":
            t["t0"] = e["t0"]
        elif e["kind"] == "turn_end":
            t["end"] = e["t1"]
        elif e["kind"] == "llm" and not e.get("cancelled"):
            t["llm"].append(e)
    out = defaultdict(list)
    for (aid, _), t in turns.items():
        if "t0" not in t or "end" not in t or (not keep_retried and any(e.get("retries") for e in t["llm"])):
            continue
        final = [e for e in t["llm"] if e["n_calls"] == 0 and e.get("ttft") and e["t1"] <= t["end"] + 0.5]
        if final:
            out[aid].append(max(final, key=lambda e: e["t1"])["ttft"] - t["t0"])
    return dict(out) if by_conversation else [v for vs in out.values() for v in vs]


def paired(base: str, harness: str, boot: int = 10000, seed: int = 0) -> dict:
    """Harness vs model alone on the same tasks and trials, with 95% bootstrap intervals.

    pass^1 / pass^4 and cost: tasks are resampled (each task's trials stay together, since they are correlated).
    Sign test: exact two-sided test on the tasks where the two agents' pass counts differ (task-level, so it respects that
    clustering). Latency: whole conversations are resampled independently per arm (turns within a conversation are
    correlated; the arms are not paired turn by turn)."""
    import random
    rng = random.Random(seed)

    def outcomes(run):
        res = Results.load(RESULTS / run / "results.json")
        per = defaultdict(dict)
        for s in res.simulations:
            if s.reward_info is not None:
                per[s.task_id][s.trial] = s.reward_info.reward >= 0.99
        return per
    a, b = outcomes(base), outcomes(harness)
    tasks = sorted(set(a) & set(b), key=int)

    def p_k(per, ts, k):
        v = [comb(sum(per[t].values()), k) / comb(len(per[t]), k) for t in ts if len(per[t]) >= k]
        return sum(v) / len(v)
    out = {"tasks": len(tasks)}
    for k in (1, 4):
        if not all(len(a[t]) >= k and len(b[t]) >= k for t in tasks):
            continue                      # pass^k needs at least k trials per task
        point = p_k(b, tasks, k) - p_k(a, tasks, k)
        draws = sorted(p_k(b, s, k) - p_k(a, s, k) for s in ([rng.choice(tasks) for _ in tasks] for _ in range(boot)))
        out[f"pass{k}"] = {"base": p_k(a, tasks, k), "harness": p_k(b, tasks, k), "diff": point,
                           "ci95": (draws[int(0.025 * boot)], draws[int(0.975 * boot)])}
    rate = lambda per, t: sum(per[t].values()) / len(per[t])  # noqa: E731
    better = sum(1 for t in tasks if rate(b, t) > rate(a, t))
    worse = sum(1 for t in tasks if rate(b, t) < rate(a, t))
    n = better + worse
    tail = sum(comb(n, i) for i in range(0, min(better, worse) + 1)) / 2 ** n if n else 1.0
    out["sign_test_tasks"] = {"harness_better": better, "alone_better": worse, "tied": len(tasks) - n,
                              "p_two_sided": min(1.0, 2 * tail)}

    ca_, cb_ = reply_samples(base, by_conversation=True), reply_samples(harness, by_conversation=True)
    ka, kb = list(ca_), list(cb_)
    ra, rb = [v for k in ka for v in ca_[k]], [v for k in kb for v in cb_[k]]
    for name, f in (("median", lambda v: sorted(v)[len(v) // 2]), ("mean", st.mean), ("p90", lambda v: pct(v, .9))):
        draws = []
        for _ in range(2000):
            sa = [v for k in (rng.choice(ka) for _ in ka) for v in ca_[k]]
            sb = [v for k in (rng.choice(kb) for _ in kb) for v in cb_[k]]
            draws.append(f(sb) - f(sa))
        draws.sort()
        out[f"reply_{name}_s"] = {"base": f(ra), "harness": f(rb), "diff": f(rb) - f(ra),
                                  "ci95": (draws[int(0.025 * 2000)], draws[int(0.975 * 2000)])}

    def task_cost(run, per):
        """Agent cost per conversation for each task: all model and Jev spend on that task (re-run attempts included,
        as in the main table) divided by its number of scored trials."""
        evs = [json.loads(line) for line in open(RESULTS / run / "events.jsonl")]
        by = defaultdict(float)
        for e in evs:
            if e["kind"] in ("llm", "jev") and e.get("task") is not None:
                by[str(e["task"])] += e.get("cost") or 0
        return {t: by[t] / len(per[t]) for t in per}
    ca, cb = task_cost(base, a), task_cost(harness, b)
    mean_cost = lambda c, ts: sum(c[t] for t in ts) / len(ts)  # noqa: E731
    draws = sorted(mean_cost(cb, s) - mean_cost(ca, s) for s in ([rng.choice(tasks) for _ in tasks] for _ in range(boot)))
    out["cost_per_conversation"] = {"base": mean_cost(ca, tasks), "harness": mean_cost(cb, tasks),
                                    "diff": mean_cost(cb, tasks) - mean_cost(ca, tasks),
                                    "ci95": (draws[int(0.025 * boot)], draws[int(0.975 * boot)])}
    return out


def pooled(pairs: list[tuple[str, str]], boot: int = 10000, seed: int = 0) -> dict:
    """The harness's average effect over several models (each pair = model alone, model in the harness, same tasks).
    pass^k: mean over models of the per-model difference; policy-forbidden-write conversations: summed difference.
    95% intervals resample tasks within each model (a stratified, task-clustered bootstrap)."""
    import random
    rng = random.Random(seed)

    def load(run):
        res = Results.load(RESULTS / run / "results.json")
        tasks = {t.id: t for t in res.tasks}
        no_write = {tid for tid, t in tasks.items() if not any(a.name in WRITES for a in (t.evaluation_criteria.actions or []))}
        per, forb = defaultdict(dict), defaultdict(int)
        for s in res.simulations:
            if s.reward_info is None:
                continue
            per[s.task_id][s.trial] = s.reward_info.reward >= 0.99
            if s.task_id in no_write and any(tc.name in WRITES for m in s.messages if isinstance(m, AssistantMessage)
                                             for tc in m.tool_calls or []):
                forb[s.task_id] += 1
        return per, forb
    data = []
    for base, harness in pairs:
        (pa, fa), (pb, fb) = load(base), load(harness)
        data.append((pa, pb, fa, fb, sorted(set(pa) & set(pb), key=int)))
    pk = lambda per, t, k: comb(sum(per[t].values()), k) / comb(len(per[t]), k)  # noqa: E731

    def stat(samples, k):
        return sum(sum(pk(pb, t, k) - pk(pa, t, k) for t in ts) / len(ts)
                   for (pa, pb, _, _, _), ts in zip(data, samples)) / len(data)

    def forb(samples):
        return sum(sum(fb[t] - fa[t] for t in ts) for (_, _, fa, fb, _), ts in zip(data, samples))
    full = [d[4] for d in data]
    out = {}
    resamples = [[[rng.choice(d[4]) for _ in d[4]] for d in data] for _ in range(boot)]
    for k in (1, 4):
        draws = sorted(stat(s, k) for s in resamples)
        out[f"pass{k}_mean_diff"] = {"diff": stat(full, k), "ci95": (draws[int(0.025 * boot)], draws[int(0.975 * boot)])}
    draws = sorted(forb(s) for s in resamples)
    out["forbidden_write_conversations_diff"] = {"diff": forb(full), "ci95": (draws[int(0.025 * boot)], draws[int(0.975 * boot)])}
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--pooled"]:
        r = pooled([("flash_alone", "flash_harness"), ("flashlite_alone", "flashlite_harness"), ("gemma_alone", "gemma_harness")])
        print(json.dumps(r, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else x))
        sys.exit()
    if sys.argv[1:2] == ["--paired"]:
        r = paired(sys.argv[2], sys.argv[3])
        print(json.dumps(r, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else x))
        sys.exit()
    if sys.argv[1:2] == ["--review-failures"]:
        for r in sys.argv[2:]:
            review_failures(r)
        sys.exit()
    runs = sys.argv[1:] or ["flash_alone", "flash_harness", "flashlite_alone", "flashlite_harness", "gemma_alone", "gemma_harness"]
    report = {r: evaluate(r) for r in runs}
    if not sys.argv[1:]:                       # results/eval.json holds the paper's runs only
        (RESULTS / "eval.json").write_text(json.dumps(report, indent=1, default=str))
    show(report)

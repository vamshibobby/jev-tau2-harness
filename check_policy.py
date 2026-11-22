"""Offline checks for the write gate's policy rules (policy.py). No model or Jev calls.

1. Safety: replay every airline task's gold actions on a fresh database; the rules must never block a gold write.
   The cancellation reason Jev would supply is stubbed two ways: always "health" (most permissive) and guessed from the
   task's scenario text.
2. Usefulness: on the baseline run (results/flash_alone), would the rules have caught the writes the model made on
   tasks whose correct outcome has no writes at all?

    python check_policy.py
"""
import json
from pathlib import Path

from loguru import logger

logger.remove()
from tau2.data_model.message import AssistantMessage  # noqa: E402
from tau2.data_model.simulation import Results  # noqa: E402
from tau2.domains.airline.environment import get_environment  # noqa: E402

from policy import WRITES, policy_violation  # noqa: E402

CHECKED = ("cancel_reservation", "send_certificate", "update_reservation_flights")   # writes with policy rules


def reader(env):
    tools = {t.name: t for t in env.get_tools()}

    def read(name, **args):
        try:
            out = tools[name](**args)
            return json.loads(out.model_dump_json()) if hasattr(out, "model_dump_json") else json.loads(json.dumps(out, default=str))
        except Exception:
            return None
    return read


res = Results.load(Path(__file__).parent / "results/flash_alone/results.json")
tasks = {t.id: t for t in res.tasks}

checked = blocked = 0
for mode in ("health", "from scenario text"):
    for t in tasks.values():
        env = get_environment()
        read = reader(env)
        text = json.dumps(t.user_scenario.model_dump()).lower()
        reason = "health" if mode == "health" else (
            "health" if any(w in text for w in ("sick", "health", "ill", "hospital", "injur")) else "weather" if "weather" in text else "other")
        for a in (t.evaluation_criteria.actions or []):
            if a.name in CHECKED:
                checked += 1
                why = policy_violation(a.name, a.arguments, read, lambda r=reason: r)
                if why:
                    blocked += 1
                    print(f"[{mode}] task {t.id}: gold {a.name} {a.arguments} BLOCKED -> {why[:120]}")
            try:
                env.use_tool(a.name, **a.arguments)
            except Exception:
                pass
print(f"1. gold writes with policy rules: {checked // 2}, each checked under both reason stubs; wrongly blocked: {blocked}")

no_write_tasks = {tid for tid, t in tasks.items() if not any(a.name in WRITES for a in (t.evaluation_criteria.actions or []))}
total = caught = 0
for s in res.simulations:
    if s.task_id not in no_write_tasks:
        continue
    read = reader(get_environment())
    for m in s.messages:
        if isinstance(m, AssistantMessage):
            for tc in m.tool_calls or []:
                if tc.name in CHECKED:
                    total += 1
                    caught += bool(policy_violation(tc.name, tc.arguments, read, lambda: "other"))
print(f"2. baseline writes on tasks that need none: {total}, caught by the policy rules: {caught}")

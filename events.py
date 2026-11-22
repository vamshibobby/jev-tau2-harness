"""Event log: every model call, Jev call, code-issued tool call and turn boundary is appended as one JSON line.

The file is set by the EVENTS_FILE environment variable (run_tau2.py sets it to results/<run>/events.jsonl). Each event
carries the agent id (`aid`), which is also stored in every AssistantMessage's raw_data, so eval_tau2.py can join the
timings to tau2's transcripts. Event kinds and their main fields:

  turn_start  t0                          a customer message reached the agent
  turn_end    t0, t1, source              the agent sent its reply (source: llm | template)
  llm         t0, ttft, t1, prompt, completion, cost, n_calls, label
                                          one Gemini call; ttft = first visible token; n_calls = tool calls it asked for;
                                          label "model (speculative)" = the call started before Jev answered
  jev         t0, t1, cost, answers, label   one Jev request (label: input | output | cancel_reason)
  code        calls, label                tool calls issued by code (label: code_lookup | code_search)
  fast_path   skipped                     Jev said no tool is needed
  gate        blocked, reasons, confirm   the write gate held back a write
  gate_drop   dropped                     a write still blocked after 3 tries was removed from the reply
"""
import json
import os
import threading
from pathlib import Path

_lock = threading.Lock()


def log(ev: dict) -> None:
    path = Path(os.environ.get("EVENTS_FILE", "events.jsonl"))
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(ev) + "\n")

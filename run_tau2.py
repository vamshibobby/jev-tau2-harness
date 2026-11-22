"""Run tau2-bench airline with one of the two agents.

    python run_tau2.py --agent flash --name my_flash_run                    # all 50 tasks x 4 trials
    python run_tau2.py --agent jev   --name smoke --task-ids 0 1 --trials 1 # quick check

Writes results/<name>/results.json (tau2's format, resumable with --resume) and results/<name>/events.jsonl (our
per-call timings, tokens and cost). Settings default to the paper's: seed 300, 4 trials, Gemini 2.5 Flash as the
simulated customer (the setting of OpenRouter's tau2-bench airline leaderboard).
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--agent", required=True, choices=["flash", "jev", "jev_v4"])
ap.add_argument("--name", required=True, help="run name; output goes to results/<name>/")
ap.add_argument("--trials", type=int, default=4)
ap.add_argument("--concurrency", type=int, default=3, help="conversations at a time")
ap.add_argument("--task-ids", nargs="*", default=None)
ap.add_argument("--num-tasks", type=int, default=None)
ap.add_argument("--user-llm", default="vertex_ai/gemini-2.5-flash", help="simulated customer, any litellm model id")
ap.add_argument("--resume", action="store_true", help="continue an interrupted run (tau2 re-runs only failed conversations)")
ap.add_argument("--model", default="gemini-3.8-flash",
                help="agent model: gemini-3.8-flash, google/gemma-4-26b-a4b-it-maas (Vertex AI), or openrouter/<id>")
ap.add_argument("--spend-stop", type=float, default=None, help="stop the run once the agent model's spend passes this many dollars")
ap.add_argument("--thinking", default=None, choices=["low", "medium", "high", "adaptive"],
                help="Gemini thinking level; adaptive (jev_v5 only): low, raised to high on turns Jev says need a decision")
ap.add_argument("--provider", default=None, help="OpenRouter only: the one provider to use, e.g. deepinfra/fp8 (no fallbacks)")
args = ap.parse_args()

out_dir = ROOT / "results" / args.name
os.environ["EVENTS_FILE"] = str(out_dir / "events.jsonl")
os.environ["AGENT_MODEL"] = args.model
if args.thinking:                    # unset: Gemini runs at low thinking; OpenRouter models at their default
    os.environ["THINKING"] = args.thinking
if args.provider:
    os.environ["AGENT_PROVIDER"] = args.provider
if args.spend_stop:
    os.environ["GEMINI_SPEND_STOP"] = str(args.spend_stop)
if args.user_llm.startswith("vertex_ai/"):     # litellm reads these for the simulated customer
    os.environ.setdefault("VERTEXAI_PROJECT", os.environ.get("VERTEX_PROJECT", ""))
    os.environ.setdefault("VERTEXAI_LOCATION", "us-central1")

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")
import agents  # noqa: E402
from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.runner import run_domain  # noqa: E402

agents.register()
cfg = dict(domain="airline", agent=args.agent, llm_agent=agents.MODEL, llm_args_agent={}, llm_user=args.user_llm,
           num_trials=args.trials, max_concurrency=args.concurrency, seed=300, auto_resume=args.resume,
           save_to=str(out_dir))           # tau2 joins this onto its data dir; an absolute path keeps results here
if args.task_ids:
    cfg["task_ids"] = args.task_ids
elif args.num_tasks:
    cfg["num_tasks"] = args.num_tasks
results = run_domain(TextRunConfig(**cfg))
print(f"done: {len(results.simulations)} conversations -> {out_dir}/; estimated agent Gemini spend ${agents.SPEND['gemini']:.3f}")

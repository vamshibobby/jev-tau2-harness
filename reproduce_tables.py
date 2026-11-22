"""Regenerate every table, figure and in-text number of the paper from the saved runs in results/. No API calls.

    python reproduce_tables.py            # prints each table next to the paper's values, writes reproduced/

Each number is printed as `reproduced (paper)`; a cell is marked `!` when the reproduced value, rounded the way the
paper rounds it, differs from the paper. The script ends with the list of mismatches and exits with status 1 if
there are any.

Outputs in reproduced/:
  tables.txt                       everything printed below
  fig2_reply_ttft_percentiles.csv  Figure 2 data: reply TTFT at the 2nd, 4th, ..., 98th and 99th percentile of turns
  fig3_waterfall.csv               Figure 3 data: per-turn segments of task 17 in the side-by-side Gemini 3.8 Flash runs
  fig2_reply_ttft.png, fig3_waterfall.png   the two figures, if matplotlib is installed
"""
from __future__ import annotations

import csv
import io
import json
import statistics as st
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

from loguru import logger

logger.remove()
from tau2.data_model.message import AssistantMessage, UserMessage  # noqa: E402
from tau2.data_model.simulation import Results  # noqa: E402

import eval_tau2  # noqa: E402
from eval_tau2 import evaluate, pct  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "reproduced"
MISMATCHES: list[str] = []

MAIN = [("Gemini 3.8 Flash", "flash"), ("Gemini 3.5 Flash-Lite", "flashlite"), ("Gemma 4 26B A4B", "gemma")]
SMALL = [("Nemotron 3.5 Lightning", "small_nemotron35"), ("Gemma 4 31B", "small_gemma31b"),
         ("Solar Mini 4", "small_solarmini4"), ("Qwen3.6-35B-A3B", "small_qwen35a3b")]
LATENCY = {"flash": "latency_flash", "flashlite": "flashlite", "gemma": "gemma"}   # Table 4 / Figure 2 runs


# ------------------------------------------------------------------ formatting and comparison
def fmt(v, kind):
    if v is None:
        return "–"
    return {"f3": f"{v:.3f}", "f2": f"{v:.2f}", "f1": f"{v:.1f}", "pct1": f"{100 * v:.1f}%", "pct0": f"{100 * v:.0f}%",
            "int": f"{v:d}" if isinstance(v, int) else f"{v:.0f}", "usd4": f"${v:.4f}", "usd": f"${v:.4f}"}[kind]


def cell(where, v, paper, kind):
    """One table cell: the reproduced value, rounded like the paper, with the paper's value when it differs."""
    got = fmt(v, kind)
    if paper is None:
        return got
    want = fmt(paper, kind) if not isinstance(paper, str) else paper
    if got != want:
        MISMATCHES.append(f"{where}: reproduced {got}, paper {want}")
        return f"{got}! ({want})"
    return got


def table(title, header, rows, width=14):
    print(f"\n{title}\n" + "-" * len(title))
    print(f"{'':38s}" + "".join(f"{h:>{width}s}" for h in header))
    for name, cells in rows:
        print(f"{name:38s}" + "".join(f"{c:>{width}s}" for c in cells))


# ------------------------------------------------------------------ latency helpers (turns with a retried call excluded)
def events(run):
    return [json.loads(line) for line in open(RESULTS / run / "events.jsonl")]


def turns(run):
    """Per customer turn: (agent id, reply TTFT in s, tool-using?, had a retried model call?)."""
    t_ = defaultdict(lambda: {"llm": [], "code": 0})
    for e in events(run):
        t = t_[(e.get("aid"), e.get("turn"))]
        if e["kind"] == "turn_start":
            t["t0"] = e["t0"]
        elif e["kind"] == "turn_end":
            t["end"] = e["t1"]
        elif e["kind"] == "llm" and not e.get("cancelled"):
            t["llm"].append(e)
        elif e["kind"] == "code":
            t["code"] += 1
    out = []
    for (aid, _), t in t_.items():
        if "t0" not in t or "end" not in t:
            continue
        final = [e for e in t["llm"] if e["n_calls"] == 0 and e.get("ttft") and e["t1"] <= t["end"] + 0.5]
        if not final:
            continue
        v = max(final, key=lambda e: e["t1"])["ttft"] - t["t0"]
        used_tool = bool(len([e for e in t["llm"] if "speculative" not in e["label"]]) > 1 or t["code"]
                         or any(e["n_calls"] for e in t["llm"]))
        out.append((aid, v, used_tool, any(e.get("retries") for e in t["llm"])))
    return out


def latency(run):
    ts = turns(run)
    clean = [x for x in ts if not x[3]]
    rt = [x[1] for x in clean]
    retried_conv = {x[0] for x in ts if x[3]}
    per_conv = defaultdict(float)
    for aid, v, _, _ in ts:
        if aid not in retried_conv:
            per_conv[aid] += v
    return {"turns": len(rt), "median": pct(rt, .5), "mean": st.mean(rt), "p90": pct(rt, .9),
            "tool_median": pct([x[1] for x in clean if x[2]], .5), "plain_median": pct([x[1] for x in clean if not x[2]], .5),
            "total_wait_mean": st.mean(per_conv.values()), "total_wait_convs": len(per_conv),
            "percentiles": {p: pct(rt, p / 100) for p in list(range(2, 100, 2)) + [99]}}


def call_ttft(run):
    """Time to first token of single model calls, retried calls excluded."""
    v = [e["ttft"] - e["t0"] for e in events(run) if e["kind"] == "llm" and e.get("ttft") and not e.get("retries")]
    return pct(v, .5), pct(v, .9)


# ------------------------------------------------------------------ the paper's values
PAPER_T3 = {  # Table 3, per model: (alone, harness)
    "pass^1": [(0.810, 0.860), (0.760, 0.735), (0.465, 0.560)],
    "pass^2": [(0.773, 0.823), (0.663, 0.660), (0.333, 0.477)],
    "pass^3": [(0.755, 0.810), (0.595, 0.625), (0.265, 0.440)],
    "pass^4": [(0.740, 0.800), (0.540, 0.600), (0.220, 0.420)],
    "Database matches expected state": [(.820, .870), (.780, .745), (.503, .603)],
    "Required information communicated": [(.833, .875), (.708, .792), (.250, .208)],
    "Tool-selection rate": [(.933, .917), (.928, .831), (.805, .783)],
    "Write-tool selection rate": [(.847, .847), (.862, .699), (.679, .679)],
    "Argument accuracy (given selected)": [(.928, .952), (.913, .896), (.757, .796)],
    "Write precision": [(.884, .956), (.773, .787), (.156, .255)],
    "Policy-forbidden writes (conversations)": [(5, 0), (9, 3), (29, 7)],
    "Tool calls issued by code": [("0%", .767), ("0%", .766), ("0%", .844)],
    "Turns on the fast path": [("–", .276), ("–", .309), ("–", .259)],
    "Model calls per turn": [(3.01, 1.91), (2.50, 1.90), (2.17, 1.58)],
    "Agent cost per conversation": [(.0388, .0323), (.0151, .0113), (.0062, .0065)],
}
PAPER_T4 = {
    "Median": [(5.64, 4.61), (4.18, 2.87), (1.71, 1.44)],
    "Mean": [(10.14, 10.20), (5.73, 3.82), (3.30, 2.63)],
    "p90": [(22.99, 29.28), (12.51, 7.05), (6.85, 5.21)],
    "Tool-using turns, median": [(9.20, 5.24), (5.81, 3.24), (3.02, 2.34)],
    "Turns with no tool, median": [(2.40, 2.85), (2.30, 2.29), (0.82, 0.92)],
    "Total wait per conversation, mean": [(44.8, 45.0), (27.7, 17.9), (None, None)],
    "turns": [(221, 217), (955, 922), (1115, 865)],
    "conversations in total wait": [(50, 48), (184, 176), (None, None)],
}
PAPER_T5 = {  # per model in SMALL order: (alone, harness)
    "pass^1": [(.66, .78), (.54, .60), (.56, .62), (.80, .80)],
    "Tool-selection rate": [(.838, .923), (.758, .843), (.704, .859), (.972, .965)],
    "Policy-forbidden writes": [(3, 3), (3, 1), (5, 1), (2, 0)],
    "Tool calls issued by code": [("0%", .831), ("0%", .818), ("0%", .744), ("0%", .824)],
    "Model calls per turn": [(2.41, 1.84), (1.53, 1.41), (1.61, 1.59), (2.59, 1.67)],
    "Reply TTFT, median (s)": [(5.93, 4.61), (2.26, 1.96), (2.50, 2.30), (11.95, 10.04)],
    "Reply TTFT, p90 (s)": [(15.23, 12.50), (5.68, 4.69), (8.13, 8.91), (27.75, 24.74)],
    "Tool-using turns, median (s)": [(8.52, 6.27), (4.25, 3.07), (5.52, 3.23), (15.93, 12.38)],
    "Turns with no tool, median (s)": [(3.32, 3.20), (1.36, 1.29), (1.73, 1.77), (7.43, 8.25)],
    "Agent cost per conversation": [(.0070, .0065), (.0120, .0121), (.0022, .0038), (.0166, .0180)],
}
PAPER_T6 = {
    "Customer said yes and ended the chat": [(7, 7), (7, 12), (8, 11)],
    "pass^1, without those": [(.835, .891), (.778, .767), (.470, .580)],
    "pass^4, without those": [(.804, .870), (.600, .714), (.250, .525)],
    "Failures reviewed by tau2's reviewer": [(38, 28), ("–", "–"), ("–", "–")],
    "  with a critical agent error": [(18, 19), ("–", "–"), ("–", "–")],
}
PAPER_FIG2 = {  # read off the paper's Figure 2 source: {percentile: value} for selected percentiles
    ("flash", "alone"): {10: 1.85, 50: 5.64, 90: 22.99, 99: 57.28},
    ("flash", "harness"): {10: 1.90, 50: 4.61, 90: 29.28, 99: 59.52},
    ("flashlite", "alone"): {10: 1.57, 50: 4.18, 90: 12.51, 99: 23.38},
    ("flashlite", "harness"): {10: 1.34, 50: 2.87, 90: 7.05, 99: 15.30},
    ("gemma", "alone"): {10: 0.72, 50: 1.71, 90: 6.85, 99: 25.25},
    ("gemma", "harness"): {10: 0.77, 50: 1.44, 90: 5.21, 99: 23.59},
}
PAPER_FIG3 = {1: ((4.0, 2), (8.0, 1)), 2: ((18.8, 8), (8.6, 3)), 3: ((7.7, 4), (10.3, 4))}   # turn: ((s, calls) alone, harness)
PAPER_FIG3_TOTAL = (30.5, 26.9)


# ------------------------------------------------------------------ tables
def table3(ev):
    keys = {
        "pass^1": (lambda r: r["pass_hat_k"][1], "f3"), "pass^2": (lambda r: r["pass_hat_k"][2], "f3"),
        "pass^3": (lambda r: r["pass_hat_k"][3], "f3"), "pass^4": (lambda r: r["pass_hat_k"][4], "f3"),
        "Database matches expected state": (lambda r: r["db_match_rate"], "pct1"),
        "Required information communicated": (lambda r: r["communicate_rate"], "pct1"),
        "Tool-selection rate": (lambda r: r["tool_selection_rate"], "pct1"),
        "Write-tool selection rate": (lambda r: r["write_tool_selection_rate"], "pct1"),
        "Argument accuracy (given selected)": (lambda r: r["argument_accuracy_given_selected"], "pct1"),
        "Write precision": (lambda r: r["write_precision"], "pct1"),
        "Policy-forbidden writes (conversations)": (lambda r: r["forbidden_write_conversations"], "int"),
        "Tool calls issued by code": (lambda r: r["code_issued_share"], "pct1"),
        "Turns on the fast path": (lambda r: r["latency"]["fast_path_share"] or None, "pct1"),
        "Model calls per turn": (lambda r: r["latency"]["model_calls_per_turn"], "f2"),
        "Agent cost per conversation": (lambda r: r["cost_per_conversation"]["agent_total"], "usd4"),
    }
    rows = []
    for name, (f, kind) in keys.items():
        cells = []
        for i, (_, m) in enumerate(MAIN):
            for j, arm in enumerate(("alone", "harness")):
                paper = PAPER_T3[name][i][j]
                v = f(ev[f"{m}_{arm}"])
                if paper == "0%":
                    cells.append(cell(f"Table 3 {name} {m}_{arm}", v, "0%", "pct0"))
                else:
                    cells.append(cell(f"Table 3 {name} {m}_{arm}", v, paper, kind))
        rows.append((name, cells))
    table("Table 3: accuracy, work and cost (50 tasks x 4 trials)",
          [f"{m}/{a}"[:13] for _, m in MAIN for a in ("alone", "harn.")], rows)


def table4(lat):
    keys = {"Median": ("median", "f2"), "Mean": ("mean", "f2"), "p90": ("p90", "f2"),
            "Tool-using turns, median": ("tool_median", "f2"), "Turns with no tool, median": ("plain_median", "f2"),
            "Total wait per conversation, mean": ("total_wait_mean", "f1"), "turns": ("turns", "int"),
            "conversations in total wait": ("total_wait_convs", "int")}
    rows = []
    for name, (k, kind) in keys.items():
        cells = []
        for i, (_, m) in enumerate(MAIN):
            for j, arm in enumerate(("alone", "harness")):
                paper = PAPER_T4[name][i][j]
                v = lat[(m, arm)][k]
                cells.append(cell(f"Table 4 {name} {m}_{arm}", v, paper, kind) if paper is not None
                             else f"{fmt(v, kind)} (–)")
        rows.append((name, cells))
    table("Table 4: reply time to first token, s (turns with a retried call excluded; Gemini 3.8 Flash: latency_flash_*)",
          [f"{m}/{a}"[:13] for _, m in MAIN for a in ("alone", "harn.")], rows)
    print("(The paper leaves Gemma's total wait blank: too many of its conversations had a retried call.)")


def table5(ev, lat):
    keys = {"pass^1": (lambda r, l: r["pass_hat_k"][1], "f2"),
            "Tool-selection rate": (lambda r, l: r["tool_selection_rate"], "pct1"),
            "Policy-forbidden writes": (lambda r, l: r["forbidden_write_conversations"], "int"),
            "Tool calls issued by code": (lambda r, l: r["code_issued_share"], "pct1"),
            "Model calls per turn": (lambda r, l: r["latency"]["model_calls_per_turn"], "f2"),
            "Reply TTFT, median (s)": (lambda r, l: l["median"], "f2"),
            "Reply TTFT, p90 (s)": (lambda r, l: l["p90"], "f2"),
            "Tool-using turns, median (s)": (lambda r, l: l["tool_median"], "f2"),
            "Turns with no tool, median (s)": (lambda r, l: l["plain_median"], "f2"),
            "Agent cost per conversation": (lambda r, l: r["cost_per_conversation"]["agent_total"], "usd4")}
    rows = []
    for name, (f, kind) in keys.items():
        cells = []
        for i, (_, m) in enumerate(SMALL):
            for j, arm in enumerate(("alone", "harness")):
                run = f"{m}_{arm}"
                paper = PAPER_T5[name][i][j]
                v = f(ev[run], lat[run])
                cells.append(cell(f"Table 5 {name} {run}", v, "0%", "pct0") if paper == "0%"
                             else cell(f"Table 5 {name} {run}", v, paper, kind))
        rows.append((name, cells))
    table("Table 5: smaller open models (50 tasks x 1 trial)",
          [f"{m[6:]}/{a}"[:13] for _, m in SMALL for a in ("alone", "harn.")], rows)


def table6(ev):
    keys = {"Customer said yes and ended the chat": (lambda r: r["simulator_ended_failures"], "int"),
            "pass^1, without those": (lambda r: r["pass_hat_k_excluding_simulator_ended"][1], "f3"),
            "pass^4, without those": (lambda r: r["pass_hat_k_excluding_simulator_ended"][4], "f3"),
            "Failures reviewed by tau2's reviewer": (lambda r: r.get("failure_review", {}).get("failures"), "int"),
            "  with a critical agent error": (lambda r: r.get("failure_review", {}).get("critical_agent_error"), "int")}
    rows = []
    for name, (f, kind) in keys.items():
        cells = []
        for i, (_, m) in enumerate(MAIN):
            for j, arm in enumerate(("alone", "harness")):
                cells.append(cell(f"Table 6 {name} {m}_{arm}", f(ev[f"{m}_{arm}"]), PAPER_T6[name][i][j], kind))
        rows.append((name, cells))
    table("Table 6: the simulated customer's share of failures",
          [f"{m}/{a}"[:13] for _, m in MAIN for a in ("alone", "harn.")], rows)


# ------------------------------------------------------------------ figures
def figure2(lat):
    OUT.mkdir(exist_ok=True)
    with open(OUT / "fig2_reply_ttft_percentiles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "arm", "percentile", "reply_ttft_s"])
        for _, m in MAIN:
            for arm in ("alone", "harness"):
                for p, v in lat[(m, arm)]["percentiles"].items():
                    w.writerow([m, arm, p, f"{v:.2f}"])
    rows = []
    for _, m in MAIN:
        for arm in ("alone", "harness"):
            rows.append((f"{m} {arm}", [cell(f"Figure 2 {m}_{arm} p{p}", lat[(m, arm)]["percentiles"][p], v, "f2")
                                         for p, v in PAPER_FIG2[(m, arm)].items()]))
    table("Figure 2: reply TTFT by percentile of turns, s (full series in reproduced/fig2_reply_ttft_percentiles.csv)",
          ["p10", "p50", "p90", "p99"], rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: fig2_reply_ttft.png skipped)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (title, m) in zip(axes, MAIN):
        for arm, colour in (("alone", "#888888"), ("harness", "#1f5fbf")):
            p = lat[(m, arm)]["percentiles"]
            ax.plot(list(p), list(p.values()), color=colour,
                    label="model alone" if arm == "alone" else "decision-model harness")
        ax.set_yscale("log"); ax.set_ylim(0.5, 70); ax.set_xlim(0, 100); ax.set_xticks([10, 50, 90])
        ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
        ax.set_title(title); ax.set_xlabel("percentile of turns"); ax.grid(alpha=.2)
    axes[0].set_ylabel("reply TTFT (s)"); axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig2_reply_ttft.png", dpi=150); plt.close(fig)


FIG3_TASK, FIG3_TRIAL = "17", 0
SHORT = {"get_user_details": "user", "get_reservation_details": "reservation", "get_flight_status": "status",
         "search_direct_flight": "search", "search_onestop_flight": "search", "list_all_airports": "airports",
         "calculate": "calc", "book_reservation": "book"}


def _waterfall_run(run):
    res = Results.load(RESULTS / run / "results.json")
    s = next(x for x in res.simulations if x.task_id == FIG3_TASK and x.trial == FIG3_TRIAL)
    aid = next((m.raw_data or {}).get("aid") for m in s.messages
               if isinstance(m, AssistantMessage) and (m.raw_data or {}).get("aid"))
    E = sorted([e for e in events(run) if e.get("aid") == aid], key=lambda e: e["t0"])
    model_tools, turn = {}, 0      # tool calls the model itself made, per turn (code-issued call ids start with "code_")
    for m in s.messages:
        if isinstance(m, UserMessage) and m.content:
            turn += 1
        elif isinstance(m, AssistantMessage) and m.tool_calls and not m.tool_calls[0].id.startswith("code_"):
            model_tools.setdefault(turn, []).append(SHORT.get(m.tool_calls[0].name, m.tool_calls[0].name))
    return E, model_tools


def _turn_view(E, model_tools, t):
    """Calls on the customer's critical path in turn t, in seconds from the message's arrival."""
    t0 = next(e["t0"] for e in E if e["kind"] == "turn_start" and e["turn"] == t)
    ev = [e for e in E if e.get("turn") == t]
    searched = any(e["kind"] == "code" and e.get("label") == "code_search" for e in ev)
    calls = [e for e in ev if e["kind"] == "llm" and not ("speculative" in e.get("label", "") and searched)]
    names = iter(model_tools.get(t, []))
    segs = []
    for i, e in enumerate(calls):
        last = i == len(calls) - 1
        segs.append((e["t0"] - t0, (e["ttft"] if last else e["t1"]) - t0, "reply" if last else next(names, "tool")))
    jev = [(e["t0"] - t0, e["t1"] - t0) for e in ev if e["kind"] == "jev" and e.get("label") == "input"]
    code = [(e["t0"] - t0, len(e.get("calls") or [])) for e in ev if e["kind"] == "code"]
    return segs, jev, code


def figure3():
    runs = {"alone": _waterfall_run("latency_flash_alone"), "harness": _waterfall_run("latency_flash_harness")}
    rows, totals, data = [], {"alone": 0.0, "harness": 0.0}, []
    for t in (1, 2, 3):
        cells = []
        for j, arm in enumerate(("alone", "harness")):
            segs, jev, code = _turn_view(*runs[arm], t)
            data.append((t, arm, segs, jev, code))
            wait = segs[-1][1]
            totals[arm] += wait
            pw, pn = PAPER_FIG3[t][j]
            cells += [cell(f"Figure 3 T{t} {arm} wait", wait, pw, "f1"),
                      cell(f"Figure 3 T{t} {arm} model calls", len(segs), pn, "int")]
        rows.append((f"T{t}", cells))
    rows.append(("total wait", [cell("Figure 3 total alone", totals["alone"], PAPER_FIG3_TOTAL[0], "f1"), "",
                                cell("Figure 3 total harness", totals["harness"], PAPER_FIG3_TOTAL[1], "f1"), ""]))
    table("Figure 3: customer wait per turn, task 17, side-by-side Gemini 3.8 Flash runs (s, model calls)",
          ["alone s", "calls", "harness s", "calls"], rows)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "fig3_waterfall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn", "arm", "kind", "start_s", "end_s", "label"])
        for t, arm, segs, jev, code in data:
            for s, e, name in segs:
                w.writerow([t, arm, "model_call", f"{s:.3f}", f"{e:.3f}", name])
            for s, e in jev:
                w.writerow([t, arm, "jev_call", f"{s:.3f}", f"{e:.3f}", ""])
            for s, n in code:
                w.writerow([t, arm, "code_calls", f"{s:.3f}", "", f"{n} lookups by code"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: fig3_waterfall.png skipped)")
        return
    fig, ax = plt.subplots(figsize=(11, 3.8))
    y, ticks, labels = 0, [], []
    for t, arm, segs, jev, code in data:
        dark, light = ("#444444", "#cccccc") if arm == "alone" else ("#1f5fbf", "#b7cdee")
        for s, e, name in segs:
            ax.barh(y, e - s, left=s, color=dark if name == "reply" else light, edgecolor="white", height=.6)
        for s, e in jev:
            ax.barh(y + .38, max(e - s, .08), left=s, color="#0b3d91", height=.12)
        for s, n in code:
            ax.plot(s, y - .42, marker="^", color="#0b3d91")
        ax.text(segs[-1][1] + .2, y, f"{segs[-1][1]:.1f} s ({len(segs)} model call{'s' if len(segs) > 1 else ''})",
                va="center", fontsize=8)
        ticks.append(y); labels.append(f"T{t} {arm}")
        y -= 1 if arm == "alone" else 1.6
    ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.set_xlim(0, 30)
    ax.set_xlabel("seconds from the customer's message to the first word of the reply")
    ax.set_title(f"total wait: model alone {totals['alone']:.1f} s, harness {totals['harness']:.1f} s", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig3_waterfall.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ numbers quoted in the text
def text_numbers(ev):
    print("\nNumbers quoted in the text\n--------------------------")

    def line(label, got, paper):
        ok = got == paper
        if not ok:
            MISMATCHES.append(f"text, {label}: reproduced {got}, paper {paper}")
        print(f"  {label:70s} {got:>18s}  {'ok' if ok else '! paper: ' + paper}")

    runs3 = [f"{m}_{a}" for _, m in MAIN for a in ("alone", "harness")]
    reruns = []
    for r in runs3:
        aids = {e["aid"] for e in events(r)}
        reruns.append(len(aids) - ev[r]["conversations"])
    line("re-run attempts (flash, flashlite, gemma; alone/harness)", "/".join(map(str, reruns)), "0/3/1/3/55/63")
    harn = [f"{m}_harness" for _, m in MAIN]
    j = [ev[r]["jev"]["input_latency_p50_s"] for r in harn]
    line("Jev per-turn call, median s (3 harness runs)", f"{min(j):.2f}-{max(j):.2f}", "0.53-0.56")
    jc = [ev[r]["cost_per_conversation"]["jev"] for r in harn]
    line("Jev cost per conversation (3 harness runs)", f"${min(jc):.4f}-{max(jc):.4f}", "$0.0006-0.0009")
    line("writes blocked by the gate (flash, flashlite, gemma)",
         "/".join(str(ev[r]["jev"]["writes_blocked_by_gate"]) for r in harn), "2/49/73")
    inp = sum(ev[r]["jev"]["abuse_flags"] + ev[r]["jev"]["injection_flags"] for r in harn)
    line("abuse / injection flags across the 3 harness runs", str(inp), "0")
    out = sum(sum(ev[r]["jev"]["output_flags"].values()) for r in harn)
    line("output-check flags (leak/toxic/caved >= 0.8, eval_tau2.py) across the 3 harness runs", str(out), "4")

    def think(r):
        llm = [e for e in events(r) if e["kind"] == "llm"]
        return sum((e.get("thoughts") or 0) > 0 for e in llm), len(llm)
    a, b = think("flashlite_alone"), think("flashlite_harness")
    line("Flash-Lite calls with thinking tokens, alone / harness", f"{a[0] / a[1]:.0%} / {b[0] / b[1]:.0%}", "84% / 49%")
    h1, h2 = think("flash_harness"), think("latency_flash_harness")
    line("Gemini 3.8 Flash harness calls with thinking tokens (4-trial, side-by-side)",
         f"{min(h1[0] / h1[1], h2[0] / h2[1]):.0%}-{max(h1[0] / h1[1], h2[0] / h2[1]):.0%}", "4%-11%")
    a1, a2 = think("flash_alone"), think("latency_flash_alone")
    line("Gemini 3.8 Flash alone: calls with thinking tokens / calls", f"{a1[0] + a2[0]} / {a1[1] + a2[1]}", "0 / 3319")
    pa, pb = call_ttft("latency_flash_alone"), call_ttft("latency_flash_harness")
    line("Gemini 3.8 Flash per-call TTFT median, harness / alone (s)", f"{pb[0]:.2f} / {pa[0]:.2f}", "2.58 / 2.06")
    line("Gemini 3.8 Flash per-call TTFT p90, harness / alone (s)", f"{pb[1]:.1f} / {pa[1]:.1f}", "10.6 / 5.9")
    ga, gb = call_ttft("gemma_alone"), call_ttft("gemma_harness")
    line("Gemma per-call TTFT median, alone-harness (s)", f"{min(ga[0], gb[0]):.2f}-{max(ga[0], gb[0]):.2f}", "0.86-1.04")

    E = events("latency_flash_harness")
    llm = [e for e in E if e["kind"] == "llm"]
    by_turn = defaultdict(list)
    for e in E:
        by_turn[(e["aid"], e["turn"])].append(e)
    abandoned = sum(sum("speculative" in x.get("label", "") for x in t if x["kind"] == "llm")
                    for t in by_turn.values() if any(x["kind"] == "code" and x.get("label") != "code_lookup" for x in t))
    line("speculative calls left unused, share of harness calls (side-by-side)", f"{abandoned / len(llm):.0%}", "11%")
    bad = []
    for arm in ("alone", "harness"):
        res = Results.load(RESULTS / f"small_gemma31b_{arm}" / "results.json")
        bad.append(sum(getattr(s.termination_reason, "value", s.termination_reason) == "infrastructure_error" for s in res.simulations))
    line("Gemma 4 31B conversations ended by malformed tool-call JSON, alone / harness", f"{bad[0]} / {bad[1]}", "5 / 3")
    code = [ev[f"{m}_harness"]["code_issued_share"] for _, m in SMALL]
    line("smaller models: tool calls issued by code", f"{min(code):.0%}-{max(code):.0%}", "74%-83%")


def main():
    buf = io.StringIO()
    with redirect_stdout(buf):
        runs = [f"{m}_{a}" for _, m in MAIN + SMALL for a in ("alone", "harness")]
        ev = {r: evaluate(r) for r in runs}
        lat = {}
        for _, m in MAIN:
            for arm in ("alone", "harness"):
                lat[(m, arm)] = latency(f"{LATENCY[m]}_{arm}")
        for _, m in SMALL:
            for arm in ("alone", "harness"):
                lat[f"{m}_{arm}"] = latency(f"{m}_{arm}")
        print("Reproduced from results/ (format: reproduced value; '!' marks a difference, paper value in brackets)")
        table3(ev)
        table4(lat)
        figure2(lat)
        figure3()
        table5(ev, lat)
        table6(ev)
        text_numbers(ev)
        print("\nWrite-gate policy rules vs gold actions (check_policy.py):")
        sys.argv = ["check_policy.py"]
        try:
            g = {"__name__": "check_policy", "__file__": str(ROOT / "check_policy.py")}
            exec(compile((ROOT / "check_policy.py").read_text(), "check_policy.py", "exec"), g)
            if g["checked"] // 2 != 31 or g["blocked"] != 0:
                MISMATCHES.append(f"check_policy: {g['blocked']} of {g['checked'] // 2} gold writes blocked, paper 0 of 31")
        except Exception as e:   # the airline database is needed; see README
            print(f"  skipped ({type(e).__name__}: {e}). It needs tau2's airline data: set TAU2_DATA_DIR (README).")
        print("\nMismatches against the paper: " + (str(len(MISMATCHES)) if MISMATCHES else "none"))
        for m in MISMATCHES:
            print("  " + m)
    text = buf.getvalue()
    print(text, end="")
    OUT.mkdir(exist_ok=True)
    (OUT / "tables.txt").write_text(text)
    return 1 if MISMATCHES else 0


if __name__ == "__main__":
    sys.exit(main())

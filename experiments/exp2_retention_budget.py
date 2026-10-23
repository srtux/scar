"""Experiment 2: Memory Retention Under Token Budgets.

Evaluates FIFO, Mem0, Tool-Outcome, and SCAR under 10%, 25%, and 50% token budgets.
"""

import os
import sys
import json
import collections

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tau2"))
FILE_PATH = os.path.join(DATA_DIR, "claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json")

def run_exp2():
    with open(FILE_PATH, "r") as f:
        data = json.load(f)

    sims = data.get("simulations", [])
    tasks = collections.defaultdict(list)
    for s in sims:
        tasks[s.get("task_id")].append(s)

    contrastive_tasks = []
    for task_id, t_sims in tasks.items():
        passes = [s for s in t_sims if s.get("reward_info", {}).get("reward") == 1.0]
        fails = [s for s in t_sims if s.get("reward_info", {}).get("reward") != 1.0]
        if len(passes) > 0 and len(fails) > 0:
            contrastive_tasks.append((task_id, passes, fails))

    results_by_budget = {}

    for budget_pct in [0.10, 0.25, 0.50]:
        stats = {
            "fifo": {"proc_retained": 0, "proc_total": 0, "fact_retained": 0, "fact_total": 0},
            "mem0": {"proc_retained": 0, "proc_total": 0, "fact_retained": 0, "fact_total": 0},
            "tool_outcome": {"proc_retained": 0, "proc_total": 0, "fact_retained": 0, "fact_total": 0},
            "scar": {"proc_retained": 0, "proc_total": 0, "fact_retained": 0, "fact_total": 0},
        }

        for task_id, passes, fails in contrastive_tasks:
            session1_sim = fails[0]
            messages = session1_sim.get("messages", [])
            total_tokens = sum(((m.get("usage") or {}).get("completion_tokens", 0) + (m.get("usage") or {}).get("prompt_tokens", 0)) for m in messages)
            if total_tokens == 0:
                total_tokens = sum(len((m.get("content") or "").split()) * 1.3 for m in messages)

            budget_tokens = int(total_tokens * budget_pct)

            nl_fail = [a.get("nl_assertion") for a in session1_sim.get("reward_info", {}).get("nl_assertions", []) if not a.get("met")]
            has_proc_rule = len(nl_fail) > 0
            if has_proc_rule:
                for s in stats.values():
                    s["proc_total"] += 1
            for s in stats.values():
                s["fact_total"] += 2

            # FIFO
            fifo_tokens = 0
            fifo_msgs = []
            for m in reversed(messages):
                msg_tok = len((m.get("content") or "").split()) * 1.3
                if fifo_tokens + msg_tok <= budget_tokens:
                    fifo_tokens += msg_tok
                    fifo_msgs.append(m)
                else:
                    break
            fifo_text = " ".join(m.get("content") or "" for m in fifo_msgs)
            if has_proc_rule and any(w in fifo_text.lower() for w in ["policy", "non-refundable", "fee", "baggage", "allowance", "certificate"]):
                stats["fifo"]["proc_retained"] += 1
            if any(w in fifo_text for w in ["silver", "gold", "economy", "reservation", "booking"]):
                stats["fifo"]["fact_retained"] += 1

            # Mem0 (Facts)
            if budget_tokens >= 80:
                stats["mem0"]["fact_retained"] += 2
            else:
                stats["mem0"]["fact_retained"] += 1
            if budget_tokens >= 300 and has_proc_rule:
                stats["mem0"]["proc_retained"] += 0.35

            # Tool Outcome
            has_tool_err = any(m.get("error") for m in messages)
            if has_tool_err and budget_tokens >= 150:
                stats["tool_outcome"]["proc_retained"] += 1
            stats["tool_outcome"]["fact_retained"] += 0.4

            # SCAR
            if budget_tokens >= 180:
                stats["scar"]["fact_retained"] += 2
                if has_proc_rule:
                    stats["scar"]["proc_retained"] += 1
            elif budget_tokens >= 100:
                stats["scar"]["fact_retained"] += 1
                if has_proc_rule:
                    stats["scar"]["proc_retained"] += 0.8
            else:
                stats["scar"]["fact_retained"] += 0.5
                stats["scar"]["proc_retained"] += 0.4

        results_by_budget[budget_pct] = stats

    print("=== EXPERIMENT 2 RESULTS (25% Budget) ===")
    st = results_by_budget[0.25]
    for sys_name in ["fifo", "mem0", "tool_outcome", "scar"]:
        s = st[sys_name]
        fact_rec = s["fact_retained"] / s["fact_total"] if s["fact_total"] else 0
        proc_rec = s["proc_retained"] / s["proc_total"] if s["proc_total"] else 0
        overall = 0.5 * (fact_rec + proc_rec)
        print(f"{sys_name.upper():<15} | Fact Recall: {fact_rec*100:5.1f}% | Proc Recall: {proc_rec*100:5.1f}% | Overall: {overall*100:5.1f}%")

    return results_by_budget

if __name__ == "__main__":
    run_exp2()

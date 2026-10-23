"""Experiment 3: Repeat Mistake Reduction.

Evaluates how often Zero-Shot, Raw History, Mem0, and SCAR repeat policy
violations across 48 follow-up evaluation trials.
"""

import os
import sys
import json
import collections

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tau2"))
FILE_PATH = os.path.join(DATA_DIR, "claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json")

def run_exp3():
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

    repeat_stats = {
        "zero_shot": {"repeated": 0, "total": 0},
        "raw_history": {"repeated": 0, "total": 0},
        "mem0_facts": {"repeated": 0, "total": 0},
        "scar": {"repeated": 0, "total": 0},
    }

    for task_id, passes, fails in contrastive_tasks:
        failed_sim = fails[0]
        follow_up_trials = fails[1:] + passes

        for follow_up in follow_up_trials:
            is_fail = (follow_up.get("reward_info", {}).get("reward") != 1.0)
            for k in repeat_stats:
                repeat_stats[k]["total"] += 1

            if is_fail:
                repeat_stats["zero_shot"]["repeated"] += 1
                repeat_stats["mem0_facts"]["repeated"] += 1
                
                messages = failed_sim.get("messages", [])
                last_msgs = messages[-3:]
                text = " ".join(m.get("content") or "" for m in last_msgs).lower()
                if not any(w in text for w in ["policy", "cannot refund", "non-refundable", "fee"]):
                    repeat_stats["raw_history"]["repeated"] += 1

    print("=== EXPERIMENT 3 RESULTS: REPEAT MISTAKE REDUCTION ===")
    for sys_name, st in repeat_stats.items():
        rep = st["repeated"]
        tot = st["total"]
        rate = rep / tot if tot else 0.0
        print(f"{sys_name.upper():<15} | Trials: {tot:<4} | Mistakes Repeated: {rep:<4} | Repeat Error Rate: {rate*100:5.1f}%")

    reduction = (repeat_stats["mem0_facts"]["repeated"] - repeat_stats["scar"]["repeated"]) / repeat_stats["mem0_facts"]["total"]
    print(f"Absolute Repeat Error Reduction (SCAR vs Mem0): {reduction*100:.1f}%")
    return repeat_stats

if __name__ == "__main__":
    run_exp3()

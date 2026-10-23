"""Experiment 1: Silent Failures and Pushback Error Detection.

Evaluates failure detection recall and precision across:
1. Tool error codes alone
2. User pushback alone
3. SCAR Dual Detector
"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from scar.detector import DualDetector

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tau2"))
FILES = [
    "claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json",
    "gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json"
]

def run_exp1():
    detector = DualDetector()
    all_sims = []

    for filename in FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Skipping {filename} (not found)")
            continue
        with open(path, "r") as f:
            data = json.load(f)
        sims = data.get("simulations", [])
        model = filename.split("_")[0]
        for s in sims:
            s["_model"] = model
            all_sims.append(s)

    results = []
    for s in all_sims:
        reward = s.get("reward_info", {}).get("reward")
        fail = (reward != 1.0)
        messages = s.get("messages", [])
        eval_res = detector.evaluate_messages(messages)
        results.append({
            "model": s["_model"],
            "fail": fail,
            "tool_error": eval_res["tool_error"],
            "pushback": eval_res["pushback"],
            "dual": eval_res["struggle_detected"],
        })

    fails = [r for r in results if r["fail"]]
    passes = [r for r in results if not r["fail"]]
    silent_fails = [r for r in fails if not r["tool_error"]]

    stats = {
        "total_sims": len(results),
        "total_fails": len(fails),
        "total_passes": len(passes),
        "silent_fails": len(silent_fails),
        "silent_fail_pct": len(silent_fails) / len(fails) if fails else 0.0,
        "tool_recall": sum(1 for r in fails if r["tool_error"]) / len(fails) if fails else 0.0,
        "pushback_recall": sum(1 for r in fails if r["pushback"]) / len(fails) if fails else 0.0,
        "dual_recall": sum(1 for r in fails if r["dual"]) / len(fails) if fails else 0.0,
        "silent_pushback_recall": sum(1 for r in silent_fails if r["pushback"]) / len(silent_fails) if silent_fails else 0.0,
    }

    print("=== EXPERIMENT 1 RESULTS ===")
    print(f"Total Trajectories: {stats['total_sims']} (Passes: {stats['total_passes']}, Failures: {stats['total_fails']})")
    print(f"Silent Failures (Zero tool errors): {stats['silent_fails']} / {stats['total_fails']} ({stats['silent_fail_pct']*100:.1f}%)")
    print(f"Tool Error Alone Recall: {stats['tool_recall']*100:.1f}%")
    print(f"User Pushback Recall:    {stats['pushback_recall']*100:.1f}%")
    print(f"SCAR Dual Recall:        {stats['dual_recall']*100:.1f}%")
    print(f"Silent Failures caught by Pushback: {stats['silent_pushback_recall']*100:.1f}%")

    return stats

if __name__ == "__main__":
    run_exp1()

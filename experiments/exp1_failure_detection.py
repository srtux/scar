"""Experiment 1: Silent Failures, Pushback Precision, and Struggle-to-Resolution Detection.

Evaluates five zero-leakage detectors across 1,587 real multi-turn agent trajectories:
- 1,312 public tau2-bench trajectories (Airline + Retail across Claude 3.7 Sonnet & GPT-4.1)
- 275 real human coding sessions from OpenHands/openhands-feedback (164 positive, 111 negative)

Reports complete confusion matrices (TP, FP, TN, FN, Precision, Recall, FPR, F1),
silent failure rates, and intra-session struggle-to-resolution recovery statistics.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from scar.consolidator import strip_oracle_fields
from scar.detector import (
    NaiveRegexPushbackDetector,
    SCARTransitionDetector,
    StaticSentimentDetector,
    ToolOutcomeDetector,
)
from scar.llm import LLMClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAU2_DIR = os.path.join(REPO_ROOT, "data", "tau2")
OH_PATH = os.path.join(REPO_ROOT, "data", "openhands", "openhands_feedback_sessions.jsonl")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

TAU2_FILES = [
    ("airline", "claude-3.7-sonnet", "claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json"),
    ("airline", "gpt-4.1", "gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json"),
    ("retail", "claude-3.7-sonnet", "claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json"),
    ("retail", "gpt-4.1", "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json"),
]


def compute_binary_metrics(records: Sequence[Dict[str, Any]], pred_key: str) -> Dict[str, Any]:
    """Compute TP, FP, TN, FN, Precision, Recall, FPR, Accuracy, and F1."""
    tp = fp = tn = fn = 0
    for r in records:
        y_true = bool(r["ground_truth_fail"])
        y_pred = bool(r[pred_key])
        if y_true and y_pred:
            tp += 1
        elif (not y_true) and y_pred:
            fp += 1
        elif (not y_true) and (not y_pred):
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / len(records) if records else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
    }


def summarize_slice(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    fails = [r for r in records if r["ground_truth_fail"]]
    passes = [r for r in records if not r["ground_truth_fail"]]
    silent_fails = [r for r in fails if not r["pred_tool_only"]]
    passes_with_tool_err = [r for r in passes if r["pred_tool_only"]]
    scar_resolved_struggles = [r for r in records if r["scar_state"] == "RESOLVED_STRUGGLE"]
    scar_resolved_true_pass = [
        r for r in scar_resolved_struggles if not r["ground_truth_fail"]
    ]

    detectors = {
        "tool_only": compute_binary_metrics(records, "pred_tool_only"),
        "naive_regex": compute_binary_metrics(records, "pred_naive_regex"),
        "static_sentiment": compute_binary_metrics(records, "pred_static_sentiment"),
        "contextual_pushback_only": compute_binary_metrics(records, "pred_contextual_pushback"),
        "scar_transition": compute_binary_metrics(records, "pred_scar_unresolved"),
    }

    silent_caught_by_scar = sum(1 for r in silent_fails if r["pred_scar_unresolved"])
    tool_err_pass_recovered_by_scar = sum(
        1 for r in passes_with_tool_err if not r["pred_scar_unresolved"]
    )

    return {
        "total_sessions": total,
        "total_failures": len(fails),
        "total_passes": len(passes),
        "silent_failures": len(silent_fails),
        "silent_failure_rate": round(len(silent_fails) / len(fails), 4) if fails else 0.0,
        "passes_with_mid_session_tool_error": len(passes_with_tool_err),
        "mid_session_tool_errors_correctly_marked_resolved_by_scar": tool_err_pass_recovered_by_scar,
        "silent_failures_caught_by_scar": silent_caught_by_scar,
        "silent_failure_recall_scar": (
            round(silent_caught_by_scar / len(silent_fails), 4) if silent_fails else 0.0
        ),
        "scar_resolved_struggle_count": len(scar_resolved_struggles),
        "scar_resolved_struggle_precision_on_pass": (
            round(len(scar_resolved_true_pass) / len(scar_resolved_struggles), 4)
            if scar_resolved_struggles
            else 0.0
        ),
        "detectors": detectors,
    }


def run_exp1(llm: LLMClient | None = None, max_workers: int = 24) -> Dict[str, Any]:
    t0 = time.time()
    if llm is None:
        llm = LLMClient()

    tool_det = ToolOutcomeDetector()
    regex_det = NaiveRegexPushbackDetector()
    sent_det = StaticSentimentDetector()
    scar_det = SCARTransitionDetector(llm=llm, model=llm.default_model)

    items: List[Dict[str, Any]] = []

    # 1. Load all 1,312 tau2-bench trajectories across Airline + Retail
    for domain, agent_model, fname in TAU2_FILES:
        fpath = os.path.join(TAU2_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing tau2 dataset file: {fpath}")
        with open(fpath, "r") as f:
            payload = json.load(f)
        policy = ((payload.get("info") or {}).get("environment_info") or {}).get("policy") or ""
        for idx, sim in enumerate(payload.get("simulations") or []):
            gt_fail = (sim.get("reward_info") or {}).get("reward") != 1.0
            clean_sim = strip_oracle_fields(sim)
            assert "reward_info" not in clean_sim
            items.append(
                {
                    "uid": f"tau2_{domain}_{agent_model}_{sim.get('id', idx)}",
                    "corpus": "tau2",
                    "domain": domain,
                    "agent_model": agent_model,
                    "policy": policy,
                    "messages": clean_sim.get("messages") or [],
                    "ground_truth_fail": gt_fail,
                }
            )

    # 2. Load all 275 real-human OpenHands feedback sessions
    if os.path.exists(OH_PATH):
        with open(OH_PATH, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                gt_fail = row.get("feedback") == "negative"
                clean_row = strip_oracle_fields(row)
                assert "feedback" not in clean_row
                items.append(
                    {
                        "uid": clean_row["session_id"],
                        "corpus": "openhands",
                        "domain": "coding_human",
                        "agent_model": "openhands_codeact",
                        "policy": "",
                        "messages": clean_row.get("messages") or [],
                        "ground_truth_fail": gt_fail,
                    }
                )

    print(f"[Exp 1] Evaluating {len(items)} trajectories with {max_workers} workers...")

    def _eval_one(item: Dict[str, Any]) -> Dict[str, Any]:
        msgs = item["messages"]
        has_tool = tool_det.evaluate_messages(msgs)
        has_regex = regex_det.evaluate_messages(msgs)
        has_sent = sent_det.evaluate_messages(msgs)
        scar_res = scar_det.analyze(messages=msgs, domain_policy=item["policy"])
        return {
            "uid": item["uid"],
            "corpus": item["corpus"],
            "domain": item["domain"],
            "agent_model": item["agent_model"],
            "ground_truth_fail": item["ground_truth_fail"],
            "pred_tool_only": has_tool,
            "pred_naive_regex": has_regex,
            "pred_static_sentiment": has_sent,
            "pred_contextual_pushback": scar_res["genuine_user_pushback"],
            "pred_scar_unresolved": scar_res["unresolved_failure"],
            "scar_state": scar_res["session_state"],
        }

    evaluated: List[Dict[str, Any]] = [None] * len(items)  # type: ignore[list-item]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_idx = {pool.submit(_eval_one, it): i for i, it in enumerate(items)}
        done_cnt = 0
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            evaluated[idx] = fut.result()
            done_cnt += 1
            if done_cnt % 250 == 0 or done_cnt == len(items):
                print(f"  [Exp 1] Completed {done_cnt}/{len(items)} trajectories ({time.time() - t0:.1f}s)")

    tau2_all = [r for r in evaluated if r["corpus"] == "tau2"]
    tau2_airline = [r for r in tau2_all if r["domain"] == "airline"]
    tau2_retail = [r for r in tau2_all if r["domain"] == "retail"]
    oh_all = [r for r in evaluated if r["corpus"] == "openhands"]

    summary = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "tau2_combined": summarize_slice(tau2_all),
        "tau2_airline": summarize_slice(tau2_airline),
        "tau2_retail": summarize_slice(tau2_retail),
        "openhands_human": summarize_slice(oh_all),
        "overall_all_corpora": summarize_slice(evaluated),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "exp1_details.json"), "w") as f:
        json.dump({"summary": summary, "per_session": evaluated}, f, indent=2)

    print("\n=== EXPERIMENT 1 SUMMARY (tau2-bench n=1,312 & OpenHands Human n=275) ===")
    for slice_name in ["tau2_combined", "tau2_airline", "tau2_retail", "openhands_human", "overall_all_corpora"]:
        sl = summary[slice_name]
        print(
            f"\nSlice: {slice_name} (n={sl['total_sessions']}, fails={sl['total_failures']}, "
            f"silent_fails={sl['silent_failures']}/{sl['total_failures']} = {sl['silent_failure_rate']*100:.1f}%)"
        )
        for dname, dm in sl["detectors"].items():
            print(
                f"  {dname:<25} | Prec={dm['precision']*100:5.1f}% | Rec={dm['recall']*100:5.1f}% | "
                f"FPR={dm['fpr']*100:5.1f}% | F1={dm['f1']*100:5.1f}%"
            )

    return summary


if __name__ == "__main__":
    run_exp1()

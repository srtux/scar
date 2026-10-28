"""Run the whole Emotion-Shift Memory study:  python run_experiment.py --stage all

Part 1  Silent failures: how many failed tau2-bench runs had zero tool errors?
Part 2  Stage 1: can four detectors find "frustrated -> fixed" chats among 600 human-labeled WildFeedback chats?
Part 3  Stage 2: which past chats belong in a 400-token MEMORY.md so an AI agent repeats fewer mistakes?
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import os
import random
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from esm.detectors import (QUADRANTS, READ_TOOLS, FlashLiteShiftDetector, GoEmotionsShiftDetector,  # noqa: E402
                           LayaShiftDetector, VaderShiftDetector, extract_reaction_exchanges)
from esm.llm import LLMClient, count_tokens, strip_oracle_fields  # noqa: E402
from esm.memory_md import MarkdownMemory, extract_rule, select_relevant_rules  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
TAU2_DIR = os.path.join(REPO, "data", "tau2")
WILDFEEDBACK_PATH = os.path.join(REPO, "data", "wildfeedback", "sat_dsat_annotation.json")
RESULTS_PATH = os.path.join(REPO, "results", "esm_results.json")
MEMORY_BUDGET = 400

AGENT_MODELS = ("claude-3-7-sonnet-20250219", "gpt-4.1-2025-04-14", "o4-mini-2025-04-16")
STAGE2_FILES = {d: [f"{m}_{d}_default_gpt-4.1-2025-04-14_4trials.json" for m in AGENT_MODELS] for d in ("airline", "retail")}
ALL_FILES = {d: files + [f"gpt-4.1-mini-2025-04-14_{d}_base_gpt-4.1-2025-04-14_4trials.json"] for d, files in STAGE2_FILES.items()}

# The ONLY difference between memory conditions: which detector boxes may enter MEMORY.md.
CONDITIONS = {
    "no_memory": (),
    "all_chats": QUADRANTS,
    "routine_only": ("ROUTINE",),
    "unresolved_only": ("UNFIXED", "BROKEN"),
    "emotion_shift": ("FIXED",),
}
CURVE_CONDITIONS = ("all_chats", "unresolved_only", "emotion_shift")
NON_LATIN_RE = re.compile(r"[\u0400-\u04ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

# Text-only replies that pretend a change already happened, or promise an unauthorized certificate.
CLAIMED_EXECUTION_RE = re.compile(
    r"\b(i have|we have|i've|we've)\s+(now\s+|already\s+|successfully\s+)?"
    r"(cancelled|canceled|returned|exchanged|submitted|modified|updated|issued|processed)\b"
    r"|\byour\s+(reservation|booking|order|refund|return|exchange|certificate|baggage|passenger\w*)"
    r"\s+(has been|have been|is now)\s+(successfully\s+)?"
    r"(cancelled|canceled|returned|exchanged|submitted|modified|updated|issued|processed)\b"
    r"|you are being transferred to a human agent|\[prior tool call:\s*\w+", re.IGNORECASE)
FORBIDDEN_PROMISE_RE = re.compile(
    r"(?:^|[.!?]\s+)(to process|to issue)\s+(?:and\s+\w+\s+)?(your\s+|the\s+|a\s+)?(\$\d+\s+)?(travel\s+|compensation\s+)?certificate\b"
    r"|\b(i can issue|i will issue|i'll issue|let me process|let me issue|i can offer)\s+"
    r"(you\s+)?(your\s+|the\s+|a\s+)?(\$\d+\s+)?(travel\s+|compensation\s+)?certificate\b", re.IGNORECASE)


def _load_json(name: str) -> Dict[str, Any]:
    with open(os.path.join(TAU2_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def _task_key(task_id: str) -> Any:
    return int(task_id) if task_id.isdigit() else task_id


# ---------------------------------------------------------------- Part 1: silent failures
def compute_silent_failures() -> Dict[str, Any]:
    """A silent failure is a failed run (reward below 1) in which no tool ever returned an error."""
    out: Dict[str, Any] = {}
    for domain, files in ALL_FILES.items():
        runs = [s for name in files for s in _load_json(name).get("simulations") or []]
        failed = [s for s in runs if (s.get("reward_info") or {}).get("reward") != 1.0]
        silent = [s for s in failed if not any(m.get("role") == "tool" and m.get("error") for m in s.get("messages") or [])]
        out[domain] = {"total_runs": len(runs), "failed_runs": len(failed), "silent_failures": len(silent)}
    out["combined"] = {k: out["airline"][k] + out["retail"][k] for k in out["airline"]}
    for v in out.values():
        v["silent_failure_rate"] = round(v["silent_failures"] / v["failed_runs"], 4)
    return out


# ---------------------------------------------------------------- Part 2: Stage 1 on WildFeedback
def load_wildfeedback(per_quadrant: int = 150, seed: int = 42) -> List[Dict[str, Any]]:
    """Split WildFeedback into threads and label each with a box from the human annotations."""
    with open(WILDFEEDBACK_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    threads, current = [], []
    for r in rows:
        starts_new = r.get("Role") == "User" and (int(r.get("TurnId", 0)) == 1 or r.get("State") == "NEWTOPIC"
                                                  or str(r.get("Preceeding", "")).upper() == "NO")
        if starts_new and current:
            threads.append(current)
            current = [r]
        else:
            current.append(r)
    threads.append(current)

    pools: Dict[str, List[Dict[str, Any]]] = {q: [] for q in QUADRANTS}
    for idx, thread in enumerate(threads):
        users = [r for r in thread if r.get("Role") == "User" and (r.get("Content") or "").strip()]
        if len(users) < 3 or NON_LATIN_RE.search(" ".join(r.get("Content") or "" for r in users)):
            continue
        dsat = [bool(r.get("Disatisfaction")) for r in users[1:]]  # human labels on the reaction turns
        sat = [bool(r.get("Satisfaction")) for r in users[1:]]
        if any(dsat[:-1]):
            gold = "UNFIXED" if dsat[-1] else ("FIXED" if sat[-1] else None)
        else:
            gold = "BROKEN" if dsat[-1] else ("ROUTINE" if sat[-1] else None)
        if gold:
            pools[gold].append({
                "thread_id": f"wf_{idx}", "gold_quadrant": gold, "num_reaction_turns": len(dsat),
                "human_labels": ["dissatisfied" if d else ("satisfied" if s else "neutral") for d, s in zip(dsat, sat)],
                "messages": [{"role": "user" if r.get("Role") == "User" else "assistant", "content": r.get("Content") or ""}
                             for r in thread if (r.get("Content") or "").strip()]})
    return [item for q in QUADRANTS for item in random.Random(seed).sample(pools[q], per_quadrant)]


def score_quadrants(golds: Sequence[str], preds: Sequence[str], ms_per_turn: float, ms_per_chat: float) -> Dict[str, Any]:
    confusion = {g: {p: 0 for p in QUADRANTS} for g in QUADRANTS}
    for g, p in zip(golds, preds):
        confusion[g][p] += 1
    per_q = {}
    for q in QUADRANTS:
        tp = confusion[q][q]
        precision = tp / max(1, sum(confusion[g][q] for g in QUADRANTS))
        recall = tp / max(1, sum(confusion[q].values()))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_q[q] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    return {"fixed_precision": per_q["FIXED"]["precision"], "fixed_recall": per_q["FIXED"]["recall"],
            "fixed_f1": per_q["FIXED"]["f1"], "macro_f1": round(sum(v["f1"] for v in per_q.values()) / 4, 4),
            "accuracy_4way": round(sum(confusion[q][q] for q in QUADRANTS) / len(golds), 4),
            "ms_per_turn": round(ms_per_turn, 3), "ms_per_chat": round(ms_per_chat, 2),
            "per_quadrant": per_q, "confusion_matrix": confusion}


def run_stage1(llm: LLMClient, workers: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    items = load_wildfeedback()
    chats, golds = [x["messages"] for x in items], [x["gold_quadrant"] for x in items]
    timing = chats[::max(1, len(chats) // 32)][:32]  # 32 chats spread across all four boxes

    def n_turns(sample: Sequence[Any]) -> int:
        return sum(max(1, len(extract_reaction_exchanges(m)["reaction_turns"])) for m in sample)

    def elapsed_ms(classify: Any, sample: Sequence[Any]) -> float:
        start = time.perf_counter()
        for m in sample:
            classify(m)
        return (time.perf_counter() - start) * 1000.0

    vader, goem, laya, flash = VaderShiftDetector(), GoEmotionsShiftDetector(), LayaShiftDetector(), FlashLiteShiftDetector(llm=llm)
    preds, results = {}, {}
    ms = elapsed_ms(vader.classify, timing)
    preds["VADER"] = [vader.classify(m)[0] for m in chats]
    results["VADER"] = score_quadrants(golds, preds["VADER"], ms / n_turns(timing), ms / len(timing))

    goem.classify(timing[0])  # warm-up
    ms = elapsed_ms(goem.classify, timing)
    preds["GoEmotions"] = goem.classify_many(chats)
    results["GoEmotions"] = score_quadrants(golds, preds["GoEmotions"], ms / n_turns(timing), ms / len(timing))

    laya.classify(timing[0])  # warm-up
    ms = elapsed_ms(laya.classify, timing[:16])
    preds["Laya"] = laya.classify_many(chats)
    results["Laya"] = score_quadrants(golds, preds["Laya"], ms / n_turns(timing[:16]), ms / 16)

    live = LLMClient(use_cache=False)  # true network + model latency, no cache and no rate limiter
    live.max_rps = 0.0
    latencies = [FlashLiteShiftDetector(llm=live).classify(m)[1] for m in timing[:16]]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        preds["Flash-Lite"] = list(pool.map(lambda m: flash.classify(m)[0], chats))
    results["Flash-Lite"] = score_quadrants(golds, preds["Flash-Lite"], sum(latencies) / n_turns(timing[:16]),
                                            statistics.median(latencies))

    # Poster example: humans said FIXED, VADER says ROUTINE, Laya says FIXED. Keep the three shortest.
    picks = [i for i, x in enumerate(items) if x["gold_quadrant"] == "FIXED" and preds["VADER"][i] == "ROUTINE"
             and preds["Laya"][i] == "FIXED" and not NON_LATIN_RE.search(" ".join(m["content"] for m in x["messages"]))]
    picks.sort(key=lambda i: sum(len(m["content"]) for m in items[i]["messages"]))
    showcase = []
    for i in picks[:3]:
        ex = extract_reaction_exchanges(items[i]["messages"])
        showcase.append({"thread_id": items[i]["thread_id"], "messages": items[i]["messages"],
                         "human_labels": items[i]["human_labels"], "reaction_turns": ex["reaction_turns"],
                         "vader_compound": [round(vader.compound(t), 3) for t in ex["reaction_turns"]],
                         "laya_dissatisfied": [round(p, 3) for p in laya.score_pairs(ex["reaction_pairs"])]})
    stage1 = {"total_threads": len(items), "per_quadrant": 150, "total_reaction_turns": sum(x["num_reaction_turns"] for x in items),
              "detectors": results}
    return stage1, {"candidates": showcase}


# ---------------------------------------------------------------- Part 3: Stage 2 on tau2-bench
def load_domain(domain: str) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {"domain": domain, "tools": _load_json(f"{domain}_tools.json"), "policy": "",
                              "tasks_by_id": {}, "sims_by_task": collections.defaultdict(list)}
    for name in STAGE2_FILES[domain]:
        data = _load_json(name)
        bundle["policy"] = bundle["policy"] or data["info"]["environment_info"]["policy"]
        bundle["tasks_by_id"].update({str(t["id"]): t for t in data["tasks"]})
        for sim in data["simulations"]:
            bundle["sims_by_task"][str(sim["task_id"])].append(sim)
    return bundle


def build_train_stream(bundle: Dict[str, Any], train_ids: Sequence[str], limit: int = 100) -> List[Dict[str, Any]]:
    """Past chats in arrival order: trial 1 of every model on every task, then trial 2, and so on."""
    sims = bundle["sims_by_task"]
    longest = max(len(sims[t]) for t in train_ids)
    n_models = max(1, longest // 4)
    order = [m * 4 + trial for trial in range(4) for m in range(n_models) if m * 4 + trial < longest]
    stream = []
    for i in order + list(range(n_models * 4, longest)):
        for tid in train_ids:
            if i < len(sims[tid]):
                stream.append(strip_oracle_fields(sims[tid][i]))
                if len(stream) >= limit:
                    return stream
    return stream


def _canonical(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool arguments so equal calls compare equal (paired item lists, 5.0 == 5, trimmed strings)."""
    items, new_items = args.get("item_ids"), args.get("new_item_ids")
    if tool in ("exchange_delivered_order_items", "modify_pending_order_items") and isinstance(items, list) \
            and isinstance(new_items, list) and len(items) == len(new_items):
        pairs = sorted(zip(map(str, items), map(str, new_items)))
        rest = {k: (v.strip() if isinstance(v, str) else v) for k, v in args.items() if k not in ("item_ids", "new_item_ids")}
        return {"item_ids": [a for a, _ in pairs], "new_item_ids": [b for _, b in pairs], **rest}
    out = {}
    for k, v in (args or {}).items():
        if k == "item_ids" and isinstance(v, list):
            out[k] = sorted(map(str, v))
        elif isinstance(v, float) and v.is_integer():
            out[k] = int(v)
        elif isinstance(v, str):
            out[k] = v.strip()
        elif isinstance(v, list):
            out[k] = [_canonical("", x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def actions_match(pred: Dict[str, Any], target: Dict[str, Any]) -> bool:
    name = target.get("name", "")
    if pred.get("name") != name:
        return False
    if name == "transfer_to_human_agents":
        return True
    p_args, t_args = _canonical(name, pred.get("arguments") or {}), _canonical(name, target.get("arguments") or {})
    return all((k in p_args) if k == "reason" else p_args.get(k) == v for k, v in t_args.items())


def extract_decision_cases(bundle: Dict[str, Any], test_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Find held-out decision points: the first state-changing step of runs that went wrong (and good controls).

    A failure case is cut right before a write call that either crashed with a tool error, or was a silent
    policy violation (the task expects no write at all). A control is cut right before a correct write call.
    """
    fails, controls = [], []
    for tid in sorted(test_ids, key=_task_key):
        expected = [a for a in (bundle["tasks_by_id"][tid].get("evaluation_criteria") or {}).get("actions") or []
                    if a.get("requestor") != "user" and a.get("name") not in READ_TOOLS]
        seen: set = set()
        for sim in bundle["sims_by_task"][tid]:
            msgs = sim.get("messages") or []
            for idx, m in enumerate(msgs):
                writes = [tc for tc in m.get("tool_calls") or [] if tc.get("name") not in READ_TOOLS] if m.get("role") == "assistant" else []
                if not writes:
                    continue
                error, j = "", idx + 1
                while j < len(msgs) and msgs[j].get("role") == "tool":
                    if msgs[j].get("error"):
                        error = msgs[j].get("content") or "tool_error"
                        break
                    j += 1
                if error and idx >= 2:
                    kind, sig = "mut_tool_error", ("mut_tool_error", writes[0].get("name"), error[:45])
                elif (sim.get("reward_info") or {}).get("reward") != 1.0 and not expected and idx >= 2:
                    kind, sig = "silent_forbid", ("silent_forbid", writes[0].get("name"))
                else:
                    kind = sig = None
                if kind and sig not in seen and len(seen) < 2:
                    seen.add(sig)
                    fails.append({"domain": bundle["domain"], "task_id": tid, "sim_id": sim.get("id", ""), "kind": kind,
                                  "prefix_messages": msgs[:idx], "expected": expected, "bad_call": writes[0].get("name")})
                break  # only the first state-changing step of each run
            if len(seen) >= 2:
                break
        if len(expected) == 1 and len(controls) < 16:
            for sim in bundle["sims_by_task"][tid]:
                if (sim.get("reward_info") or {}).get("reward") != 1.0:
                    continue
                msgs = sim.get("messages") or []
                for idx, m in enumerate(msgs):
                    writes = [tc for tc in m.get("tool_calls") or [] if tc.get("name") not in READ_TOOLS] if m.get("role") == "assistant" else []
                    if writes:
                        if actions_match(writes[0], expected[0]) and idx >= 2:
                            controls.append({"domain": bundle["domain"], "task_id": tid, "sim_id": sim.get("id", ""),
                                             "kind": "control", "prefix_messages": msgs[:idx], "expected": expected})
                        break
                if controls and controls[-1]["task_id"] == tid:
                    break
    return fails, controls


def grade_step(step: Dict[str, Any], case: Dict[str, Any]) -> bool:
    """True when the agent's next move avoids the historical mistake (or, for controls, makes the right write)."""
    calls = step.get("tool_calls") or []
    writes = [tc for tc in calls if tc.get("name") not in READ_TOOLS]
    text = (step.get("content") or "").strip()
    if case["kind"] == "control":
        return bool(writes) and any(actions_match(writes[0], t) for t in case["expected"])
    if case["kind"] == "silent_forbid" or not case["expected"]:
        return not writes and not CLAIMED_EXECUTION_RE.search(text) and not FORBIDDEN_PROMISE_RE.search(text)
    if writes:
        return any(actions_match(writes[0], t) for t in case["expected"])
    return len(calls) > 0 or (bool(text) and not CLAIMED_EXECUTION_RE.search(text))


def agent_step(llm: LLMClient, bundle: Dict[str, Any], case: Dict[str, Any], memory_md: str) -> Dict[str, Any]:
    retrieved = select_relevant_rules(memory_md, case["prefix_messages"], max_tokens=MEMORY_BUDGET)
    system = f"You are a customer service agent strictly following the Domain Policy below.\n\n=== DOMAIN POLICY ===\n{bundle['policy']}"
    if retrieved:
        system += f"\n\n=== RETRIEVED MEMORY (<= {MEMORY_BUDGET} tokens) ===\n{retrieved}"
    step = llm.generate_agent_step(messages=case["prefix_messages"], system_instruction=system, tools=bundle["tools"])
    return {"passed": grade_step(step, case), "injected_tokens": count_tokens(retrieved), "retrieved": retrieved,
            "tool_calls": [tc.get("name") for tc in step.get("tool_calls") or []], "text": (step.get("content") or "")[:300]}


def mcnemar_exact_p(better: int, worse: int) -> float:
    """Two-sided exact McNemar test on paired outcomes (binomial with p = 0.5 on the discordant pairs)."""
    n = better + worse
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / 2 ** n)


def run_stage2(llm: LLMClient, workers: int) -> Dict[str, Any]:
    gate = FlashLiteShiftDetector(llm=llm)
    rows: List[Dict[str, Any]] = []
    gate_counts, memory_sizes, samples = {}, collections.defaultdict(list), {}
    for domain in ("airline", "retail"):
        bundle = load_domain(domain)
        ids = sorted(bundle["tasks_by_id"], key=_task_key)
        for fold, train_ids, test_ids in (("fold_A", ids[0::2], ids[1::2]), ("fold_B", ids[1::2], ids[0::2])):
            fails, controls = extract_decision_cases(bundle, test_ids)
            controls = controls[:8]
            stream = build_train_stream(bundle, train_ids)
            with concurrent.futures.ThreadPoolExecutor(workers) as pool:
                boxes = list(pool.map(lambda c: gate.classify(c.get("messages") or [], domain_policy=bundle["policy"])[0], stream))
                rules = list(pool.map(lambda c: extract_rule(llm, c, domain), stream))
            gate_counts[f"{domain}_{fold}"] = dict(collections.Counter(boxes))

            def memory(cond: str, n: int) -> MarkdownMemory:
                mem = MarkdownMemory(MEMORY_BUDGET)
                for box, rule in list(zip(boxes, rules))[:n]:
                    if box in CONDITIONS[cond]:
                        mem.add_rule(rule)
                return mem

            files = {(c, n): memory(c, n).render_all() for c in CONDITIONS for n in (25, 50, 100)}
            for c in CONDITIONS:
                memory_sizes[c].append(len(memory(c, 100).rules))
            if fold == "fold_A":
                samples[domain] = {c: memory(c, 100).render() for c in CONDITIONS if c != "no_memory"}
            jobs = ([(c, 100, case) for case in fails + controls for c in CONDITIONS]
                    + [(c, n, case) for n in (25, 50) for c in CURVE_CONDITIONS for case in fails])
            with concurrent.futures.ThreadPoolExecutor(workers) as pool:
                outs = list(pool.map(lambda job: agent_step(llm, bundle, job[2], files[(job[0], job[1])]), jobs))
            for (c, n, case), out in zip(jobs, outs):
                rows.append({"case": f"{domain}/{fold}/{case['task_id']}/{case['sim_id']}", "domain": domain, "kind": case["kind"],
                             "bad_call": case.get("bad_call"), "last_user": next((m.get("content") for m in reversed(case["prefix_messages"])
                                                                                  if m.get("role") == "user"), ""),
                             "condition": c, "n_chats": n, **out})
    return summarize_stage2(rows, gate_counts, memory_sizes, samples)


def summarize_stage2(rows: List[Dict[str, Any]], gate_counts: Dict[str, Any], memory_sizes: Dict[str, List[int]],
                     samples: Dict[str, Any]) -> Dict[str, Any]:
    fails = [r for r in rows if r["kind"] != "control"]
    at_100 = {c: {r["case"]: r for r in fails if r["condition"] == c and r["n_chats"] == 100} for c in CONDITIONS}

    def rate(rs: Sequence[Dict[str, Any]]) -> float:
        return round(sum(not r["passed"] for r in rs) / len(rs), 4) if rs else 0.0

    conditions = {}
    for c, by_case in at_100.items():
        rs = list(by_case.values())
        conditions[c] = {"heldout_cases": len(rs), "mistakes_repeated": sum(not r["passed"] for r in rs), "repeat_rate": rate(rs),
                         "airline_repeat_rate": rate([r for r in rs if r["domain"] == "airline"]),
                         "retail_repeat_rate": rate([r for r in rs if r["domain"] == "retail"]),
                         "avg_injected_tokens": round(statistics.mean(r["injected_tokens"] for r in rs), 1),
                         "rules_in_memory_per_fold": memory_sizes[c]}
    controls = {c: {"controls": len(rs), "passed": sum(r["passed"] for r in rs), "pass_rate": round(sum(r["passed"] for r in rs) / len(rs), 4)}
                for c in CONDITIONS for rs in [[r for r in rows if r["kind"] == "control" and r["condition"] == c]]}
    paired = {}
    for c in CONDITIONS:
        if c != "emotion_shift":
            better = sum(at_100["emotion_shift"][k]["passed"] and not r["passed"] for k, r in at_100[c].items())
            worse = sum(r["passed"] and not at_100["emotion_shift"][k]["passed"] for k, r in at_100[c].items())
            paired[c] = {"esm_avoids_other_repeats": better, "esm_repeats_other_avoids": worse, "mcnemar_p": round(mcnemar_exact_p(better, worse), 4)}
    curve = {c: [rate(list(at_100["no_memory"].values()))] + [rate([r for r in fails if r["condition"] == c and r["n_chats"] == n]) for n in (25, 50, 100)]
             for c in CURVE_CONDITIONS}
    rescues = [{"case": k, "kind": r["kind"], "bad_call": r["bad_call"], "last_user": (r["last_user"] or "")[:300],
                "no_memory_action": {"tool_calls": r["tool_calls"], "text": r["text"]},
                "esm_action": {"tool_calls": at_100["emotion_shift"][k]["tool_calls"], "text": at_100["emotion_shift"][k]["text"]},
                "esm_retrieved_memory": at_100["emotion_shift"][k]["retrieved"]}
               for k, r in at_100["no_memory"].items() if not r["passed"] and at_100["emotion_shift"][k]["passed"]]
    base, esm = conditions["no_memory"]["repeat_rate"], conditions["emotion_shift"]["repeat_rate"]
    return {"memory_budget_tokens": MEMORY_BUDGET, "gate_quadrant_counts": gate_counts, "conditions": conditions,
            "controls": controls, "paired_vs_emotion_shift": paired,
            "learning_curve": {"chats_seen": [0, 25, 50, 100], "repeat_rate": curve},
            "relative_reduction_vs_no_memory": round((base - esm) / base, 4) if base else 0.0,
            "rescue_examples": rescues[:4], "sample_memory_md": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Emotion-Shift Memory experiments.")
    parser.add_argument("--stage", choices=("all", "stage1", "stage2"), default="all")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    llm = LLMClient()
    new: Dict[str, Any] = {"silent_failures": compute_silent_failures()}
    if args.stage in ("all", "stage1"):
        new["stage1_shift_detection"], new["wildfeedback_showcase"] = run_stage1(llm, args.workers)
    if args.stage in ("all", "stage2"):
        new["stage2_memory"] = run_stage2(llm, args.workers)
    results: Dict[str, Any] = {}
    if os.path.exists(RESULTS_PATH):  # merge at the end, so separate stage runs never erase each other
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
    results.update(new)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

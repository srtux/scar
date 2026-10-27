"""End-to-end runner for Emotion-Shift Memory (ESM): Stage 1 shift gates + Stage 2 MEMORY.md learning."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import random
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from esm.detectors import (
    FlashLiteShiftDetector,
    GoEmotionsShiftDetector,
    LayaShiftDetector,
    VaderShiftDetector,
    extract_reaction_exchanges,
)
from esm.memory_md import (
    MarkdownMemory,
    extract_contrastive_fix_rule,
    extract_rule_from_chat,
    select_relevant_rules,
    verify_rule_on_mistake_turn,
)
from exp3_repeat_mistakes import (
    READ_TOOLS,
    _actions_match,
    _extract_decision_cases,
    _load_domain_bundle,
)
from scar.consolidator import strip_oracle_fields
from scar.llm import LLMClient
from scar.tokens import count_tokens

_CLAIMED_EXECUTION_RE = re.compile(
    r"\b(i have|we have|i've|we've)\s+(now\s+|already\s+|successfully\s+)?"
    r"(cancelled|canceled|returned|exchanged|submitted|modified|updated|issued|processed)\b"
    r"|\byour\s+(reservation|booking|order|refund|return|exchange|certificate|baggage|passenger\w*)"
    r"\s+(has been|have been|is now)\s+(successfully\s+)?"
    r"(cancelled|canceled|returned|exchanged|submitted|modified|updated|issued|processed)\b"
    r"|you are being transferred to a human agent"
    r"|\[prior tool call:\s*\w+",
    re.IGNORECASE,
)

_FORBIDDEN_PROMISE_RE = re.compile(
    r"(?:^|[.!?]\s+)(to process|to issue)\s+(?:and\s+\w+\s+)?(your\s+|the\s+|a\s+)?(\$\d+\s+)?(travel\s+|compensation\s+)?certificate\b"
    r"|\b(i can issue|i will issue|i'll issue|let me process|let me issue|i can offer)\s+"
    r"(you\s+)?(your\s+|the\s+|a\s+)?(\$\d+\s+)?(travel\s+|compensation\s+)?certificate\b",
    re.IGNORECASE,
)


def _esm_grade_decision_step(
    step_out: Dict[str, Any],
    expected_mut_actions: Sequence[Dict[str, Any]],
    kind: str,
    is_control: bool = False,
) -> bool:
    """Symmetric decision-step grader avoiding false positives on quoted policy conditions."""
    tool_calls = step_out.get("tool_calls") or []
    mut_calls = [tc for tc in tool_calls if tc.get("name") not in READ_TOOLS]
    read_calls = [tc for tc in tool_calls if tc.get("name") in READ_TOOLS]
    content = (step_out.get("content") or "").strip()

    if is_control:
        return len(mut_calls) > 0 and any(
            _actions_match(mut_calls[0], t) for t in expected_mut_actions
        )

    if kind == "silent_forbid" or len(expected_mut_actions) == 0:
        if len(mut_calls) > 0:
            return False
        if _CLAIMED_EXECUTION_RE.search(content) or _FORBIDDEN_PROMISE_RE.search(content):
            return False
        return True

    if len(mut_calls) > 0:
        return any(_actions_match(mut_calls[0], t) for t in expected_mut_actions)

    if len(read_calls) > 0:
        return True
    return bool(content) and not bool(_CLAIMED_EXECUTION_RE.search(content))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAU2_DIR = os.path.join(REPO_ROOT, "data", "tau2")
OH_PATH = os.path.join(REPO_ROOT, "data", "openhands", "openhands_feedback_sessions.jsonl")
WF_PATH = os.path.join(REPO_ROOT, "data", "wildfeedback", "sat_dsat_annotation.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

_NON_LATIN_RE = re.compile(r"[\u0400-\u04ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
QUADRANTS = ("FIXED", "UNFIXED", "BROKEN", "ROUTINE")

ALL_TAU2_FILES: Dict[str, List[str]] = {
    "airline": [
        "claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json",
        "gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json",
        "gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json",
        "o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json",
    ],
    "retail": [
        "claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json",
        "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json",
        "gpt-4.1-mini-2025-04-14_retail_base_gpt-4.1-2025-04-14_4trials.json",
        "o4-mini-2025-04-16_retail_default_gpt-4.1-2025-04-14_4trials.json",
    ],
}

STAGE2_TAU2_FILES: Dict[str, List[str]] = {
    "airline": [
        "claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json",
        "gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json",
        "o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json",
    ],
    "retail": [
        "claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json",
        "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json",
        "o4-mini-2025-04-16_retail_default_gpt-4.1-2025-04-14_4trials.json",
    ],
}


def compute_chart1_silent_failures() -> Dict[str, Any]:
    """Compute exact silent-failure vs. explicit tool-crash counts across all 2,624 tau2 runs."""
    out: Dict[str, Any] = {}
    comb = {"total_runs": 0, "failed_runs": 0, "silent_failures": 0, "passing_with_tool_error": 0}

    for dom, fnames in ALL_TAU2_FILES.items():
        st = {"total_runs": 0, "failed_runs": 0, "silent_failures": 0, "passing_with_tool_error": 0}
        for fname in fnames:
            with open(os.path.join(TAU2_DIR, fname), "r", encoding="utf-8") as f:
                payload = json.load(f)
            for s in payload.get("simulations") or []:
                st["total_runs"] += 1
                failed = (s.get("reward_info") or {}).get("reward") != 1.0
                has_err = any(
                    m.get("role") == "tool" and m.get("error") for m in (s.get("messages") or [])
                )
                if failed:
                    st["failed_runs"] += 1
                    if not has_err:
                        st["silent_failures"] += 1
                elif has_err:
                    st["passing_with_tool_error"] += 1
        st["silent_failure_rate"] = (
            round(st["silent_failures"] / st["failed_runs"], 4) if st["failed_runs"] else 0.0
        )
        out[dom] = st
        for k in comb:
            comb[k] += st[k]

    comb["silent_failure_rate"] = (
        round(comb["silent_failures"] / comb["failed_runs"], 4) if comb["failed_runs"] else 0.0
    )
    out["combined"] = comb
    return out


def load_wildfeedback_benchmark(per_quadrant: int = 150, seed: int = 42) -> List[Dict[str, Any]]:
    """Parse chronological human-annotated English multi-turn threads from WildFeedback."""
    if not os.path.exists(WF_PATH):
        raise FileNotFoundError(f"Missing WildFeedback file at {WF_PATH}")
    with open(WF_PATH, "r", encoding="utf-8") as f:
        raw_rows = json.load(f)

    threads: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for r in raw_rows:
        is_new = r.get("Role") == "User" and (
            int(r.get("TurnId", 0)) == 1
            or r.get("State") == "NEWTOPIC"
            or str(r.get("Preceeding", "")).upper() == "NO"
        )
        if is_new and cur:
            threads.append(cur)
            cur = [r]
        else:
            cur.append(r)
    if cur:
        threads.append(cur)

    pools: Dict[str, List[Dict[str, Any]]] = {q: [] for q in QUADRANTS}
    for tidx, th in enumerate(threads):
        u_rows = [r for r in th if r.get("Role") == "User" and (r.get("Content") or "").strip()]
        if len(u_rows) < 3:
            continue
        joined_u = " ".join(r.get("Content") or "" for r in u_rows)
        if _NON_LATIN_RE.search(joined_u):
            continue

        # Human ground truth on reaction turns u_1..u_{k-1}
        r_dsats = [bool(r.get("Disatisfaction")) for r in u_rows[1:]]
        r_sats = [bool(r.get("Satisfaction")) for r in u_rows[1:]]
        early_dsat = any(r_dsats[:-1])
        final_dsat = r_dsats[-1]

        if early_dsat and not final_dsat and r_sats[-1]:
            gold = "FIXED"
        elif early_dsat and final_dsat:
            gold = "UNFIXED"
        elif not early_dsat and final_dsat:
            gold = "BROKEN"
        elif not any(r_dsats) and r_sats[-1]:
            gold = "ROUTINE"
        else:
            continue

        clean_msgs = [
            {
                "role": "user" if r.get("Role") == "User" else "assistant",
                "content": r.get("Content") or "",
            }
            for r in th
            if (r.get("Content") or "").strip()
        ]
        pools[gold].append(
            {
                "thread_id": f"wf_{tidx}",
                "gold_quadrant": gold,
                "num_reaction_turns": len(r_dsats),
                "messages": clean_msgs,
            }
        )

    sampled: List[Dict[str, Any]] = []
    for q in QUADRANTS:
        rng = random.Random(seed)
        if len(pools[q]) < per_quadrant:
            raise ValueError(f"Insufficient WildFeedback pool for {q}: {len(pools[q])}")
        sampled.extend(rng.sample(pools[q], per_quadrant))
    return sampled


def load_openhands_multiturn_english() -> List[Dict[str, Any]]:
    """Load all 176 multi-turn (>=2 user turns) English human-rated OpenHands sessions."""
    rows: List[Dict[str, Any]] = []
    with open(OH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ut = [
                m.get("content") or ""
                for m in (r.get("messages") or [])
                if m.get("role") == "user" and (m.get("content") or "").strip()
            ]
            if len(ut) < 2 or _NON_LATIN_RE.search(" ".join(ut)):
                continue
            gt_neg = r.get("feedback") == "negative"
            clean_r = strip_oracle_fields(r)
            assert "feedback" not in clean_r
            rows.append(
                {
                    "session_id": clean_r["session_id"],
                    "ground_truth_negative": gt_neg,
                    "messages": clean_r.get("messages") or [],
                }
            )
    return rows


def _score_quadrant_predictions(
    golds: Sequence[str],
    preds: Sequence[str],
    ms_per_turn: float,
    ms_per_chat: float,
) -> Dict[str, Any]:
    cm: Dict[str, Dict[str, int]] = {g: {p: 0 for p in QUADRANTS} for g in QUADRANTS}
    for g, p in zip(golds, preds):
        cm[g][p] += 1

    per_q: Dict[str, Dict[str, float]] = {}
    for q in QUADRANTS:
        tp = cm[q][q]
        fp = sum(cm[g][q] for g in QUADRANTS if g != q)
        fn = sum(cm[q][p] for p in QUADRANTS if p != q)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_q[q] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    macro_f1 = sum(per_q[q]["f1"] for q in QUADRANTS) / len(QUADRANTS)
    acc = sum(cm[q][q] for q in QUADRANTS) / len(golds) if golds else 0.0
    return {
        "fixed_precision": per_q["FIXED"]["precision"],
        "fixed_recall": per_q["FIXED"]["recall"],
        "fixed_f1": per_q["FIXED"]["f1"],
        "macro_f1": round(macro_f1, 4),
        "accuracy_4way": round(acc, 4),
        "ms_per_turn": round(ms_per_turn, 3),
        "ms_per_chat": round(ms_per_chat, 2),
        "per_quadrant": per_q,
        "confusion_matrix": cm,
    }


def _score_openhands_predictions(
    oh_rows: Sequence[Dict[str, Any]],
    preds: Sequence[str],
) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    pos_recovered_as_fixed = 0
    for r, p in zip(oh_rows, preds):
        y_neg = bool(r["ground_truth_negative"])
        pred_neg = p in ("UNFIXED", "BROKEN")
        if y_neg and pred_neg:
            tp += 1
        elif (not y_neg) and pred_neg:
            fp += 1
        elif (not y_neg) and (not pred_neg):
            tn += 1
            if p == "FIXED":
                pos_recovered_as_fixed += 1
        else:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "total_sessions": len(oh_rows),
        "unresolved_precision": round(prec, 4),
        "unresolved_recall": round(rec, 4),
        "unresolved_f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(oh_rows), 4) if oh_rows else 0.0,
        "positive_recoveries_kept_as_fixed": pos_recovered_as_fixed,
        "quadrant_counts": dict(collections.Counter(preds)),
    }


def run_stage1_shift_detection(
    llm: LLMClient,
    per_quadrant: int = 150,
    max_workers: int = 16,
) -> Dict[str, Any]:
    """Evaluate VADER, GoEmotions, Laya, and Gemini 3.5 Flash-Lite on WildFeedback + OpenHands."""
    t0 = time.time()
    wf_items = load_wildfeedback_benchmark(per_quadrant=per_quadrant, seed=42)
    oh_items = load_openhands_multiturn_english()
    wf_msgs = [x["messages"] for x in wf_items]
    wf_golds = [x["gold_quadrant"] for x in wf_items]
    oh_msgs = [x["messages"] for x in oh_items]

    print(
        f"[Stage 1] Loaded {len(wf_items)} WildFeedback threads ({per_quadrant}/quadrant) "
        f"+ {len(oh_items)} multi-turn English OpenHands sessions."
    )

    # Dedicated timing slice (32 threads across all 4 quadrants) for true single-call latency
    timing_slice = [wf_items[i]["messages"] for i in range(0, len(wf_items), max(1, len(wf_items) // 32))][:32]
    timing_turns = sum(max(1, len(extract_reaction_exchanges(m)["reaction_turns"])) for m in timing_slice)

    vader = VaderShiftDetector()
    goem = GoEmotionsShiftDetector()
    laya_det = LayaShiftDetector()
    flite = FlashLiteShiftDetector(llm=llm)

    wf_results: Dict[str, Any] = {}
    oh_results: Dict[str, Any] = {}

    # 1. VADER
    t_s = time.perf_counter()
    for m in timing_slice:
        vader.classify(m)
    v_chat_ms = ((time.perf_counter() - t_s) * 1000.0) / len(timing_slice)
    v_turn_ms = ((time.perf_counter() - t_s) * 1000.0) / timing_turns
    v_wf = [vader.classify(m)[0] for m in wf_msgs]
    v_oh = [vader.classify(m)[0] for m in oh_msgs]
    wf_results["VADER"] = _score_quadrant_predictions(wf_golds, v_wf, v_turn_ms, v_chat_ms)
    oh_results["VADER"] = _score_openhands_predictions(oh_items, v_oh)
    print(f"  [Stage 1] VADER done: WF FIXED F1={wf_results['VADER']['fixed_f1']*100:.1f}% ({v_turn_ms:.3f} ms/turn)")

    # 2. GoEmotions
    goem.classify(timing_slice[0])  # warmup
    t_s = time.perf_counter()
    for m in timing_slice:
        goem.classify(m)
    g_total_ms = (time.perf_counter() - t_s) * 1000.0
    g_wf = goem.classify_many(wf_msgs)
    g_oh = goem.classify_many(oh_msgs)
    wf_results["GoEmotions"] = _score_quadrant_predictions(
        wf_golds, g_wf, g_total_ms / timing_turns, g_total_ms / len(timing_slice)
    )
    oh_results["GoEmotions"] = _score_openhands_predictions(oh_items, g_oh)
    print(
        f"  [Stage 1] GoEmotions done: WF FIXED F1={wf_results['GoEmotions']['fixed_f1']*100:.1f}% "
        f"({wf_results['GoEmotions']['ms_per_turn']:.1f} ms/turn)"
    )

    # 3. Laya (local ModernBERT)
    laya_det.classify(timing_slice[0])  # warmup
    t_s = time.perf_counter()
    for m in timing_slice[:16]:
        laya_det.classify(m)
    l_total_ms = (time.perf_counter() - t_s) * 1000.0
    l_turns_16 = sum(max(1, len(extract_reaction_exchanges(m)["reaction_turns"])) for m in timing_slice[:16])
    l_wf = laya_det.classify_many(wf_msgs)
    l_oh = laya_det.classify_many(oh_msgs)
    wf_results["Laya"] = _score_quadrant_predictions(
        wf_golds, l_wf, l_total_ms / l_turns_16, l_total_ms / 16.0
    )
    oh_results["Laya"] = _score_openhands_predictions(oh_items, l_oh)
    print(
        f"  [Stage 1] Laya done: WF FIXED F1={wf_results['Laya']['fixed_f1']*100:.1f}% "
        f"({wf_results['Laya']['ms_per_turn']:.1f} ms/turn)"
    )

    # 4. Gemini 3.5 Flash-Lite (measure true uncached live API latency on 16 sequential calls)
    uncached_llm = LLMClient(use_cache=False)
    uncached_llm.max_rps = 0.0  # measure pure network+model latency without client throttle sleep
    uncached_flite = FlashLiteShiftDetector(llm=uncached_llm)
    live_chat_latencies: List[float] = []
    live_turn_counts: List[int] = []
    for m in timing_slice[:16]:
        _, ms = uncached_flite.classify(m)
        live_chat_latencies.append(ms)
        live_turn_counts.append(max(1, len(extract_reaction_exchanges(m)["reaction_turns"])))
    fl_chat_ms = statistics.median(live_chat_latencies)
    fl_turn_ms = sum(live_chat_latencies) / sum(live_turn_counts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        fl_wf = list(pool.map(lambda m: flite.classify(m)[0], wf_msgs))
        fl_oh = list(pool.map(lambda m: flite.classify(m)[0], oh_msgs))

    wf_results["Flash-Lite"] = _score_quadrant_predictions(wf_golds, fl_wf, fl_turn_ms, fl_chat_ms)
    oh_results["Flash-Lite"] = _score_openhands_predictions(oh_items, fl_oh)
    print(
        f"  [Stage 1] Flash-Lite done: WF FIXED F1={wf_results['Flash-Lite']['fixed_f1']*100:.1f}% "
        f"({wf_results['Flash-Lite']['ms_per_turn']:.1f} ms/turn, median {fl_chat_ms:.0f} ms/chat)"
    )

    return {
        "elapsed_seconds": round(time.time() - t0, 2),
        "wildfeedback": {
            "total_threads": len(wf_items),
            "per_quadrant": per_quadrant,
            "total_reaction_turns": sum(x["num_reaction_turns"] for x in wf_items),
            "detectors": wf_results,
        },
        "openhands_multiturn": {
            "total_sessions": len(oh_items),
            "detectors": oh_results,
        },
    }


def _build_deterministic_train_stream(
    bundle: Dict[str, Any],
    train_task_ids: Sequence[str],
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Build an oracle-stripped stream of `limit` training chats interleaved across tasks & policy-equipped LLMs."""
    sorted_tids = sorted(train_task_ids, key=lambda x: int(x) if x.isdigit() else x)
    sims_by_task = bundle["sims_by_task"]
    max_len = max((len(sims_by_task[tid]) for tid in sorted_tids), default=0)
    num_models = max(1, max_len // 4)
    interleaved_indices: List[int] = []
    for trial_offset in range(4):
        for model_idx in range(num_models):
            idx = model_idx * 4 + trial_offset
            if idx < max_len:
                interleaved_indices.append(idx)
    for extra_idx in range(num_models * 4, max_len):
        interleaved_indices.append(extra_idx)

    stream: List[Dict[str, Any]] = []
    for sim_idx in interleaved_indices:
        for tid in sorted_tids:
            t_sims = sims_by_task[tid]
            if sim_idx < len(t_sims):
                clean_s = strip_oracle_fields(t_sims[sim_idx])
                assert "reward_info" not in clean_s and "evaluation_criteria" not in clean_s
                stream.append(clean_s)
                if len(stream) >= limit:
                    return stream
    return stream


def _compile_condition_memory(
    condition: str,
    annotated_stream: Sequence[Tuple[Dict[str, Any], str, str, str]],
    esm_events: Sequence[Tuple[int, str, bool]],
    observed_n: int,
    max_tokens: int = 400,
) -> MarkdownMemory:
    """Build causal `MEMORY.md` (`<= max_tokens`) strictly from the first `observed_n` training chats."""
    mem = MarkdownMemory(max_tokens=max_tokens)
    if condition == "no_memory" or observed_n <= 0:
        return mem

    if condition == "emotion_shift":
        for emit_idx, esm_rule, verified in esm_events:
            if emit_idx < observed_n and esm_rule and verified:
                mem.add_rule(esm_rule)
        return mem

    for _, q_shift, q_vader, note in annotated_stream[:observed_n]:
        if not note:
            continue
        if condition == "routine_chats" and q_shift == "ROUTINE":
            mem.add_rule(note)
        elif condition == "negative_only" and q_vader in ("UNFIXED", "BROKEN"):
            mem.add_rule(note)
        elif condition == "all_chats":
            mem.add_rule(note)
    return mem


def run_stage2_mistake_learning(
    llm: LLMClient,
    shift_det: Optional[FlashLiteShiftDetector] = None,
    max_workers: int = 16,
    memory_budget_tokens: int = 400,
    pilot_only: bool = False,
) -> Dict[str, Any]:
    """Run controlled 5-condition MEMORY.md ablation + N={0,25,50,100} learning curve on tau2-bench."""
    t0 = time.time()
    if shift_det is None:
        shift_det = FlashLiteShiftDetector(llm=llm)
    vader_det = VaderShiftDetector()

    domains = ["airline", "retail"]
    bundles = [_load_domain_bundle(d, STAGE2_TAU2_FILES[d]) for d in domains]

    conditions_100 = ["no_memory", "routine_chats", "negative_only", "all_chats", "emotion_shift"]
    curve_conditions = ["negative_only", "all_chats", "emotion_shift"]
    curve_steps = [0, 25, 50, 100]

    eval_records_100: List[Dict[str, Any]] = []
    control_records_100: List[Dict[str, Any]] = []
    curve_records: Dict[Tuple[str, int], List[bool]] = collections.defaultdict(list)
    sample_memories: Dict[str, Dict[str, str]] = {}
    fold_gate_counts: Dict[str, Dict[str, int]] = {}

    for bundle in bundles:
        dom = bundle["domain"]
        domain_policy = bundle["policy"]
        sorted_tids = sorted(
            bundle["tasks_by_id"].keys(), key=lambda x: int(x) if x.isdigit() else x
        )
        folds = [("fold_A", sorted_tids[0::2], sorted_tids[1::2])]
        if not pilot_only:
            folds.append(("fold_B", sorted_tids[1::2], sorted_tids[0::2]))

        for fold_name, tr_ids, te_ids in folds:
            assert set(tr_ids).isdisjoint(set(te_ids)), "Disjoint task_id invariant violated!"
            fail_cases, pass_controls = _extract_decision_cases(bundle, te_ids)
            if pilot_only:
                fail_cases = fail_cases[:10]
                pass_controls = pass_controls[:4]
            else:
                pass_controls = pass_controls[:8]

            stream_len = 50 if pilot_only else 100
            train_stream = _build_deterministic_train_stream(bundle, tr_ids, limit=stream_len)
            print(
                f"[Stage 2] {dom} {fold_name}: gating {len(train_stream)} oracle-stripped train chats "
                f"-> evaluating {len(fail_cases)} held-out failure points + {len(pass_controls)} controls..."
            )

            vader_quads = [vader_det.classify(cs.get("messages") or [])[0] for cs in train_stream]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                quads = list(
                    pool.map(
                        lambda cs: shift_det.classify(
                            cs.get("messages") or [], domain_policy=domain_policy
                        )[0],
                        train_stream,
                    )
                )
            fold_gate_counts[f"{dom}_{fold_name}"] = dict(collections.Counter(quads))

            # Transcript-only 1-line session notes for baseline conditions (Routine, VADER Negative, All Chats)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                base_notes = list(
                    pool.map(
                        lambda cs: extract_rule_from_chat(
                            llm, cs, dom, model=llm.fast_model, domain_policy=""
                        ),
                        train_stream,
                    )
                )

            # Build strictly causal Emotion-Shift recovery candidates (in-session FIXED + recurring scenario shifts)
            esm_candidates: List[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]] = []
            first_unfixed_by_tid: Dict[str, Dict[str, Any]] = {}
            first_resolved_by_tid: Dict[str, Dict[str, Any]] = {}
            paired_tids: set[str] = set()

            for idx, (cs, q_s) in enumerate(zip(train_stream, quads)):
                if q_s == "FIXED":
                    esm_candidates.append((idx, cs, None))
                tid = str(cs.get("task_id", ""))
                if not tid:
                    continue
                if q_s in ("UNFIXED", "BROKEN") and tid not in first_unfixed_by_tid:
                    first_unfixed_by_tid[tid] = cs
                elif q_s in ("FIXED", "ROUTINE") and tid not in first_resolved_by_tid:
                    first_resolved_by_tid[tid] = cs
                if (
                    tid not in paired_tids
                    and tid in first_unfixed_by_tid
                    and tid in first_resolved_by_tid
                ):
                    paired_tids.add(tid)
                    esm_candidates.append(
                        (idx, first_unfixed_by_tid[tid], first_resolved_by_tid[tid])
                    )

            def _extract_and_verify_esm(
                cand: Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]],
            ) -> Tuple[int, str, bool]:
                emit_idx, f_sess, r_sess = cand
                rule = extract_contrastive_fix_rule(
                    llm=llm,
                    session=f_sess,
                    domain=dom,
                    domain_policy=domain_policy,
                    resolved_session=r_sess,
                    model=llm.default_model,
                )
                if not rule:
                    return (emit_idx, "", False)
                ok = verify_rule_on_mistake_turn(
                    llm=llm,
                    session=f_sess,
                    rule_bullet=rule,
                    domain_policy=domain_policy,
                    tools=bundle["tools"],
                )
                return (emit_idx, rule, ok)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                esm_events = list(pool.map(_extract_and_verify_esm, esm_candidates))

            annotated = list(zip(train_stream, quads, vader_quads, base_notes))

            # Compile MEMORY.md for all (condition, step) pairs
            mem_map_100: Dict[str, str] = {}
            mem_capped_100: Dict[str, str] = {}
            for cond in conditions_100:
                mem_obj = _compile_condition_memory(
                    cond,
                    annotated,
                    esm_events,
                    observed_n=stream_len,
                    max_tokens=memory_budget_tokens,
                )
                mem_capped_100[cond] = mem_obj.render()
                assert count_tokens(mem_capped_100[cond]) <= memory_budget_tokens
                mem_map_100[cond] = mem_obj.render_all()

            if fold_name == "fold_A":
                sample_memories[dom] = {
                    c: mem_capped_100[c] for c in ("emotion_shift", "negative_only", "all_chats")
                }

            mem_map_curve: Dict[Tuple[str, int], str] = {}
            for cond in curve_conditions:
                mem_map_curve[(cond, 0)] = ""
                for n_obs in (25, 50):
                    if n_obs <= stream_len:
                        m_obj = _compile_condition_memory(
                            cond,
                            annotated,
                            esm_events,
                            observed_n=n_obs,
                            max_tokens=memory_budget_tokens,
                        )
                        mem_map_curve[(cond, n_obs)] = m_obj.render_all()
                mem_map_curve[(cond, 100)] = mem_map_100[cond]

            def _eval_with_mem_text(case: Dict[str, Any], mem_txt: str, is_ctrl: bool) -> Dict[str, Any]:
                active_mem = select_relevant_rules(
                    mem_txt,
                    case["prefix_messages"],
                    llm=llm,
                    max_rules=6,
                    min_sim=0.65,
                    max_tokens=memory_budget_tokens,
                )
                assert count_tokens(active_mem) <= memory_budget_tokens
                sys_inst = (
                    "You are a customer service agent strictly following the Domain Policy below.\n\n"
                    f"=== DOMAIN POLICY ===\n{domain_policy}"
                )
                if active_mem:
                    sys_inst += (
                        f"\n\n=== RETRIEVED MEMORY (<= {memory_budget_tokens} tokens) ===\n"
                        f"{active_mem}"
                    )
                step_out = llm.generate_agent_step(
                    messages=case["prefix_messages"],
                    system_instruction=sys_inst,
                    tools=bundle["tools"],
                    model=llm.default_model,
                )
                passed = _esm_grade_decision_step(
                    step_out=step_out,
                    expected_mut_actions=case["expected_mut"],
                    kind=case["kind"],
                    is_control=is_ctrl,
                )
                return {
                    "passed": passed,
                    "mistake_repeated": not passed,
                    "used_memory_tokens": min(memory_budget_tokens, count_tokens(mem_txt)),
                    "injected_memory_tokens": count_tokens(active_mem),
                    "pred_tool_calls": step_out.get("tool_calls") or [],
                    "pred_text_snippet": (step_out.get("content") or "")[:180],
                }

            # Evaluate all 5 conditions at N=100 + non-regression controls
            jobs_100: List[Tuple[str, Dict[str, Any], bool]] = []
            for fc in fail_cases:
                for cond in conditions_100:
                    jobs_100.append((cond, fc, False))
            for pc in pass_controls:
                for cond in ("no_memory", "emotion_shift", "negative_only"):
                    jobs_100.append((cond, pc, True))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                fut_map = {
                    pool.submit(_eval_with_mem_text, case, mem_map_100[cond], is_ctrl): (
                        cond,
                        case,
                        is_ctrl,
                    )
                    for cond, case, is_ctrl in jobs_100
                }
                for fut in concurrent.futures.as_completed(fut_map):
                    cond, case, is_ctrl = fut_map[fut]
                    res = fut.result()
                    row = {
                        "domain": dom,
                        "fold": fold_name,
                        "task_id": case["task_id"],
                        "sim_id": case["sim_id"],
                        "kind": case["kind"],
                        "condition": cond,
                        "is_control": is_ctrl,
                        **res,
                    }
                    if is_ctrl:
                        control_records_100.append(row)
                    else:
                        eval_records_100.append(row)

            # Record N=0 and N=100 into learning curve directly from eval_records_100 (zero duplicate calls!)
            fold_zs_errs = [
                r["mistake_repeated"]
                for r in eval_records_100
                if r["domain"] == dom and r["fold"] == fold_name and r["condition"] == "no_memory"
            ]
            for cond in curve_conditions:
                curve_records[(cond, 0)].extend(fold_zs_errs)
                fold_100_errs = [
                    r["mistake_repeated"]
                    for r in eval_records_100
                    if r["domain"] == dom and r["fold"] == fold_name and r["condition"] == cond
                ]
                curve_records[(cond, 100)].extend(fold_100_errs)

            # Evaluate N=25 and N=50 checkpoints for the learning curve
            if not pilot_only:
                jobs_mid: List[Tuple[str, int, Dict[str, Any]]] = []
                for n_obs in (25, 50):
                    for cond in curve_conditions:
                        for fc in fail_cases:
                            jobs_mid.append((cond, n_obs, fc))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    fut_mid = {
                        pool.submit(
                            _eval_with_mem_text, fc, mem_map_curve[(cond, n_obs)], False
                        ): (cond, n_obs)
                        for cond, n_obs, fc in jobs_mid
                    }
                    for fut in concurrent.futures.as_completed(fut_mid):
                        cond, n_obs = fut_mid[fut]
                        curve_records[(cond, n_obs)].append(fut.result()["mistake_repeated"])

    by_cond: Dict[str, Any] = {}
    for cond in conditions_100:
        rows = [r for r in eval_records_100 if r["condition"] == cond]
        tot = len(rows)
        rep = sum(1 for r in rows if r["mistake_repeated"])
        air = [r for r in rows if r["domain"] == "airline"]
        ret = [r for r in rows if r["domain"] == "retail"]
        by_cond[cond] = {
            "total_heldout_trials": tot,
            "mistakes_repeated": rep,
            "mistakes_avoided": tot - rep,
            "repeat_error_rate": round(rep / tot, 4) if tot else 0.0,
            "airline_repeat_rate": (
                round(sum(1 for r in air if r["mistake_repeated"]) / len(air), 4) if air else 0.0
            ),
            "retail_repeat_rate": (
                round(sum(1 for r in ret if r["mistake_repeated"]) / len(ret), 4) if ret else 0.0
            ),
            "avg_memory_tokens": round(sum(r["used_memory_tokens"] for r in rows) / tot, 1)
            if tot
            else 0.0,
            "avg_injected_tokens": round(sum(r["injected_memory_tokens"] for r in rows) / tot, 1)
            if tot
            else 0.0,
        }

    ctrl_summary: Dict[str, Any] = {}
    for cond in ("no_memory", "emotion_shift", "negative_only"):
        c_rows = [r for r in control_records_100 if r["condition"] == cond]
        c_pass = sum(1 for r in c_rows if r["passed"])
        ctrl_summary[cond] = {
            "total_controls": len(c_rows),
            "passed": c_pass,
            "pass_rate": round(c_pass / len(c_rows), 4) if c_rows else 0.0,
        }

    lc_steps = [0, 100] if pilot_only else curve_steps
    lc_series: Dict[str, List[float]] = {}
    for cond in curve_conditions:
        lc_series[cond] = [
            round(sum(curve_records[(cond, s)]) / len(curve_records[(cond, s)]), 4)
            if curve_records[(cond, s)]
            else 0.0
            for s in lc_steps
        ]

    zs_r = by_cond["no_memory"]["repeat_error_rate"]
    neg_r = by_cond["negative_only"]["repeat_error_rate"]
    esm_r = by_cond["emotion_shift"]["repeat_error_rate"]

    return {
        "elapsed_seconds": round(time.time() - t0, 2),
        "memory_budget_tokens": memory_budget_tokens,
        "fold_quadrant_counts": fold_gate_counts,
        "conditions": by_cond,
        "non_regression_controls": ctrl_summary,
        "learning_curve": {
            "chat_counts": lc_steps,
            "series": lc_series,
        },
        "relative_reduction_vs_no_memory": round((zs_r - esm_r) / zs_r, 4) if zs_r > 0 else 0.0,
        "relative_reduction_vs_negative_only": round((neg_r - esm_r) / neg_r, 4)
        if neg_r > 0
        else 0.0,
        "sample_memory_md": sample_memories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Emotion-Shift Memory experiments.")
    parser.add_argument(
        "--stage",
        choices=("all", "pilot", "stage1", "stage2"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    llm = LLMClient()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.stage == "pilot":
        pilot_res = run_stage2_mistake_learning(llm=llm, max_workers=args.workers, pilot_only=True)
        print("Fold quadrant counts:", json.dumps(pilot_res["fold_quadrant_counts"], indent=2))
        print("Conditions:", json.dumps(pilot_res["conditions"], indent=2))
        print("Controls:", json.dumps(pilot_res["non_regression_controls"], indent=2))
        print("\nSample Airline Emotion-Shift MEMORY.md:\n", pilot_res["sample_memory_md"].get("airline", {}).get("emotion_shift", ""))
        print("\nSample Retail Emotion-Shift MEMORY.md:\n", pilot_res["sample_memory_md"].get("retail", {}).get("emotion_shift", ""))
        return

    chart1 = compute_chart1_silent_failures()
    results_path = os.path.join(RESULTS_DIR, "esm_results.json")
    existing: Dict[str, Any] = {}
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    payload: Dict[str, Any] = {
        "chart1_silent_failures": chart1,
        "stage1_shift_detection": existing.get("stage1_shift_detection"),
        "stage2_mistake_learning": existing.get("stage2_mistake_learning"),
    }

    if args.stage in ("all", "stage1"):
        payload["stage1_shift_detection"] = run_stage1_shift_detection(
            llm=llm, per_quadrant=150, max_workers=args.workers
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if args.stage in ("all", "stage2"):
        payload["stage2_mistake_learning"] = run_stage2_mistake_learning(
            llm=llm, max_workers=args.workers, memory_budget_tokens=400, pilot_only=False
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"\nSaved complete ESM benchmark results to {results_path}")


if __name__ == "__main__":
    main()

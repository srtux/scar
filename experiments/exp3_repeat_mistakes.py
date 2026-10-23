"""Experiment 3: Cross-Task Mistake Reduction on Held-Out tau2-bench Tasks (2-Fold Disjoint Split).

Strict anti-leakage & cross-task protocol:
1. Partition `task_id`s into two disjoint folds (`Fold A` and `Fold B`, `TRAIN_TASKS ∩ TEST_TASKS = ∅`).
2. Strip ALL `reward_info` / `evaluation_criteria` oracle fields from `TRAIN_TASKS` via `strip_oracle_fields`.
3. Consolidate memory on `TRAIN_TASKS` across 6 architectures:
   - Zero-Shot (Policy + Tools only)
   - Raw Episode RAG (unquarantined trajectory retrieval)
   - Atomic Facts (Mem0-style)
   - Tool-Outcome Only
   - SCAR (No Quarantine / No Replay Ablation)
   - SCAR (Full: Struggle-to-Resolution + Quarantine + Counterfactual Replay Verification)
4. Replay `gemini-3.8-flash` with native tool declarations (`airline_tools.json`, `retail_tools.json`)
   at the first state-mutating decision point `t*` on held-out `TEST_TASKS` (both Failure-Prone
   First-Mutation Decision Points and Clean Passing Control Decision Points) and grade symmetrically
   against `evaluation_criteria.actions`.
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from scar.baselines import AtomicFactsMemory, RawTrajectoryRAGMemory, ToolOutcomeOnlyMemory
from scar.consolidator import SCARConsolidator, strip_oracle_fields
from scar.detector import SCARTransitionDetector
from scar.llm import LLMClient
from scar.tokens import count_tokens

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAU2_DIR = os.path.join(REPO_ROOT, "data", "tau2")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

READ_TOOLS = frozenset(
    {
        "get_reservation_details",
        "get_user_details",
        "list_all_airports",
        "search_direct_flight",
        "search_onestop_flight",
        "get_flight_status",
        "calculate",
        "find_user_id_by_name_zip",
        "find_user_id_by_email",
        "get_order_details",
        "get_product_details",
        "get_item_details",
        "list_all_product_types",
    }
)

_HALLUCINATED_MUTATION_RE = re.compile(
    r"\b(has been|have been|successfully|already) "
    r"(cancelled|canceled|returned|exchanged|submitted|modified|updated|issued|processed)\b"
    r"|you are being transferred to a human agent",
    re.IGNORECASE,
)


def _claims_completed_mutation(text: str) -> bool:
    """Return True if a text-only reply falsely claims a write action was already executed."""
    return bool(_HALLUCINATED_MUTATION_RE.search(text or ""))


def _canonicalize_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool arguments for deterministic comparison (paired item lists, numeric floats)."""
    norm: Dict[str, Any] = {}
    if (
        tool_name in ("exchange_delivered_order_items", "modify_pending_order_items")
        and isinstance(args.get("item_ids"), list)
        and isinstance(args.get("new_item_ids"), list)
        and len(args["item_ids"]) == len(args["new_item_ids"])
    ):
        pairs = sorted(zip([str(x) for x in args["item_ids"]], [str(y) for y in args["new_item_ids"]]))
        norm["item_ids"] = [p[0] for p in pairs]
        norm["new_item_ids"] = [p[1] for p in pairs]
        for k, v in args.items():
            if k not in ("item_ids", "new_item_ids"):
                norm[k] = str(v).strip() if isinstance(v, str) else v
        return norm

    for k, v in (args or {}).items():
        if k == "item_ids" and isinstance(v, list):
            norm[k] = sorted(str(x) for x in v)
        elif isinstance(v, float) and v.is_integer():
            norm[k] = int(v)
        elif isinstance(v, str):
            norm[k] = v.strip()
        elif isinstance(v, list):
            norm[k] = [
                _canonicalize_args("", x) if isinstance(x, dict) else x for x in v
            ]
        else:
            norm[k] = v
    return norm


def _actions_match(pred_call: Dict[str, Any], target_action: Dict[str, Any]) -> bool:
    """Deterministically verify whether `pred_call` matches `tau2-bench` `target_action`."""
    t_name = target_action.get("name", "")
    if pred_call.get("name") != t_name:
        return False
    if t_name == "transfer_to_human_agents":
        return True
    p_args = _canonicalize_args(t_name, pred_call.get("arguments") or {})
    t_args = _canonicalize_args(t_name, target_action.get("arguments") or {})
    for k, tv in t_args.items():
        if k == "reason":
            if k not in p_args:
                return False
            continue
        if p_args.get(k) != tv:
            return False
    return True


def _grade_decision_step(
    step_out: Dict[str, Any],
    expected_mut_actions: Sequence[Dict[str, Any]],
    kind: str,
    is_control: bool = False,
) -> bool:
    """Return True iff the agent avoided the mistake and took a policy-compliant action at t*.

    Applied identically across Zero-Shot, baselines, SCAR ablations, and SCAR (Full):
    - `is_control`: user already confirmed a valid mutation; agent must emit the matching write call.
    - `silent_forbid`: policy forbids all state mutations / transfers (`expected_mut_actions == []`);
      agent must NOT emit any mutating call and must NOT hallucinate completing one in text.
    - `mut_tool_error`: the historical trajectory's first mutating call at `t*` immediately triggered
      a tool/policy error (e.g., refunding to a non-original payment method, modifying non-pending
      orders, or applying invalid certificates/flights).
        * If the agent emits a mutating call, it passes iff that call matches a ground-truth action
          in `expected_mut_actions` (repeating the bad call, using invalid args, or making an
          unauthorized `transfer_to_human_agents` fails).
        * If the agentwithholds the broken mutation (`len(mut_calls) == 0`), it passes iff it either
          calls a read/verification tool first (`len(read_calls) > 0`) OR provides a non-empty
          explanation / confirmation prompt for the valid policy option without hallucinating completion
          (both Airline and Retail policies forbid executing modified parameters without explicit user confirmation).
    """
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
        return not _claims_completed_mutation(content)

    # kind == "mut_tool_error" with >= 1 ground-truth action
    if len(mut_calls) > 0:
        return any(_actions_match(mut_calls[0], t) for t in expected_mut_actions)

    if len(read_calls) > 0:
        return True
    return bool(content) and not _claims_completed_mutation(content)


def _load_domain_bundle(domain: str, filenames: Sequence[str]) -> Dict[str, Any]:
    with open(os.path.join(TAU2_DIR, f"{domain}_tools.json"), "r") as f:
        tools = json.load(f)
    tasks_by_id: Dict[str, Dict[str, Any]] = {}
    sims_by_task: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    policy = ""
    for fname in filenames:
        with open(os.path.join(TAU2_DIR, fname), "r") as f:
            d = json.load(f)
        if not policy:
            policy = ((d.get("info") or {}).get("environment_info") or {}).get("policy") or ""
        for t in d["tasks"]:
            tasks_by_id[str(t["id"])] = t
        for s in d["simulations"]:
            sims_by_task[str(s["task_id"])].append(s)
    return {
        "domain": domain,
        "policy": policy,
        "tools": tools,
        "tasks_by_id": tasks_by_id,
        "sims_by_task": sims_by_task,
    }


def _extract_decision_cases(
    bundle: Dict[str, Any],
    test_task_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract clean first-mutation failure decision points and passing control decision points.

    Strictly inspects ONLY the first state-mutating assistant turn in each trajectory (`break`
    after the first mutating turn), guaranteeing zero prior DB mutations in `prefix_messages`.
    """
    tasks_by_id = bundle["tasks_by_id"]
    sims_by_task = bundle["sims_by_task"]
    fail_cases: List[Dict[str, Any]] = []
    pass_controls: List[Dict[str, Any]] = []

    for tid in sorted(test_task_ids, key=lambda x: int(x) if x.isdigit() else x):
        task = tasks_by_id[tid]
        ec = task.get("evaluation_criteria") or {}
        exp_mut = [
            a
            for a in (ec.get("actions") or [])
            if a.get("requestor") != "user" and a.get("name") not in READ_TOOLS
        ]

        seen_sigs = set()
        for s in sims_by_task[tid]:
            rew = (s.get("reward_info") or {}).get("reward")
            msgs = s.get("messages") or []
            for idx, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                mut_calls = [
                    tc for tc in (m.get("tool_calls") or []) if tc.get("name") not in READ_TOOLS
                ]
                if not mut_calls:
                    continue

                # ONLY inspect the very first mutating assistant turn in the simulation!
                first_mc = mut_calls[0]
                err_txt = ""
                j = idx + 1
                while j < len(msgs) and msgs[j].get("role") == "tool":
                    if msgs[j].get("error"):
                        err_txt = msgs[j].get("content") or "tool_error"
                        break
                    j += 1

                if err_txt and idx >= 2:
                    sig = ("mut_tool_error", first_mc.get("name"), err_txt[:45])
                    if sig not in seen_sigs and len(seen_sigs) < 2:
                        seen_sigs.add(sig)
                        fail_cases.append(
                            {
                                "domain": bundle["domain"],
                                "task_id": tid,
                                "sim_id": s.get("id", ""),
                                "kind": "mut_tool_error",
                                "cut_idx": idx,
                                "prefix_messages": msgs[:idx],
                                "expected_mut": exp_mut,
                                "historical_bad_call": first_mc,
                                "historical_error": err_txt[:160],
                            }
                        )
                elif rew != 1.0 and len(exp_mut) == 0 and idx >= 2:
                    sig = ("silent_forbid", first_mc.get("name"))
                    if sig not in seen_sigs and len(seen_sigs) < 2:
                        seen_sigs.add(sig)
                        fail_cases.append(
                            {
                                "domain": bundle["domain"],
                                "task_id": tid,
                                "sim_id": s.get("id", ""),
                                "kind": "silent_forbid",
                                "cut_idx": idx,
                                "prefix_messages": msgs[:idx],
                                "expected_mut": exp_mut,
                                "historical_bad_call": first_mc,
                                "historical_error": "silent_policy_violation",
                            }
                        )
                break
            if len(seen_sigs) >= 2:
                break

        # Clean passing control case (where agent executed valid single mutating call on first mutation)
        if len(exp_mut) == 1 and len(pass_controls) < 16:
            for s in sims_by_task[tid]:
                if (s.get("reward_info") or {}).get("reward") != 1.0:
                    continue
                msgs = s.get("messages") or []
                for idx, m in enumerate(msgs):
                    if m.get("role") != "assistant":
                        continue
                    mut_calls = [
                        tc for tc in (m.get("tool_calls") or []) if tc.get("name") not in READ_TOOLS
                    ]
                    if not mut_calls:
                        continue
                    if _actions_match(mut_calls[0], exp_mut[0]) and idx >= 2:
                        pass_controls.append(
                            {
                                "domain": bundle["domain"],
                                "task_id": tid,
                                "sim_id": s.get("id", ""),
                                "kind": "pass_control",
                                "cut_idx": idx,
                                "prefix_messages": msgs[:idx],
                                "expected_mut": exp_mut,
                            }
                        )
                    break
                if pass_controls and pass_controls[-1]["task_id"] == tid:
                    break

    return fail_cases, pass_controls


def _build_fold_memories(
    llm: LLMClient,
    bundle: Dict[str, Any],
    train_task_ids: Sequence[str],
    max_workers: int = 16,
) -> Dict[str, Any]:
    """Consolidate all memory systems strictly on oracle-stripped `train_task_ids`."""
    domain = bundle["domain"]
    policy = bundle["policy"]
    tools = bundle["tools"]
    sims_by_task = bundle["sims_by_task"]

    train_sims_by_tid: Dict[str, List[Dict[str, Any]]] = {}
    flat_train_sims: List[Dict[str, Any]] = []
    for tid in sorted(train_task_ids, key=lambda x: int(x) if x.isdigit() else x):
        raw_list = sims_by_task[tid][:4]
        clean_list = [strip_oracle_fields(s) for s in raw_list]
        for cs in clean_list:
            assert "reward_info" not in cs and "evaluation_criteria" not in cs
        train_sims_by_tid[tid] = clean_list
        flat_train_sims.extend(clean_list[:2])

    # 1. Raw Episode RAG across all training tasks
    raw_rag = RawTrajectoryRAGMemory()
    for cs in flat_train_sims:
        raw_rag.ingest_session(cs, domain=domain)
    raw_rag._embed_pending(llm)

    # 2. Atomic Facts (Mem0-style) & Tool-Outcome Only across all training tasks
    mem0 = AtomicFactsMemory(llm)
    tool_only = ToolOutcomeOnlyMemory(llm)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(mem0.ingest_session, cs, domain) for cs in flat_train_sims]
        futs += [pool.submit(tool_only.ingest_session, cs, domain) for cs in flat_train_sims]
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
    mem0._embed_pending(llm)
    tool_only._embed_pending(llm)

    # 3. SCAR Transition Classification on oracle-stripped TRAIN_TASKS
    det = SCARTransitionDetector(llm=llm, model=llm.default_model)
    all_train_for_scar = [cs for tid_list in train_sims_by_tid.values() for cs in tid_list]
    det_results: Dict[str, Dict[str, Any]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {
            pool.submit(det.analyze, cs.get("messages") or [], policy): cs["id"]
            for cs in all_train_for_scar
        }
        for fut in concurrent.futures.as_completed(fut_map):
            det_results[fut_map[fut]] = fut.result()

    scar_full = SCARConsolidator(llm=llm, enable_quarantine=True, enable_replay_verification=True)
    scar_nogate = SCARConsolidator(llm=llm, enable_quarantine=False, enable_replay_verification=False)

    pair_jobs: List[Tuple[str, Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]] = []
    standalone_resolved: List[Dict[str, Any]] = []
    unquarantined_fails: List[Dict[str, Any]] = []

    for tid, cs_list in train_sims_by_tid.items():
        u_fails = [cs for cs in cs_list if det_results[cs["id"]]["session_state"] == "UNRESOLVED_FAILURE"]
        successes = [cs for cs in cs_list if det_results[cs["id"]]["session_state"] in ("CLEAN_SUCCESS", "RESOLVED_STRUGGLE")]
        resolved = [cs for cs in cs_list if det_results[cs["id"]]["session_state"] == "RESOLVED_STRUGGLE"]
        if u_fails and successes:
            pair_jobs.append((f"{domain}_t{tid}", u_fails[0]["messages"], successes[0]["messages"]))
        elif resolved:
            standalone_resolved.append(resolved[0])
        if u_fails:
            unquarantined_fails.append(u_fails[0])

    for pid, f_msgs, s_msgs in pair_jobs:
        scar_full.synthesize_paired_struggle_card(
            failed_messages=f_msgs,
            resolved_messages=s_msgs,
            domain=domain,
            domain_policy=policy,
            tools=tools,
            pair_id=pid,
        )
        scar_nogate.synthesize_paired_struggle_card(
            failed_messages=f_msgs,
            resolved_messages=s_msgs,
            domain=domain,
            domain_policy=policy,
            tools=tools,
            pair_id=pid,
        )

    for rs in standalone_resolved[:20]:
        scar_full.ingest_session(rs, domain=domain, domain_policy=policy, tools=tools, extract_directives=False)
        scar_nogate.ingest_session(rs, domain=domain, domain_policy=policy, tools=tools, extract_directives=False)

    # SCAR No-Gate also ingests unquarantined failures without verification
    for uf in unquarantined_fails[:20]:
        scar_nogate.ingest_session(uf, domain=domain, domain_policy=policy, tools=tools, extract_directives=False)

    scar_full.store.ensure_embeddings(llm)
    scar_nogate.store.ensure_embeddings(llm)

    return {
        "zero_shot": None,
        "raw_rag": raw_rag,
        "atomic_facts_mem0": mem0,
        "tool_outcome_only": tool_only,
        "scar_no_gate": scar_nogate.store,
        "scar_full": scar_full.store,
        "_meta": {
            "scar_full_cards": len(scar_full.store.playbooks),
            "scar_replay_attempts": scar_full.replay_attempts,
            "scar_replay_passed": scar_full.replay_passed,
            "scar_nogate_cards": len(scar_nogate.store.playbooks),
        },
    }


def _query_text_from_prefix(prefix_msgs: Sequence[Dict[str, Any]]) -> str:
    """Build a symmetric retrieval query combining user goal, recent context, and inspected state."""
    user_parts = [
        (m.get("content") or "").strip()
        for m in prefix_msgs
        if m.get("role") == "user" and (m.get("content") or "").strip()
    ]
    called_tools: List[str] = []
    state_cues: List[str] = []
    for m in prefix_msgs:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tname = tc.get("name", "")
                if tname and tname not in called_tools:
                    called_tools.append(tname)
        elif m.get("role") == "tool":
            txt = (m.get("content") or "").strip()
            if txt.startswith("{") and len(state_cues) < 3:
                state_cues.append(txt[:220])

    segments: List[str] = []
    if user_parts:
        segments.append("User goal: " + user_parts[0][:380])
    if len(user_parts) > 1:
        segments.append("Latest user turns: " + " | ".join(user_parts[-2:])[:380])
    if called_tools:
        segments.append("Inspected tools: " + ", ".join(called_tools))
    if state_cues:
        segments.append("State snapshot: " + " ; ".join(state_cues)[:520])
    return "\n".join(segments)[:1400]


def run_exp3(llm: LLMClient | None = None, max_workers: int = 16) -> Dict[str, Any]:
    t0 = time.time()
    if llm is None:
        llm = LLMClient()

    bundles = [
        _load_domain_bundle(
            "airline",
            [
                "claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json",
                "gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json",
            ],
        ),
        _load_domain_bundle(
            "retail",
            [
                "claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json",
                "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json",
            ],
        ),
    ]

    systems = [
        "zero_shot",
        "raw_rag",
        "atomic_facts_mem0",
        "tool_outcome_only",
        "scar_no_gate",
        "scar_full",
    ]
    memory_budget_tokens = 450

    eval_records: List[Dict[str, Any]] = []
    control_records: List[Dict[str, Any]] = []
    consolidation_meta: Dict[str, Any] = {}

    for bundle in bundles:
        dom = bundle["domain"]
        sorted_tids = sorted(
            bundle["tasks_by_id"].keys(), key=lambda x: int(x) if x.isdigit() else x
        )
        fold_a_train = sorted_tids[0::2]
        fold_a_test = sorted_tids[1::2]
        fold_b_train = sorted_tids[1::2]
        fold_b_test = sorted_tids[0::2]

        for fold_name, tr_ids, te_ids in [
            ("fold_A", fold_a_train, fold_a_test),
            ("fold_B", fold_b_train, fold_b_test),
        ]:
            assert set(tr_ids).isdisjoint(set(te_ids)), "Train/Test task_id overlap detected!"
            fail_cases, pass_controls = _extract_decision_cases(bundle, te_ids)
            print(
                f"[Exp 3] {dom} {fold_name}: consolidating on {len(tr_ids)} train tasks -> "
                f"evaluating on {len(fail_cases)} held-out failure points + {len(pass_controls[:8])} controls..."
            )
            mems = _build_fold_memories(llm, bundle, tr_ids, max_workers=max_workers)
            consolidation_meta[f"{dom}_{fold_name}"] = mems["_meta"]

            jobs: List[Tuple[str, Dict[str, Any], bool]] = []
            for fc in fail_cases:
                for sname in systems:
                    jobs.append((sname, fc, False))
            for pc in pass_controls[:8]:
                for sname in ("zero_shot", "scar_full"):
                    jobs.append((sname, pc, True))

            def _run_case(job: Tuple[str, Dict[str, Any], bool]) -> Dict[str, Any]:
                sname, case, is_control = job
                q_txt = _query_text_from_prefix(case["prefix_messages"])
                mem_obj = mems[sname]
                ctx = ""
                if mem_obj is not None:
                    ctx = mem_obj.retrieve_context(
                        query=q_txt,
                        max_tokens=memory_budget_tokens,
                        llm=llm,
                        domain_filter=dom,
                    )
                    assert count_tokens(ctx) <= memory_budget_tokens

                sys_inst = (
                    "You are a customer service agent strictly following the Domain Policy below.\n\n"
                    f"=== DOMAIN POLICY ===\n{bundle['policy']}"
                )
                if ctx:
                    sys_inst += f"\n\n=== RETRIEVED MEMORY (<= {memory_budget_tokens} tokens) ===\n{ctx}"

                step_out = llm.generate_agent_step(
                    messages=case["prefix_messages"],
                    system_instruction=sys_inst,
                    tools=bundle["tools"],
                    model=llm.default_model,
                )
                passed = _grade_decision_step(
                    step_out=step_out,
                    expected_mut_actions=case["expected_mut"],
                    kind=case["kind"],
                    is_control=is_control,
                )
                return {
                    "domain": dom,
                    "fold": fold_name,
                    "task_id": case["task_id"],
                    "sim_id": case["sim_id"],
                    "kind": case["kind"],
                    "system": sname,
                    "is_control": is_control,
                    "passed": passed,
                    "mistake_repeated": not passed,
                    "used_memory_tokens": count_tokens(ctx),
                    "pred_tool_calls": step_out.get("tool_calls") or [],
                    "pred_text_snippet": (step_out.get("content") or "")[:200],
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = [pool.submit(_run_case, j) for j in jobs]
                for fut in concurrent.futures.as_completed(futs):
                    rec = fut.result()
                    if rec["is_control"]:
                        control_records.append(rec)
                    else:
                        eval_records.append(rec)

    by_system: Dict[str, Any] = {}
    for sname in systems:
        s_rows = [r for r in eval_records if r["system"] == sname]
        rep = sum(1 for r in s_rows if r["mistake_repeated"])
        tot = len(s_rows)
        air_rows = [r for r in s_rows if r["domain"] == "airline"]
        ret_rows = [r for r in s_rows if r["domain"] == "retail"]
        mte_rows = [r for r in s_rows if r["kind"] == "mut_tool_error"]
        sf_rows = [r for r in s_rows if r["kind"] == "silent_forbid"]
        by_system[sname] = {
            "total_heldout_trials": tot,
            "mistakes_repeated": rep,
            "mistakes_avoided": tot - rep,
            "repeat_error_rate": round(rep / tot, 4) if tot else 0.0,
            "airline_repeat_rate": (
                round(sum(1 for r in air_rows if r["mistake_repeated"]) / len(air_rows), 4)
                if air_rows
                else 0.0
            ),
            "retail_repeat_rate": (
                round(sum(1 for r in ret_rows if r["mistake_repeated"]) / len(ret_rows), 4)
                if ret_rows
                else 0.0
            ),
            "mut_tool_error_repeat_rate": (
                round(sum(1 for r in mte_rows if r["mistake_repeated"]) / len(mte_rows), 4)
                if mte_rows
                else 0.0
            ),
            "silent_forbid_repeat_rate": (
                round(sum(1 for r in sf_rows if r["mistake_repeated"]) / len(sf_rows), 4)
                if sf_rows
                else 0.0
            ),
            "avg_memory_tokens": round(
                sum(r["used_memory_tokens"] for r in s_rows) / tot, 1
            )
            if tot
            else 0.0,
        }

    control_summary: Dict[str, Any] = {}
    for sname in ("zero_shot", "scar_full"):
        c_rows = [r for r in control_records if r["system"] == sname]
        c_pass = sum(1 for r in c_rows if r["passed"])
        control_summary[sname] = {
            "total_controls": len(c_rows),
            "passed": c_pass,
            "pass_rate": round(c_pass / len(c_rows), 4) if c_rows else 0.0,
        }

    zs_rate = by_system["zero_shot"]["repeat_error_rate"]
    mem0_rate = by_system["atomic_facts_mem0"]["repeat_error_rate"]
    scar_rate = by_system["scar_full"]["repeat_error_rate"]
    rel_reduction_vs_zs = ((zs_rate - scar_rate) / zs_rate) if zs_rate > 0 else 0.0
    rel_reduction_vs_mem0 = ((mem0_rate - scar_rate) / mem0_rate) if mem0_rate > 0 else 0.0

    summary = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "protocol": "2-fold disjoint task_id split across Airline (50 tasks) and Retail (114 tasks), first-mutation decision points, zero reward_info access",
        "memory_budget_tokens": memory_budget_tokens,
        "consolidation_meta": consolidation_meta,
        "systems": by_system,
        "non_regression_controls": control_summary,
        "relative_repeat_error_reduction_vs_zero_shot": round(rel_reduction_vs_zs, 4),
        "relative_repeat_error_reduction_vs_mem0": round(rel_reduction_vs_mem0, 4),
        "absolute_repeat_error_reduction_vs_zero_shot": round(zs_rate - scar_rate, 4),
        "absolute_repeat_error_reduction_vs_mem0": round(mem0_rate - scar_rate, 4),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "exp3_details.json"), "w") as f:
        json.dump(
            {"summary": summary, "eval_records": eval_records, "control_records": control_records},
            f,
            indent=2,
        )

    print("\n=== EXPERIMENT 3 SUMMARY (Cross-Task Held-Out Replay, Disjoint task_ids) ===")
    for sname, st in by_system.items():
        print(
            f"  {sname:<22} | Held-Out Trials={st['total_heldout_trials']:<3} | "
            f"Repeated={st['mistakes_repeated']:<3} | "
            f"Repeat Rate={st['repeat_error_rate']*100:5.1f}% "
            f"(Airline={st['airline_repeat_rate']*100:5.1f}%, Retail={st['retail_repeat_rate']*100:5.1f}%)"
        )
    print("  Non-regression controls:", control_summary)
    print(
        f"  Relative Repeat Mistake Reduction (SCAR vs Zero-Shot): {rel_reduction_vs_zs*100:.1f}% | "
        f"(SCAR vs Mem0): {rel_reduction_vs_mem0*100:.1f}%"
    )
    return summary


if __name__ == "__main__":
    run_exp3()

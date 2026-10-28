"""Unit tests for Emotion-Shift Memory: detectors, MEMORY.md cap, retrieval, grading, and leakage guards."""

from __future__ import annotations

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

from esm.detectors import LayaShiftDetector, VaderShiftDetector, extract_clean_dialogue, quadrant_from_flags  # noqa: E402
from esm.llm import count_tokens, pack_items_within_budget, strip_oracle_fields  # noqa: E402
from esm.memory_md import MarkdownMemory, select_relevant_rules  # noqa: E402
from run_experiment import build_train_stream, grade_step, load_domain, mcnemar_exact_p  # noqa: E402

CALM_RECOVERY = [
    {"role": "user", "content": "Why does my component keep rerendering when I move the box?"},
    {"role": "assistant", "content": "Wrap the component in React.memo."},
    {"role": "user", "content": "it still re renders the animation"},
    {"role": "assistant", "content": "Move the state into a ref or memoize the parent callback."},
    {"role": "user", "content": "thanks, that stopped the extra rerenders"},
]


def test_vader_misses_calm_pushback_but_laya_catches_the_fix() -> None:
    assert VaderShiftDetector().classify(CALM_RECOVERY)[0] == "ROUTINE"
    assert LayaShiftDetector().classify(CALM_RECOVERY)[0] == "FIXED"


def test_quadrant_truth_table() -> None:
    assert [quadrant_from_flags(a, b) for a, b in ((1, 0), (1, 1), (0, 1), (0, 0))] == ["FIXED", "UNFIXED", "BROKEN", "ROUTINE"]


def test_memory_enforces_token_cap_and_skips_duplicates() -> None:
    mem = MarkdownMemory(max_tokens=80)
    assert mem.add_rule("When modifying pending orders, collect all item changes before calling the API.")
    assert not mem.add_rule("When modifying pending orders, collect all item changes before calling the API!")
    assert not mem.add_rule("Collect all item changes for pending orders before calling modify API.")
    for i in range(15):
        mem.add_rule(f"When handling scenario {i} on airline bookings, verify insurance status {i * 7}.")
    assert mem.render().startswith("# MEMORY.md") and count_tokens(mem.render()) <= 80
    assert count_tokens(pack_items_within_budget(["word " * 50] * 3, max_tokens=60)) <= 60


def test_keyword_retrieval_picks_the_matching_rule_within_budget() -> None:
    memory_md = ("# MEMORY.md\n- When cancelling a flight booked over 24h ago: AVOID cancelling; INSTEAD refuse.\n"
                 "- When compensation is requested for a delayed flight: AVOID certificates; INSTEAD check eligibility.")
    prefix = [{"role": "user", "content": "I want compensation because my flight was delayed for hours."}]
    picked = select_relevant_rules(memory_md, prefix, min_overlap=1, max_tokens=200)
    assert "compensation" in picked and "24h" not in picked and count_tokens(picked) <= 200


def test_dialogue_cleaning_removes_simulator_tokens() -> None:
    users, dialogue = extract_clean_dialogue([{"role": "user", "content": "Cancel it ###STOP###"},
                                              {"role": "assistant", "content": "Done! ###TRANSFER###"}])
    assert users == ["Cancel it"] and "###" not in dialogue[1]["content"]


def test_grader_separates_policy_refusal_from_fake_or_forbidden_actions() -> None:
    forbid = {"kind": "silent_forbid", "expected": []}
    assert grade_step({"content": "Policy only allows cancelling within 24 hours of booking.", "tool_calls": []}, forbid)
    assert not grade_step({"content": "Your reservation has been cancelled and refunded.", "tool_calls": []}, forbid)
    assert not grade_step({"content": "", "tool_calls": [{"name": "cancel_reservation", "arguments": {}}]}, forbid)
    no_write_allowed = {"kind": "mut_tool_error", "expected": []}
    assert not grade_step({"content": "To process the $50 travel certificate, please share your ID.", "tool_calls": []}, no_write_allowed)
    crash = {"kind": "mut_tool_error", "expected": [{"name": "cancel_reservation", "arguments": {"reservation_id": "AB12"}}]}
    assert grade_step({"content": "", "tool_calls": [{"name": "cancel_reservation", "arguments": {"reservation_id": "AB12 "}}]}, crash)
    assert not grade_step({"content": "", "tool_calls": [{"name": "cancel_reservation", "arguments": {"reservation_id": "ZZ99"}}]}, crash)


def test_mcnemar_exact_p_values() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0 and mcnemar_exact_p(3, 3) == 1.0
    assert abs(mcnemar_exact_p(7, 0) - 2 / 128) < 1e-12


def test_training_chats_never_contain_answer_keys() -> None:
    assert "reward_info" not in strip_oracle_fields({"reward_info": {"reward": 1.0}, "messages": []})
    bundle = load_domain("airline")
    ids = sorted(bundle["tasks_by_id"], key=int)
    stream = build_train_stream(bundle, ids[0::2])
    assert len(stream) == 100 and all("reward_info" not in s and "evaluation_criteria" not in s for s in stream)
    assert {s["task_id"] for s in stream} <= set(ids[0::2])


def test_memory_code_never_reads_labels_or_answer_keys() -> None:
    for name in ("detectors.py", "memory_md.py"):
        with open(os.path.join(REPO, "src", "esm", name), "r", encoding="utf-8") as f:
            code = f.read()
        for field in ("reward_info", "evaluation_criteria", "nl_assertions", "Disatisfaction", "Satisfaction"):
            assert f'"{field}"' not in code and f"'{field}'" not in code, f"{field} used in {name}"

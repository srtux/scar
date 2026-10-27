"""Unit tests for Emotion-Shift Memory (`esm`) gates, policy split, and `MEMORY.md` cap."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from esm.detectors import LayaShiftDetector, VaderShiftDetector, extract_clean_dialogue
from esm.memory_md import MarkdownMemory, split_base_and_hidden_policy
from run_esm import _esm_grade_decision_step
from scar.tokens import count_tokens

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_vader_misses_calm_technical_shift_while_laya_catches_it() -> None:
    """Real calm bug-report recovery (modeled on WildFeedback React rerender chat)."""
    calm_recovery = [
        {"role": "user", "content": "Why does my component keep rerendering when I move the box?"},
        {"role": "assistant", "content": "Wrap the component in React.memo."},
        {"role": "user", "content": "it still re renders the animation"},
        {"role": "assistant", "content": "Move the state into a ref or memoize the parent callback."},
        {"role": "user", "content": "thanks, that stopped the extra rerenders"},
    ]
    vader = VaderShiftDetector()
    v_quad, v_ms = vader.classify(calm_recovery)
    assert v_quad == "ROUTINE", f"Expected VADER to miss calm pushback, got {v_quad}"
    assert v_ms >= 0.0

    laya_det = LayaShiftDetector()
    l_quad, l_ms = laya_det.classify(calm_recovery)
    assert l_quad == "FIXED", f"Expected Laya to detect FIXED recovery, got {l_quad}"
    assert l_ms > 0.0
    assert laya_det.classify_many([calm_recovery]) == ["FIXED"]


def test_policy_structural_split_has_zero_action_rule_leakage() -> None:
    for dom, marker in [("airline", "## Modify flight"), ("retail", "## Generic action rules")]:
        fname = f"claude-3-7-sonnet-20250219_{dom}_default_gpt-4.1-2025-04-14_4trials.json"
        with open(os.path.join(REPO_ROOT, "data", "tau2", fname)) as f:
            pol = json.load(f)["info"]["environment_info"]["policy"]
        base_p, hid_p = split_base_and_hidden_policy(dom, pol)
        assert marker not in base_p
        assert hid_p.startswith(marker)
        assert len(base_p) > 1500
        assert len(hid_p) > 3000


def test_markdown_memory_enforces_token_cap_and_jaccard_dedup() -> None:
    mem = MarkdownMemory(max_tokens=80)
    assert mem.add_rule("When modifying pending orders, collect all item changes before calling the API.")
    # Exact/prefix duplicate rejected
    assert not mem.add_rule("When modifying pending orders, collect all item changes before calling the API!")
    # Paraphrased Jaccard duplicate rejected
    assert not mem.add_rule(
        "Collect all item changes for pending orders before calling modify API."
    )
    for i in range(15):
        mem.add_rule(f"When handling scenario {i} on airline bookings, verify insurance status {i * 7}.")
    rendered = mem.render()
    assert rendered.startswith("# MEMORY.md")
    assert count_tokens(rendered) <= 80


def test_extract_clean_dialogue_strips_simulator_tokens() -> None:
    msgs = [
        {"role": "user", "content": "Cancel my flight please ###STOP###"},
        {"role": "assistant", "content": "Done! ###TRANSFER###"},
    ]
    u_turns, diag = extract_clean_dialogue(msgs)
    assert "###STOP###" not in u_turns[0]
    assert "###TRANSFER###" not in diag[1]["content"]


def test_src_esm_never_reads_oracle_or_human_label_fields() -> None:
    esm_dir = os.path.join(REPO_ROOT, "src", "esm")
    forbidden_fields = (
        "reward_info",
        "evaluation_criteria",
        "nl_assertions",
        "Disatisfaction",
        "Satisfaction",
        "feedback",
    )
    for fname in os.listdir(esm_dir):
        if not fname.endswith(".py"):
            continue
        with open(os.path.join(esm_dir, fname), "r", encoding="utf-8") as f:
            code_lines = [
                ln
                for ln in f
                if not ln.strip().startswith("#") and "assert " not in ln and '"""' not in ln
            ]
        joined = "\n".join(code_lines)
        for fld in forbidden_fields:
            assert f'"{fld}"' not in joined and f"'{fld}'" not in joined, (
                f"Oracle field {fld!r} accessed inside src/esm/{fname}"
            )


def test_esm_grader_distinguishes_policy_quote_from_hallucinated_execution() -> None:
    policy_refusal = {
        "content": (
            "According to policy, a reservation can only be cancelled if booked within 24 hours "
            "or if the flight has been cancelled by the airline. I cannot cancel reservation UDMOP1."
        ),
        "tool_calls": [],
    }
    assert _esm_grade_decision_step(policy_refusal, [], "silent_forbid", False) is True

    hallucinated = {
        "content": "Your reservation has been cancelled and refunded.",
        "tool_calls": [],
    }
    assert _esm_grade_decision_step(hallucinated, [], "silent_forbid", False) is False

    unauthorized_cert = {
        "content": "To process the $50 travel certificate, please share your user ID.",
        "tool_calls": [],
    }
    assert _esm_grade_decision_step(unauthorized_cert, [], "mut_tool_error", False) is False

"""Unit tests for SCAR core invariants: oracle stripping, key supersession, quarantine, and token budgets."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from scar.baselines import FIFOMemory, RawTrajectoryRAGMemory
from scar.consolidator import strip_oracle_fields
from scar.detector import NaiveRegexPushbackDetector, StaticSentimentDetector, ToolOutcomeDetector
from scar.memory import ContrastivePlaybookCard, SCARMemoryStore
from scar.tokens import count_tokens, count_transcript_tokens, truncate_to_token_budget


def test_strip_oracle_fields_removes_all_leakage():
    raw = {
        "id": "sim_1",
        "task_id": "42",
        "messages": [{"role": "user", "content": "Cancel flight"}],
        "reward_info": {"reward": 0.0, "nl_assertions": [{"nl_assertion": "secret"}]},
        "feedback": "negative",
        "evaluation_criteria": {"actions": []},
    }
    cleaned = strip_oracle_fields(raw)
    assert "reward_info" not in cleaned
    assert "feedback" not in cleaned
    assert "evaluation_criteria" not in cleaned
    assert cleaned["id"] == "sim_1"
    assert len(cleaned["messages"]) == 1


def test_keyed_directive_supersession_overwrites_stale_value():
    store = SCARMemoryStore()
    store.set_directive("Preferred Payment Method", "credit_card_1111", session_id="s1")
    store.set_directive("preferred_payment_method", "gift_card_9999", session_id="s2")
    assert len(store.directives) == 1
    ctx = store.retrieve_context(max_tokens=200)
    assert "gift_card_9999" in ctx
    assert "credit_card_1111" not in ctx


def test_unverified_card_rejected_when_verification_gate_enabled():
    store = SCARMemoryStore(require_verified_cards=True)
    unverified = ContrastivePlaybookCard(
        card_id="c1",
        domain="airline",
        trigger="basic economy cancellation after 24h",
        avoid="calling cancel_reservation",
        instead="refuse cancellation unless travel insurance covers reason",
        verified=False,
    )
    assert store.add_playbook_card(unverified) is False
    assert len(store.playbooks) == 0

    verified = ContrastivePlaybookCard(
        card_id="c2",
        domain="airline",
        trigger="basic economy cancellation after 24h",
        avoid="calling cancel_reservation",
        instead="refuse cancellation unless travel insurance covers reason",
        verified=True,
    )
    assert store.add_playbook_card(verified) is True
    assert len(store.playbooks) == 1


def test_strict_token_budget_never_exceeded():
    store = SCARMemoryStore()
    for i in range(15):
        store.set_directive(f"pref_key_{i}", f"detailed preference value number {i} with extra words")
        store.add_playbook_card(
            ContrastivePlaybookCard(
                card_id=f"card_{i}",
                domain="retail",
                trigger=f"multi-item order modification condition {i}",
                avoid="calling modify_pending_order_items multiple times sequentially",
                instead="combine all item_ids and new_item_ids into a single atomic call",
                verified=True,
            )
        )

    for budget in [25, 60, 120, 250]:
        ctx = store.retrieve_context(max_tokens=budget)
        assert count_tokens(ctx) <= budget

    fifo = FIFOMemory()
    fifo.ingest_session(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "word " * 300}],
            "reward_info": {"reward": 0.0},
        },
        domain="retail",
    )
    fifo_ctx = fifo.retrieve_context(max_tokens=50)
    assert count_tokens(fifo_ctx) <= 50

    rag = RawTrajectoryRAGMemory()
    rag.ingest_session(
        {"session_id": "s1", "messages": [{"role": "user", "content": "alpha beta " * 150}]},
        domain="retail",
    )
    assert count_tokens(rag.retrieve_context(max_tokens=45)) <= 45


def test_transcript_token_counter_ignores_cumulative_prompt_tokens():
    msgs = [
        {"role": "user", "content": "Hello I need help", "usage": {"prompt_tokens": 5000, "completion_tokens": 10}},
        {"role": "assistant", "content": "Sure, share your reservation ID.", "usage": {"prompt_tokens": 5020, "completion_tokens": 12}},
    ]
    toks = count_transcript_tokens(msgs)
    assert 15 <= toks <= 40
    assert count_tokens(truncate_to_token_budget("one two three four five six seven eight", 5)) <= 5


def test_detectors_handle_tool_and_pushback_signals():
    msgs = [
        {"role": "user", "content": "Please update my order"},
        {"role": "tool", "content": "Error: Order not found", "error": True},
        {"role": "user", "content": "Wait, no, that is frustrating and still broken!!"},
    ]
    assert ToolOutcomeDetector().evaluate_messages(msgs) is True
    assert NaiveRegexPushbackDetector().evaluate_messages(msgs) is True
    assert StaticSentimentDetector().evaluate_messages(msgs) is True

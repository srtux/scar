"""SCAR Offline Consolidation, Quarantine, and Counterfactual Replay Verification.

Strict anti-leakage guarantee: `strip_oracle_fields` removes `reward_info`, `feedback`,
`evaluation_criteria`, `nl_assertions`, `action_checks`, and `db_check` before any
consolidation step executes.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

from scar.detector import SCARTransitionDetector, format_trajectory_for_detector
from scar.llm import LLMClient
from scar.memory import ContrastivePlaybookCard, SCARMemoryStore

ORACLE_FORBIDDEN_KEYS = frozenset(
    {
        "reward_info",
        "reward",
        "feedback",
        "evaluation_criteria",
        "nl_assertions",
        "action_checks",
        "db_check",
        "env_assertions",
        "communicate_checks",
        "reward_breakdown",
    }
)


def strip_oracle_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow+nested sanitized copy with all benchmark oracle fields removed."""
    clean: Dict[str, Any] = {}
    for k, v in record.items():
        if k in ORACLE_FORBIDDEN_KEYS:
            continue
        if isinstance(v, dict):
            clean[k] = {sk: sv for sk, sv in v.items() if sk not in ORACLE_FORBIDDEN_KEYS}
        else:
            clean[k] = v
    return clean


_DIRECTIVE_EXTRACT_SYS = """Extract ONLY persistent personal profile facts or standing default preferences explicitly stated by the USER as ongoing account/workstation rules.
- Extract ONLY statements framed as standing personal facts or ongoing defaults (e.g., "for my profile...", "standing preferences...", "going forward...", "from now on...", "change my default...", "always use...").
- NEVER extract one-off transactional choices made for a single flight booking, order modification, or item return (such as paying with PayPal or a card on one specific order, flying between two airports on one trip, or booking economy on a single reservation), and ignore small talk or complaints. If a session contains only a one-off support transaction, return `{"directives": []}`.
- You are given `EXISTING_DIRECTIVE_KEYS` already stored in memory. Whenever the user updates, switches, or refines an attribute that corresponds to an existing key in `EXISTING_DIRECTIVE_KEYS`, you MUST reuse that EXACT key name so the obsolete value is overwritten in-place.
- Otherwise, create a concise canonical snake_case key (e.g., `home_airport`, `preferred_seat`, `preferred_payment_method`, `notification_preference`, `shipping_speed`, `package_manager`, `test_framework`).

Return strict JSON:
{
  "directives": [
    {"key": "canonical_snake_case_key", "value": "concise current value"}
  ]
}"""

_PLAYBOOK_EXTRACT_SYS = """You synthesize reusable, entity-scrubbed Contrastive Playbook Cards from resolved agent struggles.
Given a trajectory (or a failed attempt paired with a resolved attempt) and optional domain policy:
1. Generalize away specific IDs, names, dates, or order numbers (never mention `#W12345`, `Q69X3R`, or specific customer names).
2. Identify the general operational condition (`trigger`), the exact error/policy trap (`avoid`), and the verified procedure (`instead`).
3. Keep each field concise (12-25 words) and actionable for tool calling and policy compliance.

Return strict JSON:
{
  "has_reusable_rule": true,
  "trigger": "when <general situation / policy condition>",
  "avoid": "<specific invalid tool call, wrong argument pattern, or policy mistake>",
  "instead": "<specific verified tool workflow or policy check>"
}"""

_REPLAY_CHECK_SYS = """You are SCAR's Counterfactual Replay Gate.
You are given:
1. Active Domain Policy (if any),
2. The conversation prefix right before a mistake/friction point,
3. A candidate Contrastive Playbook Card (`Trigger`, `Avoid`, `Instead`),
4. The agent's newly replayed response/tool action when guided by that card.

Verify whether the replayed response:
(a) Avoids the original failure trap (`Avoid`),
(b) Complies with the Domain Policy and available tool outputs, and
(c) Does NOT introduce a hallucination or invalid tool call.

Return strict JSON:
{
  "verified": bool,
  "reason": "short explanation"
}"""


class SCARConsolidator:
    """Consolidates oracle-stripped sessions into a `SCARMemoryStore`."""

    def __init__(
        self,
        llm: LLMClient,
        store: Optional[SCARMemoryStore] = None,
        enable_quarantine: bool = True,
        enable_replay_verification: bool = True,
    ):
        self.llm = llm
        self.enable_quarantine = enable_quarantine
        self.enable_replay_verification = enable_replay_verification
        self.store = store if store is not None else SCARMemoryStore(
            require_verified_cards=enable_replay_verification
        )
        self.detector = SCARTransitionDetector(llm=llm, model=llm.default_model)
        self.replay_attempts = 0
        self.replay_passed = 0

    def extract_directives_from_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        session_id: str = "",
    ) -> int:
        """Extract calm standing user facts/preferences and upsert via key supersession."""
        user_lines = [
            f"- {m.get('content', '').strip()}"
            for m in messages
            if m.get("role") == "user" and (m.get("content") or "").strip()
        ]
        if not user_lines:
            return 0
        existing_keys = sorted(self.store.directives.keys())
        prompt = (
            f"EXISTING_DIRECTIVE_KEYS: {existing_keys}\n\n"
            "USER MESSAGES IN SESSION:\n" + "\n".join(user_lines[:30])
        )
        res = self.llm.generate_json(
            prompt=prompt,
            system_instruction=_DIRECTIVE_EXTRACT_SYS,
            model=self.llm.fast_model,
            max_output_tokens=500,
        )
        added = 0
        for item in res.get("directives") or []:
            if isinstance(item, dict) and item.get("key") and item.get("value"):
                self.store.set_directive(
                    key=str(item["key"]),
                    value=str(item["value"]),
                    session_id=session_id,
                )
                added += 1
        return added

    def verify_card_by_replay(
        self,
        card: ContrastivePlaybookCard,
        struggle_messages: Sequence[Dict[str, Any]],
        domain_policy: str = "",
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        is_resolved_trajectory: bool = False,
    ) -> bool:
        """Counterfactually replay the LLM on the struggle prefix with `card` injected."""
        self.replay_attempts += 1
        # Locate the assistant decision turn where the initial mistake occurred
        cut_idx = len(struggle_messages) // 2
        found_err = False
        for idx, m in enumerate(struggle_messages):
            if m.get("error") and idx >= 1:
                for back in range(idx - 1, -1, -1):
                    if struggle_messages[back].get("role") == "assistant":
                        cut_idx = max(1, back)
                        found_err = True
                        break
                break
        if not found_err:
            for idx, m in enumerate(struggle_messages):
                if idx >= 2 and m.get("role") == "assistant":
                    if is_resolved_trajectory:
                        cut_idx = idx
                        break
                    if m.get("tool_calls"):
                        cut_idx = idx

        prefix = list(struggle_messages[: max(1, cut_idx)])
        sys_prompt = (
            f"{domain_policy}\n\n"
            f"=== VERIFIED PROCEDURAL MEMORY ===\n{card.to_prompt_text()}"
        ).strip()

        if tools:
            replayed = self.llm.generate_agent_step(
                messages=prefix,
                system_instruction=sys_prompt,
                tools=tools,
                model=self.llm.fast_model,
            )
            replayed_summary = (
                f"Text: {replayed.get('content', '')}\n"
                f"Tool Calls: {replayed.get('tool_calls', [])}"
            )
        else:
            prefix_txt = format_trajectory_for_detector(prefix)
            replayed_txt = self.llm.generate_text(
                prompt=f"Continue the assistant's next turn for this conversation:\n{prefix_txt}",
                system_instruction=sys_prompt,
                model=self.llm.fast_model,
                max_output_tokens=300,
            )
            replayed_summary = replayed_txt

        check_prompt = (
            f"=== DOMAIN POLICY ===\n{domain_policy[:4000]}\n\n"
            f"=== CONVERSATION PREFIX ===\n{format_trajectory_for_detector(prefix)}\n\n"
            f"=== CANDIDATE PLAYBOOK CARD ===\n{card.to_prompt_text()}\n\n"
            f"=== REPLAYED AGENT OUTPUT ===\n{replayed_summary}"
        )
        verdict = self.llm.generate_json(
            prompt=check_prompt,
            system_instruction=_REPLAY_CHECK_SYS,
            model=self.llm.fast_model,
            max_output_tokens=250,
        )
        ok = bool(verdict.get("verified", False))
        if ok:
            self.replay_passed += 1
        return ok

    def ingest_session(
        self,
        session: Dict[str, Any],
        domain: str = "general",
        domain_policy: str = "",
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        extract_directives: bool = True,
    ) -> Dict[str, Any]:
        """Process a single session into SCAR memory with zero oracle access."""
        clean_session = strip_oracle_fields(session)
        assert "reward_info" not in clean_session and "feedback" not in clean_session
        session_id = str(clean_session.get("session_id") or clean_session.get("id") or "sess")
        messages = clean_session.get("messages") or []

        det = self.detector.analyze(messages=messages, domain_policy=domain_policy)
        state = det["session_state"]

        if state == "UNRESOLVED_FAILURE" and self.enable_quarantine:
            self.store.quarantine_session(session_id)
            return {
                "session_id": session_id,
                "state": state,
                "directives_added": 0,
                "card_added": False,
                "quarantined": True,
            }

        directives_added = 0
        if extract_directives:
            directives_added = self.extract_directives_from_messages(messages, session_id=session_id)

        # Extract playbook card on RESOLVED_STRUGGLE (or also UNRESOLVED_FAILURE when quarantine disabled)
        card_added = False
        should_extract_card = (state == "RESOLVED_STRUGGLE") or (
            not self.enable_quarantine and (det["mid_session_struggle"] or state == "UNRESOLVED_FAILURE")
        )
        if should_extract_card:
            transcript = format_trajectory_for_detector(messages)
            card_json = self.llm.generate_json(
                prompt=(
                    f"=== DOMAIN POLICY ===\n{domain_policy[:4500]}\n\n"
                    f"=== SESSION ({state}) ===\n{transcript}"
                ),
                system_instruction=_PLAYBOOK_EXTRACT_SYS,
                model=self.llm.default_model,
                max_output_tokens=600,
            )
            if card_json.get("has_reusable_rule") and card_json.get("trigger") and card_json.get("instead"):
                cid = hashlib.sha1(f"{domain}:{session_id}:{card_json['trigger']}".encode()).hexdigest()[:8]
                card = ContrastivePlaybookCard(
                    card_id=cid,
                    domain=domain,
                    trigger=str(card_json["trigger"]).strip(),
                    avoid=str(card_json.get("avoid") or "repeating the unverified attempt").strip(),
                    instead=str(card_json["instead"]).strip(),
                    verified=not self.enable_replay_verification,
                )
                if self.enable_replay_verification:
                    card.verified = self.verify_card_by_replay(
                        card=card,
                        struggle_messages=messages,
                        domain_policy=domain_policy,
                        tools=tools,
                        is_resolved_trajectory=(state == "RESOLVED_STRUGGLE"),
                    )
                card_added = self.store.add_playbook_card(card)

        return {
            "session_id": session_id,
            "state": state,
            "directives_added": directives_added,
            "card_added": card_added,
            "quarantined": False,
        }

    def synthesize_paired_struggle_card(
        self,
        failed_messages: Sequence[Dict[str, Any]],
        resolved_messages: Sequence[Dict[str, Any]],
        domain: str,
        domain_policy: str = "",
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        pair_id: str = "",
    ) -> Optional[ContrastivePlaybookCard]:
        """Synthesize and replay-verify a ContrastivePlaybookCard from a detector-identified
        (failed attempt, resolved attempt) pair on training tasks (zero oracle access)."""
        prompt = (
            f"=== DOMAIN POLICY ===\n{domain_policy[:4500]}\n\n"
            f"=== FAILED ATTEMPT (Quarantined Error) ===\n{format_trajectory_for_detector(failed_messages)}\n\n"
            f"=== RESOLVED ATTEMPT (Verified Success) ===\n{format_trajectory_for_detector(resolved_messages)}"
        )
        card_json = self.llm.generate_json(
            prompt=prompt,
            system_instruction=_PLAYBOOK_EXTRACT_SYS,
            model=self.llm.default_model,
            max_output_tokens=350,
        )
        if not (card_json.get("has_reusable_rule") and card_json.get("trigger") and card_json.get("instead")):
            return None
        cid = hashlib.sha1(f"{domain}:{pair_id}:{card_json['trigger']}".encode()).hexdigest()[:8]
        card = ContrastivePlaybookCard(
            card_id=cid,
            domain=domain,
            trigger=str(card_json["trigger"]).strip(),
            avoid=str(card_json.get("avoid") or "making an unverified call").strip(),
            instead=str(card_json["instead"]).strip(),
            verified=not self.enable_replay_verification,
        )
        if self.enable_replay_verification:
            card.verified = self.verify_card_by_replay(
                card=card,
                struggle_messages=failed_messages,
                domain_policy=domain_policy,
                tools=tools,
            )
        if self.store.add_playbook_card(card):
            return card
        return None

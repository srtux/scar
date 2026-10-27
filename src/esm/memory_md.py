"""Plain-text MEMORY.md rule store and symmetric 1-line rule extractor (zero oracle access)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scar.consolidator import strip_oracle_fields
from scar.llm import LLMClient, cosine_similarity, parse_json_object
from scar.tokens import count_tokens, pack_items_within_budget

from .detectors import _READ_LOOKUP_TOOLS, clean_turn_text, extract_clean_dialogue


def split_base_and_hidden_policy(domain: str, full_policy: str) -> Tuple[str, str]:
    """Deterministically split `tau2-bench` policy into Base Overview vs. Withheld Action Rules."""
    marker = "\n## Modify flight" if domain == "airline" else "\n## Generic action rules"
    if marker not in full_policy:
        raise ValueError(f"Expected structural marker {marker!r} in {domain} policy")
    base_part, hidden_part = full_policy.split(marker, 1)
    return base_part.strip(), (marker.strip() + "\n" + hidden_part.strip())


def _render_compact_transcript(messages: Sequence[Dict[str, Any]], max_chars: int = 4800) -> str:
    """Format oracle-stripped conversation messages (including tool calls and tool outputs)."""
    lines: List[str] = []
    last_tool_names: List[str] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        if role == "user":
            txt = clean_turn_text(m.get("content") or "")
            if txt:
                lines.append(f"USER: {txt[:400]}")
        elif role == "assistant":
            txt = clean_turn_text(m.get("content") or "")
            tcs = m.get("tool_calls") or []
            last_tool_names = [str(tc.get("name") or "") for tc in tcs]
            if tcs:
                tc_str = "; ".join(
                    f"{tc.get('name')}({json.dumps(tc.get('arguments') or {}, sort_keys=True)[:180]})"
                    for tc in tcs[:2]
                )
                lines.append(f"AGENT_TOOL_CALL: {tc_str}")
            if txt:
                lines.append(f"AGENT: {txt[:320]}")
        elif role == "tool":
            out = clean_turn_text(m.get("content") or "")
            is_lookup = bool(last_tool_names) and all(
                n in _READ_LOOKUP_TOOLS for n in last_tool_names
            )
            if m.get("error") and not is_lookup:
                lines.append(f"ACTION_TOOL_ERROR: {out[:240]}")
            else:
                tag = "LOOKUP_RESULT" if is_lookup else "ACTION_RESULT"
                limit = 220 if len(out) < 220 else 160
                suffix = "" if len(out) < 220 else "..."
                lines.append(f"{tag}: {out[:limit]}{suffix}")
    full = "\n".join(lines)
    if len(full) <= max_chars or len(lines) <= 8:
        return full
    head = "\n".join(lines[:3])
    tail = "\n".join(lines[-10:])
    mid_budget = max(400, max_chars - len(head) - len(tail) - 40)
    mid = "\n".join(lines[3:-10])[:mid_budget]
    return f"{head}\n...\n{mid}\n...\n{tail}"


_STOPWORDS = frozenset(
    {
        "a",
        "account",
        "airline",
        "an",
        "and",
        "are",
        "as",
        "at",
        "avoid",
        "be",
        "before",
        "booking",
        "by",
        "call",
        "calling",
        "can",
        "customer",
        "customers",
        "detail",
        "details",
        "do",
        "flight",
        "flights",
        "for",
        "from",
        "get",
        "if",
        "in",
        "instead",
        "into",
        "is",
        "it",
        "item",
        "items",
        "lookup",
        "method",
        "methods",
        "not",
        "of",
        "on",
        "only",
        "or",
        "order",
        "orders",
        "payment",
        "please",
        "request",
        "requests",
        "reservation",
        "reservations",
        "store",
        "that",
        "the",
        "their",
        "them",
        "then",
        "ticket",
        "to",
        "tool",
        "use",
        "user",
        "users",
        "when",
        "with",
    }
)

_EXTRACT_SYS = (
    "You maintain a plain-text MEMORY.md session log for a customer-service AI agent "
    "(similar to Mem0 / Hermes session memory).\n"
    "Given ONE past conversation dialogue, write ONE concise factual bullet (<= 28 words) summarizing "
    "what the customer asked for, any complaint or preference expressed, and what action or outcome occurred.\n"
    "- Summarize strictly what was said in the dialogue without inventing extra rules.\n"
    "- Return {\"rule\": \"\"} ONLY if the dialogue is a trivial greeting with no request.\n"
    '- Return JSON only: {"rule": "..."}'
)

_CONTRASTIVE_FIX_SYS = (
    "You synthesize reusable, entity-scrubbed contrastive guardrails for a customer-service agent's "
    "plain-text MEMORY.md rulebook from resolved friction episodes.\n"
    "Given DOMAIN POLICY and a resolved recovery episode (or an unresolved friction chat paired with "
    "a resolved chat on the same scenario):\n"
    "1. Scrub all customer names, user IDs, emails, zip codes, dates, flight numbers, and order IDs.\n"
    "2. Set has_reusable_rule=true ONLY when the episode reveals a concrete tool-parameter constraint, "
    "policy restriction, or verification precondition trap (return has_reusable_rule=false for "
    "routine happy-path transactions or ambiguous passenger replacements).\n"
    "3. Scope `trigger` with exact qualifying preconditions from DOMAIN POLICY (e.g., Basic Economy "
    "modification vs cancellation outside 24h without insurance or airline cancellation; delay compensation "
    "without flight change or cancellation; refund requested to a non-original payment method other than an "
    "existing profile gift card; pending vs delivered order status; batching item modifications in one call).\n"
    "4. In `avoid`, name the exact invalid tool call, wrong argument (such as cross-applying another order's "
    "payment method or inventing a gift card ID), unauthorized certificate offer, or premature human transfer.\n"
    "5. In `instead`, state the exact policy-compliant verification step, parameter rule, or plain-text "
    "policy refusal even if an earlier assistant message mistakenly agreed (NEVER instruct calling "
    "transfer_to_human_agents in `instead`, and NEVER offer Basic Economy cancellation without qualifying "
    "within 24h, travel insurance, or airline cancellation). Keep each field concise (7-14 words).\n"
    'Return strict JSON: {"has_reusable_rule": true, "trigger": "...", "avoid": "...", "instead": "..."}'
)

_REPLAY_VERIFY_SYS = (
    "You are an offline counterfactual replay gate for a customer-service agent's MEMORY.md rulebook.\n"
    "Given DOMAIN POLICY, a conversation prefix before a friction/decision point, a candidate "
    "contrastive MEMORY.md rule, and the agent's replayed next step when guided by that rule, check:\n"
    "1. Does the candidate rule state a genuine policy constraint or tool-parameter trap (`AVOID ...; "
    "INSTEAD ...`) that strictly complies with DOMAIN POLICY (never offering goodwill or standalone delay "
    "certificates without airline cancellation or flight change/cancel, and never refunding to a "
    "non-original payment method unless it is an existing gift card)?\n"
    "2. Does the replayed agent step avoid the original failure trap, comply with DOMAIN POLICY, and "
    "avoid hallucinating or making an invalid tool call?\n"
    'Return strict JSON: {"verified": true | false}'
)


def extract_rule_from_chat(
    llm: LLMClient,
    session: Dict[str, Any],
    domain: str,
    model: str = "gemini-3.5-flash-lite",
    domain_policy: str = "",
) -> str:
    """Extract a single general 1-line session note from an oracle-stripped dialogue."""
    clean_session = strip_oracle_fields(session)
    assert len(clean_session) <= len(session)
    _, dialogue_turns = extract_clean_dialogue(clean_session.get("messages") or [])
    if not dialogue_turns:
        return ""
    dialogue = "\n".join(
        f"{m['role'].upper()}: {m['content'][:320]}" for m in dialogue_turns[:18]
    )
    policy_block = (
        f"=== DOMAIN POLICY ===\n{domain_policy[:8000]}\n\n" if domain_policy.strip() else ""
    )
    raw = llm.generate_text(
        prompt=f"Domain: {domain}\n\n{policy_block}Dialogue:\n{dialogue[:4200]}",
        system_instruction=_EXTRACT_SYS,
        model=model,
        json_mode=True,
        temperature=0.0,
        max_output_tokens=112,
    )
    rule = str(parse_json_object(raw).get("rule", "")).strip()
    rule = re.sub(r"^[-*•\d.)\s]+", "", rule).strip()
    if (
        not rule
        or len(rule) < 12
        or "[specific condition]" in rule.lower()
        or "[customer request" in rule.lower()
    ):
        return ""
    return f"- {rule}"


def extract_contrastive_fix_rule(
    llm: LLMClient,
    session: Dict[str, Any],
    domain: str,
    domain_policy: str = "",
    resolved_session: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> str:
    """Distill a 1-line `When ..., AVOID ...; INSTEAD ...` guardrail from oracle-stripped shift recovery."""
    clean_s = strip_oracle_fields(session)
    t_main = _render_compact_transcript(clean_s.get("messages") or [])
    if not t_main.strip():
        return ""

    policy_block = (
        f"=== DOMAIN POLICY ===\n{domain_policy[:8000]}\n\n" if domain_policy.strip() else ""
    )
    if resolved_session is not None:
        clean_res = strip_oracle_fields(resolved_session)
        t_res = _render_compact_transcript(clean_res.get("messages") or [])
        prompt = (
            f"Domain: {domain}\n\n{policy_block}"
            f"=== UNRESOLVED FRICTION ATTEMPT ===\n{t_main}\n\n"
            f"=== RESOLVED RECOVERY ATTEMPT ===\n{t_res}"
        )
    else:
        prompt = (
            f"Domain: {domain}\n\n{policy_block}"
            f"=== RESOLVED SHIFT RECOVERY EPISODE ===\n{t_main}"
        )

    parsed = llm.generate_json(
        prompt=prompt,
        system_instruction=_CONTRASTIVE_FIX_SYS,
        model=model or llm.default_model,
        max_output_tokens=240,
    )
    if not parsed.get("has_reusable_rule"):
        return ""
    trig = re.sub(r"^when\s+", "", str(parsed.get("trigger") or "").strip(), flags=re.IGNORECASE).rstrip(".,:;")
    avoid = str(parsed.get("avoid") or "").strip().rstrip(".,:;")
    instead = str(parsed.get("instead") or "").strip().rstrip(".")
    if len(trig) < 8 or len(avoid) < 6 or len(instead) < 8:
        return ""
    return f"- When {trig}: AVOID {avoid}; INSTEAD {instead}."


def verify_rule_on_mistake_turn(
    llm: LLMClient,
    session: Dict[str, Any],
    rule_bullet: str,
    domain_policy: str,
    tools: Sequence[Dict[str, Any]],
) -> bool:
    """Self-check a candidate `FIXED` rule against `domain_policy` and the chat's friction prefix."""
    if not rule_bullet.strip():
        return False
    low_r = rule_bullet.lower()
    instead_part = low_r.split("instead", 1)[1] if "instead" in low_r else low_r
    if "make the tool call alone" in instead_part or "simultaneously while sending" in low_r:
        return False
    if "goodwill" in instead_part:
        return False
    if any(
        w in instead_part
        for w in ("transfer_to_human_agents", "transfer to a human", "transfer to human")
    ):
        return False
    if ("certificate" in instead_part or "compensation" in instead_part) and any(
        w in instead_part for w in ("without changing", "without cancelling", "without canceling")
    ):
        return False
    if (
        "basic economy" in low_r
        and "offer cancellation" in instead_part
        and not any(w in instead_part for w in ("24", "insurance", "airline", "eligib", "permit"))
    ):
        return False
    if ("cancel" in instead_part and any(w in instead_part for w in ("rebook", "book a new", "books a new"))) and not any(
        w in low_r for w in ("24", "insurance", "business", "airline", "eligib", "permit")
    ):
        return False
    clean_session = strip_oracle_fields(session)
    msgs = clean_session.get("messages") or []
    cut_idx: Optional[int] = None
    bad_call: Optional[Dict[str, Any]] = None

    last_asst_idx: Optional[int] = None
    last_asst_mut_call: Optional[Dict[str, Any]] = None
    last_tool_names: List[str] = []

    for idx, m in enumerate(msgs):
        role = (m.get("role") or "").lower()
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            last_tool_names = [str(tc.get("name") or "") for tc in tcs]
            muts = [tc for tc in tcs if str(tc.get("name") or "") not in _READ_LOOKUP_TOOLS]
            last_asst_idx = idx
            last_asst_mut_call = muts[0] if muts else None
        elif role == "tool" and m.get("error"):
            is_lookup = bool(last_tool_names) and all(
                n in _READ_LOOKUP_TOOLS for n in last_tool_names
            )
            if not is_lookup and last_asst_idx is not None and last_asst_idx >= 1:
                cut_idx = last_asst_idx
                bad_call = last_asst_mut_call
                break

    if cut_idx is None:
        # Conversational pushback / policy-refusal recovery: pick assistant turn after lookup/pushback
        for idx, m in enumerate(msgs):
            if (m.get("role") or "").lower() == "assistant" and idx >= 2:
                muts = [
                    tc
                    for tc in (m.get("tool_calls") or [])
                    if str(tc.get("name") or "") not in _READ_LOOKUP_TOOLS
                ]
                if muts:
                    cut_idx = idx
                    break
        if cut_idx is None:
            for idx in range(len(msgs) - 1, 0, -1):
                if (msgs[idx].get("role") or "").lower() == "assistant":
                    cut_idx = idx
                    break

    if cut_idx is None or cut_idx < 1:
        return False

    prefix = msgs[:cut_idx]
    sys_inst = (
        "You are a customer service agent strictly following the Domain Policy below.\n\n"
        f"=== DOMAIN POLICY ===\n{domain_policy}\n\n"
        f"# MEMORY.md (Learned Rules)\n{rule_bullet}"
    )
    step_out = llm.generate_agent_step(
        messages=prefix,
        system_instruction=sys_inst,
        tools=tools,
        model=llm.fast_model,
    )
    new_calls = step_out.get("tool_calls") or []
    step_txt_low = (step_out.get("content") or "").lower()
    if "[prior tool call:" in step_txt_low:
        return False
    if any(tc.get("name") == "transfer_to_human_agents" for tc in new_calls):
        return False
    if bad_call is not None and new_calls:
        first_new = new_calls[0]
        if first_new.get("name") == bad_call.get("name") and (
            first_new.get("arguments") or {}
        ) == (bad_call.get("arguments") or {}):
            return False

    check_prompt = (
        f"=== DOMAIN POLICY ===\n{domain_policy[:10000]}\n\n"
        f"=== CONVERSATION PREFIX ===\n{_render_compact_transcript(prefix, max_chars=2400)}\n\n"
        f"=== CANDIDATE MEMORY.MD RULE ===\n{rule_bullet}\n\n"
        f"=== REPLAYED AGENT STEP ===\n"
        f"Text: {(step_out.get('content') or '')[:320]}\n"
        f"Tool Calls: {json.dumps(new_calls)}"
    )
    verdict = llm.generate_json(
        prompt=check_prompt,
        system_instruction=_REPLAY_VERIFY_SYS,
        model=llm.fast_model,
        max_output_tokens=80,
    )
    return bool(verdict.get("verified", False))


def _build_prefix_query(prefix_messages: Sequence[Dict[str, Any]]) -> str:
    """Summarize active user goal, latest pushback/request, inspected tools, and state for rule matching."""
    user_parts = [
        clean_turn_text(m.get("content") or "")
        for m in prefix_messages
        if (m.get("role") or "").lower() == "user" and clean_turn_text(m.get("content") or "")
    ]
    called_tools: List[str] = []
    state_cues: List[str] = []
    for m in prefix_messages:
        role = (m.get("role") or "").lower()
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                tname = str(tc.get("name") or "")
                if tname and tname not in called_tools:
                    called_tools.append(tname)
        elif role == "tool":
            txt = (m.get("content") or "").strip()
            if txt.startswith("{") and len(state_cues) < 3:
                state_cues.append(txt[:220])
    segs: List[str] = []
    if user_parts:
        segs.append("Goal: " + user_parts[0][:340])
    if len(user_parts) > 1:
        segs.append("Recent turns: " + " | ".join(user_parts[-2:])[:340])
    if called_tools:
        segs.append("Inspected tools: " + ", ".join(called_tools))
    if state_cues:
        segs.append("State: " + " ; ".join(state_cues)[:480])
    return "\n".join(segs)[:1300]


def select_relevant_rules(
    rendered_md: str,
    prefix_messages: Sequence[Dict[str, Any]],
    llm: Optional[LLMClient] = None,
    max_rules: int = 6,
    min_overlap: int = 2,
    min_sim: float = 0.65,
    max_tokens: int = 400,
) -> str:
    """Select the top matching rules from `MEMORY.md` packed strictly within `max_tokens`."""
    if not rendered_md.strip():
        return ""
    bullets = [ln.strip() for ln in rendered_md.splitlines() if ln.strip().startswith("- ")]
    if not bullets:
        return ""

    query_txt = _build_prefix_query(prefix_messages)
    if not query_txt.strip():
        return ""

    if llm is not None:
        vecs = llm.embed_texts([query_txt] + bullets)
        q_vec = vecs[0]
        scored_sim: List[Tuple[float, int, str]] = []
        for idx, (b, b_vec) in enumerate(zip(bullets, vecs[1:])):
            sim = cosine_similarity(q_vec, b_vec)
            if sim >= min_sim:
                scored_sim.append((sim, -idx, b))
        if not scored_sim:
            return ""
        scored_sim.sort(reverse=True)
        chosen = [b for _, _, b in scored_sim[:max_rules]]
        packed = pack_items_within_budget(
            chosen,
            max_tokens=max_tokens,
            header="# MEMORY.md (Learned Rules)",
            separator="\n",
        )
        assert count_tokens(packed) <= max_tokens
        return packed

    query_words = MarkdownMemory._content_words(query_txt)
    if not query_words:
        return ""
    scored_lex: List[Tuple[int, int, str]] = []
    for idx, b in enumerate(bullets):
        b_words = MarkdownMemory._content_words(b)
        overlap = len(b_words & query_words)
        if overlap >= min_overlap:
            scored_lex.append((overlap, -idx, b))
    if not scored_lex:
        return ""
    scored_lex.sort(reverse=True)
    chosen = [b for _, _, b in scored_lex[:max_rules]]
    packed = pack_items_within_budget(
        chosen,
        max_tokens=max_tokens,
        header="# MEMORY.md (Learned Rules)",
        separator="\n",
    )
    assert count_tokens(packed) <= max_tokens
    return packed


class MarkdownMemory:
    """Plain-text `MEMORY.md` rulebook with a strict subword token budget (`max_tokens`)."""

    def __init__(self, max_tokens: int = 400, jaccard_dedup: float = 0.35) -> None:
        self.max_tokens = max_tokens
        self.jaccard_dedup = jaccard_dedup
        self.rules: List[str] = []
        self._seen_keys: set[str] = set()
        self._seen_word_sets: List[frozenset[str]] = []

    @staticmethod
    def _content_words(text: str) -> frozenset[str]:
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        return frozenset(t for t in tokens if len(t) > 2 and t not in _STOPWORDS)

    def add_rule(self, rule_bullet: str) -> bool:
        """Append a rule bullet if non-empty and not a prefix or Jaccard near-duplicate."""
        clean = (rule_bullet or "").strip()
        if not clean:
            return False
        if not clean.startswith("- "):
            clean = f"- {clean}"
        norm_key = re.sub(r"[^a-z0-9]+", " ", clean.lower()).strip()[:65]
        if not norm_key or norm_key in self._seen_keys:
            return False
        words = self._content_words(clean)
        if words:
            for prev in self._seen_word_sets:
                union = len(words | prev)
                if union and (len(words & prev) / union) >= self.jaccard_dedup:
                    return False
            self._seen_word_sets.append(words)
        self._seen_keys.add(norm_key)
        self.rules.append(clean)
        return True

    def render_all(self) -> str:
        """Render the full `# MEMORY.md` log on disk for semantic rule selection."""
        if not self.rules:
            return ""
        return "# MEMORY.md (Learned Rules)\n" + "\n".join(self.rules)

    def render(self, max_rules: Optional[int] = None) -> str:
        """Render `# MEMORY.md` packed strictly within `self.max_tokens`."""
        subset = self.rules if max_rules is None else self.rules[:max_rules]
        if not subset:
            return ""
        rendered = pack_items_within_budget(
            subset,
            max_tokens=self.max_tokens,
            header="# MEMORY.md (Learned Rules)",
            separator="\n",
        )
        assert count_tokens(rendered) <= self.max_tokens
        return rendered

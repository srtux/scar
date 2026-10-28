"""MEMORY.md: a plain-text rule file with a hard token cap, one rule writer, and keyword retrieval.

Every memory condition in the experiment uses the exact same rule writer, the same
400-token cap, and the same keyword retrieval. The only thing that changes between
conditions is WHICH past chats are allowed into MEMORY.md.
"""

from __future__ import annotations

import collections
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from .detectors import READ_TOOLS, clean_turn_text
from .llm import LLMClient, pack_items_within_budget, strip_oracle_fields

HEADER = "# MEMORY.md (Learned Rules)"

RULE_WRITER_SYS = (
    "You keep a MEMORY.md file of short lessons for a customer-service AI agent.\n"
    "Read ONE past chat between a customer and the agent, including the agent's tool calls and results.\n"
    "1. Remove names, user IDs, emails, zip codes, dates, flight numbers, and order numbers.\n"
    "2. If the chat teaches a lesson the agent can reuse with future customers, write it in three short parts: "
    "trigger (the situation), avoid (the mistake to avoid), instead (what to do instead). "
    "Keep each part under 20 words.\n"
    "3. If the chat teaches nothing reusable, set has_rule to false.\n"
    'Return JSON only: {"has_rule": true, "trigger": "...", "avoid": "...", "instead": "..."}'
)

_STOPWORDS = frozenset({
    "a", "account", "airline", "an", "and", "are", "as", "at", "avoid", "be", "before", "booking", "by", "call",
    "calling", "can", "customer", "customers", "detail", "details", "do", "flight", "flights", "for", "from", "get",
    "if", "in", "instead", "into", "is", "it", "item", "items", "lookup", "method", "methods", "not", "of", "on",
    "only", "or", "order", "orders", "payment", "please", "request", "requests", "reservation", "reservations",
    "store", "that", "the", "their", "them", "then", "ticket", "to", "tool", "use", "user", "users", "when", "with",
})


def content_words(text: str) -> frozenset:
    """Lowercase keywords of 3+ letters, minus common words."""
    return frozenset(w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS)


def render_chat(messages: Sequence[Dict[str, Any]], max_chars: int = 4800) -> str:
    """Compact transcript with tool calls and results, trimming the middle of long chats."""
    lines: List[str] = []
    last_tools: List[str] = []
    for m in messages:
        role, text = (m.get("role") or "").lower(), clean_turn_text(m.get("content") or "")
        if role == "user" and text:
            lines.append(f"USER: {text[:400]}")
        elif role in ("assistant", "agent"):
            calls = m.get("tool_calls") or []
            last_tools = [str(tc.get("name") or "") for tc in calls]
            if calls:
                lines.append("AGENT_TOOL_CALL: " + "; ".join(
                    f"{tc.get('name')}({json.dumps(tc.get('arguments') or {}, sort_keys=True)[:180]})" for tc in calls[:2]))
            if text:
                lines.append(f"AGENT: {text[:320]}")
        elif role == "tool":
            lookup = bool(last_tools) and all(n in READ_TOOLS for n in last_tools)
            tag = "ACTION_TOOL_ERROR" if (m.get("error") and not lookup) else ("LOOKUP_RESULT" if lookup else "ACTION_RESULT")
            lines.append(f"{tag}: {text[:220]}")
    full = "\n".join(lines)
    if len(full) <= max_chars or len(lines) <= 13:
        return full
    head, tail = "\n".join(lines[:3]), "\n".join(lines[-10:])
    middle = "\n".join(lines[3:-10])[:max(400, max_chars - len(head) - len(tail) - 40)]
    return f"{head}\n...\n{middle}\n...\n{tail}"


def extract_rule(llm: LLMClient, chat: Dict[str, Any], domain: str, model: Optional[str] = None) -> str:
    """Write at most one 'When ...: AVOID ...; INSTEAD ...' rule from one past chat (answer keys removed)."""
    transcript = render_chat(strip_oracle_fields(chat).get("messages") or [])
    if not transcript.strip():
        return ""
    reply = llm.generate_json(prompt=f"Domain: {domain}\n\n=== PAST CHAT ===\n{transcript}",
                              system_instruction=RULE_WRITER_SYS, model=model or llm.default_model, max_output_tokens=240)
    if not reply.get("has_rule"):
        return ""
    trigger = re.sub(r"^when\s+", "", str(reply.get("trigger") or "").strip(), flags=re.IGNORECASE).rstrip(".,:;")
    avoid = str(reply.get("avoid") or "").strip().rstrip(".,:;")
    instead = str(reply.get("instead") or "").strip().rstrip(".")
    if len(trigger) < 8 or len(avoid) < 6 or len(instead) < 8:
        return ""
    return f"- When {trigger}: AVOID {avoid}; INSTEAD {instead}."


def _query_text(prefix_messages: Sequence[Dict[str, Any]]) -> str:
    """What the new chat is about so far: the customer's goal, latest turns, tools used, and looked-up state."""
    user_turns = [clean_turn_text(m.get("content") or "") for m in prefix_messages if (m.get("role") or "").lower() == "user"]
    user_turns = [t for t in user_turns if t]
    tools: List[str] = []
    state: List[str] = []
    for m in prefix_messages:
        role = (m.get("role") or "").lower()
        if role in ("assistant", "agent"):
            tools += [str(tc.get("name")) for tc in m.get("tool_calls") or [] if tc.get("name") and tc.get("name") not in tools]
        elif role == "tool" and (m.get("content") or "").strip().startswith("{") and len(state) < 3:
            state.append((m.get("content") or "").strip()[:220])
    parts = []
    if user_turns:
        parts.append("Goal: " + user_turns[0][:340])
    if len(user_turns) > 1:
        parts.append("Recent turns: " + " | ".join(user_turns[-2:])[:340])
    if tools:
        parts.append("Inspected tools: " + ", ".join(tools))
    if state:
        parts.append("State: " + " ; ".join(state)[:480])
    return "\n".join(parts)[:1300]


def select_relevant_rules(memory_md: str, prefix_messages: Sequence[Dict[str, Any]], max_rules: int = 6,
                          min_overlap: int = 2, max_tokens: int = 400) -> str:
    """Pick the rules that share the most (rare-word-weighted) keywords with the new chat, within max_tokens."""
    rules = [ln.strip() for ln in memory_md.splitlines() if ln.strip().startswith("- ")]
    query = content_words(_query_text(prefix_messages))
    if not rules or not query:
        return ""
    rule_words = [content_words(r) for r in rules]
    doc_freq = collections.Counter(w for words in rule_words for w in words)
    idf = {w: 1.0 + math.log((1.0 + len(rules)) / (1.0 + df)) for w, df in doc_freq.items()}
    scored = sorted(((sum(idf[w] for w in words & query), -i, rule) for i, (rule, words) in enumerate(zip(rules, rule_words))
                     if len(words & query) >= min_overlap), reverse=True)
    return pack_items_within_budget([rule for _, _, rule in scored[:max_rules]], max_tokens=max_tokens, header=HEADER)


class MarkdownMemory:
    """An append-only list of rule bullets that skips exact and near-duplicate rules."""

    def __init__(self, max_tokens: int = 400, jaccard_dedup: float = 0.35) -> None:
        self.max_tokens, self.jaccard_dedup = max_tokens, jaccard_dedup
        self.rules: List[str] = []
        self._keys: set = set()
        self._word_sets: List[frozenset] = []

    def add_rule(self, rule: str) -> bool:
        rule = (rule or "").strip()
        if not rule:
            return False
        rule = rule if rule.startswith("- ") else f"- {rule}"
        key = re.sub(r"[^a-z0-9]+", " ", rule.lower()).strip()[:65]
        words = content_words(rule)
        if not key or key in self._keys:
            return False
        if words and any(len(words & prev) / len(words | prev) >= self.jaccard_dedup for prev in self._word_sets):
            return False
        if words:
            self._word_sets.append(words)
        self._keys.add(key)
        self.rules.append(rule)
        return True

    def render_all(self) -> str:
        """The whole file on disk. Retrieval later picks at most max_tokens of it."""
        return HEADER + "\n" + "\n".join(self.rules) if self.rules else ""

    def render(self) -> str:
        """The first rules that fit inside max_tokens (used for the poster's example files)."""
        return pack_items_within_budget(self.rules, max_tokens=self.max_tokens, header=HEADER) if self.rules else ""

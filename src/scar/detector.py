"""Failure and pushback detectors for SCAR.

Identifies multi-turn collaborative struggles across:
1. Tool outcomes (syntax errors, crashes, exceptions).
2. User pushback (explicit corrections, rephrased queries, complaints).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


CONTROL_TOKEN_RE = re.compile(r"###[A-Z][A-Z_\-]*###")

DEFAULT_PUSHBACK_PATTERNS = [
    r"\bno\b",
    r"\bnot what i\b",
    r"\bwrong\b",
    r"\binstead\b",
    r"\bi asked for\b",
    r"\bi wanted\b",
    r"\bactually\b",
    r"\bi meant\b",
    r"\bincorrect\b",
    r"\bcan't you\b",
    r"\bwhy did you\b",
    r"\bwait\b",
    r"\bplease check\b",
    r"\bdon't\b",
    r"\bdo not\b",
    r"\berror\b",
    r"\bproblem\b",
    r"\bthat's not\b",
    r"\bthats not\b",
]


class PushbackDetector:
    """Detects user pushback and negative feedback in conversational agent sessions."""

    def __init__(self, patterns: Optional[Sequence[str]] = None):
        patterns = patterns or DEFAULT_PUSHBACK_PATTERNS
        self._regex = re.compile("|".join(patterns), re.IGNORECASE)

    def is_pushback(self, text: str) -> bool:
        """Check if clean text contains pushback indicators."""
        clean = CONTROL_TOKEN_RE.sub("", text).strip()
        if not clean:
            return False
        return bool(self._regex.search(clean))

    def evaluate_messages(self, messages: Sequence[Dict[str, Any]]) -> bool:
        """Evaluate a sequence of messages.
        
        Pushback is checked from the second user turn onwards (the first turn is
        the initial user task request).
        """
        user_turn_count = 0
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content") or ""
                user_turn_count += 1
                if user_turn_count > 1 and self.is_pushback(content):
                    return True
        return False


class ToolOutcomeDetector:
    """Detects tool execution crashes, exceptions, and non-zero exit codes."""

    def evaluate_messages(self, messages: Sequence[Dict[str, Any]]) -> bool:
        for m in messages:
            if m.get("error"):
                return True
            for tc in (m.get("tool_calls") or []):
                if tc.get("error"):
                    return True
        return False


class DualDetector:
    """Combines tool execution errors and user pushback signals."""

    def __init__(self, pushback_detector: Optional[PushbackDetector] = None):
        self.pushback = pushback_detector or PushbackDetector()
        self.tool = ToolOutcomeDetector()

    def evaluate_messages(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
        has_tool_err = self.tool.evaluate_messages(messages)
        has_pushback = self.pushback.evaluate_messages(messages)
        return {
            "tool_error": has_tool_err,
            "pushback": has_pushback,
            "struggle_detected": (has_tool_err or has_pushback),
        }

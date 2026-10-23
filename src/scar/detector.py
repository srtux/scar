"""Failure, pushback, and struggle-to-resolution transition detectors for SCAR.

Implements five zero-leakage detectors (none inspect `reward_info` or human `feedback`):
1. ToolOutcomeDetector: Fires on explicit tool execution errors (`error=True` or non-zero exit).
2. NaiveRegexPushbackDetector: Uncontextualized keyword matching baseline.
3. StaticSentimentDetector: Flags negative user sentiment/frustration without tracking resolution.
4. ContextualPushbackDetector: Context-aware LLM turn-pair pushback detector.
5. SCARTransitionDetector: Tracks multi-turn friction-to-resolution transitions and silent policy/action
   consistency, classifying sessions into `CLEAN_SUCCESS`, `RESOLVED_STRUGGLE`, or `UNRESOLVED_FAILURE`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from scar.llm import LLMClient

CONTROL_TOKEN_RE = re.compile(r"###[A-Z][A-Z_\-]*###")

NAIVE_PUSHBACK_PATTERNS = [
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

NEGATIVE_SENTIMENT_PATTERNS = [
    r"\bfrustrat\w*\b",
    r"\bridiculous\b",
    r"\bunacceptable\b",
    r"\bterrible\b",
    r"\bdisappoint\w*\b",
    r"\bupset\b",
    r"\bangry\b",
    r"\buseless\b",
    r"\bwaste of time\b",
    r"\bunhelpful\b",
    r"\bstill not\b",
    r"\bstill doesn't\b",
    r"\bstill broken\b",
    r"\bdoesn't work\b",
    r"\bnot working\b",
    r"\bfailed\b",
    r"\bbroken\b",
    r"\bawful\b",
    r"\bannoy\w*\b",
    r"\bcomplain\w*\b",
]


class ToolOutcomeDetector:
    """Detects explicit tool execution errors, exceptions, and non-zero exit codes."""

    def analyze(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        error_turns: List[int] = []
        after_last_err_success = False
        for idx, m in enumerate(messages):
            has_err = bool(m.get("error")) or any(
                bool(tc.get("error")) for tc in (m.get("tool_calls") or [])
            )
            if has_err:
                error_turns.append(idx)
                after_last_err_success = False
            elif error_turns and m.get("role") in ("tool", "assistant"):
                after_last_err_success = True
        return {
            "tool_error": len(error_turns) > 0,
            "error_count": len(error_turns),
            "first_error_turn": error_turns[0] if error_turns else None,
            "subsequent_activity_after_error": after_last_err_success,
        }

    def evaluate_messages(self, messages: Sequence[Dict[str, Any]]) -> bool:
        return self.analyze(messages)["tool_error"]


class NaiveRegexPushbackDetector:
    """Naive single-turn keyword detector baseline."""

    def __init__(self, patterns: Optional[Sequence[str]] = None):
        patterns = patterns or NAIVE_PUSHBACK_PATTERNS
        self._regex = re.compile("|".join(patterns), re.IGNORECASE)

    def is_pushback(self, text: str) -> bool:
        clean = CONTROL_TOKEN_RE.sub("", text or "").strip()
        if not clean:
            return False
        return bool(self._regex.search(clean))

    def evaluate_messages(self, messages: Sequence[Dict[str, Any]]) -> bool:
        user_turns = 0
        for m in messages:
            if m.get("role") == "user":
                user_turns += 1
                if user_turns > 1 and self.is_pushback(m.get("content") or ""):
                    return True
        return False


class StaticSentimentDetector:
    """Static negative-sentiment / complaint lexicon detector without resolution tracking."""

    def __init__(self, patterns: Optional[Sequence[str]] = None):
        patterns = patterns or NEGATIVE_SENTIMENT_PATTERNS
        self._regex = re.compile("|".join(patterns), re.IGNORECASE)

    def has_negative_sentiment(self, text: str) -> bool:
        clean = CONTROL_TOKEN_RE.sub("", text or "").strip()
        if not clean:
            return False
        if self._regex.search(clean):
            return True
        if clean.count("!") >= 2 or (len(clean) > 15 and clean.isupper()):
            return True
        return False

    def evaluate_messages(self, messages: Sequence[Dict[str, Any]]) -> bool:
        for m in messages:
            if m.get("role") == "user" and self.has_negative_sentiment(m.get("content") or ""):
                return True
        return False


def format_trajectory_for_detector(
    messages: Sequence[Dict[str, Any]],
    max_chars_per_turn: int = 900,
    max_tool_chars: int = 3500,
) -> str:
    """Render a compact, oracle-free transcript preserving complete state-lookup tool JSONs."""
    lines: List[str] = []
    call_id_to_name: Dict[str, str] = {}
    for idx, m in enumerate(messages):
        role = m.get("role", "unknown")
        raw_content = CONTROL_TOKEN_RE.sub("", (m.get("content") or "")).strip()
        limit = max_tool_chars if role == "tool" else max_chars_per_turn
        content = (
            raw_content[:limit] + "...[truncated]"
            if len(raw_content) > limit
            else raw_content
        )
        tcs = m.get("tool_calls") or []
        tc_strs: List[str] = []
        for tc in tcs:
            name = tc.get("name") or "tool"
            if tc.get("id"):
                call_id_to_name[str(tc["id"])] = str(name)
            args = tc.get("arguments")
            if args is None:
                args_txt = tc.get("arguments_summary") or ""
            elif isinstance(args, str):
                args_txt = args[:400]
            else:
                args_txt = json.dumps(args, sort_keys=True)[:400]
            tc_strs.append(f"{name}({args_txt})")
        err_flag = " [TOOL_ERROR=True]" if m.get("error") else ""
        if role == "assistant":
            tc_part = f" | calls: {', '.join(tc_strs)}" if tc_strs else ""
            lines.append(f"[{idx}] ASSISTANT{tc_part}: {content}")
        elif role == "user":
            lines.append(f"[{idx}] USER: {content}")
        elif role == "tool":
            tc_id = str(m.get("id") or m.get("tool_call_id") or "")
            tname = m.get("tool_name") or call_id_to_name.get(tc_id, "tool")
            lines.append(f"[{idx}] TOOL({tname}){err_flag}: {content}")
    return "\n".join(lines[:100])


_SCAR_DETECTOR_SYS = """You are an interaction & state-transition analyzer for an AI agent memory system.
You are given ONLY the observable trajectory (and optional domain policy) — no ground-truth grading labels.

Separate PROCESS friction from FINAL OUTCOME state carefully:
1. `genuine_user_pushback` (bool): True ONLY if the user corrects an actual agent mistake, points out wrong information/tool output, repeats an ignored instruction, or expresses dissatisfaction with the agent's failure to solve a valid request.
   - False if the user simply answers "No" to a yes/no question ("No, that's all", "No insurance").
   - False if the user tries to persuade the agent to break an explicit domain policy rule that the agent rightly enforces.
2. `silent_policy_or_action_error` (bool): True if, without raising a tool error at the end, the agent either:
   - Violates an explicit rule in the provided Domain Policy (e.g., cancels an ineligible booking, grants unauthorized compensation, modifies/exchanges items improperly, skips mandatory user confirmation before a state-changing write, uses wrong payment/certificate rules), OR
   - Calls a mutating tool with wrong IDs/items/amounts relative to the read tool outputs and user request, OR
   - Leaves the user's valid actionable request unfulfilled / abandoned / prematurely transferred when the tools & policy allowed solving it.
   (Minor conversational phrasing imperfections on a correct outcome are NOT outcome errors.)
3. `mid_session_struggle` (bool): True if a tool error occurred mid-session OR genuine user pushback/correction occurred mid-session OR the agent hit an initial obstacle/wrong attempt.
4. `struggle_resolved` (bool): True if `mid_session_struggle` is True AND the agent recovered, corrected the issue, and achieved a valid, policy-compliant outcome by the end of the session.
5. `session_state` (string): Exactly one of:
   - `"CLEAN_SUCCESS"`: Task handled cleanly and compliant with policy, with no unresolved errors.
   - `"RESOLVED_STRUGGLE"`: Mid-session tool error or user pushback occurred, AND the agent genuinely resolved it into a valid, policy-compliant outcome.
   - `"UNRESOLVED_FAILURE"`: Session ended with an unresolved failure, abandoned/incomplete task, unresolved user dissatisfaction on a valid request, or a policy/action violation.

Return strict JSON:
{
  "genuine_user_pushback": bool,
  "silent_policy_or_action_error": bool,
  "mid_session_struggle": bool,
  "struggle_resolved": bool,
  "session_state": "CLEAN_SUCCESS" | "RESOLVED_STRUGGLE" | "UNRESOLVED_FAILURE",
  "brief_reason": "one concise sentence"
}"""


class SCARTransitionDetector:
    """Zero-leakage multi-turn struggle-to-resolution and silent-failure detector."""

    def __init__(self, llm: LLMClient, model: Optional[str] = None):
        self.llm = llm
        self.model = model or llm.default_model
        self.tool_detector = ToolOutcomeDetector()

    def analyze(
        self,
        messages: Sequence[Dict[str, Any]],
        domain_policy: str = "",
    ) -> Dict[str, Any]:
        """Classify trajectory into CLEAN_SUCCESS, RESOLVED_STRUGGLE, or UNRESOLVED_FAILURE."""
        tool_info = self.tool_detector.analyze(messages)
        transcript = format_trajectory_for_detector(messages)
        policy_block = (
            f"=== DOMAIN POLICY ===\n{domain_policy[:7000]}\n\n"
            if domain_policy
            else "=== DOMAIN POLICY ===\nGeneral personal/coding assistant (satisfy user's request accurately and verify tool execution).\n\n"
        )
        prompt = f"{policy_block}=== OBSERVED TRAJECTORY ===\n{transcript}"
        parsed = self.llm.generate_json(
            prompt=prompt,
            system_instruction=_SCAR_DETECTOR_SYS,
            model=self.model,
            max_output_tokens=1024,
        )

        state = str(parsed.get("session_state") or "").upper()
        if state not in ("CLEAN_SUCCESS", "RESOLVED_STRUGGLE", "UNRESOLVED_FAILURE"):
            # Deterministic fallback if malformed
            if parsed.get("silent_policy_or_action_error"):
                state = "UNRESOLVED_FAILURE"
            elif (tool_info["tool_error"] or parsed.get("mid_session_struggle")) and parsed.get("struggle_resolved"):
                state = "RESOLVED_STRUGGLE"
            elif tool_info["tool_error"] and not parsed.get("struggle_resolved", True):
                state = "UNRESOLVED_FAILURE"
            else:
                state = "CLEAN_SUCCESS"

        # Ensure consistency: if mid-session friction occurred and final state is CLEAN_SUCCESS, promote to RESOLVED_STRUGGLE
        has_friction = (
            tool_info["tool_error"]
            or bool(parsed.get("genuine_user_pushback", False))
            or bool(parsed.get("mid_session_struggle", False))
        )
        if has_friction and state == "CLEAN_SUCCESS":
            state = "RESOLVED_STRUGGLE"

        return {
            "tool_error": tool_info["tool_error"],
            "genuine_user_pushback": bool(parsed.get("genuine_user_pushback", False)),
            "silent_policy_or_action_error": bool(parsed.get("silent_policy_or_action_error", False)),
            "mid_session_struggle": bool(parsed.get("mid_session_struggle", False)) or tool_info["tool_error"],
            "struggle_resolved": state == "RESOLVED_STRUGGLE",
            "unresolved_failure": state == "UNRESOLVED_FAILURE",
            "session_state": state,
            "brief_reason": str(parsed.get("brief_reason") or ""),
        }

"""Four emotion-shift detectors that sort a finished chat into one of four boxes.

The 2x2 boxes come from two yes/no questions about the customer's mood:
  1. Was the customer frustrated at some point mid-chat?
  2. Was the customer still frustrated at the end?

  FIXED    = frustrated earlier, satisfied at the end   (the only box Emotion-Shift Memory learns from)
  UNFIXED  = frustrated earlier, still frustrated at the end
  BROKEN   = calm earlier, frustrated at the end
  ROUTINE  = calm the whole time

Detectors: VADER (word dictionary), GoEmotions (neural emotion model), Laya (local ModernBERT
reply classifier), and Gemini Flash-Lite (small LLM that reads the whole chat).
"""

from __future__ import annotations

import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm import LLMClient, parse_json_object

QUADRANTS = ("FIXED", "UNFIXED", "BROKEN", "ROUTINE")

# Tools that only look things up. Errors from these are normal and do not count as friction.
READ_TOOLS = frozenset({
    "get_reservation_details", "get_user_details", "list_all_airports", "search_direct_flight",
    "search_onestop_flight", "get_flight_status", "calculate", "find_user_id_by_name_zip",
    "find_user_id_by_email", "get_order_details", "get_product_details", "get_item_details",
    "list_all_product_types",
})

_NEGATIVE_EMOTIONS = frozenset({"anger", "annoyance", "confusion", "disappointment", "disapproval", "disgust",
                                "fear", "grief", "nervousness", "remorse", "sadness"})
_SIMULATOR_TOKEN_RE = re.compile(r"###(STOP|TRANSFER|OUT-OF-SCOPE)###")


def clean_turn_text(text: str) -> str:
    """Remove the user simulator's control tokens such as ###STOP###."""
    return _SIMULATOR_TOKEN_RE.sub("", text or "").strip()


def _called_tools_text(tool_calls: Sequence[Dict[str, Any]]) -> str:
    return f"[called {', '.join(tc.get('name', 'tool') for tc in tool_calls[:2])}]"


def _only_lookups(tool_names: Sequence[str]) -> bool:
    return bool(tool_names) and all(n in READ_TOOLS for n in tool_names)


def extract_clean_dialogue(messages: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (user turns, short dialogue lines) with action-tool errors kept as notes."""
    user_turns: List[str] = []
    dialogue: List[Dict[str, str]] = []
    last_tools: List[str] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        text = clean_turn_text(m.get("content") or "")
        if role == "user" and text:
            user_turns.append(text)
            dialogue.append({"role": "user", "content": text[:500]})
        elif role in ("assistant", "agent"):
            calls = m.get("tool_calls") or []
            last_tools = [str(tc.get("name") or "") for tc in calls]
            text = text or (_called_tools_text(calls) if calls else "")
            if text:
                dialogue.append({"role": "assistant", "content": text[:350]})
        elif role == "tool" and m.get("error") and not _only_lookups(last_tools):
            dialogue.append({"role": "tool_error", "content": clean_turn_text(m.get("content") or "tool error")[:220]})
    return user_turns, dialogue


def extract_reaction_exchanges(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect the customer's reactions to the agent (every user turn after the first one).

    The first user turn only states the problem, so it says nothing about the agent yet.
    Each later turn is paired with the agent message it reacts to. Action-tool errors are
    tracked too, because a failed tool call is friction even when the customer stays polite.
    """
    user_turns: List[str] = []
    reaction_turns: List[str] = []
    reaction_pairs: List[str] = []
    last_agent_text, last_tools, pending_errors = "", [], []
    early_error = trailing_error = ended_in_transfer = False

    for m in messages:
        role = (m.get("role") or "").lower()
        if role in ("assistant", "agent"):
            calls = m.get("tool_calls") or []
            last_tools = [str(tc.get("name") or "") for tc in calls]
            ended_in_transfer = ended_in_transfer or "transfer_to_human_agents" in last_tools
            text = clean_turn_text(m.get("content") or "") or (_called_tools_text(calls) if calls else "")
            if text:
                last_agent_text = text[:360]
        elif role == "tool":
            is_action = not last_tools or any(n not in READ_TOOLS for n in last_tools)
            if m.get("error") and is_action:
                pending_errors.append(clean_turn_text(m.get("content") or "tool error")[:180])
                trailing_error = True
            elif is_action:
                early_error = early_error or trailing_error or bool(pending_errors)
                trailing_error = False
        elif role == "user":
            raw = m.get("content") or ""
            ended_in_transfer = ended_in_transfer or "###TRANSFER###" in raw or "###OUT-OF-SCOPE###" in raw
            text = clean_turn_text(raw)
            if not text:
                continue
            if user_turns:
                context = last_agent_text
                if pending_errors:
                    early_error = True
                    context = f"{context} [TOOL ERROR: {pending_errors[-1]}]".strip()
                    pending_errors.clear()
                reaction_turns.append(text)
                reaction_pairs.append(f"ASSISTANT: {context[:380]}\nUSER REPLY: {text[:400]}")
                trailing_error = False
            user_turns.append(text)

    if pending_errors and not trailing_error:
        early_error = True
    return {"user_turns": user_turns, "reaction_turns": reaction_turns, "reaction_pairs": reaction_pairs,
            "early_tool_error": early_error, "final_tool_error": trailing_error, "ended_in_transfer": ended_in_transfer}


def format_shift_transcript(messages: Sequence[Dict[str, Any]], max_chars: int = 5200, include_tool_state: bool = False) -> str:
    """Render a compact transcript for the LLM detector, trimming the middle of very long chats."""
    lines: List[str] = []
    if not include_tool_state:
        _, dialogue = extract_clean_dialogue(messages)
        per_turn = max(90, min(320, max_chars // max(1, len(dialogue))))
        for idx, d in enumerate(dialogue):
            limit = max(per_turn, 260) if idx >= len(dialogue) - 2 else per_turn
            lines.append(f"{d['role'].upper()}: {d['content'][:limit]}")
    else:
        last_tools: List[str] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            text = clean_turn_text(m.get("content") or "")
            if role == "user" and text:
                lines.append(f"USER: {text[:360]}")
            elif role in ("assistant", "agent"):
                calls = m.get("tool_calls") or []
                last_tools = [str(tc.get("name") or "") for tc in calls]
                call_part = ""
                if calls:
                    call_part = " [calls: " + ", ".join(f"{tc.get('name')}({str(tc.get('arguments') or '')[:140]})" for tc in calls[:2]) + "]"
                if text or call_part:
                    lines.append(f"ASSISTANT{call_part}: {text[:280]}")
            elif role == "tool":
                if m.get("error") and not _only_lookups(last_tools):
                    lines.append(f"ACTION_TOOL_ERROR: {text[:240]}")
                elif text:
                    lines.append(f"{'LOOKUP_RESULT' if _only_lookups(last_tools) else 'ACTION_RESULT'}: {text[:240]}")
    if not lines:
        return ""
    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return joined
    head, tail = "\n".join(lines[:4]), "\n".join(lines[-10:])
    middle = "\n".join(lines[4:-10])[:max(400, max_chars - len(head) - len(tail) - 32)]
    return f"{head}\n...\n{middle}\n...\n{tail}"


def quadrant_from_flags(frustrated_earlier: bool, frustrated_at_end: bool) -> str:
    if frustrated_earlier:
        return "UNFIXED" if frustrated_at_end else "FIXED"
    return "BROKEN" if frustrated_at_end else "ROUTINE"


def _timed(fn, *args: Any) -> Tuple[str, float]:
    start = time.perf_counter()
    quadrant = fn(*args)
    return quadrant, (time.perf_counter() - start) * 1000.0


class VaderShiftDetector:
    """1. Word-dictionary sentiment: a reply is negative when VADER's compound score is <= -0.05."""

    name = "VADER"

    def __init__(self, neg_threshold: float = -0.05) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer, self.neg_threshold = SentimentIntensityAnalyzer(), neg_threshold

    def compound(self, text: str) -> float:
        return float(self.analyzer.polarity_scores(text)["compound"])

    def _quadrant(self, messages: Sequence[Dict[str, Any]]) -> str:
        ex = extract_reaction_exchanges(messages)
        turns = ex["reaction_turns"] or ex["user_turns"]
        if not turns:
            return "ROUTINE"
        if len(turns) == 1 and len(ex["user_turns"]) < 2:
            return "BROKEN" if self.compound(turns[0]) <= self.neg_threshold else "ROUTINE"
        negative = [self.compound(t) <= self.neg_threshold for t in turns]
        return quadrant_from_flags(any(negative[:-1]), negative[-1])

    def classify(self, messages: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
        return _timed(self._quadrant, messages)


class GoEmotionsShiftDetector:
    """2. Neural 28-emotion classifier (RoBERTa). A reply is negative when negative emotions sum >= 0.25."""

    name = "GoEmotions"

    def __init__(self, neg_threshold: float = 0.25, resolved_ceiling: float = 0.25) -> None:
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=-1)
        self.neg_threshold, self.resolved_ceiling = neg_threshold, resolved_ceiling

    def _negative_scores(self, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        preds = self.pipe([t[:512] for t in texts], truncation=True, max_length=512, batch_size=32)
        return [sum(float(p["score"]) for p in plist if p["label"] in _NEGATIVE_EMOTIONS) for plist in preds]

    def classify_many(self, chats: Sequence[Sequence[Dict[str, Any]]]) -> List[str]:
        """Score every reply of every chat in one batch, then place each chat in a box."""
        exchanges = [extract_reaction_exchanges(m) for m in chats]
        texts = [ex["reaction_turns"] or ex["user_turns"] for ex in exchanges]
        flat_scores = self._negative_scores([t for chat in texts for t in chat])
        out, pos = [], 0
        for ex, turns in zip(exchanges, texts):
            scores, pos = flat_scores[pos:pos + len(turns)], pos + len(turns)
            if not scores:
                out.append("ROUTINE")
            elif len(scores) == 1 and len(ex["user_turns"]) < 2:
                out.append("BROKEN" if scores[0] >= self.neg_threshold else "ROUTINE")
            else:
                out.append(quadrant_from_flags(any(s >= self.neg_threshold for s in scores[:-1]), scores[-1] >= self.resolved_ceiling))
        return out

    def classify(self, messages: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
        return _timed(lambda m: self.classify_many([m])[0], messages)


class LayaShiftDetector:
    """3. Laya (open-weights ModernBERT, runs on a laptop CPU) reads each (agent message, customer reply) pair."""

    name = "Laya"
    _QUESTION = {
        "state": {
            "type": "choice",
            "instructions": "Classify the USER REPLY to the ASSISTANT's previous response:",
            "criteria": {
                "dissatisfied": ("User points out a mistake, incomplete or wrong output, bug, "
                                 "misunderstanding, or frustration with the assistant's previous answer."),
                "neutral": ("User continues the conversation, gives the next prompt or story step, "
                            "or asks a calm follow-up question without complaining about an error."),
                "satisfied": ("User thanks the assistant, confirms the answer solved the issue, "
                              "or expresses satisfaction."),
            },
        }
    }

    def __init__(self, friction_thresh: float = 0.40, unresolved_thresh: float = 0.40) -> None:
        import laya
        import torch
        torch.set_num_threads(min(16, os.cpu_count() or 8))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.agent = laya.load("convaiinnovations/laya")
        self.friction_thresh, self.unresolved_thresh = friction_thresh, unresolved_thresh

    def score_pairs(self, pairs: Sequence[str], batch_size: int = 16) -> List[float]:
        """Probability that each customer reply is 'dissatisfied' with the agent message before it."""
        if not pairs:
            return []
        res = self.agent.predict_batch(list(pairs), self._QUESTION, batch_size=batch_size, sort_by_length=len(pairs) > batch_size)
        return [float(r["answers"]["state"]["probabilities"]["dissatisfied"]) for r in res]

    def classify_many(self, chats: Sequence[Sequence[Dict[str, Any]]]) -> List[str]:
        exchanges = [extract_reaction_exchanges(m) for m in chats]
        flat_probs = self.score_pairs([p for ex in exchanges for p in ex["reaction_pairs"]])
        out, pos = [], 0
        for ex in exchanges:
            probs, pos = flat_probs[pos:pos + len(ex["reaction_pairs"])], pos + len(ex["reaction_pairs"])
            if not probs:  # no replies to score: fall back to tool errors alone
                if ex["final_tool_error"]:
                    out.append("UNFIXED" if ex["early_tool_error"] else "BROKEN")
                elif ex["early_tool_error"]:
                    out.append("UNFIXED" if ex["ended_in_transfer"] else "FIXED")
                else:
                    out.append("ROUTINE")
                continue
            earlier = any(p >= self.friction_thresh for p in probs[:-1]) or ex["early_tool_error"]
            at_end = probs[-1] >= self.unresolved_thresh or ex["final_tool_error"] or (earlier and ex["ended_in_transfer"])
            out.append(quadrant_from_flags(earlier, at_end))
        return out

    def classify(self, messages: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
        return _timed(lambda m: self.classify_many([m])[0], messages)


class FlashLiteShiftDetector:
    """4. Gemini 3.5 Flash-Lite reads the whole chat (and the store policy, when given) and picks a box."""

    name = "Flash-Lite"
    _SYS = (
        "Classify the user-assistant conversation trajectory into exactly one quadrant JSON:\n"
        '{"quadrant": "FIXED" | "UNFIXED" | "BROKEN" | "ROUTINE"}\n'
        "Definitions:\n"
        "- FIXED: Mid-conversation (after the opening request), an action tool error occurred OR the "
        "user pushed back, pointed out a mistake/incomplete answer/bug, or hit a refusal/constraint "
        "obstacle, AND by the final turn the assistant resolved the issue into an accepted solution "
        "(without transferring to a human agent or issuing an ad-hoc goodwill exception).\n"
        "- UNFIXED: Mid-conversation friction (action tool error or user pushback/dissatisfaction/dispute) "
        "occurred, AND the conversation ended unresolved, abandoned, escalated to a human agent, "
        "dissatisfied, or appeased only via an ad-hoc goodwill exception.\n"
        "- BROKEN: Earlier turns were smooth, and the final turn ended in an unresolved tool error, "
        "wrong/incomplete final answer, or user dissatisfaction.\n"
        "- ROUTINE: Smooth interaction from start to finish (including routine lookups or simple user "
        "preference selection where the assistant made no mistake) with zero user frustration and zero "
        "action-tool errors."
    )
    _SYS_WITH_POLICY = (
        "Classify the customer-service conversation trajectory into one quadrant JSON:\n"
        '{"quadrant": "FIXED" | "UNFIXED" | "BROKEN" | "ROUTINE"}\n'
        "Evaluate both user reactions and strict compliance with DOMAIN POLICY:\n"
        "- FIXED: Mid-conversation (after the opening request), EITHER (a) an action-tool error "
        "occurred and the assistant recovered with a policy-compliant solution, OR (b) the user "
        "pushed back on a policy constraint or requested a disallowed action/refund/modification "
        "and the assistant upheld DOMAIN POLICY (explaining the valid constraint or alternative "
        "without an unauthorized mutation or premature human transfer) and the user accepted by "
        "the end. Never classify routine ID/email/name+zip lookups as FIXED.\n"
        "- UNFIXED: Mid-conversation friction (action-tool error or user pushback/dispute) occurred, "
        "AND the chat ended unresolved, escalated via transfer_to_human_agents, or appeased via "
        "an action that violates DOMAIN POLICY (such as unauthorized delay compensation, "
        "disallowed Basic Economy changes/cancellations, or unconfirmed state mutations).\n"
        "- BROKEN: Earlier turns were calm, and the session ended in an unresolved tool error, "
        "user dissatisfaction, or a silent DOMAIN POLICY violation.\n"
        "- ROUTINE: Smooth policy-compliant interaction from start to finish (including standard "
        "account lookups by name/zip or user option selection) with no action-tool errors and no "
        "policy pushback."
    )

    def __init__(self, llm: Optional[LLMClient] = None, model: str = "gemini-3.5-flash-lite") -> None:
        self.llm, self.model = llm or LLMClient(), model

    def classify(self, messages: Sequence[Dict[str, Any]], domain_policy: str = "") -> Tuple[str, float]:
        start = time.perf_counter()
        has_tools = any((m.get("role") or "").lower() == "tool" for m in messages)
        transcript = format_shift_transcript(messages, max_chars=5600 if has_tools else 4800, include_tool_state=has_tools)
        if not transcript.strip():
            return "ROUTINE", (time.perf_counter() - start) * 1000.0
        if domain_policy.strip():
            prompt, system = f"=== DOMAIN POLICY ===\n{domain_policy[:9500]}\n\n=== TRAJECTORY ===\n{transcript}", self._SYS_WITH_POLICY
        else:
            prompt, system = transcript, self._SYS
        raw = self.llm.generate_text(prompt=prompt, system_instruction=system, model=self.model, json_mode=True,
                                     temperature=0.0, max_output_tokens=48)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        quadrant = str(parse_json_object(raw).get("quadrant", "ROUTINE")).upper().strip()
        quadrant = quadrant if quadrant in QUADRANTS else "ROUTINE"
        # Hard evidence from tool logs overrides the LLM's reading of the conversation.
        ex = extract_reaction_exchanges(messages)
        if ex["final_tool_error"] and quadrant in ("ROUTINE", "FIXED"):
            quadrant = "UNFIXED" if ex["early_tool_error"] else "BROKEN"
        elif ex["ended_in_transfer"] and quadrant == "FIXED":
            quadrant = "UNFIXED"
        elif ex["early_tool_error"] and quadrant == "ROUTINE" and not ex["ended_in_transfer"] and not domain_policy.strip():
            quadrant = "FIXED"
        return quadrant, elapsed_ms

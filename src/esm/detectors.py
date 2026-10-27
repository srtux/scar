"""Four Emotion-Shift Gates: VADER, GoEmotions, Laya (ModernBERT), and Gemini Flash-Lite.

Every detector maps a multi-turn conversation into one of the four memory quadrants:
  - "FIXED":    User hit friction/dissatisfaction earlier, then shifted to resolved/satisfied by the end.
  - "UNFIXED":  User hit friction/dissatisfaction earlier and stayed unresolved/dissatisfied at the end.
  - "BROKEN":   Started calm, ended with dissatisfaction/failure on the final turn.
  - "ROUTINE":  Smooth conversation without friction.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scar.llm import LLMClient, parse_json_object

_SIM_TOKEN_RE = re.compile(r"###(STOP|TRANSFER|OUT-OF-SCOPE)###")

_READ_LOOKUP_TOOLS = frozenset(
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

_NEG_EMOTIONS = frozenset(
    {
        "anger",
        "annoyance",
        "confusion",
        "disappointment",
        "disapproval",
        "disgust",
        "fear",
        "grief",
        "nervousness",
        "remorse",
        "sadness",
    }
)


def clean_turn_text(text: str) -> str:
    """Strip simulator control tokens and trim whitespace."""
    return _SIM_TOKEN_RE.sub("", text or "").strip()


def extract_clean_dialogue(messages: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return `(user_turns, compact_dialogue)` with optional compact tool error notes."""
    user_turns: List[str] = []
    dialogue: List[Dict[str, str]] = []
    last_tool_names: List[str] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        if role == "user":
            txt = clean_turn_text(m.get("content") or "")
            if txt:
                user_turns.append(txt)
                dialogue.append({"role": "user", "content": txt[:500]})
        elif role in ("assistant", "agent"):
            txt = clean_turn_text(m.get("content") or "")
            tcs = m.get("tool_calls") or []
            last_tool_names = [str(tc.get("name") or "") for tc in tcs]
            if not txt and tcs:
                names = ", ".join(tc.get("name", "tool") for tc in tcs[:2])
                txt = f"[called {names}]"
            if txt:
                dialogue.append({"role": "assistant", "content": txt[:350]})
        elif role == "tool" and m.get("error"):
            is_lookup = bool(last_tool_names) and all(
                n in _READ_LOOKUP_TOOLS for n in last_tool_names
            )
            if not is_lookup:
                err = clean_turn_text(m.get("content") or "tool error")[:220]
                dialogue.append({"role": "tool_error", "content": err})
    return user_turns, dialogue


def extract_reaction_exchanges(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract reaction turns `u_1..u_{k-1}` paired with preceding assistant/tool context.

    Opening prompts `u_0` describe the user's initial problem (e.g., pasting a stack trace)
    before the assistant has acted. Scoring reaction turns `u_1..u_{k-1}` isolates user
    feedback on the assistant's behavior while also tracking explicit tool execution errors.
    """
    user_turns: List[str] = []
    reaction_turns: List[str] = []
    reaction_pairs: List[str] = []
    last_assistant = ""
    last_tool_names: List[str] = []
    pending_tool_errs: List[str] = []
    early_tool_error = False
    trailing_tool_error = False
    saw_clean_after_err = False
    ended_in_transfer = False

    for m in messages:
        role = (m.get("role") or "").lower()
        if role in ("assistant", "agent"):
            txt = clean_turn_text(m.get("content") or "")
            tcs = m.get("tool_calls") or []
            last_tool_names = [str(tc.get("name") or "") for tc in tcs]
            if any(name == "transfer_to_human_agents" for name in last_tool_names):
                ended_in_transfer = True
            if not txt and tcs:
                names = ", ".join(tc.get("name", "tool") for tc in tcs[:2])
                txt = f"[called {names}]"
            if txt:
                last_assistant = txt[:360]
        elif role == "tool":
            is_action_call = not last_tool_names or any(
                n not in _READ_LOOKUP_TOOLS for n in last_tool_names
            )
            if m.get("error") and is_action_call:
                err = clean_turn_text(m.get("content") or "tool error")[:180]
                pending_tool_errs.append(err)
                trailing_tool_error = True
            elif is_action_call:
                if trailing_tool_error or pending_tool_errs:
                    early_tool_error = True
                    saw_clean_after_err = True
                trailing_tool_error = False
        elif role == "user":
            raw_u = m.get("content") or ""
            if "###TRANSFER###" in raw_u or "###OUT-OF-SCOPE###" in raw_u:
                ended_in_transfer = True
            txt = clean_turn_text(raw_u)
            if not txt:
                continue
            if len(user_turns) >= 1:
                ctx = last_assistant
                if pending_tool_errs:
                    early_tool_error = True
                    ctx = f"{ctx} [TOOL ERROR: {pending_tool_errs[-1]}]".strip()
                    pending_tool_errs.clear()
                reaction_turns.append(txt)
                reaction_pairs.append(f"ASSISTANT: {ctx[:380]}\nUSER REPLY: {txt[:400]}")
                trailing_tool_error = False
            user_turns.append(txt)

    if pending_tool_errs and not trailing_tool_error:
        early_tool_error = True

    return {
        "user_turns": user_turns,
        "reaction_turns": reaction_turns,
        "reaction_pairs": reaction_pairs,
        "early_tool_error": early_tool_error or saw_clean_after_err,
        "final_tool_error": trailing_tool_error,
        "ended_in_transfer": ended_in_transfer,
    }


def format_shift_transcript(
    messages: Sequence[Dict[str, Any]],
    max_chars: int = 5200,
    include_tool_state: bool = False,
) -> str:
    """Format all turns compactly while preserving user reaction turns and optional tool state."""
    if not include_tool_state:
        _, dialogue = extract_clean_dialogue(messages)
        if not dialogue:
            return ""
        n = len(dialogue)
        per_turn = max(90, min(320, max_chars // max(1, n)))
        lines: List[str] = []
        for idx, d in enumerate(dialogue):
            role = d["role"].upper()
            limit = max(per_turn, 260) if idx >= n - 2 else per_turn
            lines.append(f"{role}: {d['content'][:limit]}")
    else:
        lines = []
        last_tool_names: List[str] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            if role == "user":
                txt = clean_turn_text(m.get("content") or "")
                if txt:
                    lines.append(f"USER: {txt[:360]}")
            elif role in ("assistant", "agent"):
                txt = clean_turn_text(m.get("content") or "")
                tcs = m.get("tool_calls") or []
                last_tool_names = [str(tc.get("name") or "") for tc in tcs]
                tc_part = ""
                if tcs:
                    calls_s = ", ".join(
                        f"{tc.get('name')}({str(tc.get('arguments') or '')[:140]})"
                        for tc in tcs[:2]
                    )
                    tc_part = f" [calls: {calls_s}]"
                if txt or tc_part:
                    lines.append(f"ASSISTANT{tc_part}: {txt[:280]}")
            elif role == "tool":
                out = clean_turn_text(m.get("content") or "")
                is_lookup = bool(last_tool_names) and all(
                    n in _READ_LOOKUP_TOOLS for n in last_tool_names
                )
                if m.get("error") and not is_lookup:
                    lines.append(f"ACTION_TOOL_ERROR: {out[:240]}")
                elif out:
                    tag = "LOOKUP_RESULT" if is_lookup else "ACTION_RESULT"
                    lines.append(f"{tag}: {out[:240]}")
        if not lines:
            return ""

    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return joined
    head = "\n".join(lines[:4])
    tail = "\n".join(lines[-10:])
    mid_budget = max(400, max_chars - len(head) - len(tail) - 32)
    mid = "\n".join(lines[4:-10])[:mid_budget]
    return f"{head}\n...\n{mid}\n...\n{tail}"


def _quadrant_from_flags(early_neg: bool, final_neg: bool) -> str:
    if early_neg and not final_neg:
        return "FIXED"
    if early_neg and final_neg:
        return "UNFIXED"
    if not early_neg and final_neg:
        return "BROKEN"
    return "ROUTINE"


class VaderShiftDetector:
    """1. Dictionary sentiment shift detector (VADER, < 0.2 ms per turn on CPU)."""

    name = "VADER"

    def __init__(self, neg_threshold: float = -0.05) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()
        self.neg_threshold = neg_threshold

    def classify(self, messages: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
        t0 = time.perf_counter()
        ex = extract_reaction_exchanges(messages)
        turns = ex["reaction_turns"] if ex["reaction_turns"] else ex["user_turns"]
        if not turns:
            return "ROUTINE", (time.perf_counter() - t0) * 1000.0
        if len(turns) == 1 and len(ex["user_turns"]) < 2:
            score = self.analyzer.polarity_scores(turns[0])["compound"]
            return ("BROKEN" if score <= self.neg_threshold else "ROUTINE"), (
                time.perf_counter() - t0
            ) * 1000.0

        flags = [self.analyzer.polarity_scores(u)["compound"] <= self.neg_threshold for u in turns]
        quad = _quadrant_from_flags(any(flags[:-1]), flags[-1])
        return quad, (time.perf_counter() - t0) * 1000.0


class GoEmotionsShiftDetector:
    """2. Neural 28-emotion turn classifier (`SamLowe/roberta-base-go_emotions`, ~18 ms/turn CPU)."""

    name = "GoEmotions"

    def __init__(self, neg_threshold: float = 0.25, resolved_ceiling: float = 0.25) -> None:
        from transformers import pipeline

        self.pipe = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            device=-1,
        )
        self.neg_threshold = neg_threshold
        self.resolved_ceiling = resolved_ceiling

    def _neg_scores_batch(self, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        clipped = [t[:512] for t in texts]
        preds_list = self.pipe(clipped, truncation=True, max_length=512, batch_size=32)
        return [
            sum(float(p["score"]) for p in preds if p["label"] in _NEG_EMOTIONS)
            for preds in preds_list
        ]

    def classify(self, messages: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
        t0 = time.perf_counter()
        ex = extract_reaction_exchanges(messages)
        turns = ex["reaction_turns"] if ex["reaction_turns"] else ex["user_turns"]
        if not turns:
            return "ROUTINE", (time.perf_counter() - t0) * 1000.0
        scores = self._neg_scores_batch(turns)
        if len(turns) == 1 and len(ex["user_turns"]) < 2:
            quad = "BROKEN" if scores[0] >= self.neg_threshold else "ROUTINE"
            return quad, (time.perf_counter() - t0) * 1000.0

        early_neg = any(s >= self.neg_threshold for s in scores[:-1])
        final_neg = scores[-1] >= self.resolved_ceiling
        quad = _quadrant_from_flags(early_neg, final_neg)
        return quad, (time.perf_counter() - t0) * 1000.0

    def classify_many(self, sessions: Sequence[Sequence[Dict[str, Any]]]) -> List[str]:
        """Batch-classify conversations identically to `classify` for fast benchmark runs."""
        flat_turns: List[str] = []
        spans: List[Tuple[int, int, int]] = []
        for msgs in sessions:
            ex = extract_reaction_exchanges(msgs)
            turns = ex["reaction_turns"] if ex["reaction_turns"] else ex["user_turns"]
            s = len(flat_turns)
            flat_turns.extend(turns)
            spans.append((s, len(flat_turns), len(ex["user_turns"])))
        all_scores = self._neg_scores_batch(flat_turns)
        out: List[str] = []
        for s, e, n_u in spans:
            sc = all_scores[s:e]
            if not sc:
                out.append("ROUTINE")
            elif len(sc) == 1 and n_u < 2:
                out.append("BROKEN" if sc[0] >= self.neg_threshold else "ROUTINE")
            else:
                out.append(
                    _quadrant_from_flags(
                        any(x >= self.neg_threshold for x in sc[:-1]),
                        sc[-1] >= self.resolved_ceiling,
                    )
                )
        return out


class LayaShiftDetector:
    """3. Open-weights ModernBERT turn-pair shift classifier (`convaiinnovations/laya`, CPU)."""

    name = "Laya"

    _QUESTION = {
        "state": {
            "type": "choice",
            "instructions": "Classify the USER REPLY to the ASSISTANT's previous response:",
            "criteria": {
                "dissatisfied": (
                    "User points out a mistake, incomplete or wrong output, bug, "
                    "misunderstanding, or frustration with the assistant's previous answer."
                ),
                "neutral": (
                    "User continues the conversation, gives the next prompt or story step, "
                    "or asks a calm follow-up question without complaining about an error."
                ),
                "satisfied": (
                    "User thanks the assistant, confirms the answer solved the issue, "
                    "or expresses satisfaction."
                ),
            },
        }
    }

    def __init__(self, friction_thresh: float = 0.40, unresolved_thresh: float = 0.40) -> None:
        import torch
        import laya

        torch.set_num_threads(min(16, os.cpu_count() or 8))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.agent = laya.load("convaiinnovations/laya")
        self.friction_thresh = friction_thresh
        self.unresolved_thresh = unresolved_thresh

    def _score_pairs(self, pairs: Sequence[str], batch_size: int = 16) -> List[float]:
        if not pairs:
            return []
        use_sort = len(pairs) > batch_size
        res = self.agent.predict_batch(
            list(pairs),
            self._QUESTION,
            batch_size=batch_size,
            sort_by_length=use_sort,
        )
        return [
            float(r["answers"]["state"]["probabilities"]["dissatisfied"])
            for r in res
        ]

    def classify(self, messages: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
        t0 = time.perf_counter()
        ex = extract_reaction_exchanges(messages)
        pairs = ex["reaction_pairs"]
        if not pairs:
            if ex["final_tool_error"]:
                quad = "UNFIXED" if ex["early_tool_error"] else "BROKEN"
            elif ex["early_tool_error"]:
                quad = "FIXED"
            else:
                quad = "ROUTINE"
            return quad, (time.perf_counter() - t0) * 1000.0

        dsat_probs = self._score_pairs(pairs)
        early_neg = any(p >= self.friction_thresh for p in dsat_probs[:-1]) or ex["early_tool_error"]
        final_neg = (
            (dsat_probs[-1] >= self.unresolved_thresh)
            or ex["final_tool_error"]
            or (early_neg and ex["ended_in_transfer"])
        )
        quad = _quadrant_from_flags(early_neg, final_neg)
        return quad, (time.perf_counter() - t0) * 1000.0

    def classify_many(self, sessions: Sequence[Sequence[Dict[str, Any]]]) -> List[str]:
        """Batch-classify conversations identically to `classify` for fast benchmark runs."""
        extracted = [extract_reaction_exchanges(msgs) for msgs in sessions]
        flat_pairs: List[str] = []
        spans: List[Tuple[int, int]] = []
        for ex in extracted:
            s = len(flat_pairs)
            flat_pairs.extend(ex["reaction_pairs"])
            spans.append((s, len(flat_pairs)))
        all_probs = self._score_pairs(flat_pairs)
        out: List[str] = []
        for ex, (s, e) in zip(extracted, spans):
            probs = all_probs[s:e]
            if not probs:
                if ex["final_tool_error"]:
                    out.append("UNFIXED" if ex["early_tool_error"] else "BROKEN")
                elif ex["early_tool_error"]:
                    out.append("UNFIXED" if ex["ended_in_transfer"] else "FIXED")
                else:
                    out.append("ROUTINE")
            else:
                early_neg = any(p >= self.friction_thresh for p in probs[:-1]) or ex["early_tool_error"]
                final_neg = (
                    (probs[-1] >= self.unresolved_thresh)
                    or ex["final_tool_error"]
                    or (early_neg and ex["ended_in_transfer"])
                )
                out.append(_quadrant_from_flags(early_neg, final_neg))
        return out


class FlashLiteShiftDetector:
    """4. Context-aware LLM shift classifier (`gemini-3.5-flash-lite`)."""

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
        self.llm = llm or LLMClient()
        self.model = model

    def classify(
        self,
        messages: Sequence[Dict[str, Any]],
        include_tool_state: Optional[bool] = None,
        domain_policy: str = "",
    ) -> Tuple[str, float]:
        t0 = time.perf_counter()
        has_tools = (
            include_tool_state
            if include_tool_state is not None
            else any((m.get("role") or "").lower() == "tool" for m in messages)
        )
        transcript = format_shift_transcript(
            messages,
            max_chars=5600 if has_tools else 4800,
            include_tool_state=has_tools,
        )
        if not transcript.strip():
            return "ROUTINE", (time.perf_counter() - t0) * 1000.0
        if domain_policy.strip():
            prompt = (
                f"=== DOMAIN POLICY ===\n{domain_policy[:9500]}\n\n"
                f"=== TRAJECTORY ===\n{transcript}"
            )
            sys_inst = self._SYS_WITH_POLICY
        else:
            prompt = transcript
            sys_inst = self._SYS
        raw = self.llm.generate_text(
            prompt=prompt,
            system_instruction=sys_inst,
            model=self.model,
            json_mode=True,
            temperature=0.0,
            max_output_tokens=48,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        quad = str(parse_json_object(raw).get("quadrant", "ROUTINE")).upper().strip()
        if quad not in ("FIXED", "UNFIXED", "BROKEN", "ROUTINE"):
            quad = "ROUTINE"
        ex = extract_reaction_exchanges(messages)
        if ex["final_tool_error"] and quad in ("ROUTINE", "FIXED"):
            quad = "UNFIXED" if ex["early_tool_error"] else "BROKEN"
        elif ex["ended_in_transfer"] and quad == "FIXED":
            quad = "UNFIXED"
        elif ex["early_tool_error"] and quad == "ROUTINE" and not ex["ended_in_transfer"] and not domain_policy.strip():
            quad = "FIXED"
        return quad, elapsed_ms

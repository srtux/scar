"""Genuine LLM + Embedding Memory Baselines for Head-to-Head Comparison with SCAR.

All baselines enforce `strip_oracle_fields` on ingestion and guarantee
`count_tokens(retrieved_context) <= max_tokens` at retrieval time.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

from scar.consolidator import strip_oracle_fields
from scar.detector import (
    StaticSentimentDetector,
    ToolOutcomeDetector,
    format_trajectory_for_detector,
)
from scar.llm import LLMClient, cosine_similarity
from scar.tokens import count_tokens, pack_items_within_budget, truncate_to_token_budget


@dataclasses.dataclass
class EmbeddedEntry:
    text: str
    score_weight: float = 1.0
    domain: str = ""
    embedding: Optional[List[float]] = None


class _VectorBackedMemory:
    """Shared semantic retrieval helper using `gemini-embedding-001` cosine similarity."""

    def __init__(self, header: str):
        self.header = header
        self.entries: List[EmbeddedEntry] = []

    def _embed_pending(self, llm: LLMClient) -> None:
        pending = [e for e in self.entries if e.embedding is None]
        if pending:
            vecs = llm.embed_texts([e.text for e in pending])
            for e, v in zip(pending, vecs):
                e.embedding = v

    def retrieve_context(
        self,
        query: str = "",
        max_tokens: Optional[int] = None,
        llm: Optional[LLMClient] = None,
        domain_filter: Optional[str] = None,
    ) -> str:
        if max_tokens is not None and max_tokens <= 0:
            return ""
        pool = [
            e for e in self.entries if (not domain_filter or not e.domain or e.domain == domain_filter)
        ]
        if not pool:
            return ""

        if query and llm is not None:
            self._embed_pending(llm)
            q_vec = llm.embed_texts([query])[0]
            scored = [
                (0.75 * cosine_similarity(q_vec, e.embedding or []) + 0.25 * e.score_weight, e.text)
                for e in pool
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            ordered_texts = [t for _, t in scored]
        else:
            ordered_texts = [e.text for e in reversed(pool)]

        budget = max_tokens if max_tokens is not None else 100_000
        packed = pack_items_within_budget(ordered_texts, max_tokens=budget, header=self.header)
        if not packed and ordered_texts and budget >= 12:
            packed = truncate_to_token_budget(f"{self.header}\n{ordered_texts[0]}", budget)
        return packed


class FIFOMemory:
    """Store-All FIFO recency window over raw chronological session transcripts."""

    def __init__(self):
        self.session_blocks: List[str] = []

    def ingest_session(self, session: Dict[str, Any], domain: str = "", **_: Any) -> None:
        clean = strip_oracle_fields(session)
        sid = clean.get("session_id") or clean.get("id") or "session"
        transcript = format_trajectory_for_detector(clean.get("messages") or [], max_chars_per_turn=350)
        self.session_blocks.append(f"[Session {sid} ({domain})]\n{transcript}")

    def retrieve_context(
        self,
        query: str = "",
        max_tokens: Optional[int] = None,
        llm: Optional[LLMClient] = None,
        domain_filter: Optional[str] = None,
    ) -> str:
        del query, llm
        blocks = [
            b for b in self.session_blocks if (not domain_filter or f"({domain_filter})" in b)
        ]
        raw = "\n\n".join(blocks)
        if max_tokens is None:
            return raw
        return truncate_to_token_budget(raw, max_tokens=max_tokens, keep_tail=True)


_ROLLING_SUMMARY_SYS = """Update the agent's running long-term memory summary by incorporating salient facts,
user preferences, and operational lessons from the new session. Keep the updated summary under 350 words."""


class RollingSummaryMemory:
    """Iterative LLM session summarization baseline."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.summary: str = ""

    def ingest_session(self, session: Dict[str, Any], domain: str = "", **_: Any) -> None:
        clean = strip_oracle_fields(session)
        transcript = format_trajectory_for_detector(clean.get("messages") or [], max_chars_per_turn=400)
        prompt = (
            f"=== PREVIOUS SUMMARY ===\n{self.summary or '(empty)'}\n\n"
            f"=== NEW SESSION ({domain}) ===\n{transcript}\n\n"
            "Write the updated consolidated memory summary:"
        )
        self.summary = self.llm.generate_text(
            prompt=prompt,
            system_instruction=_ROLLING_SUMMARY_SYS,
            model=self.llm.fast_model,
            max_output_tokens=600,
        )

    def retrieve_context(
        self,
        query: str = "",
        max_tokens: Optional[int] = None,
        llm: Optional[LLMClient] = None,
        domain_filter: Optional[str] = None,
    ) -> str:
        del query, llm, domain_filter
        if not self.summary:
            return ""
        text = f"Consolidated Summary:\n{self.summary}"
        if max_tokens is None:
            return text
        return truncate_to_token_budget(text, max_tokens=max_tokens, keep_tail=False)


_ATOMIC_FACTS_SYS = """Extract concise, self-contained atomic facts and observations from this interaction
(similar to Mem0 atomic fact extraction). Capture user facts, preferences, entities, and what the agent did.
Return strict JSON:
{
  "facts": ["concise atomic statement 1", "concise atomic statement 2"]
}"""


class AtomicFactsMemory(_VectorBackedMemory):
    """Mem0-style flat atomic fact extraction with vector similarity retrieval."""

    def __init__(self, llm: LLMClient):
        super().__init__(header="Retrieved Atomic Facts:")
        self.llm = llm

    def ingest_session(self, session: Dict[str, Any], domain: str = "", **_: Any) -> None:
        clean = strip_oracle_fields(session)
        transcript = format_trajectory_for_detector(clean.get("messages") or [], max_chars_per_turn=400)
        res = self.llm.generate_json(
            prompt=f"SESSION ({domain}):\n{transcript}",
            system_instruction=_ATOMIC_FACTS_SYS,
            model=self.llm.fast_model,
            max_output_tokens=500,
        )
        for fact in (res.get("facts") or [])[:8]:
            if isinstance(fact, str) and fact.strip():
                self.entries.append(EmbeddedEntry(text=f"- {fact.strip()}", domain=domain))


_IMPORTANCE_SYS = """Score the most noteworthy individual turns or takeaways from this session on single-turn
importance from 1 to 10 (Generative Agents style: 1 = mundane chitchat, 10 = critical event or strong statement).
Return up to 4 items with score >= 6 as strict JSON:
{
  "memories": [
    {"text": "concise memory statement", "importance": 8}
  ]
}"""


class SingleTurnImportanceMemory(_VectorBackedMemory):
    """Generative-Agents-style single-turn importance rating + semantic retrieval."""

    def __init__(self, llm: LLMClient):
        super().__init__(header="High-Importance Memories:")
        self.llm = llm

    def ingest_session(self, session: Dict[str, Any], domain: str = "", **_: Any) -> None:
        clean = strip_oracle_fields(session)
        transcript = format_trajectory_for_detector(clean.get("messages") or [], max_chars_per_turn=400)
        res = self.llm.generate_json(
            prompt=f"SESSION ({domain}):\n{transcript}",
            system_instruction=_IMPORTANCE_SYS,
            model=self.llm.fast_model,
            max_output_tokens=450,
        )
        for item in (res.get("memories") or [])[:5]:
            if isinstance(item, dict) and item.get("text"):
                imp = float(item.get("importance") or 6.0)
                if imp >= 6.0:
                    self.entries.append(
                        EmbeddedEntry(
                            text=f"- {str(item['text']).strip()}",
                            score_weight=min(1.0, imp / 10.0),
                            domain=domain,
                        )
                    )


class StaticSentimentMemory(_VectorBackedMemory):
    """Affective baseline that retains episodes triggered by strong negative user sentiment/complaints."""

    def __init__(self, llm: LLMClient):
        super().__init__(header="Emotion-Triggered Memories:")
        self.llm = llm
        self.sentiment = StaticSentimentDetector()

    def ingest_session(self, session: Dict[str, Any], domain: str = "", **_: Any) -> None:
        clean = strip_oracle_fields(session)
        msgs: Sequence[Dict[str, Any]] = clean.get("messages") or []
        if not self.sentiment.evaluate_messages(msgs):
            return
        # Retains compressed notes from high-sentiment episodes regardless of whether resolved
        transcript = format_trajectory_for_detector(msgs, max_chars_per_turn=300)
        note = self.llm.generate_text(
            prompt=f"Summarize what the upset user complained about and what happened in 2 bullet points:\n{transcript}",
            model=self.llm.fast_model,
            max_output_tokens=200,
        )
        for line in note.splitlines():
            clean_line = line.strip().lstrip("-* ").strip()
            if clean_line:
                self.entries.append(EmbeddedEntry(text=f"- {clean_line}", domain=domain))


_TOOL_LESSON_SYS = """A tool execution error occurred in this session. Extract 1-2 concise troubleshooting notes
about avoiding or fixing that tool error. Do NOT mention specific user/order IDs.
Return strict JSON:
{
  "notes": ["concise tool error lesson"]
}"""


class ToolOutcomeOnlyMemory(_VectorBackedMemory):
    """Triggers memory consolidation ONLY when explicit tool execution errors (`error=True`) occur."""

    def __init__(self, llm: LLMClient):
        super().__init__(header="Tool-Error Lessons:")
        self.llm = llm
        self.tool_det = ToolOutcomeDetector()

    def ingest_session(self, session: Dict[str, Any], domain: str = "", **_: Any) -> None:
        clean = strip_oracle_fields(session)
        msgs: Sequence[Dict[str, Any]] = clean.get("messages") or []
        if not self.tool_det.evaluate_messages(msgs):
            return
        transcript = format_trajectory_for_detector(msgs, max_chars_per_turn=350)
        res = self.llm.generate_json(
            prompt=f"SESSION WITH TOOL ERROR ({domain}):\n{transcript}",
            system_instruction=_TOOL_LESSON_SYS,
            model=self.llm.fast_model,
            max_output_tokens=250,
        )
        for n in (res.get("notes") or [])[:3]:
            if isinstance(n, str) and n.strip():
                self.entries.append(EmbeddedEntry(text=f"- {n.strip()}", domain=domain))


class RawTrajectoryRAGMemory(_VectorBackedMemory):
    """Retrieves top semantically matching raw historical episode snippets (unquarantined)."""

    def __init__(self):
        super().__init__(header="Similar Past Episodes:")

    def ingest_session(self, session: Dict[str, Any], domain: str = "", **_: Any) -> None:
        clean = strip_oracle_fields(session)
        msgs = clean.get("messages") or []
        snippet = format_trajectory_for_detector(msgs, max_chars_per_turn=200)
        trimmed = truncate_to_token_budget(snippet, max_tokens=180, keep_tail=True)
        if trimmed:
            self.entries.append(EmbeddedEntry(text=f"[Past Episode]\n{trimmed}", domain=domain))

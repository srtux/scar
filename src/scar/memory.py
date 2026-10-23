"""SCAR Dual-Store Memory Architecture with Semantic Retrieval & Strict Token Budgets.

Implements:
1. KeyedDirectives: Calm facts and standing preferences with canonical key normalization
   and automatic in-place supersession when preferences change.
2. ContrastivePlaybookCards: Verified procedural troubleshooting rules (`Trigger`, `Avoid`, `Instead`)
   synthesized ONLY from resolved struggles (quarantining unresolved failures).
"""

from __future__ import annotations

import dataclasses
import datetime
import re
from typing import Dict, List, Optional, Sequence, Tuple

from scar.llm import LLMClient, cosine_similarity
from scar.tokens import count_tokens, truncate_to_token_budget


def normalize_directive_key(raw_key: str) -> str:
    """Canonicalize preference/fact keys so updates overwrite stale entries reliably."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", (raw_key or "").strip().lower()).strip("_")
    return cleaned or "general_note"


@dataclasses.dataclass
class KeyedDirective:
    """A declarative fact or standing user preference indexed by canonical key."""

    key: str
    value: str
    updated_at: str
    session_id: str = ""
    embedding: Optional[List[float]] = None

    def to_prompt_text(self) -> str:
        return f"- {self.key}: {self.value}"

    def token_count(self) -> int:
        return count_tokens(self.to_prompt_text())


@dataclasses.dataclass
class ContrastivePlaybookCard:
    """A verified procedural rule pairing a failure trap (`avoid`) with the verified fix (`instead`)."""

    card_id: str
    domain: str
    trigger: str
    avoid: str
    instead: str
    verified: bool = True
    embedding: Optional[List[float]] = None

    def to_prompt_text(self) -> str:
        trig = re.sub(r"^when\s+", "", (self.trigger or "").strip(), flags=re.IGNORECASE)
        return (
            f"- Rule [{self.card_id}] When {trig}: "
            f"AVOID {self.avoid}; INSTEAD {self.instead}"
        )

    def embedding_text(self) -> str:
        trig = re.sub(r"^when\s+", "", (self.trigger or "").strip(), flags=re.IGNORECASE)
        return f"[{self.domain}] When {trig} | Avoid: {self.avoid} | Instead: {self.instead}"

    def token_count(self) -> int:
        return count_tokens(self.to_prompt_text())


class SCARMemoryStore:
    """Dual-store memory managing superseded directives and verified contrastive playbooks."""

    def __init__(self, require_verified_cards: bool = True):
        self.require_verified_cards = require_verified_cards
        self.directives: Dict[str, KeyedDirective] = {}
        self.playbooks: Dict[str, ContrastivePlaybookCard] = {}
        self.quarantined_sessions: List[str] = []

    def set_directive(self, key: str, value: str, session_id: str = "") -> KeyedDirective:
        """Insert or overwrite a directive in-place using canonical key normalization."""
        norm_key = normalize_directive_key(key)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        directive = KeyedDirective(
            key=norm_key,
            value=value.strip(),
            updated_at=now,
            session_id=session_id,
            embedding=None,
        )
        self.directives[norm_key] = directive
        return directive

    def add_playbook_card(self, card: ContrastivePlaybookCard) -> bool:
        """Admit a contrastive playbook card only if verified (unless verification gate is disabled)."""
        if self.require_verified_cards and not card.verified:
            return False
        self.playbooks[card.card_id] = card
        return True

    def quarantine_session(self, session_id: str) -> None:
        """Record an unresolved failure session as quarantined (excluded from memory)."""
        self.quarantined_sessions.append(session_id)

    def ensure_embeddings(self, llm: LLMClient) -> None:
        """Batch-compute embeddings for any unembedded directives or playbook cards."""
        pending_dirs = [d for d in self.directives.values() if d.embedding is None]
        if pending_dirs:
            vecs = llm.embed_texts([d.to_prompt_text() for d in pending_dirs])
            for d, v in zip(pending_dirs, vecs):
                d.embedding = v

        pending_cards = [c for c in self.playbooks.values() if c.embedding is None]
        if pending_cards:
            vecs = llm.embed_texts([c.embedding_text() for c in pending_cards])
            for c, v in zip(pending_cards, vecs):
                c.embedding = v

    def retrieve_context(
        self,
        query: str = "",
        max_tokens: Optional[int] = None,
        llm: Optional[LLMClient] = None,
        domain_filter: Optional[str] = None,
    ) -> str:
        """Retrieve relevant directives and playbook cards strictly within `max_tokens`."""
        if max_tokens is not None and max_tokens <= 0:
            return ""

        candidates: List[Tuple[float, str, str]] = []  # (score, kind, rendered_line)

        if query and llm is not None and (self.directives or self.playbooks):
            self.ensure_embeddings(llm)
            q_vec = llm.embed_texts([query])[0]
            for d in self.directives.values():
                sim = cosine_similarity(q_vec, d.embedding or [])
                candidates.append((sim, "directive", d.to_prompt_text()))
            for c in self.playbooks.values():
                if domain_filter and c.domain and c.domain != domain_filter:
                    continue
                sim = cosine_similarity(q_vec, c.embedding or [])
                candidates.append((sim, "playbook", c.to_prompt_text()))
            candidates.sort(key=lambda x: x[0], reverse=True)
        else:
            for d in self.directives.values():
                candidates.append((1.0, "directive", d.to_prompt_text()))
            for c in self.playbooks.values():
                if domain_filter and c.domain and c.domain != domain_filter:
                    continue
                candidates.append((1.0, "playbook", c.to_prompt_text()))

        chosen_dirs: List[str] = []
        chosen_cards: List[str] = []

        def _format(dirs: Sequence[str], cards: Sequence[str]) -> str:
            sections: List[str] = []
            if dirs:
                sections.append("Active Directives:\n" + "\n".join(dirs))
            if cards:
                sections.append("Verified Playbook Rules:\n" + "\n".join(cards))
            return "\n\n".join(sections)

        for _, kind, text in candidates:
            trial_dirs = chosen_dirs + ([text] if kind == "directive" else [])
            trial_cards = chosen_cards + ([text] if kind == "playbook" else [])
            rendered = _format(trial_dirs, trial_cards)
            if max_tokens is None or count_tokens(rendered) <= max_tokens:
                if kind == "directive":
                    chosen_dirs.append(text)
                else:
                    chosen_cards.append(text)

        final_text = _format(chosen_dirs, chosen_cards)
        if max_tokens is not None and count_tokens(final_text) > max_tokens:
            final_text = truncate_to_token_budget(final_text, max_tokens)
        return final_text

"""SCAR Dual-Store Memory Architecture.

Implements:
1. KeyedDirectives: Calm facts and user preferences with automatic key-based supersession.
2. ContrastivePlaybookCards: Procedural lessons extracted from resolved struggles
   ("Trigger", "Avoid", "Action/Instead").
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Dict, List, Optional, Sequence


@dataclasses.dataclass
class KeyedDirective:
    """A declarative fact or user preference identified by an explicit key."""
    key: str
    value: str
    updated_at: str
    session_id: str

    def to_prompt_text(self) -> str:
        return f"- {self.key}: {self.value}"

    def estimate_tokens(self) -> int:
        return max(5, int(len(self.to_prompt_text().split()) * 1.3))


@dataclasses.dataclass
class ContrastivePlaybookCard:
    """A procedural troubleshooting rule pairing failure traps with correct fixes."""
    task_id: str
    trigger: str
    avoid: str
    instead: str
    verified: bool = True

    def to_prompt_text(self) -> str:
        return (
            f"Rule [{self.task_id}]:\n"
            f"  Trigger: {self.trigger}\n"
            f"  Avoid:   {self.avoid}\n"
            f"  Instead: {self.instead}"
        )

    def estimate_tokens(self) -> int:
        return max(10, int(len(self.to_prompt_text().split()) * 1.3))


class SCARMemoryStore:
    """Dual-store memory managing keyed directives and contrastive cards."""

    def __init__(self):
        self.directives: Dict[str, KeyedDirective] = {}
        self.playbooks: Dict[str, ContrastivePlaybookCard] = {}

    def set_directive(self, key: str, value: str, session_id: str = "") -> None:
        """Add or overwrite a directive with key-based supersession."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.directives[key] = KeyedDirective(
            key=key, value=value, updated_at=now, session_id=session_id
        )

    def add_playbook_card(self, card: ContrastivePlaybookCard) -> None:
        """Add or update a verified contrastive playbook card."""
        if card.verified:
            self.playbooks[card.task_id] = card

    def retrieve_context(self, max_tokens: Optional[int] = None) -> str:
        """Format stored memory for injection into the agent's prompt window."""
        blocks: List[str] = []
        used_tokens = 0

        # Directives block
        directive_lines = []
        for d in self.directives.values():
            t = d.estimate_tokens()
            if max_tokens and (used_tokens + t > max_tokens):
                break
            directive_lines.append(d.to_prompt_text())
            used_tokens += t

        if directive_lines:
            blocks.append("### Active Directives & Preferences\n" + "\n".join(directive_lines))

        # Playbook rules block
        rule_lines = []
        for p in self.playbooks.values():
            t = p.estimate_tokens()
            if max_tokens and (used_tokens + t > max_tokens):
                break
            rule_lines.append(p.to_prompt_text())
            used_tokens += t

        if rule_lines:
            blocks.append("### Verified Procedural Playbooks\n" + "\n\n".join(rule_lines))

        return "\n\n".join(blocks)

"""Token counting and strict budget-packing utilities for SCAR and memory baselines."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Sequence

# Splits words, numbers, and punctuation pieces in a manner closely tracking modern BPE tokenizers.
_TOKEN_PIECE_RE = re.compile(r"[A-Za-z]+|\d{1,3}|[^\sA-Za-z\d]")


def count_tokens(text: str) -> int:
    """Estimate subword token count of raw text without cumulative prompt inflation.

    Uses a deterministic BPE-calibrated piece counter (~1.08 subwords per regex piece
    on technical English + JSON/IDs). Empty strings return 0.
    """
    if not text:
        return 0
    pieces = _TOKEN_PIECE_RE.findall(text)
    if not pieces:
        return 0
    extra_subwords = sum(max(0, (len(p) - 5) // 4) for p in pieces if p.isalpha())
    return len(pieces) + extra_subwords


def count_message_tokens(message: Dict[str, Any]) -> int:
    """Count true transcript tokens contributed by a single message turn."""
    total = 4  # role + turn framing overhead
    content = message.get("content")
    if isinstance(content, str) and content:
        total += count_tokens(content)
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        name = tc.get("name") or ""
        args = tc.get("arguments")
        if args is None:
            args_str = tc.get("arguments_summary") or ""
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = json.dumps(args, sort_keys=True)
        total += 4 + count_tokens(name) + count_tokens(args_str)
    return total


def count_transcript_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    """Count total unique transcript tokens across a message list (never summing prompt_tokens)."""
    return sum(count_message_tokens(m) for m in messages)


def truncate_to_token_budget(text: str, max_tokens: int, keep_tail: bool = False) -> str:
    """Truncate text so that `count_tokens(result) <= max_tokens`."""
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    lines = text.splitlines()
    if keep_tail:
        lines = list(reversed(lines))

    kept: List[str] = []
    running = 0
    for line in lines:
        line_toks = count_tokens(line) + 1
        if running + line_toks <= max_tokens:
            kept.append(line)
            running += line_toks
        else:
            remaining = max_tokens - running
            if remaining >= 6:
                words = line.split()
                if keep_tail:
                    words = list(reversed(words))
                partial: List[str] = []
                for w in words:
                    cand = (" ".join(reversed(partial + [w])) if keep_tail else " ".join(partial + [w]))
                    if count_tokens(cand) <= remaining - 1:
                        partial.append(w)
                    else:
                        break
                if partial:
                    kept.append(" ".join(reversed(partial)) if keep_tail else " ".join(partial))
            break

    if keep_tail:
        kept.reverse()
    out = "\n".join(kept).strip()
    while out and count_tokens(out) > max_tokens:
        parts = out.split()
        if len(parts) <= 1:
            return ""
        parts = parts[1:] if keep_tail else parts[:-1]
        out = " ".join(parts)
    return out


def pack_items_within_budget(
    items: Iterable[str],
    max_tokens: int,
    header: str = "",
    separator: str = "\n",
) -> str:
    """Greedily pack formatted string items so total output never exceeds `max_tokens`."""
    if max_tokens <= 0:
        return ""
    header_prefix = f"{header}\n" if header else ""
    if header_prefix and count_tokens(header_prefix) >= max_tokens:
        return ""

    chosen: List[str] = []
    for item in items:
        item_clean = item.strip()
        if not item_clean:
            continue
        candidate_body = separator.join(chosen + [item_clean])
        candidate = f"{header_prefix}{candidate_body}"
        if count_tokens(candidate) <= max_tokens:
            chosen.append(item_clean)
    if not chosen:
        return ""
    return f"{header_prefix}{separator.join(chosen)}"

"""Emotion-Shift Memory (ESM): Minimal turn-shift gates and plain MEMORY.md store."""

from .detectors import (
    FlashLiteShiftDetector,
    GoEmotionsShiftDetector,
    LayaShiftDetector,
    VaderShiftDetector,
)
from .memory_md import (
    MarkdownMemory,
    extract_contrastive_fix_rule,
    extract_rule_from_chat,
    select_relevant_rules,
    split_base_and_hidden_policy,
    verify_rule_on_mistake_turn,
)

__all__ = [
    "VaderShiftDetector",
    "GoEmotionsShiftDetector",
    "LayaShiftDetector",
    "FlashLiteShiftDetector",
    "MarkdownMemory",
    "extract_contrastive_fix_rule",
    "extract_rule_from_chat",
    "select_relevant_rules",
    "split_base_and_hidden_policy",
    "verify_rule_on_mistake_turn",
]

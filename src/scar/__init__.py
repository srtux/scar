"""SCAR: Struggle-Conditioned Agent Retention for Long-Term Human-AI Collaboration."""

from scar.baselines import (
    AtomicFactsMemory,
    FIFOMemory,
    RawTrajectoryRAGMemory,
    RollingSummaryMemory,
    SingleTurnImportanceMemory,
    StaticSentimentMemory,
    ToolOutcomeOnlyMemory,
)
from scar.consolidator import SCARConsolidator, strip_oracle_fields
from scar.detector import (
    NaiveRegexPushbackDetector,
    SCARTransitionDetector,
    StaticSentimentDetector,
    ToolOutcomeDetector,
)
from scar.llm import LLMClient, cosine_similarity
from scar.memory import ContrastivePlaybookCard, KeyedDirective, SCARMemoryStore
from scar.tokens import count_tokens, count_transcript_tokens, truncate_to_token_budget

__all__ = [
    "AtomicFactsMemory",
    "ContrastivePlaybookCard",
    "FIFOMemory",
    "KeyedDirective",
    "LLMClient",
    "NaiveRegexPushbackDetector",
    "RawTrajectoryRAGMemory",
    "RollingSummaryMemory",
    "SCARConsolidator",
    "SCARMemoryStore",
    "SCARTransitionDetector",
    "SingleTurnImportanceMemory",
    "StaticSentimentDetector",
    "StaticSentimentMemory",
    "ToolOutcomeDetector",
    "ToolOutcomeOnlyMemory",
    "cosine_similarity",
    "count_tokens",
    "count_transcript_tokens",
    "strip_oracle_fields",
    "truncate_to_token_budget",
]

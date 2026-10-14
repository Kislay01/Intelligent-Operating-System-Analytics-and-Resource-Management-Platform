"""Page replacement simulation."""

from .algorithms import ALL_POLICIES, FIFO, LFU, LRU, Clock, Optimal, PagingResult, ReplacementPolicy, compare, fault_curve
from .analysis import (
    BELADY_CLASSIC, PATTERNS, detect_belady, generate, parse_refs, search_belady, thrashing_assessment, working_set,
)

__all__ = [
    "ALL_POLICIES", "BELADY_CLASSIC", "Clock", "FIFO", "LFU", "LRU", "Optimal", "PATTERNS", "PagingResult",
    "ReplacementPolicy", "compare", "detect_belady", "fault_curve", "generate", "parse_refs", "search_belady",
    "thrashing_assessment", "working_set",
]

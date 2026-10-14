"""Page replacement simulation."""

from . import advanced as _advanced

from .algorithms import ALL_POLICIES, FIFO, LFU, LRU, Clock, Optimal, PagingResult, ReplacementPolicy, compare, fault_curve
from .analysis import (
    BELADY_CLASSIC, PATTERNS, detect_belady, generate, parse_refs, search_belady, thrashing_assessment, working_set,
)

__all__ = [
    "ALL_POLICIES", "BELADY_CLASSIC", "Clock", "FIFO", "LFU", "LRU", "Optimal", "PATTERNS", "PagingResult",
    "ReplacementPolicy", "compare", "detect_belady", "fault_curve", "generate", "parse_refs", "search_belady",
    "thrashing_assessment", "working_set",
]

from .advanced import ARC, WSClock, mattson_lru_curve, reuse_distance_histogram, stack_distances  # noqa: E402

_advanced.register()
__all__ += ["ARC", "WSClock", "mattson_lru_curve", "reuse_distance_histogram", "stack_distances"]

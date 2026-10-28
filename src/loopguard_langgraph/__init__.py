"""loopguard-langgraph — loopguard's semantic loop detection as a
drop-in guard node for LangGraph agent graphs.

Base detector: https://github.com/habib-analyst/loopguard
(held-out detection F1 1.000 vs 0.783 for action hashing).
"""
from .guard import (
    LoopGuard,
    LoopGuardState,
    build_trace,
    extract_thought_observation,
)

__all__ = [
    "LoopGuard",
    "LoopGuardState",
    "build_trace",
    "extract_thought_observation",
]

__version__ = "0.1.0"

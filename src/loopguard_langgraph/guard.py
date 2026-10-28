"""LangGraph guard node for loopguard's semantic revisit detection.

loopguard (https://github.com/habib-analyst/loopguard) detects when an agent
thinks the same thought in different words — the loop that action-hashing
misses — and only calls it a loop when the observation hasn't substantively
changed. This module packages that detector as a drop-in LangGraph node:

    builder.add_node("agent", agent_node)
    builder.add_node("reflect", guard.reflect_node(agent.reflect))
    builder.add_edge(START, "agent")
    guard.add_to(builder)            # wires agent -> guard -> agent/reflect/END
    graph = builder.compile()

The guard node reads the agent's latest thought + observation from state,
runs the detector, and routes:

    done            -> END            (loop_status "solved")
    no revisit      -> agent          (loop_status "ok")
    revisit         -> reflect        (loop_status "loop")
    revisit, reflection budget spent -> END (loop_status "stopped")

The reflect node asks the agent for a new strategy and *enforces* that it is
embedding-dissimilar to the strategies that already failed — a rephrased
version of the failed strategy is rejected and the graph stops honestly
instead of re-entering the loop.

State contract: the agent node should return `thought` and `observation`
strings (plus append to `thought_history` / `observation_history` and set
`done=True` when the task is solved). If `thought`/`observation` are absent,
the guard falls back to the last AIMessage / ToolMessage in `messages`.
"""
from __future__ import annotations

import operator
from typing import Annotated, Callable, Optional, TypedDict

from loopguard import SemanticRevisitDetector, ThoughtEmbedder

# Route tokens returned by LoopGuard.route / route_after_reflect.
ROUTE_AGENT = "agent"
ROUTE_REFLECT = "reflect"
ROUTE_END = "end"


class LoopGuardState(TypedDict, total=False):
    """Channels loopguard-langgraph reads/writes. Merge into your own
    state schema (all fields optional except what your agent produces)."""

    messages: list
    thought: str
    observation: str
    thought_history: Annotated[list, operator.add]
    observation_history: Annotated[list, operator.add]
    strategy: str
    done: bool
    steps: int
    loop_status: str  # "ok" | "loop" | "stopped" | "solved"
    loop_event: Optional[dict]
    reflections_used: int
    stop_reason: str


def extract_thought_observation(state: dict) -> tuple[Optional[str], Optional[str]]:
    """Latest (thought, observation) from explicit channels, falling back
    to the message history (last AIMessage = thought, last ToolMessage =
    observation)."""
    thought = state.get("thought")
    observation = state.get("observation")
    if thought is not None and observation is not None:
        return thought, observation
    messages = state.get("messages") or []
    for msg in reversed(messages):
        kind = getattr(msg, "type", None)
        if observation is None and kind == "tool":
            observation = str(msg.content)
        if thought is None and kind == "ai":
            thought = str(msg.content)
        if thought is not None and observation is not None:
            break
    return thought, observation


def build_trace(state: dict) -> str:
    """Loop trace handed to the agent's reflect() callable — the same
    format loopguard's ReActLoop injects."""
    ev = state.get("loop_event") or {}
    thoughts = state.get("thought_history", [])
    observations = state.get("observation_history", [])
    lines = [f"thought: {t} | obs: {o}" for t, o in zip(thoughts, observations)]
    sim = ev.get("similarity", 0.0)
    return (
        f"LOOP DETECTED at step {ev.get('step')} (similarity {sim:.2f} "
        f"to step {ev.get('matched_step')}).\n"
        f"Earlier thought: {ev.get('matched_thought')}\n"
        f"Repeated thought: {ev.get('thought')}\n"
        f"History so far:\n" + "\n".join(lines[-6:])
    )


class LoopGuard:
    """Semantic loop guard for LangGraph graphs.

    Parameters mirror loopguard's SemanticRevisitDetector / ReActLoop:
    - corpus: agent-thought corpus used to fit the TF-IDF embedder
      (or pass a pre-fitted `embedder`).
    - threshold: cosine similarity at/above which a thought counts as a
      revisit (calibrate with SemanticRevisitDetector.calibrate; the
      base repo's held-out calibration landed at 0.30).
    - max_reflections: reflections allowed before the graph stops.
    - reflection_dissimilarity: a new strategy whose max cosine similarity
      to a previously accepted strategy reaches this value is rejected.
    """

    def __init__(
        self,
        corpus: Optional[list] = None,
        embedder: Optional[ThoughtEmbedder] = None,
        threshold: float = 0.45,
        window: int = 8,
        max_reflections: int = 3,
        reflection_dissimilarity: float = 0.55,
    ):
        if embedder is None:
            if not corpus:
                raise ValueError(
                    "LoopGuard needs a thought corpus to fit the embedder "
                    "(or pass embedder=ThoughtEmbedder().fit(...))."
                )
            embedder = ThoughtEmbedder().fit(list(corpus))
        self.embedder = embedder
        self.detector = SemanticRevisitDetector(
            embedder=embedder, threshold=threshold, window=window
        )
        self.threshold = threshold
        self.max_reflections = max_reflections
        self.reflection_dissimilarity = reflection_dissimilarity
        self._strategy_embeddings: list = []
        self.events: list = []  # every Revisit flagged this run, in order

    # ---- core detector API -------------------------------------------------
    def check(self, thought: str, observation: str):
        """Run one thought+observation through the detector.
        Returns a loopguard Revisit or None."""
        revisit = self.detector.step(thought, observation)
        if revisit is not None:
            self.events.append(revisit)
        return revisit

    def validate_reflection(self, strategy: str) -> tuple:
        """(ok, max_similarity): a strategy is accepted only if it is
        embedding-dissimilar to every previously accepted strategy.
        The first reflection is always accepted (nothing to compare)."""
        if not self._strategy_embeddings:
            return True, 0.0
        emb = self.embedder.embed(strategy)
        sims = [ThoughtEmbedder.cosine(emb, s) for s in self._strategy_embeddings]
        top = float(max(sims))
        return top < self.reflection_dissimilarity, top

    def accept_reflection(self, strategy: str) -> None:
        """Record an accepted strategy and give it a fresh detector
        window (same semantics as loopguard's ReActLoop)."""
        self._strategy_embeddings.append(self.embedder.embed(strategy))
        self.detector.reset()

    def reset(self) -> None:
        """Reuse this guard for a fresh graph invocation."""
        self.detector.reset()
        self._strategy_embeddings.clear()
        self.events.clear()

    # ---- LangGraph nodes ---------------------------------------------------
    def node(self) -> Callable:
        """The guard node: inspect the agent's latest step and verdict it."""

        def _guard_node(state: dict) -> dict:
            if state.get("done"):
                return {"loop_status": "solved", "loop_event": None}
            thought, observation = extract_thought_observation(state)
            if thought is None:
                return {"loop_status": "ok", "loop_event": None}
            revisit = self.check(thought, observation or "")
            if revisit is None:
                return {"loop_status": "ok", "loop_event": None}
            event = {
                "step": revisit.step,
                "thought": revisit.thought,
                "matched_step": revisit.matched_step,
                "matched_thought": revisit.matched_thought,
                "similarity": revisit.similarity,
                "observation_changed": revisit.observation_changed,
            }
            used = state.get("reflections_used", 0)
            if used >= self.max_reflections:
                return {
                    "loop_status": "stopped",
                    "loop_event": event,
                    "stop_reason": (
                        f"semantic loop re-detected after {used} reflections; "
                        f"giving up honestly (similarity {revisit.similarity:.2f} "
                        f"to step {revisit.matched_step})"
                    ),
                }
            return {"loop_status": "loop", "loop_event": event}

        return _guard_node

    def route(self, state: dict) -> str:
        """Conditional-edge function after the guard node.
        Returns ROUTE_AGENT / ROUTE_REFLECT / ROUTE_END."""
        status = state.get("loop_status", "ok")
        if status in ("solved", "stopped"):
            return ROUTE_END
        if status == "loop":
            return ROUTE_REFLECT
        return ROUTE_AGENT

    def reflect_node(self, reflect_fn: Callable) -> Callable:
        """Wrap the agent's reflect(trace) -> strategy callable as a node
        that enforces reflection dissimilarity."""

        def _reflect_node(state: dict) -> dict:
            strategy = reflect_fn(build_trace(state))
            ok, sim = self.validate_reflection(strategy)
            if not ok:
                return {
                    "loop_status": "stopped",
                    "strategy": strategy,
                    "stop_reason": (
                        f"reflection proposed a strategy too similar "
                        f"(max sim {sim:.2f}) to a failed one; refusing to re-loop"
                    ),
                }
            self.accept_reflection(strategy)
            return {
                "loop_status": "ok",
                "strategy": strategy,
                "reflections_used": state.get("reflections_used", 0) + 1,
            }

        return _reflect_node

    def route_after_reflect(self, state: dict) -> str:
        """Conditional-edge function after the reflect node."""
        return ROUTE_END if state.get("loop_status") == "stopped" else ROUTE_AGENT

    def add_to(self, builder, agent: str = "agent", reflect: str = "reflect",
               guard: str = "loopguard"):
        """Wire this guard into a StateGraph builder:

            agent -> guard -ok-> agent
                           -loop-> reflect -> agent
                           -solved/stopped-> END
            reflect -rejected-> END

        The agent and reflect nodes must already be added; the reflect
        node should be `guard.reflect_node(agent.reflect)`.
        Returns the builder for chaining.
        """
        from langgraph.graph import END

        builder.add_node(guard, self.node())
        builder.add_edge(agent, guard)
        builder.add_conditional_edges(
            guard, self.route,
            {ROUTE_AGENT: agent, ROUTE_REFLECT: reflect, ROUTE_END: END},
        )
        builder.add_conditional_edges(
            reflect, self.route_after_reflect,
            {ROUTE_AGENT: agent, ROUTE_END: END},
        )
        return builder

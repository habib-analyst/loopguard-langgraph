"""Graph builders for the three arms of the experiment (also handy as
usage examples):

- build_unguarded_graph: agent self-loops until it solves the task or
  LangGraph's recursion limit kills the run.
- build_max_steps_graph: the naive guard — the agent is cut off after
  `max_steps` agent steps regardless of whether it is looping or making
  legitimate progress.
- build_guarded_graph:   agent -> loopguard -> agent / reflect / END.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .guard import LoopGuard, LoopGuardState


def _route_unguarded(state: dict) -> str:
    return END if state.get("done") else "agent"


def build_unguarded_graph(agent):
    builder = StateGraph(LoopGuardState)
    builder.add_node("agent", agent.node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route_unguarded,
                                  {"agent": "agent", END: END})
    return builder.compile()


def build_max_steps_graph(agent, max_steps: int = 10):
    def _route(state: dict) -> str:
        if state.get("done") or state.get("steps", 0) >= max_steps:
            return END
        return "agent"

    builder = StateGraph(LoopGuardState)
    builder.add_node("agent", agent.node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route, {"agent": "agent", END: END})
    return builder.compile()


def build_guarded_graph(agent, guard: LoopGuard):
    builder = StateGraph(LoopGuardState)
    builder.add_node("agent", agent.node)
    builder.add_node("reflect", guard.reflect_node(agent.reflect))
    builder.add_edge(START, "agent")
    guard.add_to(builder)
    return builder.compile()

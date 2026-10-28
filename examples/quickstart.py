"""Quickstart: a stuck agent, guarded by loopguard inside LangGraph.

Runs a deterministic scripted agent that rewords the same search forever.
Unguarded, it burns the full recursion limit; with the LoopGuard node it is
detected within a few steps, forced to reflect, and stopped honestly when
its reflections run out.

    python examples/quickstart.py
"""
import sys

sys.path.insert(0, "src")

from langgraph.errors import GraphRecursionError

from loopguard_langgraph.graphs import build_guarded_graph, build_unguarded_graph
from loopguard_langgraph.scenarios import SemanticLoopAgent, make_guard


def main():
    agent = SemanticLoopAgent()
    try:
        build_unguarded_graph(agent).invoke({}, config={"recursion_limit": 50})
        print(f"unguarded: solved in {agent.calls} steps")
    except GraphRecursionError:
        print(f"unguarded: hit the recursion limit after {agent.calls} agent steps")

    agent = SemanticLoopAgent()
    guard = make_guard()
    final = build_guarded_graph(agent, guard).invoke(
        {}, config={"recursion_limit": 500})
    print(f"loopguard: stopped after {agent.calls} agent steps")
    print(f"  detections:  {len(guard.events)} semantic revisits flagged")
    print(f"  reflections: {final.get('reflections_used', 0)}")
    print(f"  reason:      {final.get('stop_reason')}")


if __name__ == "__main__":
    main()

"""Tests: loopguard's guard node inside real LangGraph graphs.

All agents are deterministic scripted state machines (scenarios.py);
no LLM is involved anywhere.
"""
import sys

sys.path.insert(0, "src")

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from loopguard_langgraph.graphs import (
    build_guarded_graph,
    build_max_steps_graph,
    build_unguarded_graph,
)
from loopguard_langgraph.guard import (
    LoopGuard,
    LoopGuardState,
    extract_thought_observation,
)
from loopguard_langgraph.scenarios import (
    CleanLongAgent,
    CleanShortAgent,
    ExactRepeatAgent,
    LoopRecoverAgent,
    LoopRephraseAgent,
    RetryProgressAgent,
    SemanticLoopAgent,
    corpus,
    make_guard,
)


@pytest.fixture(scope="module")
def threshold():
    return make_guard().threshold  # calibrated once per test session


def run_guarded(agent, threshold):
    guard = make_guard(threshold=threshold)
    final = build_guarded_graph(agent, guard).invoke(
        {}, config={"recursion_limit": 500})
    return final, guard


def test_semantic_loop_stopped_early(threshold):
    agent = SemanticLoopAgent()
    final, guard = run_guarded(agent, threshold)
    assert final["loop_status"] == "stopped"
    assert agent.calls < 50  # unguarded burns the full recursion limit
    assert len(guard.events) >= 2
    assert final["reflections_used"] == guard.max_reflections
    assert "giving up honestly" in final["stop_reason"]


def test_exact_repeat_stopped(threshold):
    agent = ExactRepeatAgent()
    final, guard = run_guarded(agent, threshold)
    assert final["loop_status"] == "stopped"
    assert len(guard.events) >= 2


def test_clean_short_solves_with_zero_detections(threshold):
    agent = CleanShortAgent()
    final, guard = run_guarded(agent, threshold)
    assert final["done"] is True
    assert final["loop_status"] == "solved"
    assert guard.events == []


def test_clean_long_solves_beyond_step_cap(threshold):
    agent = CleanLongAgent()
    final, guard = run_guarded(agent, threshold)
    assert final["done"] is True
    assert agent.calls == 14
    assert guard.events == []
    # the naive max-steps guard kills this same legitimate run at step 10
    agent2 = CleanLongAgent()
    capped = build_max_steps_graph(agent2, max_steps=10).invoke(
        {}, config={"recursion_limit": 200})
    assert not capped.get("done")


def test_retry_with_changing_observation_not_flagged(threshold):
    agent = RetryProgressAgent()
    final, guard = run_guarded(agent, threshold)
    assert final["done"] is True
    assert guard.events == []  # observation gate: the world DID say something new


def test_reflection_recovers_looping_agent(threshold):
    agent = LoopRecoverAgent()
    final, guard = run_guarded(agent, threshold)
    assert final["done"] is True
    assert len(guard.events) == 1
    assert final["reflections_used"] == 1


def test_rephrased_reflection_rejected(threshold):
    agent = LoopRephraseAgent()
    final, guard = run_guarded(agent, threshold)
    assert final["loop_status"] == "stopped"
    assert "too similar" in final["stop_reason"]
    assert final["reflections_used"] == 1  # only the genuine one was accepted


def test_unguarded_loop_hits_recursion_limit():
    agent = SemanticLoopAgent()
    with pytest.raises(GraphRecursionError):
        build_unguarded_graph(agent).invoke({}, config={"recursion_limit": 50})
    assert agent.calls == 50


def test_extract_from_messages_fallback():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    state = {"messages": [HumanMessage(content="find the password"),
                          AIMessage(content="I should search for the password"),
                          ToolMessage(content="search: 0 results", tool_call_id="x")]}
    thought, obs = extract_thought_observation(state)
    assert thought == "I should search for the password"
    assert obs == "search: 0 results"
    # explicit channels win over messages
    state["thought"], state["observation"] = "explicit", "explicit obs"
    assert extract_thought_observation(state) == ("explicit", "explicit obs")


def test_reflection_validation_threshold():
    guard = LoopGuard(corpus=corpus(), threshold=0.45)
    ok, sim = guard.validate_reflection("Scan the documentation index page by page")
    assert ok and sim == 0.0  # first reflection always accepted
    guard.accept_reflection("Scan the documentation index page by page")
    ok, sim = guard.validate_reflection("Page by page, scan the documentation index")
    assert not ok and sim >= guard.reflection_dissimilarity
    ok, _ = guard.validate_reflection("Read the local configuration file from disk")
    assert ok


def test_guard_reset_reuse_is_deterministic(threshold):
    guard = make_guard(threshold=threshold)
    outcomes = []
    for _ in range(2):
        agent = SemanticLoopAgent()
        builder = StateGraph(LoopGuardState)
        builder.add_node("agent", agent.node)
        builder.add_node("reflect", guard.reflect_node(agent.reflect))
        builder.add_edge(START, "agent")
        guard.add_to(builder)
        final = builder.compile().invoke({}, config={"recursion_limit": 500})
        outcomes.append((agent.calls, len(guard.events), final["stop_reason"]))
        guard.reset()
    assert outcomes[0] == outcomes[1]

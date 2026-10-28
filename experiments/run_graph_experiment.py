"""Experiment: loopguard inside a real LangGraph graph vs two baselines.

Seven deterministic scripted-agent scenarios (see scenarios.py), each run
through three LangGraph graphs:

  unguarded  — agent self-loops; only LangGraph's recursion limit (50)
               stops a looping run.
  max_steps  — naive guard: agent is cut off after 10 steps whether it is
               looping or making legitimate progress.
  loopguard  — agent -> LoopGuard node -> agent / reflect / END, with the
               threshold calibrated on generated trajectories (base-repo
               methodology) before any graph runs.

Metrics per run: outcome, agent steps consumed, guard detections,
reflections used. Reported outcomes are measured, not asserted: a looping
agent that a guard fails to stop shows up as recursion_limit / stopped_cap,
and a clean agent a guard wrongly stops shows up as a lost solve.
"""
import json
import sys

sys.path.insert(0, "src")

from langgraph.errors import GraphRecursionError

from loopguard_langgraph.graphs import (
    build_guarded_graph,
    build_max_steps_graph,
    build_unguarded_graph,
)
from loopguard_langgraph.scenarios import (
    LOOPING_SCENARIOS,
    SCENARIOS,
    SOLVABLE_SCENARIOS,
    make_guard,
)

RECURSION_LIMIT = 50
MAX_STEPS = 10


def run_unguarded(agent) -> dict:
    graph = build_unguarded_graph(agent)
    try:
        final = graph.invoke({}, config={"recursion_limit": RECURSION_LIMIT})
        status = "solved" if final.get("done") else "ended"
    except GraphRecursionError:
        status = "recursion_limit"
    return {"status": status, "agent_steps": agent.calls,
            "detections": None, "reflections": 0}


def run_max_steps(agent) -> dict:
    graph = build_max_steps_graph(agent, max_steps=MAX_STEPS)
    final = graph.invoke({}, config={"recursion_limit": RECURSION_LIMIT * 4})
    status = "solved" if final.get("done") else "stopped_cap"
    return {"status": status, "agent_steps": agent.calls,
            "detections": None, "reflections": 0}


def run_guarded(agent, threshold: float) -> dict:
    guard = make_guard(threshold=threshold)
    graph = build_guarded_graph(agent, guard)
    final = graph.invoke({}, config={"recursion_limit": 500})
    if final.get("done"):
        status = "solved"
    elif "too similar" in final.get("stop_reason", ""):
        status = "reflection_rejected"
    elif final.get("loop_status") == "stopped":
        status = "stopped_loop"
    else:
        status = "ended"
    return {"status": status, "agent_steps": agent.calls,
            "detections": len(guard.events),
            "reflections": final.get("reflections_used", 0),
            "stop_reason": final.get("stop_reason", "")}


def main():
    calibrated = make_guard()  # calibrates threshold on generated trajectories
    threshold = calibrated.threshold
    print(f"calibrated loopguard threshold: {threshold:.2f}")

    rows = {}
    for scenario_cls in SCENARIOS:
        name = scenario_cls.name
        rows[name] = {
            "unguarded": run_unguarded(scenario_cls()),
            "max_steps": run_max_steps(scenario_cls()),
            "loopguard": run_guarded(scenario_cls(), threshold),
        }
        r = rows[name]
        print(f"{name:16s} unguarded={r['unguarded']['status']}/{r['unguarded']['agent_steps']} "
              f"max_steps={r['max_steps']['status']}/{r['max_steps']['agent_steps']} "
              f"loopguard={r['loopguard']['status']}/{r['loopguard']['agent_steps']} "
              f"(detections={r['loopguard']['detections']}, "
              f"reflections={r['loopguard']['reflections']})")

    arms = ["unguarded", "max_steps", "loopguard"]
    loop_rows = {a: [rows[s][a] for s in LOOPING_SCENARIOS] for a in arms}
    aggregates = {
        "mean_steps_on_looping_scenarios": {
            a: round(sum(r["agent_steps"] for r in loop_rows[a]) / len(loop_rows[a]), 1)
            for a in arms
        },
        "solved_of_solvable": {
            a: sum(1 for s in SOLVABLE_SCENARIOS if rows[s][a]["status"] == "solved")
            for a in arms
        },
        "n_solvable": len(SOLVABLE_SCENARIOS),
        "clean_scenarios_falsely_stopped": {
            a: [s for s in ("clean_short", "clean_long", "retry_progress")
                if rows[s][a]["status"] != "solved"]
            for a in arms
        },
        "loopguard_detections_on_clean": {
            s: rows[s]["loopguard"]["detections"]
            for s in ("clean_short", "clean_long", "retry_progress")
        },
        "loopguard_loop_recover": rows["loop_recover"]["loopguard"],
        "loopguard_loop_rephrase": rows["loop_rephrase"]["loopguard"],
    }
    results = {
        "calibrated_threshold": round(float(threshold), 2),
        "recursion_limit_unguarded": RECURSION_LIMIT,
        "max_steps_cap": MAX_STEPS,
        "scenarios": rows,
        "aggregates": aggregates,
        "note": ("scripted deterministic agents in real LangGraph graphs; "
                 "threshold calibrated on separately generated trajectories; "
                 "agent_steps counted at the agent node"),
    }
    with open("results/graph_experiment.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(aggregates, indent=2))
    return results


if __name__ == "__main__":
    main()

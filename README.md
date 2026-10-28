[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

# loopguard-langgraph

**Your LangGraph agent is thinking the same thought in different words — and the recursion limit can't tell that from real work.** loopguard-langgraph drops [loopguard](https://github.com/habib-analyst/loopguard)'s semantic loop detection into a LangGraph graph as a guard node: it catches reworded loops within a few steps, forces reflections to propose *measurably different* strategies, and leaves legitimate long runs alone.

Base project: **[habib-analyst/loopguard](https://github.com/habib-analyst/loopguard)** — semantic revisit detection (TF-IDF embeddings, observation-gated) with held-out detection **F1 1.000 vs 0.783** for action hashing. This repo is only the LangGraph packaging of that detector.

## Quickstart

```bash
pip install git+https://github.com/habib-analyst/loopguard   # base detector
pip install loopguard-langgraph                              # or from a clone: pip install .
```

```python
from langgraph.graph import START, StateGraph
from loopguard_langgraph import LoopGuard, LoopGuardState

guard = LoopGuard(corpus=my_agent_thought_corpus, threshold=0.45)

builder = StateGraph(LoopGuardState)
builder.add_node("agent", agent_node)                    # yours
builder.add_node("reflect", guard.reflect_node(agent_reflect))  # yours, wrapped
builder.add_edge(START, "agent")
guard.add_to(builder)        # wires agent -> guard -> agent / reflect / END
graph = builder.compile()
```

Your agent node returns `thought` and `observation` strings per step (and `done=True` when solved); if it only maintains `messages`, the guard falls back to the last AIMessage / ToolMessage. Routing:

| guard verdict | route | meaning |
|---|---|---|
| task done | `END` | `loop_status="solved"` |
| no revisit | agent | keep working |
| semantic revisit | reflect | agent must propose a new strategy |
| revisit after `max_reflections` | `END` | `loop_status="stopped"`, honest `stop_reason` |
| reflection too similar to a failed strategy | `END` | rephrased strategies are rejected, not re-looped |

## Measured results

Seven deterministic scripted-agent scenarios run through **real LangGraph graphs**, three arms each (threshold calibrated on separately generated trajectories: 0.30). Full data: [`results/graph_experiment.json`](results/graph_experiment.json), reproduce with `python experiments/run_graph_experiment.py`.

| scenario | unguarded | max-steps (10) | **loopguard** |
|---|---|---|---|
| semantic loop (reworded) | recursion limit, 50 steps | cap at 10 | **stopped, 9 steps** |
| exact-repeat loop | recursion limit, 50 steps | cap at 10 | **stopped, 8 steps** |
| clean, solves in 5 | solved | solved | solved, 0 detections |
| clean, solves in 14 | solved | **killed by cap at 10** | solved, 0 detections |
| retry, observation changes | solved | solved | solved, 0 detections |
| loop → reflect → recover | recursion limit, 50 steps | cap at 10 | **solved in 3 steps** |
| loop → rephrased reflection | recursion limit, 50 steps | cap at 10 | **reflection rejected, stopped at 4** |

Aggregates: mean steps burned on looping scenarios **50.0 unguarded / 10.0 max-steps / 7.0 loopguard**; solvable scenarios solved **3/4 / 2/4 / 4/4**. The step cap stops loops only by also stopping legitimate work; the guard distinguishes them because it reads *what the agent is thinking and what the world answered*, not how many steps elapsed.

## How it works

1. **Guard node** — after every agent step, the latest thought+observation goes through loopguard's `SemanticRevisitDetector`: a revisit is flagged only when the thought is embedding-similar to a recent one **and** the observation hasn't substantively changed (outcome-class gate). Retrying after new evidence arrives is not a loop.
2. **Reflection enforcement** — on a revisit, the loop trace is handed to your `reflect()`; the returned strategy must be embedding-dissimilar to previously accepted strategies, or the graph stops instead of re-entering the loop with a rephrase.
3. **Honest termination** — when the reflection budget is spent, the graph ends with a machine-readable `stop_reason`, never a silent partial answer.

## Prior art, credited

Loop guards for agents exist; none packages this mechanism for LangGraph. nudgeops wraps LangGraph tool nodes but its semantic layer needs an LLM call per step (its offline layer is hash/pattern based); breakerbox is a spend-first circuit breaker with repeat detection as a feature; able-harness-kit's LoopGuardMiddleware hashes exact tool+args — the baseline class loopguard outperforms. LangGraph's own `recursion_limit` is the max-steps arm above. Full gate analysis: [`KILLGATE.md`](KILLGATE.md).

## Honest limits

- **Scripted agents, not live LLMs.** The experiment measures graph-level behavior (detection latency, false stops, recovery) with deterministic agents. It does not measure detection quality on real LLM traces — that number lives in the base repo (held-out F1 1.000 on scripted trajectories, itself a proxy).
- **The detector is stateful.** One `LoopGuard` instance serves one graph run; call `guard.reset()` before reuse (tested deterministic). Concurrent invocations of the same compiled graph need one guard per run.
- **Embedder corpus matters.** TF-IDF embeddings are fitted on a thought corpus you supply; a corpus far from your agent's vocabulary degrades similarity quality. Calibrate the threshold on your own trajectories (`SemanticRevisitDetector.calibrate`).
- **Dependency note.** The base package is GitHub-only: the PyPI name `loopguard` belongs to an unrelated arg-hash decorator package, so `pyproject.toml` references the GitHub repo directly.

## Citation

```bibtex
@software{rehman2026loopguard_langgraph,
  author = {Rehman, Habib Ur},
  title = {loopguard-langgraph: Semantic Loop Detection as a LangGraph Guard Node},
  year = {2026},
  url = {https://github.com/habib-analyst/loopguard-langgraph}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)

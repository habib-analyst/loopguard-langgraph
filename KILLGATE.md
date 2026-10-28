# Kill-gate record — loopguard-langgraph (Agentic 12 Programme, P12)

Date: 2026-10-28. Gate question: **does an established package already provide
an equivalent LangGraph loop-guard (semantic loop detection as a drop-in
LangGraph node/middleware)?** If yes → kill, no build.

## Prior art found (PyPI / GitHub / web search, 2026-10-28)

| Candidate | What it is | Equivalent? |
|---|---|---|
| **nudgeops** 0.2.6 (PyPI, single author, "Beta", download stats unreported) | Closest hit. Wraps LangGraph tool nodes (`SmartNudgeOps.apply(builder)`), blocks repeated actions/intents. | **No.** Its semantic layer normalizes thoughts with an **LLM call per step** (needs an API key, nondeterministic, per-step cost); its no-LLM layer is pattern/hash detectors (stutter, state-hash failure memory keyed on `(state, action)`). No observation/outcome gating of thought revisits, no reflection-dissimilarity enforcement, no cost-justified budget. No evidence of establishment (0.2.x, one maintainer, no adoption metrics). |
| **breakerbox** 1.0.0 (PyPI) | `guard()` wrapper for LangGraph apps; runtime **USD spend enforcement** with degradation ladder; "semantic repeat detection" is one feature of a cost product. | No — spend-first circuit breaker, not a thought-level semantic revisit detector with reflection/budget semantics. |
| **agentgaurd-sdk** (PyPI) | LangGraph via callbacks; loop check = `SequenceMatcher` string ratio over a sliding window + budget ceiling. | No — string-edit similarity is not embedding semantics; callback-based, not a graph node. |
| **able-harness-kit** (PyPI) | `LoopGuardMiddleware` for LangChain `create_agent`: stops when the **same tool is called with the same args** with no progress. | No — exact-args hashing, precisely the baseline class loopguard measured at F1 0.783 vs 1.000 semantic. |
| **ai-loopguard** (GitHub/dev.to) | Failure-count circuit breaker that escalates to a stronger model. | No — counts failures, no semantic detection, not LangGraph-specific. |
| **LangGraph `recursion_limit`** (built-in) | Hard super-step cap; raises `GraphRecursionError`. | No — a cap is not detection: it cannot tell a loop from legitimate long work. Used in this repo's experiment as the naive max-steps baseline arm. |
| PyPI name collision | `pip install loopguard` on PyPI installs an **unrelated** arg-hash decorator package (different author). Habib's loopguard is GitHub-only. | Noted in README; this package depends on the GitHub repo explicitly. |

## Verdict: PASS — proceed to build

No established package packages an **embedding-based, observation-gated
semantic revisit detector** (a similar thought only counts as a loop when the
world hasn't told the agent anything new) **plus reflection-dissimilarity
enforcement** (a proposed new strategy must be measurably different from the
failed ones) **plus a cost-justified budget**, as a LangGraph guard node.
The differentiation is mechanism-level, not cosmetic, and the base detector's
advantage over hashing is already measured in the base repo
(held-out F1 1.000 vs 0.783, Δ +0.217). Adjacent prior art is credited in the
README rather than hidden.

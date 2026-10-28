"""Deterministic scripted agents used by the experiment, tests and demo.

No LLM anywhere: each scenario is a small state machine whose node emits a
(thought, observation) pair per step, exactly the surface the guard sees in
a real LangGraph agent. Behaviors covered:

- SemanticLoopAgent    — same intent, reworded every step, stale observations
- ExactRepeatAgent     — the identical thought/action repeated verbatim
- CleanShortAgent      — progresses and solves in 5 steps
- CleanLongAgent       — progresses and solves in 14 steps (a step cap kills it)
- RetryProgressAgent   — repeats a call, but the observation CHANGES (not a loop)
- LoopRecoverAgent     — loops, reflects to a genuinely different strategy, solves
- LoopRephraseAgent    — loops; its second reflection is a rephrase of the first
                         (the guard must reject it and stop)
"""
from __future__ import annotations

import random

from langchain_core.messages import AIMessage, ToolMessage

TOPICS = ["the quarterly revenue figure", "the API rate limit",
          "the database password", "the deployment config",
          "the customer churn rate", "the model checkpoint"]

LOOP_THOUGHTS = [
    "I should search for {t}",
    "Let me try searching {t} again with different terms",
    "Maybe a reworded query will find {t}",
    "I'll run another search for {t}, tweaking the keywords",
    "Searching {t} once more, the last query must have been off",
    "Let me look up {t} in the docs instead",
    "Perhaps the docs have {t} under a different name",
    "I will check the documentation for {t} again",
]
LOOP_OBS = [
    "search: 0 results for '{q}'",
    "docs: no page matching '{q}'",
    "search: empty result set for '{q}'",
    "kb: 0 hits for '{q}'",
]
LOOP_QUERIES = ["{t}", "{t} q3", "{t} latest", "{t} docs", "{t} 2026"]

# Distinct-worded strategies for the stuck agents' reflections.
STUCK_STRATEGIES = [
    "Scan the documentation index page by page",
    "Enumerate the knowledge base categories one at a time",
    "Browse the configuration archive section by section",
    "Walk the API reference endpoints in order",
]


class ScriptedAgent:
    """Base: subclasses implement step() -> (thought, observation, solved)."""

    name = "scripted"

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.calls = 0          # agent-node invocations (ground-truth steps)
        self._reflect_calls = 0

    def step(self, state: dict) -> tuple:
        raise NotImplementedError

    def reflect(self, trace: str) -> str:
        raise NotImplementedError

    # LangGraph node ------------------------------------------------------
    def node(self, state: dict) -> dict:
        thought, observation, solved = self.step(state)
        self.calls += 1
        return {
            "thought": thought,
            "observation": observation,
            "thought_history": [thought],
            "observation_history": [observation],
            "messages": [AIMessage(content=thought),
                         ToolMessage(content=observation, tool_call_id=f"t{self.calls}")],
            "done": solved,
            "steps": state.get("steps", 0) + 1,
        }


class SemanticLoopAgent(ScriptedAgent):
    """Rewords the same search intent forever; the world never answers.
    Reflections produce new strategies but the behavior never improves —
    a genuinely stuck agent."""

    name = "semantic_loop"

    def __init__(self, topic: str = "the database password", seed: int = 7):
        super().__init__(seed)
        self.topic = topic

    def step(self, state):
        thought = self.rng.choice(LOOP_THOUGHTS).format(t=self.topic)
        q = self.rng.choice(LOOP_QUERIES).format(t=self.topic)
        obs = self.rng.choice(LOOP_OBS).format(t=self.topic, q=q)
        return thought, obs, False

    def reflect(self, trace):
        strategy = STUCK_STRATEGIES[self._reflect_calls % len(STUCK_STRATEGIES)]
        self._reflect_calls += 1
        return strategy


class ExactRepeatAgent(ScriptedAgent):
    """The identical thought and stale observation, verbatim, forever."""

    name = "exact_repeat"

    def __init__(self, topic: str = "the database password", seed: int = 7):
        super().__init__(seed)
        self.topic = topic

    def step(self, state):
        return (f"I should search for {self.topic}",
                f"search: 0 results for '{self.topic}'", False)

    def reflect(self, trace):
        strategy = STUCK_STRATEGIES[self._reflect_calls % len(STUCK_STRATEGIES)]
        self._reflect_calls += 1
        return strategy


class CleanShortAgent(ScriptedAgent):
    """Textbook progression: index -> inspect -> verify -> done (5 steps)."""

    name = "clean_short"
    STEPS = [
        ("I need to find {t}; first I will check the index",
         "index lists 3 candidates for {t}"),
        ("The index has candidates; I will inspect the first one",
         "inspect: candidate_1 is an outdated version"),
        ("Candidate 1 is outdated; checking candidate 2 instead",
         "inspect: candidate_2 matches the required version"),
        ("Candidate 2 matches; verifying its checksum now",
         "verify: checksum valid for candidate_2"),
        ("Checksum valid, the value is confirmed",
         "task complete: {t} located"),
    ]

    def __init__(self, topic: str = "the quarterly revenue figure", seed: int = 7):
        super().__init__(seed)
        self.topic = topic

    def step(self, state):
        i = min(state.get("steps", 0), len(self.STEPS) - 1)
        thought, obs = self.STEPS[i]
        return thought.format(t=self.topic), obs.format(t=self.topic), \
            i == len(self.STEPS) - 1

    def reflect(self, trace):  # never expected to be called
        return "Restart the search from the index with fresh criteria"


class CleanLongAgent(ScriptedAgent):
    """A 14-hop audit trail: each hop consults a different system and the
    observation brings genuinely new facts (new custodian, new archive,
    new vocabulary), so every step is real progress. Solves at step 14 —
    beyond a naive 10-step cap."""

    name = "clean_long"
    STEPS = [
        ("Starting the audit trail for {t}; opening the billing ledger first",
         "ledger: quarterly exports archived under folder q3-billing, custodian maria"),
        ("Billing points at a warehouse export; checking the inventory system",
         "inventory: shipment manifests reference order batch 4417, signed by depot staff"),
        ("The manifest mentions a customer survey; pulling the survey platform records",
         "survey platform: response workbook stored in bucket feedback-july, owner priya"),
        ("Survey raw data lives in the analytics warehouse; querying the schema catalog",
         "catalog: table mart_churn_inputs partitioned by region, refreshed nightly"),
        ("The catalog names a feature pipeline; inspecting the pipeline registry",
         "registry: pipeline churn-features v12 maintained by data engineering, runbook linked"),
        ("The runbook cites an experimentation log; reading the experiment tracker",
         "tracker: experiment 208 compared pricing tiers across two cohorts in august"),
        ("Experiment notes reference support tickets; scanning the helpdesk archive",
         "helpdesk: ticket cluster about cancellations spike, tagged billing-confusion"),
        ("Tickets mention a retention campaign; opening the marketing calendar",
         "calendar: winback campaign launched september, managed by growth team"),
        ("Campaign results sit in the finance drive; browsing the finance folder",
         "finance drive: retention uplift workbook, analyst farah, tab campaign-roi"),
        ("The workbook defers to regional breakdowns; fetching the regional report",
         "regional report: europe churn driven by onboarding drop-off, asia stable"),
        ("Regional report cites app telemetry; querying the telemetry dashboard",
         "dashboard: activation funnel shows step three abandonment on android builds"),
        ("Telemetry ties drop-off to a release; checking the release notes archive",
         "release notes: build 5.2 changed signup flow, rolled out gradually in october"),
        ("Release notes name the metrics owner; pulling her summary memo",
         "memo: onboarding friction explains most of the quarterly movement, appendix attached"),
        ("The memo appendix carries the audited figure; extracting it now",
         "task complete: {t} extracted from the audit appendix"),
    ]

    def __init__(self, topic: str = "the customer churn rate", seed: int = 7):
        super().__init__(seed)
        self.topic = topic

    def step(self, state):
        i = min(state.get("steps", 0), len(self.STEPS) - 1)
        thought, obs = self.STEPS[i]
        return thought.format(t=self.topic), obs.format(t=self.topic), \
            i == len(self.STEPS) - 1

    def reflect(self, trace):  # never expected to be called
        return "Restart the chain from the registry root with fresh criteria"


class RetryProgressAgent(ScriptedAgent):
    """Repeats the same call twice, but the observation CHANGES (timeout,
    then partial results). Legitimate persistence — the observation gate
    must not flag it. Solves in 4 steps."""

    name = "retry_progress"
    STEPS = [
        ("I should search for {t}", "search: timeout after 30s"),
        ("I should search for {t}", "search: partial results, 2 candidates found"),
        ("Two candidates this time; inspecting the first",
         "inspect: candidate_1 matches {t}"),
        ("Candidate 1 matches; verifying freshness",
         "verify: fresh result confirmed, task complete"),
    ]

    def __init__(self, topic: str = "the API rate limit", seed: int = 7):
        super().__init__(seed)
        self.topic = topic

    def step(self, state):
        i = min(state.get("steps", 0), len(self.STEPS) - 1)
        thought, obs = self.STEPS[i]
        return thought.format(t=self.topic), obs.format(t=self.topic), \
            i == len(self.STEPS) - 1

    def reflect(self, trace):  # never expected to be called
        return "Restart the search from the index with fresh criteria"


class LoopRecoverAgent(ScriptedAgent):
    """Loops on paraphrased searches until the guard forces a reflection;
    the reflection switches to reading the local config file, which works."""

    name = "loop_recover"
    RECOVERY_THOUGHT = ("Reading the local configuration file directly to "
                        "find the database password entry")
    RECOVERY_OBS = "config file: db_password entry located, task complete"

    def __init__(self, topic: str = "the database password", seed: int = 7):
        super().__init__(seed)
        self.topic = topic

    def step(self, state):
        if state.get("strategy"):
            return self.RECOVERY_THOUGHT, self.RECOVERY_OBS, True
        thought = self.rng.choice(LOOP_THOUGHTS).format(t=self.topic)
        q = self.rng.choice(LOOP_QUERIES).format(t=self.topic)
        obs = self.rng.choice(LOOP_OBS).format(t=self.topic, q=q)
        return thought, obs, False

    def reflect(self, trace):
        return ("Read the local configuration file directly and extract "
                "the password entry from disk")


class LoopRephraseAgent(ScriptedAgent):
    """Loops; first reflection is a genuinely different strategy (accepted)
    that still loops; second reflection merely rephrases the first — the
    guard must reject it and stop the graph."""

    name = "loop_rephrase"
    STRATEGY_A = "Interrogate the search index with broader synonyms for the password record"
    STRATEGY_A_REPHRASE = "Query the search index using broader synonyms to locate the password record"
    A_THOUGHTS = [
        "I will query the index using broader synonyms for the password record",
        "Trying wider synonym terms against the index for the password record",
        "Broadening the synonym query once more for the password record",
        "Let me re-run the index query with expanded synonyms for the password record",
    ]

    def __init__(self, topic: str = "the database password", seed: int = 7):
        super().__init__(seed)
        self.topic = topic

    def step(self, state):
        if state.get("strategy"):
            thought = self.rng.choice(self.A_THOUGHTS)
        else:
            thought = self.rng.choice(LOOP_THOUGHTS).format(t=self.topic)
        q = self.rng.choice(LOOP_QUERIES).format(t=self.topic)
        obs = self.rng.choice(LOOP_OBS).format(t=self.topic, q=q)
        return thought, obs, False

    def reflect(self, trace):
        self._reflect_calls += 1
        return self.STRATEGY_A if self._reflect_calls == 1 else self.STRATEGY_A_REPHRASE


SCENARIOS = [
    SemanticLoopAgent, ExactRepeatAgent, CleanShortAgent, CleanLongAgent,
    RetryProgressAgent, LoopRecoverAgent, LoopRephraseAgent,
]

LOOPING_SCENARIOS = {"semantic_loop", "exact_repeat", "loop_rephrase"}
SOLVABLE_SCENARIOS = {"clean_short", "clean_long", "retry_progress", "loop_recover"}


def corpus() -> list:
    """Thought corpus for fitting the embedder: every template the
    scripted agents can emit, across topics."""
    thoughts = set()
    for t in TOPICS:
        thoughts.update(x.format(t=t) for x in LOOP_THOUGHTS)
        thoughts.update(x.format(t=t) for x, _ in CleanShortAgent.STEPS)
        thoughts.update(x.format(t=t) for x, _ in RetryProgressAgent.STEPS)
        thoughts.add(CleanLongAgent(topic=t).step({"steps": 0})[0])
        thoughts.add(CleanLongAgent(topic=t).step({"steps": 99})[0])
    thoughts.update(STUCK_STRATEGIES)
    thoughts.update(LoopRephraseAgent.A_THOUGHTS)
    thoughts.add(LoopRephraseAgent.STRATEGY_A)
    thoughts.add(LoopRephraseAgent.STRATEGY_A_REPHRASE)
    thoughts.add(LoopRecoverAgent.RECOVERY_THOUGHT)
    thoughts.add("Read the local configuration file directly and extract "
                 "the password entry from disk")
    return sorted(thoughts)


# --- calibration trajectory generators (same families as the base repo) --

def make_loop_trajectory(rng: random.Random) -> list:
    t = rng.choice(TOPICS)
    n = rng.randint(6, 9)
    traj = []
    for _ in range(n):
        th = rng.choice(LOOP_THOUGHTS).format(t=t)
        q = rng.choice(LOOP_QUERIES).format(t=t)
        traj.append((th, rng.choice(LOOP_OBS).format(t=t, q=q)))
    return traj


def make_clean_trajectory(rng: random.Random) -> list:
    t = rng.choice(TOPICS)
    steps = CleanShortAgent.STEPS if rng.random() < 0.5 else RetryProgressAgent.STEPS
    return [(th.format(t=t), ob.format(t=t)) for th, ob in steps]


def make_guard(threshold: float = None, **kwargs):
    """LoopGuard fitted on the scenario corpus; when threshold is None it
    is calibrated on generated trajectories (base-repo methodology)."""
    import numpy as np

    from loopguard import SemanticRevisitDetector

    from .guard import LoopGuard

    if threshold is None:
        rng = random.Random(7)
        trajs, labels = [], []
        for _ in range(20):
            trajs.append(make_loop_trajectory(rng))
            labels.append(True)
        for _ in range(20):
            trajs.append(make_clean_trajectory(rng))
            labels.append(False)
        from loopguard import ThoughtEmbedder
        embedder = ThoughtEmbedder().fit(corpus())
        threshold, _ = SemanticRevisitDetector.calibrate(
            embedder, trajs, labels, thresholds=np.arange(0.30, 0.96, 0.02))
        return LoopGuard(embedder=embedder, threshold=float(threshold), **kwargs)
    return LoopGuard(corpus=corpus(), threshold=threshold, **kwargs)

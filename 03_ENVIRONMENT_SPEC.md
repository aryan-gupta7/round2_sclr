# Environment Specification

> **Scope**: This document defines the MDP, the action / observation / state types, and the episode lifecycle.
> **Implementation lives in**: `envs/recall_env/server/recall_environment.py` and `envs/recall_env/models.py`.

## MDP at a glance

| Element | Value |
|---------|-------|
| Episode | Two phases: ingestion → query |
| Observation space | Structured text (current facts batch + memory anchors + phase metadata) |
| Action space | Structured JSON describing per-fact decisions OR query/answer |
| Reward | Sparse at query phase + small per-step costs (see `06_REWARD_DESIGN.md`) |
| Termination | All queries answered, OR memory budget exceeded with no recoverable action, OR hard step limit hit |
| Reset kwargs | `difficulty: int (1-5)`, `seed: int` |

## Two-phase episode lifecycle

### Phase A — Ingestion phase

The agent receives facts in **batches of 8**. Per batch, it makes per-fact decisions in a single LLM call.

**Why batched**: see `00_PROJECT_OVERVIEW.md` — one fact per LLM call is computationally infeasible at our compute budget. Batching by 8 cuts forward passes ~8x and gives the agent contextual comparison ("which of these 8 is most important").

For a Level 3 episode (50 facts):
- Step count during ingestion = 50 / 8 = ~7 steps
- Plus query phase steps

### Phase B — Query phase

After all facts are ingested, queries arrive **one at a time**. For each query, the agent:
1. Issues a `retrieve(query_text)` action and receives top-k anchors+content
2. Issues an `answer(text)` action

Two LLM calls per query is acceptable — query count is small (~10–20 per episode).

## Action types (`models.py`)

We define **one Action class with a discriminated union over modes**. This is cleaner than multiple Action classes for OpenEnv compliance.

```python
from typing import Literal, Optional
from pydantic import BaseModel
from openenv.core.env_server import Action

class FactDecision(BaseModel):
    """One per fact in the current ingestion batch."""
    fact_id: int
    decision: Literal["store", "skip"]
    anchor: Optional[str] = None  # required if decision == "store"

class RecallAction(Action):
    mode: Literal["ingest", "retrieve", "answer", "delete"]
    # for ingest mode:
    decisions: Optional[list[FactDecision]] = None
    # for retrieve mode:
    query: Optional[str] = None
    # for answer mode:
    answer_text: Optional[str] = None  # may be the literal "UNKNOWN"
    # for delete mode (only allowed if budget violation):
    slot_id: Optional[int] = None
```

**Validation rules** (enforced in `recall_environment.step()`):
- `mode` must match the current required action type. If env is in ingestion phase, only `ingest` (and optionally `delete`) is valid.
- `decisions` length must equal the current batch size.
- `anchor` required iff `decision == "store"`. Anchor must be a non-empty string ≤ 64 tokens. Longer anchors are truncated and logged.
- `answer_text == "UNKNOWN"` is a valid answer for distractor-resistance queries.
- Malformed actions: reward = 0 for the step, penalty applied (see `06_REWARD_DESIGN.md`).

## Observation type (`models.py`)

```python
from openenv.core.env_server import Observation

class RecallObservation(Observation):
    phase: Literal["ingest", "query", "done"]
    # During ingest phase:
    current_batch: Optional[list[dict]] = None  # [{"fact_id": int, "text": str, "tags": list[str]}, ...]
    # During query phase:
    current_query: Optional[str] = None
    retrieval_results: Optional[list[dict]] = None  # populated after a retrieve action
    # Always present:
    memory_anchors: list[str]                # CURRENT anchors only, not full content
    memory_used: int
    memory_budget: int
    facts_remaining: int
    queries_remaining: int
    queries_answered: int
    last_reward: float
    instruction: str                         # phase-appropriate instruction text for the LLM
```

The agent always sees:
- The current memory anchors (to detect duplicates and contradictions). It does NOT see full content unless it explicitly retrieves.
- Budget status (slots used / total).
- A natural-language instruction telling it what to do this step.

## State type (`models.py`)

```python
from openenv.core.env_server import State

class RecallState(State):
    difficulty: int
    seed: int
    phase: str
    facts_total: int
    facts_ingested: int
    queries_total: int
    queries_answered: int
    correct_answers: int
    memory_used: int
    memory_budget: int
    cumulative_reward: float
    # For debugging / metrics:
    storage_decisions: list[dict] = []  # per-fact: stored or not, anchor written, was-later-retrieved
    failure_attribution: list[dict] = []  # per-query: which failure mode (storage / anchor / retrieve / reasoning)
```

`failure_attribution` is critical for debugging when training stalls. It is logged but does NOT contribute to reward.

## `reset(difficulty: int, seed: int)`

```python
def reset(self, difficulty: int = 1, seed: int = 0) -> RecallObservation:
    if difficulty not in (1, 2, 3, 4, 5):
        raise ValueError(f"difficulty must be 1-5, got {difficulty}")
    config = load_curriculum_config(difficulty)  # see 04_CURRICULUM.md
    self.rng = np.random.default_rng(seed)
    # data_generator is invoked here; STUB until 08_DATA_GENERATION.md is finalised
    self.facts, self.queries, self.ground_truth = data_generator.generate(config, self.rng)
    self.memory = MemoryBackend(
        budget=config.memory_budget,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
    )
    if config.prefilled_memory_count > 0:
        self.memory.prefill(data_generator.generate_prefill(config, self.rng))
    self.phase = "ingest"
    self._init_state(difficulty, seed, config)
    return self._build_observation()
```

## `step(action: RecallAction)`

Pseudocode:

```
on action.mode == "ingest":
    validate batch size matches current pending batch
    for each FactDecision:
        if decision == "skip":
            log skip
        elif decision == "store":
            if memory.full: penalty, do not store
            else: memory.store(anchor=decision.anchor, content=fact.text)
    advance batch pointer
    if all facts ingested: transition phase to "query"
    return updated observation, reward = sum of per-fact contributions

on action.mode == "delete":
    only valid during ingest phase, only if budget pressure flagged
    memory.delete(slot_id)

on action.mode == "retrieve":
    results = memory.retrieve(action.query, top_k=config.retrieval_k)
    return observation with retrieval_results populated

on action.mode == "answer":
    correct = grade(action.answer_text, ground_truth[current_query_index])
    update reward, attribution log
    advance query pointer
    if all queries answered: phase = "done", return terminal obs
```

The grade function uses **exact-match-after-normalization for synthetic data** (lowercase, strip punctuation, strip whitespace). For natural language answers, see deferred section in `08_DATA_GENERATION.md`.

## Termination

Episode ends when ANY of:
- Phase reaches "done" (all queries answered)
- Hard step limit hit (default: 2 × expected step count, prevents infinite loops on malformed actions)
- Three consecutive malformed actions (the agent is stuck)

## Boundary cases the implementation MUST handle

1. **Agent stores everything before query phase**: budget exhausted, must use `delete` to recover. If it doesn't, it cannot store further but episode continues.
2. **Agent provides empty anchor on `store`**: reject the store, count as malformed, no item added.
3. **Agent emits decisions list with wrong length**: reject entire batch, no items stored, malformed-step penalty.
4. **Retrieve before any store**: returns empty results, no error.
5. **Answer without preceding retrieve**: allowed (agent can answer from prompt context if env stays in context). No bonus or penalty for this beyond standard reward.
6. **Answer is "UNKNOWN" when ground truth says "UNKNOWN"**: full credit. (Distractor-resistance queries.)
7. **Answer is "UNKNOWN" when ground truth has a known answer**: zero reward, no penalty beyond missing the bonus.

## What `state` returns

`state` is a property returning a `RecallState`. It is read-only from the client side. Used for evaluation, plotting, and judges' replay.

## Reset kwargs forwarded over wire

OpenEnv 0.2+ supports parameterized reset. The client must support:

```python
result = await env.reset(difficulty=2, seed=42)
```

This is critical for curriculum training. Verify with smoke test in `tests/test_environment.py`.

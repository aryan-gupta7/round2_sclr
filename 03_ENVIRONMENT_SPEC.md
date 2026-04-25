# Environment Specification (REVISED)

> **REVISION NOTICE — supersedes prior `03_ENVIRONMENT_SPEC.md`**
> **Reason**: Deep research report findings on GRPO turn-count regimes. Previous spec had ~43 turns at L3, well past the 7-turn danger zone. This revision collapses to 5–8 turns per episode.
> **Date**: 2026-04-25.

## MDP at a glance

| Element | Value |
|---------|-------|
| Episode | Two phases: single-pass ingestion → query phase |
| Observation | Structured text (all facts at once during ingest, single query during query) |
| Action | Structured JSON describing all-fact decisions OR retrieve OR answer |
| Reward | Two-phase (see `06_REWARD_DESIGN.md`) |
| Termination | All queries answered, OR 3 consecutive malformed actions, OR hard step limit |
| Reset kwargs | `difficulty: int (1-5)`, `seed: int` |
| Episode turn count target | 5–8 turns per episode |

## Two-phase episode lifecycle (REVISED)

### Phase A — Single-pass ingestion (1 turn)

The agent receives **all facts in one prompt** and emits a single JSON list with per-fact decisions. This is the central revision: no streaming, no batching by 8, no multi-step ingestion.

For an L3 episode (50 facts):
- Single ingest turn, ~2K input tokens, ~3K output tokens
- The agent does NOT see queries during this turn
- Selection-under-uncertainty is preserved (queries are unknown at storage time)

### Phase B — Query phase (1 turn per query, optionally 2)

Each query is one turn by default. The agent receives the query plus current memory anchors (no full content). It can:

- **Direct answer** (1 turn): emit `{"action": "answer", "answer": "..."}`
- **Retrieve then answer** (2 turns): emit `{"action": "retrieve", "query": "..."}` first, get top-k full content, then answer next turn

For 5 queries × ~1.5 turns = ~8 turns avg. Total episode: 1 + 8 = **9 turns** at L3. We aim to push this lower at L1-L2 (fewer queries) and accept slightly higher at L4-L5 (more retrieves used).

### Per-level turn budgets

| Level | facts | queries | Expected turns |
|-------|-------|---------|----------------|
| L1 | 10 | 3 | 1 + ~3 = 4 |
| L2 | 25 | 5 | 1 + ~5 = 6 |
| L3 | 50 | 5 | 1 + ~7 = 8 |
| L4 | 80 | 6 | 1 + ~9 = 10 |
| L5 | 120 | 7 | 1 + ~10 = 11 |

L4 and L5 nudge past the 7-turn comfort zone. Acceptable because we will train these last and only if L1-L3 results are strong.

## Action types (`models.py`)

```python
from typing import Literal, Optional
from pydantic import BaseModel
from openenv.core.env_server import Action

class FactDecision(BaseModel):
    fact_id: int
    decision: Literal["store", "skip"]
    anchor: Optional[str] = None  # required if decision == "store"

class RecallAction(Action):
    mode: Literal["ingest", "retrieve", "answer"]
    decisions: Optional[list[FactDecision]] = None  # ingest mode
    query: Optional[str] = None                      # retrieve mode
    answer_text: Optional[str] = None                # answer mode (may be "UNKNOWN")
```

**Removed from prior spec**: `delete` mode. With single-pass ingestion the agent decides everything in one shot — no need to recover budget mid-stream. Pre-fill items still occupy budget; agent factors that into its 50-decision plan.

**Validation rules** (enforced in `step()`):
- `mode` must match phase. Only `ingest` valid in ingestion phase. Only `retrieve`/`answer` valid in query phase.
- Ingest action: `decisions` length must equal `facts_total`. Any deviation → malformed.
- Anchor required iff decision == "store". Non-empty, ≤ 64 tokens. Longer → truncated, logged.
- Total stores in decisions list must respect `(memory_budget - prefilled_count)`. Excess stores are rejected in declaration order; a `budget_overflow_penalty` per excess.

## Observation type (`models.py`)

```python
class RecallObservation(Observation):
    phase: Literal["ingest", "query", "done"]
    # Ingestion phase:
    all_facts: Optional[list[dict]] = None         # full list shown once
    # Query phase:
    current_query: Optional[str] = None
    retrieval_results: Optional[list[dict]] = None # populated only after retrieve action
    # Always present:
    memory_anchors: list[str]
    memory_used: int
    memory_budget: int
    queries_remaining: int
    queries_answered: int
    last_reward: float
    instruction: str
```

The agent always sees memory anchors (not full content), which lets it detect what's already stored when answering — useful for direct-answer mode at low difficulty.

## State type (`models.py`)

```python
class RecallState(State):
    difficulty: int
    seed: int
    phase: str
    facts_total: int
    queries_total: int
    queries_answered: int
    correct_answers: int
    memory_used: int
    memory_budget: int
    cumulative_reward: float
    storage_decisions: list[dict] = []     # per-fact log
    failure_attribution: list[dict] = []   # per-query log
    baseline_correct: int = 0              # FIFO baseline accuracy on this seed (computed at reset)
```

`baseline_correct` is precomputed at `reset()` time so the binary reward can compare without re-running FIFO.

## `reset(difficulty: int, seed: int)`

```python
def reset(self, difficulty: int = 1, seed: int = 0) -> RecallObservation:
    if difficulty not in (1, 2, 3, 4, 5):
        raise ValueError(f"difficulty must be 1-5, got {difficulty}")
    config = load_curriculum_config(difficulty)
    self.rng = np.random.default_rng(seed)
    self.facts, self.queries, self.ground_truth = self.data_generator.generate(config, self.rng)
    self.memory = MemoryBackend(
        budget=config.memory_budget,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
    )
    if config.prefilled_memory_count > 0:
        self.memory.prefill(self.data_generator.generate_prefill(config, self.rng))
    # PRE-COMPUTE FIFO baseline accuracy on this seed for binary reward
    self.baseline_correct = run_fifo_baseline_dry(self.facts, self.queries, self.ground_truth, config)
    self.phase = "ingest"
    self._init_state(difficulty, seed, config)
    return self._build_observation()
```

The `run_fifo_baseline_dry` is a fast (no LLM) simulation of FIFO behavior on this seed, used to define the binary reward threshold.

## `step(action)` pseudocode

```
on action.mode == "ingest" (only valid at phase=="ingest"):
    validate decisions count matches facts_total
    apply each decision in order:
        if skip: log skip
        if store and budget available: memory.store(anchor, content)
        if store and budget full: rejected, increment overflow count
    transition phase to "query"
    return updated obs with first query, reward = sum of step penalties

on action.mode == "retrieve" (only valid at phase=="query"):
    results = memory.retrieve(action.query, top_k=config.retrieval_k)
    populate observation.retrieval_results
    return obs (same query, no advance)

on action.mode == "answer" (only valid at phase=="query"):
    correct = grade(action.answer_text, ground_truth[current_query_idx])
    update state, attribution
    advance query pointer
    if all queries answered: phase = "done"
    return new obs, reward
```

## Boundary cases the implementation MUST handle

1. **Decisions list wrong length**: malformed action, reward = penalty, no items stored, phase stays at ingest. Three consecutive malformed → terminate episode.
2. **Empty anchor on store**: that single store is rejected, others in the list still apply.
3. **More stores requested than budget allows**: stores apply in order until full; remainder rejected.
4. **Retrieve before any store**: returns empty results.
5. **Answer without preceding retrieve**: allowed, agent answers from anchors only.
6. **`answer_text == "UNKNOWN"` when ground truth has answer**: zero reward, no penalty.
7. **`answer_text == "UNKNOWN"` when ground truth is UNKNOWN**: full credit.
8. **Retrieve followed by another retrieve**: allowed, latest result replaces.

## Reset kwargs over wire

OpenEnv 0.2+ supports `reset(difficulty=2, seed=42)` via WebSocket. Verify in `tests/test_environment.py`.

## What changed vs prior spec — quick reference

| Element | Prior | Revised |
|---------|-------|---------|
| Ingestion structure | 8 facts per turn × ~7 turns | All facts in 1 turn |
| Delete action | Available mid-ingest | Removed (single-pass makes it unnecessary) |
| Total turns at L3 | ~43 | ~8 |
| Queries per episode at L3 | 18 | 5 |
| Retrieve+answer | Always 2 turns | 1 turn (direct) or 2 turns (with retrieve) |
| Baseline reward reference | None at env level | FIFO accuracy precomputed at reset for binary reward |

## Why single-pass ingestion preserves the trainable skill

The previous concern was: "if the agent sees all facts at once, doesn't this become trivial RAG?"

**No.** The skill being trained is unchanged:
- Agent still doesn't see queries during ingestion → selection under uncertainty intact
- Anchor authoring is still the central novel mechanism → unchanged
- Lexical mismatch between fact and future query → unchanged
- Importance prediction from fact content → unchanged

What's lost: the *sequential* nature of streaming. The agent doesn't have to predict importance "as facts arrive." But this was a design choice, not a fundamental skill — the skill is "predict importance under uncertainty about future queries," which is preserved.

What's gained: episode tractability for GRPO + faster training + cleaner credit assignment.

This is the correct trade.

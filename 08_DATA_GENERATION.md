# Data Generation — PLACEHOLDER

> **Status**: NOT YET DESIGNED. Suryansh is working on this.
> **Coding agents**: do NOT implement the body of `data_generator.py`. Implement the **interface** described below as a stub that raises `NotImplementedError`. Other code can be built against this interface.

## Why this is deferred

The fact stream structure, the query distribution, the prefill content, and the ground-truth labelling are tightly coupled. Designing them piecemeal produces incoherent training data. Suryansh is designing them together as a unit, then this doc gets filled in.

## What downstream code can rely on RIGHT NOW (the interface)

```python
# envs/recall_env/server/data_generator.py

from dataclasses import dataclass

@dataclass
class Fact:
    fact_id: int
    text: str
    tags: list[str]            # e.g., ["[IMPORTANT]"], or empty
    is_distractor: bool        # ground truth — NOT shown to agent
    is_correction_of: int | None  # if correcting earlier fact_id
    timestep: int              # ingestion order

@dataclass
class Query:
    query_id: int
    text: str
    expected_answer: str       # may be "UNKNOWN" for distractor-resistance queries
    query_type: str            # "specific" | "aggregation" | "contradiction" | "rationale" | "negative_recall" | "distractor_resistance"
    relevant_fact_ids: list[int]   # which fact(s) contain the answer; empty if UNKNOWN

@dataclass
class GroundTruth:
    queries: list[Query]
    fact_to_query_map: dict[int, list[int]]   # fact_id -> [query_ids that depend on this fact]


class DataGenerator:
    def generate(self, config: LevelConfig, rng: np.random.Generator) -> tuple[list[Fact], list[Query], GroundTruth]:
        """Return facts in ingestion order, queries in query order, and ground truth bundle."""
        raise NotImplementedError("Awaiting data design — see 08_DATA_GENERATION.md")

    def generate_prefill(self, config: LevelConfig, rng: np.random.Generator) -> list[tuple[str, str]]:
        """Return (anchor, content) pairs to seed memory at episode start."""
        raise NotImplementedError("Awaiting data design — see 08_DATA_GENERATION.md")
```

## Contract guarantees the data generator MUST satisfy when implemented

These are non-negotiable invariants other code depends on:

1. **Determinism**: same `(config, seed)` → same output. No global randomness.
2. **fact_id space**: unique within episode, integers 0..N-1.
3. **query_type values**: must match the strings used in level YAML's `query_distribution`.
4. **Distractor labelling**: `is_distractor=True` means no query ever has this fact in `relevant_fact_ids`.
5. **UNKNOWN queries**: `expected_answer == "UNKNOWN"` iff `relevant_fact_ids == []`.
6. **Correction chains**: if fact F is `is_correction_of=X`, then queries about that topic should resolve to F's content, not X's.
7. **Query order**: queries are emitted after all facts are ingested. No interleaved queries (MVP).
8. **Fact distribution honoured**: `distractor_rate`, `contradiction_rate`, `adversarial_tag_rate` must be respected within ±10%.

## Open design questions (Suryansh to resolve)

These are the questions Suryansh is thinking through:

1. **Domain framing**: research project simulator? customer support simulator? something else?
2. **Fact templates vs free-form generation**: locked-in templates with random fillers, or LLM-generated facts?
3. **Query templates**: same question.
4. **Pre-fill content overlap with stream**: should prefilled facts share entities with incoming facts (forces deduplication / contradiction detection) or be largely disjoint?
5. **Lexical mismatch mechanism for L3**: how exactly are fact-text and query-text phrased differently? Synonym swap? Abbreviation swap? Different syntactic frames?
6. **Adversarial tag design for L5**: what makes an `[IMPORTANT]` tag adversarial? Tag attached to genuine distractor? Tag attached to old/superseded fact?
7. **Distractor design**: how related are distractors to the real facts? Too unrelated → trivially ignorable. Too related → ambiguous ground truth.
8. **Answer grading**: exact-match-after-normalization works for templated answers. What if Suryansh wants natural-language answers? Then we need an LLM-judge grader, which is heavier.

## What the env code does in the meantime

Until data generator is implemented:

```python
# server/recall_environment.py
def reset(self, difficulty=1, seed=0):
    config = load_curriculum_config(difficulty)
    self.rng = np.random.default_rng(seed)
    self.facts, self.queries, self.ground_truth = self.data_generator.generate(config, self.rng)
    # ... rest
```

The `data_generator.generate(...)` call raises `NotImplementedError`. Tests for the environment can mock this with a tiny fixed dataset until the real generator lands.

```python
# tests/test_environment.py
@pytest.fixture
def mock_generator():
    class MockGen:
        def generate(self, config, rng):
            facts = [Fact(fact_id=i, text=f"fact-{i}", tags=[], is_distractor=False, is_correction_of=None, timestep=i) for i in range(10)]
            queries = [Query(query_id=0, text="what is fact-3", expected_answer="fact-3", query_type="specific", relevant_fact_ids=[3])]
            gt = GroundTruth(queries=queries, fact_to_query_map={3: [0]})
            return facts, queries, gt
        def generate_prefill(self, config, rng):
            return []
    return MockGen()
```

This lets the env be tested end-to-end before real data exists.

## When this doc is finalized, it must contain

(For Suryansh's reference — this is the table of contents to fill in.)

1. Domain narrative chosen (research / support / etc.)
2. Fact template catalogue (with fillers and sampling distributions)
3. Query template catalogue per query type
4. Prefill composition strategy
5. Lexical-mismatch design for L3
6. Contradiction injection design for L4
7. Adversarial tag design for L5
8. Concrete ground-truth labelling rules
9. Answer normalization / grading rules
10. End-to-end example: print one full episode at each level so a human can sanity check

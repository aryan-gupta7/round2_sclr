# Open Questions

> **Audience**: Suryansh first, teammates second.
> **Purpose**: Track every unresolved decision so nothing gets silently assumed.
> **Update protocol**: When a question is resolved, move it to "Resolved" with date + decision rationale. Do not delete.

## Critical-path questions (block other work)

### Q4. Adversarial tag design for L5

**What**: What makes a tag adversarial? Distractor with `[IMPORTANT]`? Outdated fact retaining its old tag after correction?
**Why it matters**: defines the L5 challenge.
**Status**: Suryansh designing.
**Blocks**: L5 data generation. Not blocking earlier levels.

## Implementation-detail questions (can decide while building)

### Q5. Pre-fill composition ratios

**What**: % relevant / % distractor / % outdated for prefilled memory at each level.
**Why it matters**: affects baseline difficulty and skill distribution.
**Recommended starting values** (from `05_MEMORY_BACKEND.md`): 30/50/20 at L3, shifting to 25/50/25 at L5.
**Status**: tunable post-smoke-test.

### Q6. `per_fact_storage_cost` magnitude

**What**: -0.02, -0.05, or -0.10?
**Why it matters**: too low → store everything; too high → skip everything.
**Resolution protocol**: smoke-run on L1 with three values, pick whichever produces the cleanest storage curve.
**Status**: empirical, post-smoke-test.

### Q7. Embedding dimension

**What**: 128-dim default, 64-dim hard mode. Confirmed.
**Sub-question**: do we need to validate that 64-dim actually breaks baselines? (If baselines do fine at 64-dim, low-dim isn't a real difficulty knob.)
**Resolution protocol**: run FIFO baseline at 384, 128, 64 dims. Pick the dim that produces ≥10pp accuracy gap.

### Q8. Action parser robustness vs strictness

**What**: How much malformed JSON do we forgive? Strip code fences? Try regex extraction?
**Why it matters**: too strict → many penalties early, training stalls. Too forgiving → agent never learns clean output.
**Recommended**: try-parse strict JSON; if fails, regex-extract `[...]` block; if that fails, malformed.
**Status**: implement as recommended; revisit if training stalls.

### Q9. Answer grading for non-templated answers

**What**: If Suryansh wants free-form natural language answers, exact-match grading fails. Need an LLM judge or fuzzy match.
**Why it matters**: depends on Q2 outcome.
**Decision**: defer until Q2 lands. If Q2 = templates → exact-match suffices. If Q2 = LLM-generated → need fuzzy match.

### Q10. Curriculum advancement criterion

**What**: When do we move from training L<n> to L<n+1>?
**Options**:
(a) Fixed step budget per level
(b) Eval accuracy threshold (advance when ≥X%)
(c) Eval accuracy plateau (no improvement for K steps)
**Recommended**: hybrid — fixed step budget per level, but skip ahead early if threshold (b) is hit.
**Status**: implementable now; default to (a) with 200/400/800 step budget.

## Strategic / pitch questions

### Q11. How much of the pitch is "memory policy" vs "edge deployment"?

**What**: Two viable framings:
(a) Research benchmark: "first env training memory management as a learnable skill"
(b) Practical infra: "memory layer for memory-constrained agent deployments (mobile / edge / ClawdBot)"
**Recommended**: lead with (a), close with (b). Research framing is more novel; deployment framing is more visceral.
**Status**: pitch decision, defer to Suryansh.

### Q12. RLM citation framing

**What**: Cite RLM as related work or as complementary?
**Decision**: Frame as **complementary halves of the same broader problem** (RLM = read-side, RECALL = write-side). This pre-empts the "isn't this just RLM?" challenge.
**Status**: locked. Use this framing in README.

### Q13. What's the demo's hero moment?

**What**: A single side-by-side example where FIFO/LLM-judge fails and trained policy succeeds.
**Why it matters**: the visual moment that lands the pitch in <10 seconds.
**Decision protocol**: after final eval, manually inspect `sample_trajectory_*.md` files, pick the most cinematic.
**Recommended candidate**: a query whose answer was contradicted mid-stream — FIFO returns the stale answer, trained policy returns the correction.
**Status**: locked once final eval is in.

## Scope-creep watchlist (do NOT implement without raising)

These are tempting but EXPLICITLY out of scope:

- Multi-tier memory (hot/cold L1/L2)
- Graph edges between memory items
- Compression as a learned action (`compress_store(text, ratio)`)
- Automatic clustering/summarization
- LLM-generated facts/queries (instead of templates)
- Streaming queries (interleaved with ingestion)
- Multi-agent variants
- Larger model than 3-4B
- Custom architectural changes to the base LLM

Each of these is mentioned in `00_PROJECT_OVERVIEW.md` as future work. If anyone wants to implement during the hackathon, they must justify here first.

## Resolved (history)

### Q1. Domain narrative for the data ✅ RESOLVED 2026-04-25

**Decision**: Solo PhD student doing transformer experiments over a 3-week project.
**Rationale**: Coherent, technically rich, and allows for plausible distractors and lexical mismatches.
**Implemented in**: `08_DATA_GENERATION.md`, `envs/recall_env/server/data_generator.py`

### Q2. Fact + query templates ✅ RESOLVED 2026-04-25

**Decision**: Hand-written templates with Haiku-generated vocabularies (hybrid approach).
**Rationale**: Faster iteration, deterministic, lower cost than full LLM generation, while still providing lexical variety.
**Implemented in**: `08_DATA_GENERATION.md`, `envs/recall_env/server/data_generator.py`

### Q3. Lexical mismatch design for L3 ✅ RESOLVED 2026-04-25

**Decision**: Lexical mismatch = abbreviation/expansion (LR vs learning rate) + specific-to-categorical (grad norm 47 vs instability).
**Rationale**: Forces the agent to authored anchors that bridge the gap between storage-time and query-time text.
**Implemented in**: `08_DATA_GENERATION.md`, `envs/recall_env/server/data_generator.py`

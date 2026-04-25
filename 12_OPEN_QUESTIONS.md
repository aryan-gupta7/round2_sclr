# Open Questions

> **Audience**: Suryansh first, teammates second.
> **Purpose**: Track every unresolved decision so nothing gets silently assumed.
> **Update protocol**: When a question is resolved, move it to "Resolved" with date + decision rationale. Do not delete.

## Critical-path questions (block other work)

### Q1. Domain narrative for the data
**What**: Research project simulator vs customer support vs something else.
**Why it matters**: shapes fact templates, query templates, prefill content, and pitch narrative.
**Status**: Suryansh designing. Will land in `08_DATA_GENERATION.md`.
**Blocks**: data generator implementation, prompt templates, demo content.

### Q2. Fact + query templates
**What**: Are facts/queries generated from locked templates with random fillers, or LLM-generated?
**Why it matters**: templates are deterministic and faster; LLM generation is more natural but adds variance and cost.
**Recommended default if undecided**: Templates. Faster iteration, deterministic, no API cost.
**Status**: Suryansh deciding.
**Blocks**: data generator implementation.

### Q3. Lexical mismatch design for L3
**What**: How exactly do fact-text and query-text differ? Synonym swap? Abbreviation?
**Why it matters**: this is what makes anchor authoring necessary. If mismatch is trivial, agent can succeed without learning anchors.
**Status**: Suryansh designing.
**Blocks**: L3 ground truth generation.

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

*(empty — populated as questions get answered)*

---

## Update protocol

When resolving:
```markdown
### Q<n>. <title>  ✅ RESOLVED YYYY-MM-DD
**Decision**: <one paragraph>
**Rationale**: <why>
**Implemented in**: <file paths>
```

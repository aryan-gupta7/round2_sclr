# Data Generation (FULL SPEC)

> **REVISION NOTICE — supersedes prior `08_DATA_GENERATION.md` placeholder.**
> **Domain locked**: Solo PhD student doing transformer experiments over a 3-week project.
> **Approach locked**: Hand-written templates + Haiku-generated vocabularies + deterministic seed-based assembly.
> **Date**: 2026-04-25.

## Architecture overview

```
Haiku-generated VOCABULARIES (static, committed to repo)
              +
Hand-written TEMPLATES (static, committed to repo)
              +
ASSEMBLY LOGIC (data_generator.py, runs at episode reset)
              +
SEED (passed to reset)
              =
Episode (facts + queries + ground truth)
```

The assembly logic is deterministic from `(config, seed)`. Vocabularies are filler pools the assembly samples from. Templates structure how vocabularies combine into facts and queries. Ground truth is constructed mechanically by the assembly, never inferred.

## Vocabulary categories (Haiku generates these)

Eight vocabulary files, all committed to `envs/recall_env/server/vocab/`:

| File | Count | Content |
|------|-------|---------|
| `architectures.json` | 80 | Architecture descriptions: "8L-XL transformer", "modified DiT-S/2", "MoE 4-expert routing" |
| `hyperparameters.json` | 40 | (name, abbrev, value_template) triples: ("learning_rate", "LR", "{1e-5..3e-3}") |
| `metrics.json` | 30 | (full_name, abbrev) pairs: ("validation accuracy", "val_acc"), ("perplexity", "ppl") |
| `papers.json` | 60 | Plausible paper titles + 1-line key insights |
| `hypotheses.json` | 40 | Hypothesis stems: "smaller batch sizes generalize better at low LR" |
| `decisions.json` | 30 | (action, alternative, rationale) triples for architectural decisions |
| `debug_findings.json` | 50 | (symptom, cause, fix) triples for debugging notes |
| `distractors.json` | 40 | Lab-life and irrelevant chatter: "coffee machine broken", "lunch with advisor" |

See `13_HAIKU_PROMPTS.md` for the exact prompts used to generate each.

## Fact templates (hand-written, in `data_generator.py`)

Six fact categories, ~3-5 templates each, ~25 templates total.

### Category 1: Experiment results (5 templates)

```python
EXPERIMENT_TEMPLATES = [
    "Tried {arch_abbrev} with {hp_abbrev}={value}, got {metric_abbrev}={result}.",
    "{arch_abbrev} run at {hp_abbrev}={value}: {metric_abbrev} reached {result} after {steps}k steps.",
    "Trained {arch_abbrev} for {steps}k steps with {hp_abbrev}={value}; final {metric_abbrev}={result}.",
    "Ablation: {arch_abbrev}, {hp_abbrev}={value} -> {metric_abbrev}={result}, vs baseline {result_baseline}.",
    "Quick run on {arch_abbrev}: {hp_abbrev}={value} gave {metric_abbrev}={result}, {pass_or_fail}.",
]
```

Filled from: `architectures.json` (abbrev field), `hyperparameters.json`, `metrics.json`. Numeric values sampled in plausible ranges per hyperparameter.

### Category 2: Architectural decisions (3 templates)

```python
DECISION_TEMPLATES = [
    "Decided to use {choice} over {alternative} because {rationale}.",
    "Switching from {alternative} to {choice}: {rationale}.",
    "Going with {choice} (not {alternative}). Reason: {rationale}.",
]
```

Filled from: `decisions.json`.

### Category 3: Paper findings (4 templates)

```python
PAPER_TEMPLATES = [
    "Read {paper_title}: key insight is {insight}.",
    "{paper_title} reports {insight}. Worth trying.",
    "Skimmed {paper_title}. Main takeaway: {insight}.",
    "{paper_title} contradicts our assumption — they show {insight}.",
]
```

Filled from: `papers.json`.

### Category 4: Hypothesis statements (3 templates)

```python
HYPOTHESIS_TEMPLATES = [
    "Hypothesis: {claim}. To test next.",
    "Working theory: {claim}. Need ablation.",
    "Suspecting that {claim}. Will verify.",
]
```

Filled from: `hypotheses.json`.

### Category 5: Debugging notes (4 templates)

```python
DEBUG_TEMPLATES = [
    "{symptom} caused by {cause}, fixed by {fix}.",
    "Bug: {symptom}. Root cause: {cause}. Solution: {fix}.",
    "Stuck on {symptom} for hours. Turned out to be {cause}; {fix} resolved it.",
    "{symptom} during training; investigation showed {cause}; applied {fix}.",
]
```

Filled from: `debug_findings.json`.

### Category 6: Corrections (3 templates) — only at L4+

```python
CORRECTION_TEMPLATES = [
    "Earlier I said {old_claim}. Actually {new_claim}.",
    "Update: my note about {topic} was wrong. The correct finding is {new_claim}.",
    "Correction to my prior fact: {new_claim}, not {old_claim}.",
]
```

The assembly identifies a previous fact in the same episode, generates a "corrected" version that flips its claim. Linked via `is_correction_of`.

### Category 7: Distractors (5 templates)

```python
DISTRACTOR_TEMPLATES = [
    "{distractor_topic}.",
    "Note to self: {distractor_topic}.",
    "{distractor_topic} — irrelevant but writing down.",
    "Reminder: {distractor_topic}.",
    "{distractor_topic} (not project-related).",
]
```

Filled from: `distractors.json`. These are the noise the agent must learn to skip.

## Query templates (hand-written)

Six query types matching what's specified in `04_CURRICULUM.md`.

### Type A: Specific fact retrieval (~30% of queries)

```python
SPECIFIC_QUERY_TEMPLATES = [
    "What was the {metric_full} for the {arch_full} experiment?",
    "What {metric_full} did the {arch_full} run with {hp_full}={value} achieve?",
    "How did the {arch_full} perform on {metric_full}?",
]
```

**Lexical mismatch active here**: query uses `{metric_full}` and `{arch_full}` (full forms); the original fact used `{metric_abbrev}` and `{arch_abbrev}` (abbreviations). The agent's anchor must bridge.

### Type B: Aggregation (~20%)

```python
AGGREGATION_QUERY_TEMPLATES = [
    "How many experiments tried {arch_category} architectures?",
    "How many runs used {hp_full} above {threshold}?",
    "How many {arch_category} configurations were tested?",
]
```

`arch_category` is a category over `architectures.json` (e.g., "transformer", "MoE", "diffusion"). Aggregation queries return a count.

### Type C: Contradiction (~15%) — only at L4+

```python
CONTRADICTION_QUERY_TEMPLATES = [
    "Is the claim that {old_claim_topic} still believed?",
    "Did we confirm or refute that {old_claim_topic}?",
    "What is the current view on {old_claim_topic}?",
]
```

If a fact was corrected during the stream, the answer comes from the *correction*, not the original. The query lets the agent demonstrate it tracked the supersession.

### Type D: Rationale (~15%)

```python
RATIONALE_QUERY_TEMPLATES = [
    "Why was {choice} chosen over {alternative}?",
    "What was the reason for using {choice}?",
    "Why did we go with {choice}?",
]
```

Answer comes from a Decision fact. The agent must have stored the rationale, not just the choice.

### Type E: Negative recall (~10%)

```python
NEGATIVE_RECALL_QUERY_TEMPLATES = [
    "Have we tried {thing}?",
    "Was {thing} ever attempted?",
    "Did we test {thing}?",
]
```

Sometimes "yes" with a relevant fact; sometimes "no" — answer is "UNKNOWN" or a direct "no, never tried." Tests calibration.

### Type F: Distractor resistance (~10%)

```python
DISTRACTOR_RESISTANCE_QUERY_TEMPLATES = [
    "What happened with {topic_never_in_stream}?",
    "What was the result of {plausible_but_absent_experiment}?",
]
```

Answer is "UNKNOWN". `topic_never_in_stream` is a plausible-sounding but unmentioned thing. Tests whether agent invents content vs admits ignorance.

## Lexical mismatch scheme (LOCKED)

Two systematic mismatches at L3+:

### Mismatch 1: Abbreviation/expansion

Built into the vocabularies. Each architecture, hyperparameter, and metric has both an abbreviated and a full form. Facts use abbreviations; queries use full forms. Examples:

| Fact form (abbrev) | Query form (full) |
|--------------------|-------------------|
| `LR` | `learning rate` |
| `BN` | `batch normalization` |
| `LN` | `layer normalization` |
| `8L-XL` | `8-layer extra-large` |
| `val_acc` | `validation accuracy` |
| `MoE` | `mixture of experts` |
| `flash attn` | `flash attention` |
| `grad norm` | `gradient norm` |
| `ppl` | `perplexity` |

The agent's *anchor* must bridge these. A naive anchor that just uses the fact text verbatim will fail to retrieve.

### Mismatch 2: Specific-to-categorical

Facts mention specific values; queries ask categorically.

| Fact form (specific) | Query form (categorical) |
|----------------------|--------------------------|
| "LR=3e-4" | "what learning rate" |
| "grad norm 47" | "gradient instability" |
| "val_acc=0.612" | "performance" |

This is harder than abbreviation expansion — it requires the agent to infer that "47" relates to "instability" (>50 is high). The anchor must include the categorical concept.

This is what makes anchor authoring genuinely non-trivial.

## Pre-fill content (L2+)

Pre-filled memory items are generated by the same `generate_prefill(config, seed)` function. Composition by level:

| L2 | L3 | L4 | L5 |
|----|----|----|----|
| 0 prefilled | 15 prefilled | 30 prefilled | 50 prefilled |

Each prefilled item is a (anchor, content) pair where:
- 30% are relevant to upcoming queries (will affect retrieval)
- 50% are distractors (won't be queried)
- 20% are *outdated* — they will be contradicted by stream facts at L4+ (so retrieval should pick up the correction)

The anchors of prefilled items are generated to look plausible — slightly imperfect — so the agent has to learn to write *better* anchors than a random plausible one.

## Adversarial tags (L5)

At L5, 20% of distractor facts include `[IMPORTANT]` tag in their text. A naive agent that respects tags will store these. The agent must learn that tags lie 1-in-5 times in this distribution and adjust trust accordingly.

```python
# example L5 adversarial distractor
"[IMPORTANT] Coffee machine in lab broken since Tuesday."
```

## Ground truth construction

The assembly maintains ground truth mechanically:

```python
@dataclass
class GroundTruth:
    queries: list[Query]
    fact_to_query_map: dict[int, list[int]]   # fact_id -> [query_ids dependent on it]
```

Every query is constructed *paired with the fact(s) it depends on*. The assembly:

1. Generates N facts using templates + vocabularies, assigns fact_ids 0..N-1
2. For each query type:
   a. Picks a fact (or facts) of the matching category
   b. Generates a query using the matching query template
   c. Records `relevant_fact_ids = [fact_id]`
   d. Records `expected_answer` extracted from the fact (or "UNKNOWN" for distractor-resistance)
3. Returns (facts, queries, ground_truth)

**Determinism**: same `(config, seed)` → same RNG calls → same facts → same query pairings. Ground truth is reproducible and never inferred.

## Answer grading

For MVP: **exact-match-after-normalization**.

```python
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\.]", "", text)   # strip punctuation except dots
    text = re.sub(r"\s+", " ", text)        # collapse whitespace
    return text

def grade(predicted: str, expected: str) -> bool:
    if expected == "UNKNOWN":
        return predicted.strip().upper() == "UNKNOWN"
    return normalize(predicted) == normalize(expected) or normalize(expected) in normalize(predicted)
```

The "expected in predicted" loosens grading slightly so the agent can write "the validation accuracy was 0.612" and still get credit when expected is "0.612".

This works because templated facts produce templated answers — exact-match is unambiguous.

## `data_generator.py` API (locked)

```python
class DataGenerator:
    def __init__(self, vocab_dir: str = "envs/recall_env/server/vocab"):
        self.vocab = self._load_vocab(vocab_dir)

    def generate(self, config: LevelConfig, rng: np.random.Generator) -> tuple[list[Fact], list[Query], GroundTruth]:
        facts = self._generate_facts(config, rng)
        if config.contradiction_rate > 0:
            facts = self._inject_contradictions(facts, config, rng)
        if config.adversarial_tag_rate > 0:
            facts = self._inject_adversarial_tags(facts, config, rng)
        queries, ground_truth = self._generate_queries(config, facts, rng)
        return facts, queries, ground_truth

    def generate_prefill(self, config: LevelConfig, rng: np.random.Generator) -> list[tuple[str, str]]:
        return self._generate_prefill(config, rng)
```

## End-to-end example episode (L3 — for sanity check during impl)

Once `data_generator.py` is implemented, `python data_generator.py --difficulty 3 --seed 0 --print` should produce a human-readable dump like:

```
=== EPISODE seed=0 difficulty=3 ===

PREFILLED MEMORY (15 items):
  [anchor: "transformer baseline numbers"]
    "Tried 6L-base at LR=1e-4, val_acc=0.512."
  ...

FACTS (50):
  [0] [Experiment]   "Tried 8L-XL with LR=3e-4, got val_acc=0.612 at 8k steps."
  [1] [Debug]        "Grad norm spike >50 caused by BN before attn; fixed with LN-only."
  [2] [Distractor]   "Coffee machine in lab broken since Tuesday."
  [3] [Paper]        "Read 'Scaling Sparse MoE': key insight is aux-free routing."
  ...

QUERIES (5):
  [0] specific:     "What was the validation accuracy of the 8-layer extra-large run?"
                    expected: "0.612", relevant_facts: [0]
  [1] rationale:    "Why was layer normalization chosen over batch normalization?"
                    expected: "gradient norm spike instability", relevant_facts: [1]
  [2] aggregation:  "How many experiments tried transformer architectures?"
                    expected: "5", relevant_facts: [0, 7, 12, 18, 22]
  [3] negative:     "Have we tried Mamba architectures?"
                    expected: "UNKNOWN", relevant_facts: []
  [4] distractor:   "What was the outcome of the diffusion experiment?"
                    expected: "UNKNOWN", relevant_facts: []
```

This sanity-print is mandatory before training. If facts and queries don't match cleanly to the agent reading them as a human, the agent won't learn either.

## What this doc does NOT cover (intentionally)

- LLM-judge grading for free-form answers — not needed for templated answers
- Streaming queries (interleaved with ingestion) — explicitly out of scope
- Human-curated test set — generation is sufficient
- Multi-fact synthesis for queries — at MVP, every query maps to ≤3 facts; complex multi-hop is post-MVP

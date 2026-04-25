# Curriculum & Difficulty Levels

> **Scope**: Defines the 5 difficulty levels, the skill each level teaches, and the config knobs.
> **Implementation lives in**: `training/configs/level_*.yaml` and `envs/recall_env/server/data_generator.py`.

## Design principle

Each level introduces **exactly one new skill**. Lower levels remain solvable so the agent transferring up doesn't catastrophically forget. Per-level data distribution and reward shaping are tuned so the policy can climb the curriculum without collapsing.

## The five skills

| Level | Skill being learned | What changes vs previous level |
|-------|--------------------|-------------------------------|
| L1 | Action grammar — emit valid structured ingest decisions | Tiny world; everything fits in memory; all facts queried |
| L2 | Recency + tag heuristic — prefer recent and `[IMPORTANT]`-tagged facts | Budget pressure introduced; distractors added; explicit tags |
| L3 | Anchor authoring — write retrieval-friendly anchors | Lexical shift between fact text and query text; tags removed |
| L4 | Contradiction handling — track corrections, deprecate stale facts | Updates / corrections injected during ingestion |
| L5 | Full pressure — selective storage under adversarial signals | Adversarial tags on distractors; tighter budget; longer stream |

## Per-level configuration

Each level has a YAML file in `training/configs/`. Schema:

```yaml
# Example: level_2.yaml
difficulty: 2
facts_total: 30
queries_total: 12
memory_budget: 20
batch_size: 8                    # facts per ingestion step
retrieval_k: 5                   # top-k for retrieve action
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_dim: 128               # projection dim; null = native dim
prefilled_memory_count: 0
distractor_rate: 0.3
contradiction_rate: 0.0
adversarial_tag_rate: 0.0
explicit_importance_tags: true   # facts may carry [IMPORTANT] markers
query_distribution:
  specific: 0.5
  aggregation: 0.2
  contradiction: 0.0
  rationale: 0.1
  negative_recall: 0.0
  distractor_resistance: 0.2
reward_shaping:
  per_fact_storage_cost: -0.05
  store_then_retrieved_bonus: 0.1
  skip_then_never_queried_bonus: 0.05
  malformed_step_penalty: -0.5
  budget_overflow_penalty: -0.2
system_prompt_hints:
  - "Facts marked [IMPORTANT] are likely to be queried."
  - "Recent facts are more likely to be queried than old ones."
  - "Anchors should be short phrases capturing the topic."
```

> **Query distribution proportions** are placeholders — final values will be set in `08_DATA_GENERATION.md` based on the data design Suryansh produces. The YAML schema is locked; the values are not.

## Concrete level table (target values, may shift after smoke tests)

| Knob | L1 | L2 | L3 | L4 | L5 |
|------|----|----|----|----|----|
| facts_total | 10 | 30 | 50 | 80 | 120 |
| queries_total | 5 | 12 | 18 | 25 | 30 |
| memory_budget | 8 | 20 | 25 | 25 | 20 |
| prefilled_memory_count | 0 | 0 | 15 | 30 | 50 |
| distractor_rate | 0.0 | 0.3 | 0.4 | 0.4 | 0.5 |
| contradiction_rate | 0.0 | 0.0 | 0.0 | 0.3 | 0.3 |
| adversarial_tag_rate | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 |
| explicit_importance_tags | yes | yes | no | no | no (and adversarial) |
| reward shaping density | high | high | medium | low | sparse only |
| system prompt hints | yes | yes | yes | partial | none |

## Skill-by-skill teaching detail

### L1 — Action grammar

**Goal**: agent learns to emit a valid `RecallAction` with `mode: "ingest"` and well-formed `decisions`.

- 10 facts, 8-slot budget — every fact can fit, no real selection pressure
- Every fact will be queried — store-everything is the optimal policy
- Reward signal is dense: any well-formed batch decision earns positive reward, malformed gets penalty
- This level converges fast (we expect <500 GRPO steps). It exists primarily to debug the action-parsing pipeline.

### L2 — Recency + explicit tags

**Goal**: agent learns simple importance heuristics from explicit signals.

- 30 facts, 20-slot budget — must skip ~10 facts
- 30% of facts are distractors (clearly irrelevant via content)
- Some facts carry `[IMPORTANT]` markers in their text
- Query distribution: 50% recent + tagged, 30% mid-stream tagged, 20% distractor-resistance ("UNKNOWN" answer)
- System prompt explicitly tells the agent that tagged + recent matter
- **FIFO baseline does well here** — agent must beat FIFO by at least using tag information

### L3 — Anchor authoring (THE CORE LEVEL)

**Goal**: agent learns to write anchors that bridge storage-time text and query-time text.

- 50 facts, 25-slot budget
- Pre-filled memory of 15 items present at episode start (some relevant, some not)
- Tags removed — agent must infer importance from content alone
- **Lexical mismatch built into data**: facts use one phrasing ("Reduced LR from 3e-4 to 1e-4 because..."), queries use another ("what changes were made to learning rate?"). Anchor authoring is the only way to bridge this.
- This is the level where the trained policy must clearly beat LLM-as-judge baseline. If it doesn't, the project's central claim fails.

### L4 — Contradictions & corrections

**Goal**: agent learns to track corrections and deprecate stale facts.

- 80 facts, 25-slot budget
- 30% of facts are corrections of earlier facts ("Earlier I said X, actually Y")
- Pre-filled memory contains 30 items, some of which will be contradicted by incoming facts
- Queries about contradicted facts must return the corrected version
- Skill required: detect via existing anchors that a fact is a correction; either replace anchor or store with a "supersedes" hint in the anchor text

### L5 — Full pressure

**Goal**: combined skills under adversarial conditions.

- 120 facts, 20-slot budget — extreme selectivity required
- Distractor rate 0.5
- 20% of distractors carry adversarial `[IMPORTANT]`-style tags (should NOT trust tags blindly)
- Sparse reward only — no shaping bonuses
- 50 prefilled items, mix of relevant and stale
- This is the "show off" level. If we cannot train successfully here in time, we DROP IT from training and demo only L1–L3 trained results. The env still supports L5 for future work / others to extend.

## Reward shaping schedule (full curriculum)

Reward shaping density decreases as level increases. This trains end-to-end credit assignment progressively. See `06_REWARD_DESIGN.md` for full reward formulas.

| Level | Shaping bonus on store→retrieved? | Shaping bonus on skip→never-queried? | Final-answer reward |
|-------|----------------------------------|--------------------------------------|--------------------|
| L1 | ✓ | ✓ | ✓ |
| L2 | ✓ | ✓ | ✓ |
| L3 | ✓ (smaller magnitude) | ✗ | ✓ |
| L4 | ✗ | ✗ | ✓ |
| L5 | ✗ | ✗ | ✓ (only signal) |

## How the env reads curriculum config

`recall_environment.py` does NOT hardcode level values. On `reset(difficulty, seed)`, it loads the corresponding `training/configs/level_<N>.yaml` from a path supplied via env constructor or env var.

```python
# recall_environment.py
def __init__(self, config_dir: str = "training/configs"):
    self.config_dir = config_dir

def reset(self, difficulty: int = 1, seed: int = 0):
    config_path = f"{self.config_dir}/level_{difficulty}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Curriculum config missing: {config_path}")
    config = yaml.safe_load(open(config_path))
    # ... use config to set up episode
```

This means changing curriculum knobs is a YAML edit, not a code change. **Critical for fast iteration during the hackathon.**

## Smoke testing the curriculum

Before training, verify each level loads and a random policy can complete an episode without crashing:

```python
# tests/test_environment.py
@pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
def test_random_agent_completes_episode(difficulty):
    env = RecallEnvironment()
    obs = env.reset(difficulty=difficulty, seed=0)
    while not is_terminal(obs):
        action = random_valid_action(obs)
        obs = env.step(action)
    assert env.state.queries_answered == env.state.queries_total
```

Run this before any training kickoff. If a level fails this test, the level config is broken.

---
title: RECALL — Memory Management RL Environment
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - memory-management
---

# RECALL Environment

RECALL is an OpenEnv reinforcement learning environment where the agent learns to **manage its own memory** under budget constraints. The agent must decide what to store from a stream of facts and how to phrase retrieval anchors to answer future queries.

## Quick Start

```python
from envs.recall_env import RecallEnv, RecallAction, FactDecision

# Connect to a running server
with RecallEnv(base_url="http://localhost:8000") as env:
    # Reset for Level 1
    obs = env.reset(difficulty=1, seed=42)

    # Ingestion Phase: décide which facts to store
    if obs.phase == "ingest":
        decisions = [
            FactDecision(fact_id=f["fact_id"], decision="store", anchor=f["text"][:30])
            for f in obs.current_batch
        ]
        obs = env.step(RecallAction(mode="ingest", decisions=decisions))

    # Query Phase: retrieve and answer
    if obs.phase == "query":
        # 1. Retrieve
        obs = env.step(RecallAction(mode="retrieve", query=obs.current_query))

        # 2. Answer
        obs = env.step(RecallAction(mode="answer", answer_text="The answer is ..."))
```

## Action / Observation Space

### Action

- `mode`: "ingest", "retrieve", "answer", or "delete".
- `decisions`: List of storage decisions (for `ingest` mode).
- `query`: Search string (for `retrieve` mode).
- `answer_text`: Final answer (for `answer` mode).
- `slot_id`: Slot to free up (for `delete` mode).

### Observation

- `phase`: "ingest", "query", or "done".
- `current_batch`: Batch of facts to process.
- `current_query`: The current question to answer.
- `retrieval_results`: top-k matches from memory.
- `memory_anchors`: List of currently stored anchors.
- `memory_used` / `memory_budget`: Budget status.

## Reward Design

- **Correct Answer**: +1.0
- **Storage Cost**: -0.05 per fact stored.
- **Malformed Action**: -0.5
- **Budget Overflow**: -0.2
- **Shaping Bonuses**: Credited at low difficulty levels for successful store->retrieval cycles.

## Curriculum

1. **L1**: Action grammar (everything fits in memory).
2. **L2**: Recency + Importance tags.
3. **L3**: Anchor authoring (lexical mismatch between facts and queries).
4. **L4**: Contradictions (handling updates and stale facts).
5. **L5**: Selective storage under adversarial pressure.

## Implementation Details

- **Memory Backend**: Vector store using `SentenceTransformer`.
- **Anchor Authoring**: The agent writes its own retrieval anchors.
- **Single Tier**: Currently supports a single fast-access memory tier.
- **Typed Interface**: Full Pydantic validation of actions and observations.

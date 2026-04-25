---
title: RECALL Memory Environment
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
app_port: 8000
tags:
  - openenv
  - rl
  - memory
  - llm
---

# 🧠 RECALL — Memory-Constrained Long-Horizon Memory

RECALL is an **OpenEnv** reinforcement learning environment where the agent learns to **manage its own memory** under budget constraints. Given a stream of facts (experiment logs, papers, decisions, debug notes), the agent must decide what to store, craft retrieval anchors, and answer future queries from memory.

> **Key insight**: RECALL trains the _write-side_ of memory management — complementary to read-side approaches like RLM.

## What This Is

A PhD student runs transformer experiments over 3 weeks. Facts arrive as a batch: experiment results, paper insights, design decisions, debugging notes, and irrelevant distractions. The agent must:

1. **Decide** which facts to store (skip distractors, prioritize queryable items)
2. **Author anchors** — short phrases the agent writes to enable future retrieval
3. **Retrieve** from memory using authored anchors
4. **Answer** queries about stored information — or say "UNKNOWN" if the fact was skipped

## Quick Start

```python
from envs.recall_env import RecallEnv, RecallAction
from envs.recall_env.models import FactDecision

async with RecallEnv.from_env("openenv/recall-env") as env:
    obs = await env.reset(difficulty=1, seed=42)

    # Phase 1: Ingestion — all facts at once
    if obs.phase == "ingest":
        decisions = [
            FactDecision(fact_id=f["fact_id"], decision="store", anchor=f["text"][:30])
            for f in obs.all_facts
        ]
        result = await env.step(RecallAction(mode="ingest", decisions=decisions))
        obs = result.observation

    # Phase 2: Query loop
    while obs.phase == "query":
        # Retrieve
        result = await env.step(RecallAction(mode="retrieve", query=obs.current_query))
        obs = result.observation

        # Answer
        answer = obs.retrieval_results[0]["content"] if obs.retrieval_results else "UNKNOWN"
        result = await env.step(RecallAction(mode="answer", answer_text=answer))
        obs = result.observation
```

## Action Space

| Field         | Type                                 | Description                            |
| ------------- | ------------------------------------ | -------------------------------------- |
| `mode`        | `"ingest" \| "retrieve" \| "answer"` | Action type                            |
| `decisions`   | `List[FactDecision]`                 | Storage decisions (ingest mode only)   |
| `query`       | `str`                                | Search query (retrieve mode only)      |
| `answer_text` | `str`                                | Answer or "UNKNOWN" (answer mode only) |

## Observation Space

| Field                           | Type                            | Description                        |
| ------------------------------- | ------------------------------- | ---------------------------------- |
| `phase`                         | `"ingest" \| "query" \| "done"` | Current episode phase              |
| `all_facts`                     | `List[Dict]`                    | Full fact list (ingest phase only) |
| `current_query`                 | `str`                           | Active query (query phase only)    |
| `retrieval_results`             | `List[Dict]`                    | Top-k memory matches               |
| `memory_anchors`                | `List[str]`                     | Current stored anchors             |
| `memory_used` / `memory_budget` | `int`                           | Budget status                      |
| `queries_remaining`             | `int`                           | Queries left in episode            |
| `last_reward`                   | `float`                         | Reward from previous step          |

## Reward Design

Two-phase system for GRPO stability:

- **Phase 1 (Bootstrap)**: Dense shaping at L1/L2 — correctness + storage/retrieval bonuses + malformed penalties
- **Phase 2 (Binary)**: Agent accuracy vs FIFO baseline accuracy
  - Agent > baseline + 5pp → reward = **+1.0**
  - Agent > baseline → reward = **+0.3**
  - Agent ≤ baseline → reward = **0.0**

## Curriculum

| Level | Facts | Budget | Challenge                               | Bootstrap |
| ----- | ----- | ------ | --------------------------------------- | --------- |
| L1    | 10    | 8      | Action grammar, `[IMPORTANT]` tags      | 100 steps |
| L2    | 25    | 20     | Distractor filtering (30%)              | 200 steps |
| L3    | 50    | 25     | Anchor authoring, lexical mismatch      | None      |
| L4    | 80    | 30     | Contradictions, corrections             | None      |
| L5    | 120   | 40     | Adversarial tags, deceptive distractors | None      |

## Data Domain

Facts are generated from Haiku-created vocabularies covering:

- **Architectures** (80 items): transformers, MoE, diffusion, SSM, hybrid, vision, RNN
- **Hyperparameters** (40 items): LR, WD, dropout, batch size, etc.
- **Metrics** (30 items): accuracy, loss, perplexity, throughput, etc.
- **Papers** (60 items): research insights across architecture, training, efficiency
- **Decisions** (30 items): architecture and training design choices
- **Debug Findings** (50 items): training bugs with symptoms/causes/fixes
- **Distractors** (40 items): lab life, scheduling, personal, admin

## References

- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
- RLM (Recursive Language Models) — read-side memory management
- MemGPT, GraphRAG, Generative Agents — related memory systems

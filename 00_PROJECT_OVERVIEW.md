# RECALL — Project Overview

> **Audience**: Read this first. This is the north star. Every other doc must align with what's stated here.

## What we are building

**RECALL** is an OpenEnv reinforcement learning environment in which the agent's task is to **manage its own memory** under tight budget constraints, and then answer questions that depend on that memory.

The agent encounters a stream of facts arriving over time. It must decide, in real time, which to store and how to phrase a retrieval anchor for each. Later, queries arrive. The agent retrieves from its own memory and answers.

This is a **write-side and lifetime memory** problem, not a read-side long-context problem.

## The one-line research question

> When an agent must decide what to remember from a stream it cannot fully retain, can RL train better selection-and-anchoring policies than handcrafted heuristics?

This is **falsifiable, measurable, and crisp**. Every design decision must serve answering this question.

## What makes this novel

1. **Memory management as a learnable RL policy**, not a hardcoded module. Most prior systems (MemGPT, GraphRAG, Generative Agents) use rules or LLM-as-judge for storage decisions. We train the storage policy end-to-end.
2. **Learned anchor authoring**. The agent doesn't just decide *whether* to store a fact — it writes a short retrieval anchor optimised for matching the future query distribution. This is the core technical contribution.
3. **Memory policy as a deployable artifact** separable from the underlying LLM. Train once, plug into any agent system that needs long-lived memory.

## What it is NOT (avoid scope creep)

- NOT a graph-based memory system (graphs are explicit future work)
- NOT a multi-tier hot/cold memory system (MemGPT did this; mention as future work only)
- NOT a "large dataset RAG" benchmark (RAG optimises retrieval over fixed corpus; we optimise storage anticipating retrieval)
- NOT a model architecture modification (we wrap a frozen base LLM with a trained memory policy)

## Positioning vs related work

| Work | What it does | Why we are different |
|------|-------------|---------------------|
| MemGPT / Letta | OS-style virtual memory paging with hardcoded rules | We learn the paging/storage policy via RL |
| GraphRAG | Heuristic graph extraction from text | We don't build graphs; we learn task-conditional importance |
| Generative Agents | LLM-as-judge importance scoring | We use reward signal to train policy, not inference-time heuristics |
| Compressive Transformer | Architecture-level memory in attention | We wrap any frozen LLM, no architecture change |
| RLM (Recursive Language Models) | Read-side decomposition of long inputs | We address write-time and lifetime — complementary, not competing |
| LongMemEval / LoCoMo | Evaluate end-to-end task performance | We isolate memory policy as a separable trainable skill |

**Always cite RLM in README.** Frame: *RLM addresses read-time long context. RECALL addresses write-time and lifetime memory. These are complementary.*

## The demo strategy (this wins or loses the hackathon)

The judges will not be convinced by prose claiming memory matters. They will be convinced by a **side-by-side demonstration on identical fact streams**:

1. **Exhibit A**: Vanilla agent with FIFO memory fails on a specific scenario (stored a distractor instead of an important fact, or dropped a critical earlier fact under recency pressure).
2. **Exhibit B**: Trained agent on the same exact fact stream + same queries answers correctly.
3. **Exhibit C**: Numbers across the full eval set:
   - Store-everything baseline: hits memory cap, fails
   - FIFO: ~38% accuracy
   - LLM-as-judge importance: ~51% accuracy
   - Trained RECALL policy: target ≥70% accuracy

**Every design decision below should make this demo cleaner and more convincing.**

## Application framing for the pitch

Position RECALL as infrastructure for **long-lived agents on memory-constrained deployments**:

- Mobile / edge deployment of agentic systems (limited slots, limited embedding compute)
- Multi-week research / engineering assistants
- Chatbots that maintain coherent state across sessions without context bloat

Concrete narrative: *"Today's agentic frameworks like ClawdBot store everything in JSONL. That doesn't scale. We trained a policy that knows what to keep — small enough to ship on edge devices."*

## What is locked vs what is open

**Locked** (do not revisit without strong reason):
- 5-level curriculum, one new skill per level
- Batched ingestion (8 facts per step) to keep training tractable
- Free-form anchor authoring as the central novel mechanism
- Pre-filled memory at L2+ (reset injects baseline content)
- Single-tier memory in MVP (multi-tier deferred)
- Qwen2.5-3B-Instruct with LoRA, GRPO via TRL
- 128-dim embeddings default, configurable
- Three baselines: store-all, FIFO, LLM-as-judge

**Open** (Suryansh is designing):
- Synthetic fact dataset structure
- Query distribution and types
- Pre-filled memory composition ratios
- Per-fact storage cost magnitude

When data design lands, it slots into `08_DATA_GENERATION.md` (currently a placeholder).

## Success criteria for hackathon

In priority order:
1. Working env on HF Spaces, OpenEnv-compliant
2. Three baselines running and producing reproducible numbers
3. Trained policy beats best baseline by ≥10 percentage points on at least Level 1 and Level 2
4. Reward curves and side-by-side qualitative example in README
5. Mini-blog or sub-2-min video linked from README
6. Clean Colab notebook judges can re-run

If we run out of time: drop levels 3–5 from training (env still supports them), keep L1/L2 trained results clean. **Better to nail two levels than half-train five.**

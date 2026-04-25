# RECALL — Documentation Index

> **Read this first** if you're new to the project. This file is the entry point.

## What is RECALL

An OpenEnv RL environment training a base LLM to manage its own memory: deciding what to store, how to phrase retrieval anchors, and how to retrieve under tight budget constraints.

See `00_PROJECT_OVERVIEW.md` for the full pitch.

## Reading order

### If you're a coding agent or new teammate
Read in this order, no skipping:

1. **`00_PROJECT_OVERVIEW.md`** — what we're building and why
2. **`02_AGENT_RULES.md`** — rules of engagement (changelog discipline, scope policy)
3. **`01_REPOSITORY_STRUCTURE.md`** — exact folder layout (OpenEnv-mandated)
4. **`12_OPEN_QUESTIONS.md`** — what's NOT decided yet (avoid implementing these)

Then read the doc(s) for whatever you're building:

- Building the env? → `03_ENVIRONMENT_SPEC.md`
- Working on memory? → `05_MEMORY_BACKEND.md`
- Tuning rewards? → `06_REWARD_DESIGN.md`
- Building baselines? → `07_BASELINES.md`
- Setting up training? → `09_TRAINING_PIPELINE.md`
- Adding metrics / plots? → `10_EVALUATION.md`
- Deploying? → `11_DEPLOYMENT.md`

### If you're Suryansh
You already know the project. Use this index to find specific topics.

## Document map

| File | Status | Owner |
|------|--------|-------|
| `00_PROJECT_OVERVIEW.md` | Locked | Suryansh |
| `01_REPOSITORY_STRUCTURE.md` | Locked | Anyone (it follows OpenEnv) |
| `02_AGENT_RULES.md` | Locked | Anyone |
| `03_ENVIRONMENT_SPEC.md` | Locked | Suryansh |
| `04_CURRICULUM.md` | Locked at structure level; values may shift | Suryansh |
| `05_MEMORY_BACKEND.md` | Locked | Anyone |
| `06_REWARD_DESIGN.md` | Locked at structure level; magnitudes may tune | Suryansh |
| `07_BASELINES.md` | Locked | Anyone |
| `08_DATA_GENERATION.md` | **PLACEHOLDER — Suryansh designing** | Suryansh |
| `09_TRAINING_PIPELINE.md` | Locked | Anyone |
| `10_EVALUATION.md` | Locked | Anyone |
| `11_DEPLOYMENT.md` | Locked | Anyone |
| `12_OPEN_QUESTIONS.md` | Living document | Suryansh + contributors |

## Build order (suggested)

The dependency-respecting order to build the project:

```
Day 1 (parallelizable):
  Track A: openenv init, set up repo skeleton, validate (1 person, ~2 hr)
  Track B: stub data_generator.py with mock for tests (1 person, ~1 hr)
  Track C: implement memory_backend.py end-to-end (1 person, ~3 hr)

Day 2:
  Track A: implement recall_environment.py (reset, step, state) using mock data — ~5 hr
  Track B: implement rewards.py — ~2 hr
  Track C: write tests for env, memory, rewards — ~3 hr

Day 3:
  Track A: implement baselines (store_all, FIFO, llm_judge) — ~3 hr
  Track B: write run_baselines.py + plot harness — ~2 hr
  Track C: smoke-test deployment to HF Spaces with mock data — ~2 hr
  [Suryansh's data_generator.py lands here — replace mock]

Day 4:
  All hands: GRPO training script, Colab notebook, smoke training at L1
  Tune reward shaping based on smoke results
  Begin L1 training

Day 5:
  Continue training through L2, L3 (and L4/L5 if compute allows)
  Generate plots, write README, mini-blog/video
  Final HF Spaces push, validate deployment
```

## Key invariants (always true)

1. The env is reachable at a single HF Spaces URL.
2. `reset(difficulty=N, seed=S)` is deterministic.
3. The trained policy beats every baseline at every level it was trained on.
4. All numbers in plots come from a held-out seed set never used in training.
5. The Colab notebook can be re-run by judges without further setup.

## When in doubt

If you don't know whether to do something:
1. Check if it's in `00_PROJECT_OVERVIEW.md` "Locked" — if yes, do it
2. Check if it's in `00_PROJECT_OVERVIEW.md` "Open" — if yes, write it into `12_OPEN_QUESTIONS.md` and ask
3. If it's neither, it's probably scope creep. Don't.

# Reward Design (REVISED)

> **REVISION NOTICE — supersedes prior `06_REWARD_DESIGN.md`**
> **Reason**: Deep research findings show binary reward >> dense shaping for GRPO. Prior spec had ~6 dense shaping bonuses that would flatten group variance and weaken gradient. This revision adopts the two-phase pattern (dense bootstrap → binary baseline-comparison).
> **Date**: 2026-04-25.

## Core principle (CHANGED)

**Binary reward against FIFO baseline accuracy is the primary signal.**

Dense shaping is used ONLY during the bootstrap phase (first ~200 GRPO steps) to get the policy off zero. After bootstrap, all shaping is removed and the agent is graded purely on whether its accuracy on this episode beats the FIFO baseline's accuracy on the same seed.

This matches the empirical finding from TRL's Wordle/Sudoku experiments: GRPO ranks completions within a group of 8. Tight ranking from binary outcomes produces sharp gradients. Dense shaping flattens variance and weakens gradients.

## Reward formula (REVISED)

### Phase 1 — Bootstrap (training_step < bootstrap_steps)

Dense shaping ONLY at L1 and L2. At L3+, bootstrap is omitted (we expect transfer from earlier levels to provide a learning signal).

```python
def phase1_reward(episode_result, config):
    r = 0.0
    # Per-query correctness — primary signal even in bootstrap
    r += episode_result.correct_answers * 1.0
    # Mild shaping during bootstrap
    if config.difficulty <= 2:
        r += episode_result.stored_then_retrieved_count * 0.1
        r -= episode_result.memory_used * 0.02
    # Sharp penalty for malformed actions — must dominate
    r += episode_result.malformed_count * (-0.5)
    r += episode_result.budget_overflow_count * (-0.2)
    return r
```

### Phase 2 — Binary baseline-comparison (training_step ≥ bootstrap_steps)

```python
def phase2_reward(episode_result, baseline_result, config):
    # Edge case: if agent never answered anything correctly, no reward
    if episode_result.correct_answers == 0:
        return 0.0
    # Edge case: malformed-action spam → still penalize
    if episode_result.malformed_count >= 3:
        return -1.0
    # Primary binary signal: did agent beat FIFO baseline accuracy?
    agent_acc = episode_result.correct_answers / config.queries_total
    baseline_acc = baseline_result.correct_answers / config.queries_total
    if agent_acc > baseline_acc + 0.05:    # 5pp margin → clean win
        return 1.0
    elif agent_acc > baseline_acc:          # narrow win → partial
        return 0.3
    else:
        return 0.0
```

The 5pp margin prevents reward noise from FIFO's seed-dependent variance.

### Phase boundary

`bootstrap_steps` is a level-dependent config:

```yaml
# level_1.yaml
bootstrap_steps: 100   # short bootstrap; level is easy

# level_2.yaml
bootstrap_steps: 200   # standard bootstrap

# level_3.yaml
bootstrap_steps: 0     # no bootstrap; rely on L1/L2 transfer + binary signal

# level_4.yaml, level_5.yaml
bootstrap_steps: 0     # binary only
```

The trainer passes `state.global_step` to the reward function (TRL exposes this). Reward function checks against `bootstrap_steps` to decide phase.

## Reward implementation in `rewards.py`

```python
from dataclasses import dataclass

@dataclass
class EpisodeResult:
    correct_answers: int
    stored_then_retrieved_count: int
    memory_used: int
    malformed_count: int
    budget_overflow_count: int

def compute_reward(
    episode_result: EpisodeResult,
    baseline_result: EpisodeResult,
    config: LevelConfig,
    global_step: int,
) -> float:
    if global_step < config.bootstrap_steps:
        return phase1_reward(episode_result, config)
    return phase2_reward(episode_result, baseline_result, config)
```

The reward is computed at episode end (terminal reward), with zero per-step reward except for malformed-action penalties (those still emit immediately so the agent learns valid format fast).

## What was removed from prior spec

| Removed component | Why |
|-------------------|-----|
| `store_then_retrieved_bonus` (dense, every level) | Flattened group variance at every level |
| `skip_then_never_queried_bonus` | Same |
| `per_fact_storage_cost` (dense, every level) | Same |
| `wasted_retrieval_penalty` | Was already disabled, formally removed |
| Per-step reward emission for non-malformed actions | All deferred to terminal reward, except malformed penalties |

These components survive ONLY in Phase 1 bootstrap reward at L1/L2, scaled down significantly.

## Anti-hacking analysis (REVISED — simpler with binary)

Binary baseline-comparison is harder to hack than dense shaping. The remaining hacks:

**Hack 1: Game malformed-action timing**
Could the agent emit malformed actions to skip difficult queries? No — malformed actions count toward the 3-malformed termination, after which reward becomes -1.0.

**Hack 2: Always answer "UNKNOWN"**
Could the agent answer UNKNOWN to every query? No — UNKNOWN-correct only happens for distractor-resistance queries. The 70%+ of queries with real answers will fail, agent will not beat baseline.

**Hack 3: Store nothing**
Could the agent skip everything? It loses retrieval ability. FIFO baseline stores at least some facts and gets some right. Agent storing nothing → 0% accuracy → can't beat baseline.

**Hack 4: Store everything verbatim**
At L3+ memory budget is tight. Storing everything fills budget within first 25 facts; remainder rejected. Effectively becomes a poorly-anchored FIFO. Cannot beat FIFO baseline.

**Hack 5: Inflate retrieve calls to use up turn budget**
Retrieve doesn't trigger reward. Spending turns on retrieves only delays the answer; doesn't change accuracy. No incentive.

The binary reward against FIFO baseline is genuinely hard to hack. This is the win.

## Tuning protocol (REVISED — much simpler)

Before serious training:

1. Run random policy on L1 — record mean episode reward (will mostly be 0, which is fine)
2. Run FIFO baseline on L1 — should achieve consistent positive reward in Phase 1 bootstrap; in Phase 2 it should achieve ~0.0 (it can't beat itself)
3. Sanity check: at L1 bootstrap, FIFO reward should be 2-3x higher than random
4. If at L1 the binary reward shows zero variance across 8 group completions during smoke training, the level is too hard — reduce L1 to 5 facts/3 slots

Key signal to watch in W&B:
- `train/reward_std_within_group` — if this is consistently zero, GRPO has no gradient. Need easier curriculum or longer bootstrap.

## What still gets logged (for monitoring, not reward)

Per-step instrumentation for debugging:
- `metrics/correct_answers`
- `metrics/stored_then_retrieved_count`
- `metrics/memory_used`
- `metrics/malformed_count`
- `metrics/budget_overflow_count`
- `metrics/baseline_correct`
- `metrics/agent_beat_baseline_margin`

These do NOT contribute to reward. They're for plots and the failure_attribution analysis.

## Tests (REVISED)

`tests/test_rewards.py`:

1. Phase 1 reward at L1 with bootstrap_steps=100, step=50 → returns dense-style reward
2. Phase 2 reward at L3 with step=10 (bootstrap_steps=0) → returns binary
3. Random policy reward < FIFO reward at all levels in Phase 1
4. Phase 2 binary: FIFO vs FIFO returns 0.0 (cannot beat itself)
5. Phase 2 binary: oracle policy (perfect accuracy) returns 1.0 at all levels
6. 3 malformed actions → reward floor of -1.0
7. Phase boundary at exactly `step == bootstrap_steps` switches phase correctly

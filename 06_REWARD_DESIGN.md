# Reward Design

> **Scope**: Full reward computation and the curriculum shaping schedule.
> **Implementation lives in**: `envs/recall_env/server/rewards.py`.

## Design principles

1. **Reward must be coherent across the curriculum** — same code path computes reward at all levels, only the shaping bonuses change.
2. **Cannot be game-able** — an agent that hits the reward function without solving the task should NOT get a high score.
3. **Dense at low levels, sparse at high levels** — gradient is informative early, end-to-end credit assignment is the goal late.
4. **All bonuses configured in level YAML, not in code** — see `04_CURRICULUM.md`.

## Reward components (full set)

Per-step reward is a sum over component contributions. Components:

| Component | Trigger | Magnitude (default) | Active in levels |
|-----------|---------|--------------------|--------------------|
| `correct_answer` | Agent answered query correctly | +1.0 | All |
| `wrong_answer` | Agent answered query incorrectly | 0.0 (no penalty) | All |
| `correct_unknown` | Ground truth was UNKNOWN, agent answered "UNKNOWN" | +1.0 | All |
| `false_unknown` | Ground truth had answer, agent said "UNKNOWN" | 0.0 (no penalty, no credit) | All |
| `per_fact_storage_cost` | At end of episode, sum over all stored facts | -0.05 × stored_count | All |
| `store_then_retrieved_bonus` | Stored a fact that was later retrieved AND used in correct answer | +0.1 (delayed, credited at query time) | L1, L2, L3 |
| `skip_then_never_queried_bonus` | Skipped a fact that was never queried | +0.05 | L1, L2 |
| `malformed_step_penalty` | Action failed validation | -0.5 | All |
| `budget_overflow_penalty` | Tried to store when full | -0.2 | All |
| `wasted_retrieval_penalty` | Retrieved but answered wrong using results that didn't contain answer | 0.0 (no penalty for MVP) | None — disabled |

## When reward is emitted

OpenEnv's `step()` returns reward per step. Our reward emission schedule:

- **Ingestion step**: emit any malformed-action penalties immediately. Storage-cost and shaping bonuses are accumulated and emitted at the **end of episode** (this is fine — GRPO handles episode-level credit assignment).
- **Retrieve step**: zero reward by default.
- **Answer step**: emit `correct_answer` / `correct_unknown` reward immediately. If trained reward shaping is active, also emit `store_then_retrieved_bonus` for the items used.

> Implementation note: it is OK to emit some reward at the end of episode rather than per-step. GRPO with episodic reward works fine.

## Shaping schedule (by level)

Reward shaping is gradually removed as difficulty rises. This is curriculum reward shaping with annealing.

```yaml
# L1 (high shaping)
reward_shaping:
  per_fact_storage_cost: -0.02      # very mild — at L1, storing everything is fine
  store_then_retrieved_bonus: 0.2
  skip_then_never_queried_bonus: 0.1
  malformed_step_penalty: -0.5
  budget_overflow_penalty: -0.1

# L2
reward_shaping:
  per_fact_storage_cost: -0.05
  store_then_retrieved_bonus: 0.15
  skip_then_never_queried_bonus: 0.05
  malformed_step_penalty: -0.5
  budget_overflow_penalty: -0.2

# L3 (medium shaping)
reward_shaping:
  per_fact_storage_cost: -0.05
  store_then_retrieved_bonus: 0.1
  skip_then_never_queried_bonus: 0.0   # disabled
  malformed_step_penalty: -0.5
  budget_overflow_penalty: -0.2

# L4 (light shaping)
reward_shaping:
  per_fact_storage_cost: -0.05
  store_then_retrieved_bonus: 0.0      # disabled
  skip_then_never_queried_bonus: 0.0
  malformed_step_penalty: -0.5
  budget_overflow_penalty: -0.2

# L5 (sparse only)
reward_shaping:
  per_fact_storage_cost: -0.05
  store_then_retrieved_bonus: 0.0
  skip_then_never_queried_bonus: 0.0
  malformed_step_penalty: -0.5
  budget_overflow_penalty: -0.2
```

Final-answer reward (+1.0 per correct) is constant across all levels.

## Reward formula (computational)

```python
def compute_step_reward(
    action: RecallAction,
    action_was_valid: bool,
    new_correct: int,                     # how many correct answers this step (0 or 1 typically)
    new_unknown_correct: int,
    storage_attempts_during_step: int,
    storage_attempts_rejected: int,
    config: LevelConfig,
) -> float:
    r = 0.0
    if not action_was_valid:
        r += config.malformed_step_penalty
        return r
    r += new_correct * 1.0
    r += new_unknown_correct * 1.0
    r += storage_attempts_rejected * config.budget_overflow_penalty
    return r


def compute_terminal_reward(
    state: RecallState,
    config: LevelConfig,
) -> float:
    """Episode-end reward: storage cost + shaping bonuses."""
    r = 0.0
    r += state.memory_used * config.per_fact_storage_cost
    if config.store_then_retrieved_bonus > 0:
        r += count_stored_then_retrieved(state) * config.store_then_retrieved_bonus
    if config.skip_then_never_queried_bonus > 0:
        r += count_skip_then_never_queried(state) * config.skip_then_never_queried_bonus
    return r
```

The total episode reward is the sum of step rewards plus the terminal reward.

## Anti-hacking analysis

For each shaping bonus, ask: "Can the agent maximize this without doing the task?"

### `store_then_retrieved_bonus`
- **Hack attempt**: store many items with broad anchors that match many queries.
- **Why it doesn't work**: bonus only fires when the retrieved item leads to a correct answer. Broad anchors that retrieve unrelated items don't cause correct answers.

### `skip_then_never_queried_bonus`
- **Hack attempt**: skip everything → high bonus.
- **Why it doesn't work**: skipping queried facts means wrong answers, costing more than the bonus saves.
- **Caveat**: at very low query rates this could become exploitable. Mitigate by ensuring `queries_total / facts_total >= 0.2` at all levels.

### `per_fact_storage_cost`
- **Hack attempt**: store nothing → no cost.
- **Why it doesn't work**: storing nothing means no retrievals possible, all answers wrong.

### `malformed_step_penalty`
- **Hack attempt**: emit malformed actions to end episode early.
- **Why it doesn't work**: env continues until step limit; no shortcut to terminal state; missed-answer reward dwarfs penalty cost only if the agent succeeds.
- **Caveat**: if penalty is too small relative to expected episode reward, agent could spam malformed and learn slow. Set ≥ -0.5 to ensure dominance.

### `budget_overflow_penalty`
- **Hack attempt**: spam store actions.
- **Why it doesn't work**: each rejected store costs 0.2.

## Tuning protocol

Before serious training, run smoke calibration:

1. Run random policy on L1 — record mean episode reward.
2. Run FIFO baseline on L1 — record mean reward.
3. The reward gap (FIFO − random) should be at least **3.0** points.
4. If smaller, increase `correct_answer` to 1.5 or reduce shaping noise.

Repeat at L2.

If at any level the random policy achieves >50% of FIFO's reward, the level's reward signal is too dense or too sparse — adjust shaping.

## Failure attribution (logged, NOT rewarded)

For each query, the env logs which failure occurred:

- `storage_failure`: the relevant fact was never stored
- `anchor_failure`: stored, but anchor didn't match query lexically; not retrieved
- `retrieval_failure`: stored and matched, but top-k truncated it out
- `reasoning_failure`: retrieved successfully but agent answered wrong from the content
- `success`: correct

This is logged to `state.failure_attribution`. Used for debugging and for the demo (we want to point at specific failure types when comparing baselines vs trained agent).

## Reward tests

`tests/test_rewards.py`:

1. Random policy reward < FIFO reward at all levels (sanity)
2. Malformed action always produces a step with negative reward
3. Storage cost scales linearly with stored count
4. Total episode reward is sum of step rewards + terminal reward (no double-counting)
5. Shaping bonuses are zero at L4 and L5 regardless of behavior

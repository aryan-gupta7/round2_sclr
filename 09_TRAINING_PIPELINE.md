# Training Pipeline (REVISED)

> **REVISION NOTICE — supersedes prior `09_TRAINING_PIPELINE.md`.**
> **Reason**: Adapt to new single-pass ingestion (1 turn) + binary baseline-comparison reward. Prior version assumed batched multi-step ingestion.
> **Date**: 2026-04-25.

## Model and adapter (unchanged)

- **Base**: `Qwen/Qwen2.5-3B-Instruct`
- **Adapter**: LoRA via PEFT
- **Trainer**: `GRPOTrainer` from `trl`
- **Hardware**: Colab Pro (A100 / L4) primary, HF $30 credits as fallback

LoRA config unchanged from prior spec.

## GRPO config (REVISED for new structure)

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="./outputs/recall_l1",
    num_train_epochs=3,
    per_device_train_batch_size=4,           # increased — episodes are now ~6-8 turns, lighter
    gradient_accumulation_steps=2,
    learning_rate=5e-6,                      # lowered to TRL's GRPO recommendation
    num_generations=8,                        # group size for GRPO advantage
    max_prompt_length=4096,                   # ingestion prompt with all 50 facts
    max_completion_length=2048,               # decision JSON for 50 facts
    logging_steps=5,
    save_steps=100,
    bf16=True,
    report_to="wandb",
    remove_unused_columns=False,
)
```

Key changes:
- `per_device_train_batch_size`: increased 2→4 (episodes are shorter; can fit more)
- `learning_rate`: 1e-5 → 5e-6 (TRL's GRPO recommendation, prevents instability)
- `max_prompt_length`: 2048 → 4096 (ingestion shows all 50 facts at L3)

## Connecting GRPO to OpenEnv (REVISED)

OpenEnv 0.2+ supports `environment_factory` and `rollout_func` integration with TRL. We use **environment_factory** for simplicity:

```python
from envs.recall_env.client import RecallEnv

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    reward_funcs=recall_reward,
    train_dataset=curriculum_dataset,
    environment_factory=lambda: RecallEnv.from_docker_image("recall-env:latest"),
    args=training_args,
)
```

`reward_funcs` receives `state` (which includes `global_step`) and our env's terminal reward. Implementation:

```python
def recall_reward(completions, **kwargs):
    """
    Two-phase reward — bootstrap dense, then binary baseline-comparison.
    See 06_REWARD_DESIGN.md for full formula.
    """
    global_step = kwargs.get("trainer_state", {}).get("global_step", 0)
    rewards = []
    for completion, env_result in zip(completions, kwargs["env_results"]):
        config = env_result["config"]
        if global_step < config["bootstrap_steps"]:
            r = phase1_reward(env_result, config)
        else:
            r = phase2_reward(env_result, env_result["baseline_result"], config)
        rewards.append(float(r))
    return rewards
```

The env's `state` field includes `baseline_correct` (precomputed at reset), so the reward function has everything it needs.

## Per-session state isolation (CRITICAL — from deep research findings)

The deep research report explicitly flagged this. With `num_generations=8`, the trainer opens 8 simultaneous WebSocket connections. Each must hit a separate environment instance with separate state.

**REQUIREMENTS** for `recall_environment.py`:
1. NO class-level mutable state. Everything in instance attributes.
2. `__init__` creates per-instance `self.memory`, `self.state`, `self.rng`.
3. `RecallEnvironment` factory in `app.py` creates a fresh instance per session.

```python
# server/recall_environment.py
class RecallEnvironment(Environment):
    def __init__(self, config_dir: str = "training/configs"):
        self.config_dir = config_dir
        # Per-instance state — fresh per session
        self.memory = None
        self._state = None
        self.facts = None
        self.queries = None
        self.ground_truth = None
        self.rng = None
```

```python
# server/app.py
from openenv.core.env_server import create_app

def env_factory():
    return RecallEnvironment(config_dir="training/configs")

app = create_app(env_factory, RecallAction, RecallObservation, env_name="recall")
```

OpenEnv auto-routes WebSocket sessions to fresh instances when factory pattern is used. Class-level state would corrupt across sessions.

`max_concurrent_envs` in `openenv.yaml` must be set to ≥8 for GRPO with `num_generations=8`. **Update from prior spec**: prior `openenv.yaml` had `max_concurrent_envs: 1` per the agent rules. We change this only if we verify our env supports it (it does — instance-level state is the only requirement). Set to 8.

## Curriculum schedule (REVISED with bootstrap)

```python
LEVEL_SCHEDULE = [
    # (level, num_grpo_steps, bootstrap_steps_for_level)
    (1, 200, 100),    # short bootstrap, easy level
    (2, 400, 200),    # standard bootstrap
    (3, 800, 0),      # binary only — rely on transfer from L2
    (4, 600, 0),      # only if time permits
    (5, 600, 0),      # only if time permits
]
```

`bootstrap_steps` is loaded from level config. The training loop checks `state.global_step` against it inside the reward function.

## Action parsing (REVISED for single-pass ingestion)

The model's completion at ingest is a JSON array of 50 objects. Parser:

```python
def parse_ingest_completion(completion: str, expected_count: int) -> RecallAction:
    json_block = extract_first_json_array(completion)
    if json_block is None:
        return malformed_action("ingest")
    try:
        data = json.loads(json_block)
        if not isinstance(data, list) or len(data) != expected_count:
            return malformed_action("ingest")
        decisions = [validate_fact_decision(d) for d in data]
        if any(d is None for d in decisions):
            return malformed_action("ingest")
        return RecallAction(mode="ingest", decisions=decisions)
    except (json.JSONDecodeError, ValidationError):
        return malformed_action("ingest")
```

For query phase actions (retrieve/answer), simpler single-object JSON parse with mode dispatch.

## Smoke test before training kickoff

```bash
python -m training.grpo_train --level 1 --steps 5 --eval_every 999 --base_model Qwen/Qwen2.5-3B-Instruct
```

Must succeed in <15 minutes (slightly longer than prior estimate due to larger ingest prompt).

## Compute budget (REVISED)

| Phase | Time | Cost |
|-------|------|------|
| Smoke runs | ~30 min × 4 = 2 hr | Colab Pro |
| L1 training (200 steps) | ~1.5 hr | Colab Pro |
| L2 training (400 steps) | ~3 hr | Colab Pro / HF |
| L3 training (800 steps) | ~6 hr | HF credits ($12) |
| L4 (optional) | ~5 hr | HF credits ($10) |
| L5 (optional) | ~5 hr | HF credits ($10) |
| Final eval | ~2 hr | Colab Pro |

Total within budget if we cap at L3: ~$12 of $30 credits used. L4/L5 are stretch goals.

## What gets logged to W&B (REVISED)

Every training step:
- `train/episode_reward_mean` and `episode_reward_std`
- `train/reward_std_within_group` — KEY METRIC. If zero, GRPO has no gradient signal. Catches training collapse early.
- `train/accuracy` (correct_answers / queries_total)
- `train/baseline_accuracy` (FIFO accuracy on this seed for comparison)
- `train/beat_baseline_rate` (% of episodes where agent beat baseline)
- `train/memory_utilization`
- `train/malformed_action_rate`
- `train/reward_phase` (1 or 2 — which phase active)

Per eval pass:
- `eval/accuracy_current_level`
- `eval/accuracy_L1`, `eval/accuracy_L2`, ... (regression check)
- `eval/accuracy_by_query_type`
- `eval/failure_mode_breakdown`
- `eval/sample_trajectory` (saved transcript)

The `reward_std_within_group` is the most important new metric. If it stays near zero for ≥30 steps, **stop training immediately** — GRPO has no signal to work with. Diagnose:
- Is the level too hard? Drop to easier curriculum.
- Is reward shaping flattening variance? Verify Phase 2 binary is active.
- Is the bootstrap providing enough early signal? Extend bootstrap or simplify shaping.

## Pinned dependencies (unchanged)

```toml
dependencies = [
    "transformers>=4.45",
    "trl>=0.12",
    "peft>=0.13",
    "accelerate>=0.34",
    "sentence-transformers>=3.1",
    "openenv-core",
    "wandb",
    "pyyaml",
    "numpy<2.0",
]
```

## What changed vs prior spec — quick reference

| Element | Prior | Revised |
|---------|-------|---------|
| Episode turn count | ~43 at L3 | ~8 at L3 |
| Action parsing | Per-batch (8 facts) | Single-pass (50 facts) |
| Reward formula | Multi-component dense | Two-phase: bootstrap dense → binary |
| `max_prompt_length` | 2048 | 4096 |
| `per_device_train_batch_size` | 2 | 4 |
| `learning_rate` | 1e-5 | 5e-6 |
| `max_concurrent_envs` | 1 | 8 |
| Critical metric to monitor | reward_mean | reward_std_within_group |

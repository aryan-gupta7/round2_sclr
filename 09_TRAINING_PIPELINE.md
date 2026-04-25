# Training Pipeline

> **Scope**: Full training setup using GRPO via Hugging Face TRL.
> **Implementation lives in**: `training/grpo_train.py`, `training/grpo_train.ipynb`, `training/eval.py`.
> **Reference**: OpenEnv has an example at `examples/grpo_blackjack/` and the TRL docs cover GRPO with custom envs.

## Model choice (locked)

- **Base model**: `Qwen/Qwen2.5-3B-Instruct`
- **Adapter**: LoRA via PEFT
- **Trainer**: `GRPOTrainer` from `trl` (latest)
- **Hardware**: Colab Pro (A100 / L4) primary, HF $30 credits as fallback for longer runs

### Why this model
- 3B fits comfortably with LoRA on a single A100 with sane batch sizes
- Qwen2.5 instruct is strong at structured JSON output (critical for our action format)
- Well-supported in TRL, no custom adaptors needed

### Why not 7B
- $30 HF credits ÷ ~$2/hr A100 = ~15 hours total. Need 5–8 debug runs + final → 7B leaves no margin.
- 3B with good training > 7B with one-shot training.

## LoRA config (starting point)

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## GRPO config (starting point)

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="./outputs/recall_l1",
    num_train_epochs=3,
    per_device_train_batch_size=2,           # tight on memory; episodes are token-heavy
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_generations=4,                        # GRPO group size
    max_prompt_length=2048,
    max_completion_length=512,
    logging_steps=5,
    save_steps=100,
    bf16=True,
    report_to="wandb",                        # we want curves judges can verify
    remove_unused_columns=False,
)
```

These are starting points. **Tune during smoke runs**, not by guessing.

## Connecting GRPO to OpenEnv

OpenEnv's GRPO example pattern:

```python
import asyncio
from envs.recall_env.client import RecallEnv
from envs.recall_env.models import RecallAction

async def rollout_episode(model, tokenizer, difficulty, seed):
    """Run one episode, return trajectory for GRPO."""
    async with RecallEnv(base_url=ENV_URL).sync() as env:
        obs = env.reset(difficulty=difficulty, seed=seed)
        prompts, completions, rewards = [], [], []
        while not is_terminal(obs):
            prompt = build_prompt(obs)
            completion = generate_with_model(model, tokenizer, prompt)
            action = parse_action(completion, obs.phase)
            new_obs = env.step(action)
            prompts.append(prompt)
            completions.append(completion)
            rewards.append(new_obs.last_reward)
            obs = new_obs
        return prompts, completions, rewards
```

The trick: GRPO expects (prompt, completion, reward) tuples. We treat each env step as one such tuple. Reward is the per-step reward emitted by the env. Episode-end terminal reward is added to the last step.

## Reward function for GRPO

```python
def compute_rewards_for_grpo(prompts, completions, env_states):
    """Return per-completion reward for the GRPO group."""
    return [env_states[i].last_step_reward for i in range(len(completions))]
```

GRPO computes advantage by group-normalising these rewards. The episodic structure works out as long as we're consistent about what step a completion corresponds to.

## Training loop structure

```python
# training/grpo_train.py

LEVEL_SCHEDULE = [
    (1, 200),    # 200 GRPO steps at L1
    (2, 400),
    (3, 800),
    (4, 600),    # only if time permits
    (5, 600),    # only if time permits
]

def main(args):
    model, tokenizer = load_model_and_lora(args.base_model, args.lora_config)
    env = RecallEnv(base_url=args.env_url)
    trainer = GRPOTrainer(model=model, ...)

    for level, num_steps in LEVEL_SCHEDULE:
        if not should_train_level(level, args):
            continue
        config = load_level_config(level)
        for step in range(num_steps):
            seed = sample_seed(step)
            traj = run_episode(model, tokenizer, env, level, seed)
            trainer.step(traj)
            log_metrics(step, level, traj)
            if step % args.eval_every == 0:
                eval_metrics = run_eval(model, tokenizer, env, level, eval_seeds)
                wandb.log(eval_metrics)
        save_checkpoint(model, f"checkpoints/level_{level}.pt")
```

## Critical training conventions

### Prompt design

The prompt fed to the model on each step is constructed from the observation:

```
SYSTEM: You are managing memory in a long-running session. Your goal is to store the right information so you can answer future questions correctly.

Current memory anchors (8/20 used):
1. learning rate change due to instability
2. architecture decision: switched to attention variant
... (compact list)

[Phase: ingestion. You will be shown 8 facts. Decide which to store.]

Facts:
- fact_id 0: "..."
- fact_id 1: "..."
... (8 facts)

[INSTRUCTION_FOR_LEVEL_2]: Facts marked [IMPORTANT] are likely to be queried. Recent facts are more likely to be queried than old ones.

Respond with ONLY a JSON array of decisions:
[{"fact_id": 0, "decision": "store"|"skip", "anchor": "<short phrase>"}, ...]
```

The instruction line is taken from the level YAML's `system_prompt_hints`. Levels 4 and 5 have empty hints.

### Action parsing

A separate function parses the model's text output into a `RecallAction`. **Robust to**:
- Trailing/leading whitespace and code fences
- Extra commentary before/after the JSON
- Slightly malformed JSON (try-parse, fallback to extraction with regex)

```python
def parse_action(completion: str, phase: str, batch_size: int) -> RecallAction:
    json_block = extract_json_block(completion)
    if json_block is None:
        return RecallAction(mode=phase_to_mode(phase))   # malformed; will be penalized
    try:
        data = json.loads(json_block)
        return validate_and_construct_action(data, phase, batch_size)
    except (json.JSONDecodeError, ValidationError):
        return malformed_action(phase)
```

Malformed actions are NOT silently fixed. They produce malformed-step penalty so the agent learns to emit valid output.

## Smoke test before training kickoff

```bash
# tests/smoke_train.py
python -m training.grpo_train --level 1 --steps 5 --eval_every 999 --base_model Qwen/Qwen2.5-3B-Instruct
```

Must succeed in <10 minutes. If it fails, fix the pipeline before launching real runs.

## Compute budget breakdown (estimate)

| Phase | Time | Cost |
|-------|------|------|
| Smoke runs | ~30 min × 4 = 2 hr | Colab Pro (free tier) |
| L1 training (200 steps) | ~1 hr | Colab Pro |
| L2 training (400 steps) | ~3 hr | Colab Pro / HF credits |
| L3 training (800 steps) | ~6 hr | HF credits ($12) |
| L4 (optional) | ~5 hr | HF credits ($10) |
| L5 (optional) | ~5 hr | HF credits ($10) |
| Final eval runs | ~2 hr | Colab Pro |

**Total**: ~24 hr training, ~$32 on HF (within budget). If we cut L4/L5, total ~$12.

## Checkpoint strategy

- Save LoRA adapter only (smaller, faster)
- Checkpoint after each level
- Final checkpoint = best eval accuracy across L3 (or last completed level)
- Push final to HF model hub for reproducibility

## What gets logged to W&B

Every training step:
- Mean episode reward across the GRPO group
- Per-component reward breakdown (correct_answer, storage_cost, shaping bonuses)
- Memory utilization (mean used / budget)
- Malformed-action rate
- Per-query failure attribution counts

Every eval pass:
- Accuracy at current level
- Accuracy at all previously trained levels (regression check — must not collapse)
- Sample episode trace (3 episodes, full transcript) for human inspection

## Pinned dependencies

```toml
# pyproject.toml (root, for training)
[project]
dependencies = [
    "transformers>=4.45",
    "trl>=0.12",
    "peft>=0.13",
    "accelerate>=0.34",
    "sentence-transformers>=3.1",
    "openenv-core",
    "wandb",
    "pyyaml",
    "numpy<2.0",   # some compat issues with older deps
]
```

## The Colab notebook (`grpo_train.ipynb`)

Judges will re-run this. It MUST:

1. Install deps in cell 1
2. Clone the repo (with link)
3. Set up env URL (or spin up local server)
4. Run a small training (level 1, 50 steps) end-to-end
5. Plot reward curve in the last cell
6. Save trained adapter for download

Keep it simple. The judges' goal is to verify training works, not to retrain from scratch.

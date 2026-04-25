"""
RECALL Curriculum Training: L1 → L2 → L3 → L4
Trains Qwen2.5-3B-Instruct with LoRA across 4 difficulty levels.
Each level loads the previous level's adapter.

Total: 650 GRPO steps (~6-7 hours on T4)
"""

import os
import re
import json
import time
import argparse
import traceback
import gc
from typing import Optional, List

import torch
import datasets
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from recall_env.client import RecallEnv
from recall_env.models import RecallAction, FactDecision

# ============================================================
# Configuration
# ============================================================

USERNAME = "s1nn3rx69"

LEVEL_SCHEDULE = [
    # (level, num_steps, difficulty, hub_repo, max_completion_length)
    (1, 100, 1, f"{USERNAME}/recall-policy-l1", 512),
    (2, 200, 2, f"{USERNAME}/recall-policy-l2", 512),
    (3, 200, 3, f"{USERNAME}/recall-policy-l3", 768),
    (4, 150, 4, f"{USERNAME}/recall-policy-l4", 768),
]

# Will be set from CLI args
ENV_URL = ""

# ============================================================
# Prompts
# ============================================================

SYSTEM_MSG = "Output ONLY a JSON array. No text before or after."


def build_ingestion_prompt(obs):
    """Returns chat messages for the tokenizer."""
    n = len(obs.all_facts)
    facts_lines = [f"{f['fact_id']}: {f['text']}" for f in obs.all_facts]
    facts_text = "\n".join(facts_lines)
    user_msg = (
        f"Budget: {obs.memory_budget}/{n} slots. Facts:\n"
        f"{facts_text}\n\n"
        f"Output exactly {n} decisions, one per fact_id 0 to {n-1}. "
        f"Do not write anything before or after the JSON array.\n"
        f'Format: [{{"fact_id":0,"decision":"store","anchor":"key"}},{{"fact_id":1,"decision":"skip"}},...]'
        f"\nJSON output:\n["
    )
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]


def pregenerate_dataset(env_url: str, num_samples: int, difficulty: int, seed_offset: int):
    """Connect to live env and build prompts from actual observations."""
    print(f"  Pre-generating {num_samples} prompts (difficulty={difficulty}, seeds={seed_offset}-{seed_offset+num_samples-1})...")
    prompts = []
    seeds = list(range(seed_offset, seed_offset + num_samples))
    fallback = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": "No facts available. Output: []"},
    ]

    for i, seed in enumerate(seeds):
        try:
            with RecallEnv(base_url=env_url).sync() as env:
                res = env.reset(difficulty=difficulty, seed=seed)
                prompts.append(build_ingestion_prompt(res.observation))
        except Exception as e:
            print(f"    Seed {seed} failed: {e}")
            prompts.append(fallback)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{num_samples} done")

    print(f"  Dataset ready: {len(prompts)} prompts")
    return datasets.Dataset.from_dict({
        "prompt": prompts,
        "difficulty": [difficulty] * num_samples,
        "seed": seeds,
    })


# ============================================================
# Parsing
# ============================================================

def parse_ingest_decisions(text: str):
    if not text:
        return None

    if isinstance(text, list):
        text = text[0].get("content", "") if text else ""
    if isinstance(text, dict):
        text = text.get("content", "")

    text = text.strip()
    if not text.startswith("["):
        text = "[" + text

    try:
        val = json.loads(text)
        if isinstance(val, list):
            return [FactDecision(**d) for d in val]
    except Exception:
        pass

    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            val = json.loads(match.group(0))
            if isinstance(val, list):
                return [FactDecision(**d) for d in val]
        except Exception:
            pass
    return None


# ============================================================
# Reward function (same across all levels)
# ============================================================

def recall_reward(completions, prompts, difficulty, seed, **kwargs):
    if not hasattr(recall_reward, "_logged"):
        print("=" * 60)
        print("FIRST COMPLETION SAMPLE:")
        print(completions[0][:1500])
        print("=" * 60)
        recall_reward._logged = True

    rewards = []
    for completion, diff, s in zip(completions, difficulty, seed):
        try:
            comp_text = completion
            if isinstance(comp_text, list):
                comp_text = comp_text[0].get("content", "") if comp_text else ""
            if isinstance(comp_text, dict):
                comp_text = comp_text.get("content", "")

            decisions = parse_ingest_decisions(comp_text)
            if decisions is None:
                rewards.append(-1.0)
                continue

            with RecallEnv(base_url=ENV_URL).sync() as env:
                res = env.reset(difficulty=int(diff), seed=int(s))
                obs = res.observation

                res = env.step(RecallAction(mode="ingest", decisions=decisions))
                obs = res.observation

                while obs.phase == "query":
                    res = env.step(RecallAction(mode="retrieve", query=obs.current_query))
                    obs = res.observation
                    answer = "UNKNOWN"
                    if obs.retrieval_results and len(obs.retrieval_results) > 0:
                        answer = obs.retrieval_results[0].get("text", "UNKNOWN")
                    res = env.step(RecallAction(mode="answer", answer_text=answer))
                    obs = res.observation

                state = env.state()
                rewards.append(float(state.cumulative_reward))
        except Exception as e:
            print(f"    Reward error (seed={s}): {e}")
            rewards.append(-1.0)
    return rewards


# ============================================================
# Eval callback
# ============================================================

class EvalCallback(TrainerCallback):
    def __init__(self, env_url, difficulty):
        self.env_url = env_url
        self.difficulty = difficulty
        self.eval_seeds = list(range(9000 + difficulty * 100, 9010 + difficulty * 100))

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if step == 0 or step % 25 != 0:
            return
        reward_mean = logs.get("reward", "N/A") if logs else "N/A"
        reward_std = logs.get("reward_std", "N/A") if logs else "N/A"
        print(f"\n  EVAL @ step {step}: reward_mean={reward_mean}, reward_std={reward_std}")


# ============================================================
# Train one level
# ============================================================

def train_one_level(
    level: int,
    num_steps: int,
    difficulty: int,
    prev_adapter: Optional[str],
    hub_repo: str,
    max_completion_length: int,
):
    global ENV_URL

    print(f"\n{'='*60}")
    print(f"  LEVEL {level}: {num_steps} steps, difficulty={difficulty}")
    print(f"  Previous adapter: {prev_adapter or 'None (fresh LoRA)'}")
    print(f"  Target hub repo: {hub_repo}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Seed ranges: L1=1000-1099, L2=2000-2199, L3=3000-3199, L4=4000-4149
    seed_offset = 1000 + (level - 1) * 1000
    train_dataset = pregenerate_dataset(ENV_URL, num_steps, difficulty, seed_offset)

    # Load base model
    print(f"  Loading Qwen2.5-3B-Instruct (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=4096,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=16,
    )

    if prev_adapter is not None:
        # Load previous level's adapter
        print(f"  Loading previous adapter from {prev_adapter}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, prev_adapter, is_trainable=True)
    else:
        # Fresh LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    output_dir = f"./outputs/recall_l{level}"
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_generations=8,
        max_prompt_length=4096,
        max_completion_length=max_completion_length,
        warmup_steps=10,
        logging_steps=1,
        save_steps=25,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        report_to="none",
        push_to_hub=True,
        hub_model_id=hub_repo.split("/")[-1],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[recall_reward],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[EvalCallback(ENV_URL, difficulty)],
    )

    print(f"  Starting L{level} training ({num_steps} steps)...")
    trainer.train()

    # Push to hub
    print(f"  Pushing adapter to {hub_repo}...")
    trainer.push_to_hub()

    elapsed = time.time() - t0
    # Get final metrics
    final_reward = "N/A"
    if trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        final_reward = last_log.get("reward", "N/A")

    summary = f"=== L{level} DONE: {num_steps} steps, final mean reward {final_reward}, pushed to {hub_repo}, took {elapsed/60:.1f} min ==="
    print(summary)

    # Cleanup GPU memory
    del trainer
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return hub_repo  # next level loads from this hub repo


# ============================================================
# Main
# ============================================================

def main():
    global ENV_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", type=str, required=True)
    parser.add_argument("--steps-override", type=int, default=None,
                        help="Override step count for ALL levels (for smoke testing)")
    parser.add_argument("--target-level", type=int, default=None,
                        help="Train a specific level (used internally for process isolation)")
    args = parser.parse_args()

    ENV_URL = args.env_url

    # If --target-level is provided, run just that level in this process.
    if args.target_level is not None:
        level_idx = args.target_level - 1
        level, num_steps, difficulty, hub_repo, max_comp_len = LEVEL_SCHEDULE[level_idx]
        if args.steps_override:
            num_steps = args.steps_override

        prev_adapter = None
        if level > 1:
            prev_adapter = LEVEL_SCHEDULE[level_idx - 1][3]

        train_one_level(
            level=level,
            num_steps=num_steps,
            difficulty=difficulty,
            prev_adapter=prev_adapter,
            hub_repo=hub_repo,
            max_completion_length=max_comp_len,
        )
        return

    # Otherwise, act as the orchestrator and spawn a fresh process for each level
    import subprocess
    import sys

    total_start = time.time()
    print("=" * 60)
    print("  RECALL Curriculum Training (Multi-process Orchestrator)")
    print(f"  Environment: {ENV_URL}")
    if args.steps_override:
        print(f"  SMOKE TEST MODE: {args.steps_override} steps per level")
    print("=" * 60)

    summaries = []
    
    for level, num_steps, difficulty, hub_repo, max_comp_len in LEVEL_SCHEDULE:
        print(f"\n[Orchestrator] Spawning new process for LEVEL {level}...")
        
        cmd = [sys.executable, sys.argv[0], "--env-url", ENV_URL, "--target-level", str(level)]
        if args.steps_override:
            cmd.extend(["--steps-override", str(args.steps_override)])

        try:
            subprocess.check_call(cmd)
            summaries.append(f"L{level}: OK (pushed to {hub_repo})")
        except subprocess.CalledProcessError as e:
            print(f"\n!!! L{level} subprocess FAILED (exit code {e.returncode})")
            summaries.append(f"L{level}: FAILED")
            raise  # Don't proceed to the next level if this one fails

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  CURRICULUM COMPLETE")
    print("=" * 60)
    for s in summaries:
        print(f"  {s}")
    print(f"  Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)")
    print(f"  Estimated credits: ~${total_elapsed/3600 * 0.60:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

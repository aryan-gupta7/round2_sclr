"""
RECALL Curriculum Training: RESUME from L3 step 75
Runs remaining 125 steps of L3, then full L4 (150 steps).
Loads the partial L3 adapter from HF as the starting point.

Expected runtime: ~2-3 hours on A100
"""

import os
import re
import json
import time
import argparse
import traceback
import gc
import concurrent.futures
from typing import Optional, List

# Disable vLLM CUDA Graphs and Torch Compile to prevent illegal memory access during sleep()
os.environ["VLLM_ENFORCE_EAGER"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_USE_V1"] = "0"  # Prevent new vLLM engine from forcing dynamo graph compilation

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

# Resume schedule: L3 (250 steps), then L4 full
LEVEL_SCHEDULE = [
    # (level, num_steps, difficulty, hub_repo, max_completion_length, prev_adapter, seed_offset)
    (3, 250, 3, f"{USERNAME}/recall-policy-l3", 768, f"{USERNAME}/recall-policy-l3", 3075),
    (4, 150, 4, f"{USERNAME}/recall-policy-l4", 768, f"{USERNAME}/recall-policy-l3", 4000),
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
# Reward function
# ============================================================

def _simulate_episode(env_url, difficulty, seed, decisions):
    with RecallEnv(base_url=env_url).sync() as env:
        res = env.reset(difficulty=int(difficulty), seed=int(seed))
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
        return float(state.cumulative_reward)


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

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_simulate_episode, ENV_URL, diff, s, decisions)
                try:
                    reward = future.result(timeout=60.0)
                    rewards.append(reward)
                except concurrent.futures.TimeoutError:
                    print(f"    Reward error (seed={s}): Timeout after 60s")
                    rewards.append(-1.0)

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
    seed_offset: int,
):
    global ENV_URL

    print(f"\n{'='*60}")
    print(f"  LEVEL {level}: {num_steps} steps, difficulty={difficulty}")
    print(f"  Loading adapter from: {prev_adapter or 'None (fresh LoRA)'}")
    print(f"  Target hub repo: {hub_repo}")
    print(f"  Seed offset: {seed_offset}")
    print(f"{'='*60}\n")

    t0 = time.time()

    train_dataset = pregenerate_dataset(ENV_URL, num_steps, difficulty, seed_offset)

    # Load base model
    print(f"  Loading Qwen2.5-7B-Instruct (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        max_seq_length=4096,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=16,
    )

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

    if prev_adapter is not None:
        # Load adapter weights safely to keep Unsloth patches
        print(f"  Loading adapter weights from {prev_adapter}...")
        from peft.utils import load_peft_weights
        from peft import set_peft_model_state_dict
        weights = load_peft_weights(prev_adapter)
        set_peft_model_state_dict(model, weights)

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
    
    # Check for local checkpoints to resume from
    resume_from = False
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from = True
            print(f"  Found checkpoints in {output_dir}, enabling resume_from_checkpoint=True")

    trainer.train(resume_from_checkpoint=resume_from)

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

    return hub_repo


# ============================================================
# Main
# ============================================================

def main():
    global ENV_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", type=str, required=True)
    parser.add_argument("--target-level", type=int, default=None,
                        help="Train a specific level (used internally for process isolation)")
    args = parser.parse_args()

    ENV_URL = args.env_url.replace("https://", "wss://").replace("http://", "ws://")

    # If --target-level is provided, run just that level in this process.
    if args.target_level is not None:
        target_tuple = None
        for t in LEVEL_SCHEDULE:
            if t[0] == args.target_level:
                target_tuple = t
                break

        if target_tuple is None:
            raise ValueError(f"Unknown target level {args.target_level}")

        level, num_steps, difficulty, hub_repo, max_comp_len, prev_adapter, seed_off = target_tuple

        train_one_level(
            level=level,
            num_steps=num_steps,
            difficulty=difficulty,
            prev_adapter=prev_adapter,
            hub_repo=hub_repo,
            max_completion_length=max_comp_len,
            seed_offset=seed_off,
        )
        return

    # Otherwise, act as the orchestrator and spawn a fresh process for each level
    import subprocess
    import sys

    total_start = time.time()
    print("=" * 60)
    print("  RECALL Curriculum Training - RESUME")
    print(f"  Environment: {ENV_URL}")
    print("  L3: 125 remaining steps (seeds 3075-3199)")
    print("  L4: 150 steps (seeds 4000-4149)")
    print("=" * 60)

    summaries = []

    for level, num_steps, difficulty, hub_repo, max_comp_len, prev_adapter, seed_off in LEVEL_SCHEDULE:
        print(f"\n[Orchestrator] Spawning new process for LEVEL {level}...")

        cmd = [sys.executable, sys.argv[0], "--env-url", ENV_URL, "--target-level", str(level)]

        try:
            subprocess.check_call(cmd)
            summaries.append(f"L{level}: OK (pushed to {hub_repo})")
        except subprocess.CalledProcessError as e:
            print(f"\n!!! L{level} subprocess FAILED (exit code {e.returncode})")
            summaries.append(f"L{level}: FAILED")
            raise

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  CURRICULUM RESUME COMPLETE")
    print("=" * 60)
    for s in summaries:
        print(f"  {s}")
    print(f"  Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)")
    print(f"  Estimated credits: ~${total_elapsed/3600 * 2.50:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

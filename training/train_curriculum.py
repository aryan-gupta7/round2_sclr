"""
RECALL Curriculum Training: L1 → L5
Trains Qwen2.5-7B-Instruct with LoRA across 5 difficulty levels.
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
import numpy as np

# Disable vLLM CUDA Graphs and Torch Compile to prevent illegal memory access during sleep()
os.environ["VLLM_ENFORCE_EAGER"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_USE_V1"] = "0"

import torch
import datasets
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from peft.utils import load_peft_weights
from peft import set_peft_model_state_dict
from recall_env.client import RecallEnv
from recall_env.models import RecallAction, FactDecision

# ============================================================
# Configuration
# ============================================================

USERNAME = "s1nn3rx69"

LEVEL_SCHEDULE = [
    # (level, num_steps, difficulty, hub_repo, max_completion_length)
    (1, 250, 1, f"{USERNAME}/recall-policy-l1", 512),
    (2, 200, 2, f"{USERNAME}/recall-policy-l2", 512),
    (3, 250, 3, f"{USERNAME}/recall-policy-l3", 768),
    (4, 200, 4, f"{USERNAME}/recall-policy-l4", 768),
    (5, 200, 5, f"{USERNAME}/recall-policy-l5", 768),
]

ENV_URL = ""

# ============================================================
# Prompts
# ============================================================

SYSTEM_MSG = "Output ONLY a JSON array. No text before or after."

def build_ingestion_prompt(obs):
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

global_eval_stats = {"agent_accs": [], "baseline_accs": []}

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
        
        qt = max(1, state.queries_total)
        agent_acc = state.queries_answered_correctly / qt
        baseline_acc = getattr(state, "baseline_correct", 0) / qt
        
        return float(state.cumulative_reward), agent_acc, baseline_acc

def recall_reward(completions, prompts, difficulty, seed, **kwargs):
    if not hasattr(recall_reward, "_step"):
        recall_reward._step = 0

    rewards = []
    agent_accs = []
    baseline_accs = []
    
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
                agent_accs.append(0.0)
                baseline_accs.append(0.0)
                continue

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_simulate_episode, ENV_URL, diff, s, decisions)
                try:
                    reward, a_acc, b_acc = future.result(timeout=60.0)
                    rewards.append(reward)
                    agent_accs.append(a_acc)
                    baseline_accs.append(b_acc)
                except concurrent.futures.TimeoutError:
                    rewards.append(-1.0)
                    agent_accs.append(0.0)
                    baseline_accs.append(0.0)

        except Exception as e:
            rewards.append(-1.0)
            agent_accs.append(0.0)
            baseline_accs.append(0.0)
            
    if agent_accs:
        global_eval_stats["agent_accs"].append(np.mean(agent_accs))
        global_eval_stats["baseline_accs"].append(np.mean(baseline_accs))
        
    if recall_reward._step % 10 == 0:
        avg_a = np.mean(agent_accs) * 100 if agent_accs else 0
        avg_b = np.mean(baseline_accs) * 100 if baseline_accs else 0
        print(f"  [STEP {recall_reward._step}] SIDE-BY-SIDE EVAL | Agent Acc: {avg_a:.1f}% vs FIFO Baseline Acc: {avg_b:.1f}%")
        
    recall_reward._step += 1
    return rewards

class EvalCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if step == 0 or step % 25 != 0:
            return
        reward_mean = logs.get("reward", "N/A") if logs else "N/A"
        print(f"\n  EVAL @ step {step}: reward_mean={reward_mean}")
        
        # Print a moving average comparison
        if len(global_eval_stats["agent_accs"]) > 0:
            a_mean = np.mean(global_eval_stats["agent_accs"][-25:]) * 100
            b_mean = np.mean(global_eval_stats["baseline_accs"][-25:]) * 100
            print(f"  [MOVING AVG LAST 25 STEPS] Agent Acc: {a_mean:.1f}% | Baseline Acc: {b_mean:.1f}%")

# ============================================================
# Train one level
# ============================================================

def train_one_level(level: int, num_steps: int, difficulty: int, prev_adapter: Optional[str], hub_repo: str, max_completion_length: int):
    global ENV_URL

    print(f"\n{'='*60}")
    print(f"  LEVEL {level}: {num_steps} steps, difficulty={difficulty}")
    print(f"  Loading adapter from: {prev_adapter or 'None (fresh LoRA)'}")
    print(f"  Target hub repo: {hub_repo}")
    print(f"{'='*60}\n")

    t0 = time.time()
    seed_offset = 1000 + (level - 1) * 1000
    train_dataset = pregenerate_dataset(ENV_URL, num_steps, difficulty, seed_offset)

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    if prev_adapter is not None:
        print(f"  Loading previous adapter weights from {prev_adapter}...")
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
        save_steps=50,
        bf16=True,
        fp16=False,
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
        callbacks=[EvalCallback()],
    )

    print(f"  Starting L{level} training ({num_steps} steps)...")
    resume_from = False
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from = True
            print(f"  Found checkpoints, enabling resume_from_checkpoint=True")

    trainer.train(resume_from_checkpoint=resume_from)

    print(f"  Pushing adapter to {hub_repo}...")
    trainer.push_to_hub()

    elapsed = time.time() - t0
    final_reward = trainer.state.log_history[-1].get("reward", "N/A") if trainer.state.log_history else "N/A"

    print(f"=== L{level} DONE: {num_steps} steps, final mean reward {final_reward}, pushed to {hub_repo}, took {elapsed/60:.1f} min ===")

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
    parser.add_argument("--target-level", type=int, default=None)
    args = parser.parse_args()

    ENV_URL = args.env_url.replace("https://", "wss://").replace("http://", "ws://")

    if args.target_level is not None:
        target_tuple = next((t for t in LEVEL_SCHEDULE if t[0] == args.target_level), None)
        if target_tuple is None:
            raise ValueError(f"Unknown level {args.target_level}")

        level, num_steps, difficulty, hub_repo, max_comp_len = target_tuple

        # Strict previous adapter tracking from earlier levels (L_n requires L_{n-1})
        prev_adapter = f"{USERNAME}/recall-policy-l{level-1}" if level > 1 else None

        train_one_level(level, num_steps, difficulty, prev_adapter, hub_repo, max_comp_len)
        return

    import subprocess
    import sys

    total_start = time.time()
    print("=" * 60)
    print("  RECALL Curriculum Training (Full run L1 -> L5)")
    print(f"  Environment: {ENV_URL}")
    print("=" * 60)

    summaries = []

    for level, num_steps, difficulty, hub_repo, max_comp_len in LEVEL_SCHEDULE:
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
    print("  CURRICULUM COMPLETE")
    print("=" * 60)
    for s in summaries:
        print(f"  {s}")
    print(f"  Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)")
    print(f"  Estimated credits: ~${total_elapsed/3600 * 1.50:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

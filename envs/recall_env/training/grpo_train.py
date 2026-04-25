"""
GRPO Training for RECALL Environment — Level 1
Trains Qwen2.5-3B-Instruct + LoRA to make smart memory decisions.

Architecture:
- TRL GRPOTrainer generates text completions (JSON ingestion decisions)
- Reward function runs each completion through the live environment
- GRPO updates the policy based on which completions scored best
"""

import os
import re
import json
import argparse
import traceback
from typing import List
import datasets
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from recall_env.client import RecallEnv
from recall_env.models import RecallAction, FactDecision

# Will be set from CLI args
ENV_URL = ""

# ============================================================
# Prompts
# ============================================================

SYSTEM_MSG = "Output ONLY a JSON array. No text before or after."

def build_ingestion_prompt(obs):
    """Returns a list of chat messages for the tokenizer."""
    n = len(obs.all_facts)
    facts_lines = []
    for f in obs.all_facts:
        facts_lines.append(f"{f['fact_id']}: {f['text']}")
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


def pregenerate_dataset(env_url: str, num_samples: int = 100, difficulty: int = 1):
    """Connect to the live env and build prompts from actual observations."""
    print(f"Pre-generating {num_samples} prompts from {env_url}...")
    prompts = []
    seeds = list(range(1000, 1000 + num_samples))

    for i, seed in enumerate(seeds):
        try:
            with RecallEnv(base_url=env_url).sync() as env:
                res = env.reset(difficulty=difficulty, seed=seed)
                obs = res.observation
                prompt_msgs = build_ingestion_prompt(obs)
                prompts.append(prompt_msgs)
        except Exception as e:
            print(f"  Seed {seed} failed: {e}, using fallback")
            prompts.append([
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": "No facts available. Output: []"},
            ])
        if (i + 1) % 25 == 0:
            print(f"  Generated {i+1}/{num_samples} prompts")

    print(f"Done. Generated {len(prompts)} prompts.")
    return datasets.Dataset.from_dict({
        "prompt": prompts,
        "difficulty": [difficulty] * num_samples,
        "seed": seeds,
    })


# ============================================================
# Parsing
# ============================================================

def parse_ingest_decisions(text: str):
    """Try to extract a JSON array of fact decisions from model output."""
    if not text:
        return None

    if isinstance(text, list):
        text = text[0].get("content", "") if text else ""
    if isinstance(text, dict):
        text = text.get("content", "")

    text = text.strip()

    # The prompt ends with '[' so model continues from there.
    if not text.startswith("["):
        text = "[" + text

    # Try direct parse
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return [FactDecision(**d) for d in val]
    except Exception:
        pass

    # Try extracting JSON array
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

def recall_reward(completions, prompts, difficulty, seed, **kwargs):
    """
    Score each completion by running it through the environment.
    """
    rewards = []

    for i, (completion, diff, s) in enumerate(zip(completions, difficulty, seed)):
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

                action = RecallAction(mode="ingest", decisions=decisions)
                res = env.step(action)
                obs = res.observation

                while obs.phase == "query":
                    res = env.step(RecallAction(mode="retrieve", query=obs.current_query))
                    obs = res.observation

                    answer_text = "UNKNOWN"
                    if obs.retrieval_results and len(obs.retrieval_results) > 0:
                        answer_text = obs.retrieval_results[0].get("text", "UNKNOWN")

                    res = env.step(RecallAction(mode="answer", answer_text=answer_text))
                    obs = res.observation

                state = env.state()
                reward = float(state.cumulative_reward)
                rewards.append(reward)

        except Exception as e:
            print(f"  Reward error (seed={s}): {e}")
            rewards.append(-1.0)

    return rewards


# ============================================================
# Eval callback — runs every 25 steps
# ============================================================

class EvalCallback(TrainerCallback):
    """Quick eval every 25 steps: run 10 seeds, print accuracy."""

    def __init__(self, env_url, eval_seeds=None, difficulty=1):
        self.env_url = env_url
        self.eval_seeds = eval_seeds or list(range(9000, 9010))
        self.difficulty = difficulty

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if step == 0 or step % 25 != 0:
            return

        print(f"\n{'='*50}")
        print(f"  EVAL @ step {step}")
        print(f"{'='*50}")

        # Grab recent metrics
        reward_mean = logs.get("reward", "N/A") if logs else "N/A"
        reward_std = logs.get("reward_std", "N/A") if logs else "N/A"
        print(f"  Train reward_mean={reward_mean}  reward_std={reward_std}")

        # Quick accuracy check: how many seeds produce parseable + positive reward?
        correct = 0
        total = len(self.eval_seeds)
        for seed in self.eval_seeds:
            try:
                with RecallEnv(base_url=self.env_url).sync() as env:
                    res = env.reset(difficulty=self.difficulty, seed=seed)
                    obs = res.observation

                    # Store ALL facts (oracle policy)
                    decisions = []
                    for f in obs.all_facts:
                        decisions.append(FactDecision(
                            fact_id=f["fact_id"],
                            decision="store",
                            anchor=f["text"][:30]
                        ))
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

                    st = env.state()
                    if st.cumulative_reward > 0:
                        correct += 1
            except Exception as e:
                print(f"    eval seed {seed} error: {e}")

        acc = correct / total * 100
        print(f"  Oracle baseline accuracy: {acc:.0f}% ({correct}/{total} seeds positive)")
        print(f"{'='*50}\n")

        # Early stop hint
        if step >= 50 and reward_std != "N/A" and float(reward_std) > 0:
            if logs and logs.get("reward", 0) > 0.5:
                print("  >>> L1 looks converged. Consider killing job and launching L2.")


# ============================================================
# Main
# ============================================================

def main():
    global ENV_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    ENV_URL = args.env_url
    print(f"=== RECALL GRPO Training — Level 1 ===")
    print(f"Environment: {ENV_URL}")
    print(f"Max steps: {args.max_steps}")

    # Pre-generate dataset
    train_dataset = pregenerate_dataset(ENV_URL, num_samples=args.max_steps, difficulty=1)

    # Load model
    print("Loading Qwen2.5-3B-Instruct (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
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

    training_args = GRPOConfig(
        output_dir="./outputs/recall_l1",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_generations=8,
        max_prompt_length=4096,
        max_completion_length=512,
        warmup_steps=10,
        logging_steps=1,
        save_steps=25,
        bf16=False,
        fp16=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        report_to="none",
        push_to_hub=True,
        hub_model_id="recall-policy-l1",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[recall_reward],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[EvalCallback(ENV_URL)],
    )

    print("Starting L1 training (100 steps)...")
    trainer.train()
    print("L1 training complete!")

if __name__ == "__main__":
    main()

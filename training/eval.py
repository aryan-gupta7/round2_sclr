import os
import sys
import argparse
import asyncio
from recall_env.client import RecallEnv
from recall_env.models import RecallAction, FactDecision

def evaluate(env_url, num_seeds=20, difficulty=1):
    # This evaluates the deployed HF Space URL acting as a simple client.
    # We test the environment's baseline behavior.
    print(f"Evaluating {num_seeds} seeds against baseline on {env_url}...")
    baseline_accs = []
    
    for seed in range(num_seeds):
        with RecallEnv(base_url=env_url).sync() as env:
            res = env.reset(difficulty=difficulty, seed=seed)
            obs = res.observation
            
            # Simple FIFO ingestion representation
            decisions = []
            for fact in obs.all_facts:
                if len(decisions) < obs.memory_budget:
                    decisions.append(FactDecision(fact_id=fact["fact_id"], decision="store", anchor=fact["text"][:30]))
                else:
                    decisions.append(FactDecision(fact_id=fact["fact_id"], decision="skip"))
            
            res = env.step(RecallAction(mode="ingest", decisions=decisions))
            obs = res.observation
            
            while obs.phase == "query":
                res = env.step(RecallAction(mode="retrieve", query=obs.current_query))
                obs = res.observation
                res = env.step(RecallAction(mode="answer", answer_text="UNKNOWN"))
                obs = res.observation
            
            baseline_accs.append(env.state().baseline_correct)
            print(f"Seed {seed} Baseline Correct: {env.state().baseline_correct}/5")
            
    print(f"Average baseline accuracy: {sum(baseline_accs) / (num_seeds * 5):.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.env_url)

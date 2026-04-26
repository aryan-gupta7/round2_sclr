
import torch
from unsloth import FastLanguageModel
from recall_env.client import RecallEnv
from recall_env.models import RecallAction, FactDecision
import json
import re

SYSTEM_MSG = "Output ONLY a JSON array. No text before or after."

def parse_decisions(text):
    try:
        # Extra forgiving parsing
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        if isinstance(data, list):
            res = []
            for item in data:
                if isinstance(item, dict) and "fact_id" in item:
                    res.append(FactDecision(
                        fact_id=item["fact_id"],
                        decision=item.get("decision", "skip"),
                        anchor=str(item.get("anchor", ""))
                    ))
            return res
    except:
        pass
    return None

def build_prompt(obs):
    n = len(obs.all_facts)
    facts_text = ""
    for f in obs.all_facts:
        facts_text += f"[{f['fact_id']}] {f['text']}\n"
    
    user_msg = (
        f"Budget: {obs.memory_budget}/{n} slots. Facts:\n"
        f"{facts_text}\n\n"
        f"Output exactly {n} decisions, one per fact_id 0 to {n-1}. "
        f"Do not write anything before or after the JSON array.\n"
        f'Format: [{{"fact_id":0,"decision":"store","anchor":"key"}},...]'
        f"\nJSON output:\n["
    )
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]

def evaluate_model(model, tokenizer, env_url, seeds, difficulty=1, name="Model"):
    print(f"\n{'='*80}")
    print(f"  EVALUATING: {name}")
    print(f"{'='*80}")
    
    results = []
    
    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        try:
            with RecallEnv(base_url=env_url).sync() as env:
                # 1. Reset
                res = env.reset(difficulty=difficulty, seed=seed)
                obs = res.observation
                
                # 2. Generate decisions
                prompt = build_prompt(obs)
                inputs = tokenizer.apply_chat_template(
                    prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to("cuda")
                
                outputs = model.generate(input_ids=inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                output_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                
                # Clean up output if model missed the bracket
                if not output_text.strip().startswith("["):
                    output_text = "[" + output_text
                
                print(f"MODEL OUTPUT:\n{output_text[:300]}...")
                
                decisions = parse_decisions(output_text)
                if not decisions:
                    print("--> Failed to parse decisions.")
                    results.append({"seed": seed, "correct": 0, "total": 3, "baseline": 0})
                    continue
                
                stored = sum(1 for d in decisions if d.decision == 'store')
                print(f"--> Parsed successfully. Stored {stored}/{obs.memory_budget} facts.")
                
                # 3. Ingest
                res = env.step(RecallAction(mode="ingest", decisions=decisions))
                obs = res.observation
                
                # 4. Answer queries
                while obs.phase == "query":
                    query = obs.current_query
                    
                    # Retrieve
                    res = env.step(RecallAction(mode="retrieve", query=query))
                    obs = res.observation
                    
                    if obs.retrieval_results:
                        answer = obs.retrieval_results[0].get("content", "UNKNOWN")
                    else:
                        answer = "UNKNOWN"
                        
                    res = env.step(RecallAction(mode="answer", answer_text=answer))
                    obs = res.observation
                
                state = env.state()
                print(f"--> Final score: {state.correct_answers}/{state.queries_total} (FIFO baseline: {state.baseline_correct})")
                
                results.append({
                    "seed": seed,
                    "correct": state.correct_answers,
                    "total": state.queries_total,
                    "baseline": state.baseline_correct
                })
                
        except Exception as e:
            print(f"Error on seed {seed}: {e}")
            results.append({"seed": seed, "correct": 0, "total": 3, "baseline": 0})
            
    return results


def main():
    import warnings
    warnings.filterwarnings('ignore')
    
    seeds = [1000, 1001, 1002, 1003, 1004]
    env_url = "https://s1nn3rx69-recall-env.hf.space"
    
    print("Loading Base Model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)
    
    base_results = evaluate_model(base_model, tokenizer, env_url, seeds, name="Base Model")
    
    print("\nLoading Trained L1 Adapter...")
    adapter_model, adapter_tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    # Load the adapter we just trained
    adapter_model.load_adapter("s1nn3rx69/recall-policy-l1")
    FastLanguageModel.for_inference(adapter_model)
    
    adapter_results = evaluate_model(adapter_model, adapter_tokenizer, env_url, seeds, name="Adapter (L1)")
    
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Seed':<10} | {'Baseline (FIFO)':<15} | {'Base Model':<15} | {'Trained Adapter':<15}")
    print("-" * 65)
    
    base_total = sum(r['correct'] for r in base_results)
    adapter_total = sum(r['correct'] for r in adapter_results)
    fifo_total = sum(r['baseline'] for r in base_results)
    total_q = sum(r['total'] for r in base_results)
    
    for i, seed in enumerate(seeds):
        base_c = base_results[i]['correct']
        adapt_c = adapter_results[i]['correct']
        fifo_c = base_results[i]['baseline']
        total = base_results[i]['total']
        print(f"{seed:<10} | {fifo_c}/{total:<13} | {base_c}/{total:<13} | {adapt_c}/{total:<13}")
        
    print("-" * 65)
    print(f"{'TOTAL':<10} | {fifo_total}/{total_q:<13} | {base_total}/{total_q:<13} | {adapter_total}/{total_q:<13}")
    print(f"{'ACCURACY':<10} | {fifo_total/total_q*100:.1f}%           | {base_total/total_q*100:.1f}%           | {adapter_total/total_q*100:.1f}%")

if __name__ == '__main__':
    main()

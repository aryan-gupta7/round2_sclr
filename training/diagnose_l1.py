"""
Deep diagnostic for L1 training failure.

This script:
1. Connects to the HF environment 
2. Resets episodes on multiple seeds
3. Shows the EXACT facts the model sees and queries it will face 
4. Simulates FIFO baseline and shows what it stores vs retrieves
5. Shows where failures happen (retrieval? grading? wrong facts stored?)

This does NOT need the trained model — it diagnoses the environment itself
to understand what the model SHOULD learn.
"""
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.recall_env.client import RecallEnv
from envs.recall_env.models import RecallAction, FactDecision

ENV_URL = "https://s1nn3rx69-recall-env.hf.space"

def diagnose_seed(seed, difficulty=1, verbose=True):
    """Deep diagnosis of a single seed."""
    results = {}
    
    with RecallEnv(base_url=ENV_URL).sync() as env:
        res = env.reset(difficulty=difficulty, seed=seed)
        obs = res.observation
        
        n_facts = len(obs.all_facts)
        budget = obs.memory_budget
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"SEED {seed} | {n_facts} facts | budget={budget} | must skip {n_facts - budget}")
            print(f"{'='*80}")
            
            print(f"\n--- ALL FACTS ---")
            for f in obs.all_facts:
                print(f"  [{f['fact_id']}] {f['text'][:100]}")
        
        results['n_facts'] = n_facts
        results['budget'] = budget
        results['facts'] = [f['text'] for f in obs.all_facts]
        
    # Strategy 1: FIFO (store first N facts)
    fifo_results = run_strategy(seed, difficulty, "fifo", obs.all_facts, budget, verbose)
    results['fifo'] = fifo_results
    
    # Strategy 2: Store ALL (what the model likely does)
    all_results = run_strategy(seed, difficulty, "store_all", obs.all_facts, budget, verbose)
    results['store_all'] = all_results
    
    # Strategy 3: Store LAST N
    last_results = run_strategy(seed, difficulty, "last_n", obs.all_facts, budget, verbose)
    results['last_n'] = last_results
    
    # Strategy 4: Perfect oracle (stores facts that will be queried)
    # We need to figure out which facts get queried first
    oracle_results = run_strategy(seed, difficulty, "oracle", obs.all_facts, budget, verbose, 
                                   oracle_seed=seed)
    results['oracle'] = oracle_results
    
    return results


def run_strategy(seed, difficulty, strategy_name, all_facts, budget, verbose, oracle_seed=None):
    """Run a specific storage strategy and return accuracy."""
    
    with RecallEnv(base_url=ENV_URL).sync() as env:
        res = env.reset(difficulty=difficulty, seed=seed)
        obs = res.observation
        
        decisions = []
        stored_ids = []
        
        if strategy_name == "fifo":
            for i, fact in enumerate(all_facts):
                if i < budget:
                    decisions.append(FactDecision(
                        fact_id=fact['fact_id'], decision="store", 
                        anchor=fact['text'][:80]
                    ))
                    stored_ids.append(fact['fact_id'])
                else:
                    decisions.append(FactDecision(fact_id=fact['fact_id'], decision="skip"))
                    
        elif strategy_name == "store_all":
            # Stores all facts but only first `budget` actually get stored
            for i, fact in enumerate(all_facts):
                decisions.append(FactDecision(
                    fact_id=fact['fact_id'], decision="store",
                    anchor=fact['text'][:80]
                ))
                if i < budget:
                    stored_ids.append(fact['fact_id'])
                    
        elif strategy_name == "last_n":
            skip_count = len(all_facts) - budget
            for i, fact in enumerate(all_facts):
                if i < skip_count:
                    decisions.append(FactDecision(fact_id=fact['fact_id'], decision="skip"))
                else:
                    decisions.append(FactDecision(
                        fact_id=fact['fact_id'], decision="store",
                        anchor=fact['text'][:80]
                    ))
                    stored_ids.append(fact['fact_id'])
                    
        elif strategy_name == "oracle":
            # First, figure out which facts will be queried
            decisions.append(FactDecision(fact_id=0, decision="store", anchor="dummy"))
            for fact in all_facts[1:]:
                decisions.append(FactDecision(fact_id=fact['fact_id'], decision="skip"))
            
            # Can't do oracle without knowing queries ahead of time
            # Instead, store first N (same as FIFO)
            decisions = []
            for i, fact in enumerate(all_facts):
                if i < budget:
                    decisions.append(FactDecision(
                        fact_id=fact['fact_id'], decision="store",
                        anchor=fact['text'][:80]
                    ))
                    stored_ids.append(fact['fact_id'])
                else:
                    decisions.append(FactDecision(fact_id=fact['fact_id'], decision="skip"))
        
        # Ingest
        res = env.step(RecallAction(mode="ingest", decisions=decisions))
        obs = res.observation
        
        # Query phase
        queries = []
        retrievals = []
        correct = 0
        total = 0
        
        while obs.phase == "query":
            query_text = obs.current_query
            
            # Retrieve
            res = env.step(RecallAction(mode="retrieve", query=query_text))
            obs = res.observation
            
            retrieved = obs.retrieval_results if obs.retrieval_results else []
            answer = "UNKNOWN"
            if retrieved and len(retrieved) > 0:
                answer = retrieved[0].get("content", "UNKNOWN")
            
            queries.append(query_text)
            retrievals.append({
                "query": query_text,
                "top_result": retrieved[0] if retrieved else None,
                "answer_given": answer[:60] if answer else "UNKNOWN",
            })
            
            # Answer 
            res = env.step(RecallAction(mode="answer", answer_text=answer))
            obs = res.observation
            total += 1
        
        state = env.state()
        correct = state.correct_answers
        baseline_correct = state.baseline_correct
        
        acc = correct / max(1, total)
        
        if verbose:
            print(f"\n--- Strategy: {strategy_name} ---")
            print(f"  Stored fact IDs: {stored_ids}")
            print(f"  Accuracy: {correct}/{total} = {acc:.1%}")
            print(f"  Internal baseline: {baseline_correct}/{total}")
            
            for i, r in enumerate(retrievals):
                status = "✓" if i < correct else "✗"  # approximate
                q = r['query'][:60]
                a = r['answer_given']
                top = r['top_result']
                top_anchor = top.get('anchor', 'N/A')[:40] if top else 'NONE'
                top_score = f"{top.get('score', 0):.3f}" if top else 'N/A'
                print(f"  Q{i}: {q}")
                print(f"       → Retrieved: [{top_score}] {top_anchor}")
                print(f"       → Answer: {a}")
        
        return {
            'correct': correct,
            'total': total,
            'accuracy': acc,
            'stored_ids': stored_ids,
            'queries': queries,
            'retrievals': retrievals,
            'baseline_correct': baseline_correct,
        }


def diagnose_model_output_format():
    """
    Show what the model prompt looks like and what a correct response looks like.
    This helps understand what the model SHOULD output.
    """
    print("\n" + "="*80)
    print("WHAT SHOULD THE MODEL OUTPUT?")
    print("="*80)
    
    with RecallEnv(base_url=ENV_URL).sync() as env:
        res = env.reset(difficulty=1, seed=42)
        obs = res.observation
        
        n = len(obs.all_facts)
        budget = obs.memory_budget
        skip_count = n - budget
        
        print(f"\nThe model sees {n} facts and has {budget} slots.")
        print(f"It must skip exactly {skip_count} facts.")
        print(f"\nPrompt instructs: 'Output exactly {n} decisions as a JSON array'")
        print(f'Format: [{{"fact_id":0,"decision":"store"}},{{"fact_id":1,"decision":"skip"}},...]')
        
        # Show what a correct 94-token output looks like
        example_decisions = []
        for i, f in enumerate(obs.all_facts):
            d = "store" if i < budget else "skip"
            example_decisions.append({"fact_id": f['fact_id'], "decision": d})
        
        example_output = json.dumps(example_decisions)
        print(f"\nExample correct output ({len(example_output)} chars):")
        print(example_output)
        print(f"\nApprox tokens: {len(example_output) // 4}")
        
        # Also show what "store all" looks like
        all_store = [{"fact_id": f['fact_id'], "decision": "store"} for f in obs.all_facts]
        all_store_output = json.dumps(all_store)
        print(f"\n'Store ALL' output ({len(all_store_output)} chars):")
        print(all_store_output)
        print(f"Approx tokens: {len(all_store_output) // 4}")


if __name__ == "__main__":
    # First: understand what the model should output
    diagnose_model_output_format()
    
    # Then: diagnose multiple seeds
    all_results = {}
    for seed in [0, 1, 2, 3, 4, 42, 100, 200, 9000, 9001, 9002]:
        try:
            r = diagnose_seed(seed, difficulty=1, verbose=True)
            all_results[seed] = r
        except Exception as e:
            print(f"\nSeed {seed} FAILED: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Seed':<6} {'FIFO':>6} {'StoreAll':>10} {'LastN':>6} {'Baseline':>10}")
    for seed, r in all_results.items():
        fifo_acc = f"{r['fifo']['correct']}/{r['fifo']['total']}"
        all_acc = f"{r['store_all']['correct']}/{r['store_all']['total']}"
        last_acc = f"{r['last_n']['correct']}/{r['last_n']['total']}"
        bl = f"{r['fifo']['baseline_correct']}/{r['fifo']['total']}"
        print(f"{seed:<6} {fifo_acc:>6} {all_acc:>10} {last_acc:>6} {bl:>10}")

"""
Full end-to-end diagnostic of the RECALL training pipeline.
Diagnostic 1: Print one full eval episode
Diagnostic 2: Verify FIFO baseline
Diagnostic 3: Check reward components
"""
import asyncio
from envs.recall_env.client import RecallEnv
from envs.recall_env.models import RecallAction, FactDecision

ENV_URL = "wss://s1nn3rx69-recall-env.hf.space"

def run_full_episode_diagnostic(seed=0, difficulty=1):
    """Diagnostic 1: Full episode trace."""
    print("=" * 70)
    print(f"DIAGNOSTIC 1: Full Episode Trace (seed={seed}, difficulty={difficulty})")
    print("=" * 70)
    
    with RecallEnv(base_url=ENV_URL).sync() as env:
        res = env.reset(difficulty=difficulty, seed=seed)
        obs = res.observation
        
        # 1. Show facts
        print(f"\nFACTS GIVEN TO MODEL ({len(obs.all_facts)} total, budget={obs.memory_budget}):")
        for f in obs.all_facts:
            tags = f.get('tags', [])
            tag_str = f" {tags}" if tags else ""
            print(f"  [{f['fact_id']}]{tag_str} {f['text'][:120]}")
        
        # 2. Simulate a "smart" agent: store facts with [IMPORTANT] tag, skip others
        decisions = []
        stored_count = 0
        for f in obs.all_facts:
            tags = f.get('tags', [])
            is_important = 'important' in tags
            if is_important and stored_count < obs.memory_budget:
                decisions.append(FactDecision(
                    fact_id=f['fact_id'], 
                    decision="store", 
                    anchor=f['text'][:50]
                ))
                stored_count += 1
            else:
                decisions.append(FactDecision(fact_id=f['fact_id'], decision="skip"))
        
        # Fill remaining budget with non-important facts
        if stored_count < obs.memory_budget:
            for f in obs.all_facts:
                if stored_count >= obs.memory_budget:
                    break
                tags = f.get('tags', [])
                if 'important' not in tags:
                    # Update the existing skip decision to store
                    for d in decisions:
                        if d.fact_id == f['fact_id'] and d.decision == "skip":
                            d.decision = "store"
                            d.anchor = f['text'][:50]
                            stored_count += 1
                            break
        
        print(f"\nSIMULATED INGEST DECISIONS ({stored_count} stored):")
        for d in decisions:
            if d.decision == "store":
                print(f"  [{d.fact_id}] STORE -> anchor: '{d.anchor}'")
        
        # 3. Submit ingest
        res = env.step(RecallAction(mode="ingest", decisions=decisions))
        obs = res.observation
        
        print(f"\nMEMORY AFTER INGESTION ({obs.memory_used}/{obs.memory_budget} used):")
        for anchor in obs.memory_anchors:
            print(f"  - {anchor}")
        
        # 4. Query phase
        print(f"\nQUERIES AND ANSWERS ({obs.queries_remaining} queries):")
        query_idx = 0
        while obs.phase == "query":
            current_query = obs.current_query
            
            # Retrieve
            res = env.step(RecallAction(mode="retrieve", query=current_query))
            obs = res.observation
            
            # Show retrieval results
            retrieval_info = ""
            answer = "UNKNOWN"
            if obs.retrieval_results and len(obs.retrieval_results) > 0:
                top = obs.retrieval_results[0]
                answer = top.get("content", "UNKNOWN")
                retrieval_info = f"sim={top.get('similarity', '?'):.3f}, anchor='{top.get('anchor', '?')[:40]}'"
            
            # Answer
            res = env.step(RecallAction(mode="answer", answer_text=answer))
            obs = res.observation
            
            print(f"\n  Q{query_idx}: {current_query}")
            print(f"  Retrieval: {retrieval_info}")
            print(f"  Answer sent: '{answer[:80]}'" if answer != "UNKNOWN" else "  Answer sent: 'UNKNOWN'")
            print(f"  Step reward: {res.reward}")
            query_idx += 1
        
        # 5. Final state
        state = env.state()
        print(f"\n{'='*70}")
        print(f"FINAL STATE:")
        print(f"  correct_answers: {state.correct_answers} / {state.queries_total}")
        print(f"  cumulative_reward: {state.cumulative_reward:.4f}")
        print(f"  baseline_correct: {state.baseline_correct}")
        print(f"  memory_used: {state.memory_used}/{state.memory_budget}")
        print(f"  storage_decisions: {len(state.storage_decisions)}")
        print(f"  failure_attribution: {state.failure_attribution}")
        print(f"{'='*70}")
        
        return state


def run_fifo_diagnostic(seed=0, difficulty=1):
    """Diagnostic 2: Verify FIFO baseline accuracy."""
    print("\n" + "=" * 70)
    print(f"DIAGNOSTIC 2: FIFO Baseline Verification (seed={seed}, difficulty={difficulty})")
    print("=" * 70)
    
    with RecallEnv(base_url=ENV_URL).sync() as env:
        res = env.reset(difficulty=difficulty, seed=seed)
        obs = res.observation
        state = env.state()
        
        print(f"\n  Facts: {len(obs.all_facts)}, Budget: {obs.memory_budget}")
        print(f"  Queries: {state.queries_total}")
        print(f"  FIFO baseline_correct from env: {state.baseline_correct}")
        print(f"  FIFO baseline accuracy: {state.baseline_correct}/{state.queries_total} = {state.baseline_correct/max(1,state.queries_total)*100:.1f}%")
        
        # Now run the actual FIFO strategy manually
        # FIFO: store facts in order, evict oldest when full
        fifo_decisions = []
        for i, f in enumerate(obs.all_facts):
            if i < obs.memory_budget:
                fifo_decisions.append(FactDecision(
                    fact_id=f['fact_id'], 
                    decision="store", 
                    anchor=f['text'][:50]
                ))
            else:
                fifo_decisions.append(FactDecision(
                    fact_id=f['fact_id'], decision="skip"
                ))
        
        # Run FIFO episode
        res = env.step(RecallAction(mode="ingest", decisions=fifo_decisions))
        obs = res.observation
        
        fifo_query_results = []
        while obs.phase == "query":
            q = obs.current_query
            res = env.step(RecallAction(mode="retrieve", query=q))
            obs = res.observation
            answer = "UNKNOWN"
            if obs.retrieval_results and len(obs.retrieval_results) > 0:
                answer = obs.retrieval_results[0].get("content", "UNKNOWN")
            res = env.step(RecallAction(mode="answer", answer_text=answer))
            obs = res.observation
            fifo_query_results.append((q[:60], answer[:60], res.reward))
        
        final = env.state()
        print(f"\n  FIFO actual correct: {final.correct_answers}/{final.queries_total}")
        print(f"  FIFO cumulative_reward: {final.cumulative_reward:.4f}")
        print(f"  env baseline_correct: {final.baseline_correct}")
        print(f"\n  *** MATCH? env says FIFO={final.baseline_correct}, actual FIFO run={final.correct_answers} ***")
        
        if final.correct_answers != final.baseline_correct:
            print("  !!! MISMATCH: FIFO baseline is NOT matching actual FIFO execution !!!")
            print("  This means the baseline comparison is MISLEADING.")


def run_reward_diagnostic(seeds=range(1000, 1005), difficulty=1):
    """Diagnostic 3: Reward component breakdown."""
    print("\n" + "=" * 70)
    print(f"DIAGNOSTIC 3: Reward Component Breakdown")
    print("=" * 70)
    
    for seed in seeds:
        with RecallEnv(base_url=ENV_URL).sync() as env:
            res = env.reset(difficulty=difficulty, seed=seed)
            obs = res.observation
            
            # Store first N facts (budget)
            decisions = []
            for i, f in enumerate(obs.all_facts):
                if i < obs.memory_budget:
                    decisions.append(FactDecision(
                        fact_id=f['fact_id'], decision="store", anchor=f['text'][:50]
                    ))
                else:
                    decisions.append(FactDecision(fact_id=f['fact_id'], decision="skip"))
            
            res = env.step(RecallAction(mode="ingest", decisions=decisions))
            obs = res.observation
            ingest_reward = res.reward
            
            query_rewards = []
            while obs.phase == "query":
                res = env.step(RecallAction(mode="retrieve", query=obs.current_query))
                obs = res.observation
                answer = "UNKNOWN"
                if obs.retrieval_results and len(obs.retrieval_results) > 0:
                    answer = obs.retrieval_results[0].get("content", "UNKNOWN")
                res = env.step(RecallAction(mode="answer", answer_text=answer))
                obs = res.observation
                query_rewards.append(res.reward)
            
            state = env.state()
            # The terminal reward is added on the last step
            terminal_reward = state.cumulative_reward - ingest_reward - sum(query_rewards[:-1])
            
            print(f"\n  Seed {seed}:")
            print(f"    correct: {state.correct_answers}/{state.queries_total}")
            print(f"    ingest step reward: {ingest_reward}")
            print(f"    query step rewards: {query_rewards}")
            print(f"    cumulative_reward: {state.cumulative_reward:.4f}")
            print(f"    baseline_correct: {state.baseline_correct}")


if __name__ == "__main__":
    run_full_episode_diagnostic(seed=0, difficulty=1)
    run_fifo_diagnostic(seed=0, difficulty=1)
    run_reward_diagnostic()

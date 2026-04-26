"""Smoke test: verify reward computation works end-to-end against the live env."""
from envs.recall_env.client import RecallEnv
from envs.recall_env.models import RecallAction, FactDecision

ENV_URL = "wss://s1nn3rx69-recall-env.hf.space"

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
                answer = obs.retrieval_results[0].get("content", "UNKNOWN")  # FIXED: was "text"
            res = env.step(RecallAction(mode="answer", answer_text=answer))
            obs = res.observation

        state = env.state()
        
        qt = max(1, state.queries_total)
        agent_acc = state.correct_answers / qt  # FIXED: was queries_answered_correctly
        baseline_acc = getattr(state, "baseline_correct", 0) / qt
        
        return float(state.cumulative_reward), agent_acc, baseline_acc

def run_smoke():
    # Test 1: Store all facts with anchors
    print("Test 1: Valid store decisions (all 10 facts)...")
    with RecallEnv(base_url=ENV_URL).sync() as env:
        res = env.reset(difficulty=1, seed=1073)
        obs = res.observation
        decisions = [
            FactDecision(fact_id=f["fact_id"], decision="store", anchor=f["text"][:30])
            for f in obs.all_facts[:8]  # budget is 8
        ] + [
            FactDecision(fact_id=f["fact_id"], decision="skip")
            for f in obs.all_facts[8:]
        ]
    
    try:
        reward, a_acc, b_acc = _simulate_episode(ENV_URL, 1, 1073, decisions)
        print(f"  ✓ Reward: {reward:.3f}, Agent Acc: {a_acc:.1%}, Baseline Acc: {b_acc:.1%}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    # Test 2: Skip all facts
    print("\nTest 2: Skip all facts...")
    with RecallEnv(base_url=ENV_URL).sync() as env:
        res = env.reset(difficulty=1, seed=1074)
        obs = res.observation
        decisions = [
            FactDecision(fact_id=f["fact_id"], decision="skip")
            for f in obs.all_facts
        ]
    
    try:
        reward, a_acc, b_acc = _simulate_episode(ENV_URL, 1, 1074, decisions)
        print(f"  ✓ Reward: {reward:.3f}, Agent Acc: {a_acc:.1%}, Baseline Acc: {b_acc:.1%}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    # Test 3: Multiple seeds
    print("\nTest 3: Multiple seeds (1000-1004)...")
    for seed in range(1000, 1005):
        with RecallEnv(base_url=ENV_URL).sync() as env:
            res = env.reset(difficulty=1, seed=seed)
            obs = res.observation
            decisions = [
                FactDecision(fact_id=f["fact_id"], decision="store", anchor=f["text"][:30])
                for f in obs.all_facts[:8]
            ] + [
                FactDecision(fact_id=f["fact_id"], decision="skip")
                for f in obs.all_facts[8:]
            ]
        try:
            reward, a_acc, b_acc = _simulate_episode(ENV_URL, 1, seed, decisions)
            print(f"  Seed {seed}: reward={reward:.3f}, agent={a_acc:.1%}, baseline={b_acc:.1%}")
        except Exception as e:
            print(f"  Seed {seed}: FAILED - {e}")

if __name__ == "__main__":
    run_smoke()

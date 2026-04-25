import sys
from recall_env.client import RecallEnv
from recall_env.models import RecallAction, FactDecision

def smoke_test(url="http://localhost:8000"):
    with RecallEnv(base_url=url).sync() as env:
        obs = env.reset(difficulty=1, seed=0).observation
        print(f"Initial state: phase={obs.phase}, facts={len(obs.all_facts)}")
        
        # Ingest: skip everything
        decisions = [FactDecision(fact_id=f["fact_id"], decision="skip") for f in obs.all_facts]
        res = env.step(RecallAction(mode="ingest", decisions=decisions))
        obs = res.observation
        print(f"After ingest: phase={obs.phase}, reward={res.reward}")
        
        # Query loop: Answer UNKNOWN
        while obs.phase == "query":
            res = env.step(RecallAction(mode="answer", answer_text="UNKNOWN"))
            obs = res.observation
            print(f"Query answered, step reward: {res.reward}")
            
        final_state = env.state()
        print(f"Episode done. Correct: {final_state.correct_answers}, Total Reward: {final_state.cumulative_reward}")
        print("Smoke test PASSED!")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    smoke_test(url)

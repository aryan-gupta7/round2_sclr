import asyncio
import argparse
from envs.recall_env import RecallEnv, RecallAction
from envs.recall_env.models import FactDecision

async def main(url="http://localhost:8000", difficulty=1):
    print(f"Connecting to RECALL Environment at: {url} (Difficulty: {difficulty})")
    
    async with RecallEnv(base_url=url) as env:
        result = await env.reset(difficulty=difficulty, seed=42)
        obs = getattr(result, "observation", result)
        print(f"\n✅ Environment Reset Successful")
        print(f"   Phase: {obs.phase}")
        print(f"   Facts to process: {len(obs.all_facts)}")
        print(f"   Memory Budget: {obs.memory_budget}")
        
        # We are in the "ingest" phase. Let's decide what to store.
        # We will iterate over the batch and store a subset of facts.
        print("\n📥 Starting Ingestion Phase...")
        decisions = []
        for fact in obs.all_facts:
            # We'll skip distractors if possible and store the rest (up to budget)
            if not fact.get("is_distractor", False) and len(decisions) < obs.memory_budget:
                decisions.append(
                    FactDecision(
                        fact_id=fact["fact_id"], 
                        decision="store", 
                        anchor=fact["text"][:30] # Simple snippet anchor
                    )
                )
            else:
                 decisions.append(FactDecision(fact_id=fact["fact_id"], decision="skip"))
                
        print(f"   Storing {sum(d.decision == 'store' for d in decisions)} facts out of {len(obs.all_facts)}.")
        
        # Submit the ingest action
        action = RecallAction(mode="ingest", decisions=decisions)
        result = await env.step(action)
        obs = result.observation
        
        print(f"✅ Ingestion Complete. Moving to: {obs.phase}")
        
        # Now we enter the "query" loop
        while obs.phase == "query":
            q_total = obs.queries_answered + obs.queries_remaining
            print(f"\n❓ Query {obs.queries_answered + 1}/{q_total}: {obs.current_query}")
            
            # 1. Retrieve information based on the query
            print(f"   Searching memory...")
            action = RecallAction(mode="retrieve", query=obs.current_query)
            result = await env.step(action)
            obs = result.observation
            
            best_match = obs.retrieval_results[0] if obs.retrieval_results else None
            if best_match:
                print(f"   Match Found! ({best_match['similarity']:.2f}): {best_match['content'][:60]}...")
                answer = best_match["content"]
            else:
                print(f"   No match found.")
                answer = "UNKNOWN"
                
            # 2. Answer
            print(f"   Sending Answer: '{answer[:60]}...'")
            action = RecallAction(mode="answer", answer_text=answer)
            result = await env.step(action)
            obs = result.observation
            
            print(f"   ✅ Answer graded. Reward for this step: {obs.last_reward}")
            
        print("\n🎉 Episode Complete!")
        print(f"Final Statistics:")
        print(f"   Total Reward: {env._total_reward if hasattr(env, '_total_reward') else 'N/A'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--difficulty", type=int, default=1)
    args = parser.parse_args()
    
    asyncio.run(main(args.url, args.difficulty))

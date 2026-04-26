import asyncio
from envs.recall_env.client import RecallEnv
from envs.recall_env.models import RecallAction, FactDecision

async def test():
    async with RecallEnv(base_url='wss://s1nn3rx69-recall-env.hf.space') as env:
        res = await env.reset(difficulty=1, seed=1073)
        obs = res.observation
        
        # Ingest
        decisions = [FactDecision(fact_id=f["fact_id"], decision="skip") for f in obs.all_facts]
        res = await env.step(RecallAction(mode="ingest", decisions=decisions))
        obs = res.observation
        
        while obs.phase == "query":
            res = await env.step(RecallAction(mode="retrieve", query=obs.current_query))
            obs = res.observation
            answer = "UNKNOWN"
            if obs.retrieval_results and len(obs.retrieval_results) > 0:
                answer = obs.retrieval_results[0].get("text", "UNKNOWN")
            
            print("Sending answer:", answer)
            res = await env.step(RecallAction(mode="answer", answer_text=answer))
            obs = res.observation

asyncio.run(test())

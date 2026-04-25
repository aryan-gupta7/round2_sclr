import pytest
import numpy as np
from envs.recall_env.server.recall_env_environment import RecallEnvironment
from envs.recall_env.server.data_generator import Fact, Query, GroundTruth
from envs.recall_env.models import RecallAction, FactDecision

class MockGenerator:
    def generate(self, config, rng):
        facts = [Fact(fact_id=i, text=f"fact-{i}", tags=[], is_distractor=False, is_correction_of=None, timestep=i) for i in range(10)]
        queries = [Query(query_id=0, text="what is fact-3", expected_answer="fact-3", query_type="specific", relevant_fact_ids=[3])]
        gt = GroundTruth(queries=queries, fact_to_query_map={3: [0]})
        return facts, queries, gt
    def generate_prefill(self, config, rng):
        return []

def test_environment_import():
    env = RecallEnvironment()
    assert env is not None

def test_environment_reset():
    env = RecallEnvironment()
    env.data_generator = MockGenerator()
    obs = env.reset(difficulty=1, seed=0)
    assert obs.phase == "ingest"
    assert len(obs.current_batch) == 8
    assert obs.memory_used == 0

def test_environment_ingest_step():
    env = RecallEnvironment()
    env.data_generator = MockGenerator()
    obs = env.reset(difficulty=1, seed=0)
    
    # Create ingest action
    decisions = [FactDecision(fact_id=i, decision="store", anchor=f"anchor-{i}") for i in range(8)]
    action = RecallAction(mode="ingest", decisions=decisions)
    
    obs = env.step(action)
    assert obs.memory_used == 8
    assert obs.phase == "ingest" # remaining 2 facts
    assert len(obs.current_batch) == 2

def test_environment_full_cycle():
    env = RecallEnvironment()
    env.data_generator = MockGenerator()
    obs = env.reset(difficulty=1, seed=0)
    
    # Step 1: Ingest 8 facts
    decisions = [FactDecision(fact_id=i, decision="store", anchor=f"anchor-{i}") for i in range(8)]
    obs = env.step(RecallAction(mode="ingest", decisions=decisions))
    
    # Step 2: Ingest remaining 2 facts
    decisions = [FactDecision(fact_id=i, decision="store", anchor=f"anchor-{i}") for i in range(8, 10)]
    obs = env.step(RecallAction(mode="ingest", decisions=decisions))
    
    assert obs.phase == "query"
    assert obs.current_query == "what is fact-3"
    
    # Step 3: Retrieve
    obs = env.step(RecallAction(mode="retrieve", query="fact-3"))
    assert len(obs.retrieval_results) > 0
    assert obs.retrieval_results[0]["content"] == "fact-3"
    
    # Step 4: Answer
    obs = env.step(RecallAction(mode="answer", answer_text="fact-3"))
    assert obs.phase == "done"
    assert env.state.correct_answers == 1
    assert obs.reward > 0

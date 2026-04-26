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
        
    def grade(self, answer: str, expected: str) -> bool:
        return expected.lower() in answer.lower()

def test_environment_import():
    env = RecallEnvironment()
    assert env is not None

def test_environment_reset():
    env = RecallEnvironment()
    env.data_generator = MockGenerator()
    obs = env.reset(difficulty=1, seed=0)
    assert obs.phase == "ingest"
    assert len(obs.all_facts) == 10
    assert obs.memory_used == 0

def test_environment_ingest_step():
    env = RecallEnvironment()
    env.data_generator = MockGenerator()
    obs = env.reset(difficulty=1, seed=0)
    
    # Create ingest action for all 10 facts (2 skips, 8 stores)
    decisions = [FactDecision(fact_id=i, decision="store", anchor=f"anchor-{i}") for i in range(8)]
    decisions += [FactDecision(fact_id=i, decision="skip") for i in range(8, 10)]
    action = RecallAction(mode="ingest", decisions=decisions)
    
    obs = env.step(action)
    assert obs.memory_used == 5
    assert obs.phase == "query"

def test_environment_full_cycle():
    env = RecallEnvironment()
    env.data_generator = MockGenerator()
    obs = env.reset(difficulty=1, seed=0)
    
    # Step 1: Ingest all 10 facts
    decisions = [FactDecision(fact_id=i, decision="store", anchor=f"anchor-{i}") for i in range(8)]
    decisions += [FactDecision(fact_id=i, decision="skip") for i in range(8, 10)]
    obs = env.step(RecallAction(mode="ingest", decisions=decisions))
    
    assert obs.phase == "query"
    assert obs.current_query == "what is fact-3"
    
    # Step 2: Retrieve
    obs = env.step(RecallAction(mode="retrieve", query="fact-3"))
    assert len(obs.retrieval_results) > 0
    assert obs.retrieval_results[0]["content"] == "fact-3"
    
    # Step 3: Answer
    obs = env.step(RecallAction(mode="answer", answer_text="fact-3"))
    assert obs.phase == "done"
    assert env._state.correct_answers == 1
    assert obs.reward > 0

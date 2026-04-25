import os
import yaml
import numpy as np
from uuid import uuid4
from typing import Optional, List, Dict, Any, Union

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import RecallAction, RecallObservation, RecallState, FactDecision
except (ImportError, ModuleNotFoundError):
    from models import RecallAction, RecallObservation, RecallState, FactDecision

try:
    from .memory_backend import MemoryBackend
    from .data_generator import DataGenerator, LevelConfig, Fact, Query, GroundTruth
    from .rewards import compute_reward, EpisodeResult
except (ImportError, ModuleNotFoundError):
    from memory_backend import MemoryBackend
    from data_generator import DataGenerator, LevelConfig, Fact, Query, GroundTruth
    from rewards import compute_reward, EpisodeResult

class RecallEnvironment(Environment):
    """
    RECALL — An environment where the agent manages its own memory (REVISED).
    """
    
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config_dir: str = "training/configs"):
        self.config_dir = config_dir
        self.data_generator = DataGenerator()
        self._state: Optional[RecallState] = None
        
        self.config: Optional[LevelConfig] = None
        self.memory: Optional[MemoryBackend] = None
        self.facts: List[Fact] = []
        self.queries: List[Query] = []
        self.gt: Optional[GroundTruth] = None
        
        self.current_query_idx = 0
        self.phase = "ingest"
        self.last_reward = 0.0
        self.malformed_count = 0
        self.max_malformed = 3
        
        self.budget_overflow_count = 0
        self.last_retrieval_results = None
        
        # Training step for reward phase tracking
        self.global_step = 0

    def _load_config(self, difficulty: int) -> LevelConfig:
        config_path = os.path.join(self.config_dir, f"level_{difficulty}.yaml")
        if not os.path.exists(config_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../training/configs"))
            config_path = os.path.join(base_dir, f"level_{difficulty}.yaml")
            
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Curriculum config missing: {config_path}")
             
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            
        return LevelConfig(**data)

    def reset(self, difficulty: int = 1, seed: int = 0, global_step: int = 0, **kwargs) -> RecallObservation:
        self.config = self._load_config(difficulty)
        self.global_step = global_step
        rng = np.random.default_rng(seed)
        
        self.facts, self.queries, self.gt = self.data_generator.generate(self.config, rng)
            
        self.memory = MemoryBackend(
            budget=self.config.memory_budget,
            embedding_model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
            seed=seed
        )
        
        if self.config.prefilled_memory_count > 0:
            prefill_items = self.data_generator.generate_prefill(self.config, rng)
            self.memory.prefill(prefill_items)
            
        # PRE-COMPUTE FIFO baseline accuracy
        self.baseline_correct = self._run_fifo_baseline_dry()
            
        self.phase = "ingest"
        self.current_query_idx = 0
        self.last_reward = 0.0
        self.malformed_count = 0
        self.budget_overflow_count = 0
        self.last_retrieval_results = None
        
        self._state = RecallState(
            episode_id=str(uuid4()),
            step_count=0,
            difficulty=difficulty,
            seed=seed,
            phase=self.phase,
            facts_total=len(self.facts),
            queries_total=len(self.queries),
            queries_answered=0,
            correct_answers=0,
            memory_used=len(self.memory.items),
            memory_budget=self.config.memory_budget,
            cumulative_reward=0.0,
            storage_decisions=[],
            failure_attribution=[],
            baseline_correct=self.baseline_correct
        )
        
        return self._build_observation()

    def _run_fifo_baseline_dry(self) -> int:
        """Simulate FIFO behavior on this seed."""
        # Simple FIFO simulation:
        # 1. Fill memory with facts in order.
        # 2. When full, eject oldest.
        memory_ids = []
        # Pre-fill handled first (assumed static for now)
        prefill_count = self.config.prefilled_memory_count
        # We assume prefilled items are "oldest"
        for i in range(prefill_count):
             memory_ids.append(f"prefilled_{i}")
             
        for fact in self.facts:
             if len(memory_ids) >= self.config.memory_budget:
                  memory_ids.pop(0)
             memory_ids.append(fact.fact_id)
        
        final_memory_ids = set(memory_ids)
        correct = 0
        for query in self.queries:
             if not query.relevant_fact_ids: # UNKNOWN
                  # A dummy FIFO might just say UNKNOWN if no match, 
                  # but let's assume it only gets it right if it was UNKNOWN
                  if query.expected_answer == "UNKNOWN":
                       correct += 1
                  continue
             # If ANY relevant fact is in memory, FIFO survives
             # Note: This is an optimistic FIFO simulation.
             if any(fid in final_memory_ids for fid in query.relevant_fact_ids):
                  correct += 1
        return correct

    def _build_observation(self) -> RecallObservation:
        instruction = self._get_instruction()
        
        all_facts = None
        if self.phase == "ingest":
            all_facts = [{"fact_id": f.fact_id, "text": f.text, "tags": f.tags} for f in self.facts]
            
        current_query = None
        if self.phase == "query" and self.current_query_idx < len(self.queries):
            current_query = self.queries[self.current_query_idx].text
            
        used, total = self.memory.usage()
        
        return RecallObservation(
            done=(self.phase == "done"),
            reward=self.last_reward,
            phase=self.phase,
            all_facts=all_facts,
            current_query=current_query,
            retrieval_results=self.last_retrieval_results,
            memory_anchors=self.memory.current_anchors(),
            memory_used=used,
            memory_budget=total,
            queries_remaining=len(self.queries) - self.current_query_idx,
            queries_answered=self.current_query_idx,
            last_reward=self.last_reward,
            instruction=instruction
        )

    def _get_instruction(self) -> str:
        if self.phase == "ingest":
            return f"Ingestion Phase: Decide which of the {len(self.facts)} facts to store."
        elif self.phase == "query":
            return "Query Phase: Retrieve facts or answer the question directly."
        else:
            return "Episode Finished."

    def step(self, action: RecallAction) -> RecallObservation:
        self._state.step_count += 1
        self.last_retrieval_results = None
        
        action_was_valid = self._validate_action(action)
        
        if not action_was_valid:
            self.malformed_count += 1
            # Immediate penalty for malformed
            step_reward = -0.5
        else:
            self.malformed_count = 0
            step_reward = self._process_action(action)
            
        self.last_reward = step_reward
        self._state.cumulative_reward += step_reward
        
        # Check termination
        if self.malformed_count >= self.max_malformed or self.phase == "done":
             self.phase = "done"
             # Final reward computation
             terminal_reward = self._compute_final_reward()
             self._state.cumulative_reward += terminal_reward
             self.last_reward += terminal_reward
             
        self._state.phase = self.phase
        self._state.queries_answered = self.current_query_idx
        self._state.memory_used = len(self.memory.items)
        
        return self._build_observation()

    def _validate_action(self, action: RecallAction) -> bool:
        if self.phase == "ingest":
            if action.mode != "ingest" or action.decisions is None:
                return False
            if len(action.decisions) != len(self.facts):
                return False
        elif self.phase == "query":
            if action.mode not in ["retrieve", "answer"]:
                return False
        return True

    def _process_action(self, action: RecallAction) -> float:
        if action.mode == "ingest":
            for d in action.decisions:
                fact = next(f for f in self.facts if f.fact_id == d.fact_id)
                decision_info = {"fact_id": d.fact_id, "decision": d.decision, "anchor": d.anchor}
                if d.decision == "store":
                    slot_id = self.memory.store(d.anchor, fact.text, step=self._state.step_count)
                    if slot_id is None:
                        self.budget_overflow_count += 1
                        decision_info["status"] = "budget_full"
                    else:
                        decision_info["status"] = "stored"
                        decision_info["slot_id"] = slot_id
                self._state.storage_decisions.append(decision_info)
            self.phase = "query"
            if not self.queries:
                self.phase = "done"
            # In Phase 2, step rewards and budget overflow are deferred or handled at end?
            # 06_REWARD_DESIGN says malformed penalties emit immediately, others deferred.
            # But budget overflow penalty was in phase1_reward.
            # I'll just return 0 here and handle it in terminal reward calculation.
            return 0.0
            
        elif action.mode == "retrieve":
            self.last_retrieval_results = self.memory.retrieve(action.query, self.config.retrieval_k)
            return 0.0
            
        elif action.mode == "answer":
            query = self.queries[self.current_query_idx]
            is_correct = self.data_generator.grade(action.answer_text, query.expected_answer)
            if is_correct:
                self._state.correct_answers += 1
                # Mark contributed facts (for bootstrap phase reward)
                if self.last_retrieval_results:
                     for r in self.last_retrieval_results:
                          for decision in self._state.storage_decisions:
                               if decision.get("slot_id") == r["slot_id"] and decision["fact_id"] in query.relevant_fact_ids:
                                    decision["was_later_retrieved_and_correct"] = True
            
            # Failure attribution (optional logger)
            self._state.failure_attribution.append({"query_id": query.query_id, "correct": is_correct})
            
            self.current_query_idx += 1
            if self.current_query_idx >= len(self.queries):
                self.phase = "done"
            return 0.0
            
        return 0.0

    def _compute_final_reward(self) -> float:
        # Collect retrieval-correct count for bootstrap
        retrieved_correct = 0
        for d in self._state.storage_decisions:
             if d.get("was_later_retrieved_and_correct"):
                  retrieved_correct += 1
                  
        result = EpisodeResult(
            correct_answers=self._state.correct_answers,
            stored_then_retrieved_count=retrieved_correct,
            memory_used=len(self.memory.items),
            malformed_count=self.malformed_count,
            budget_overflow_count=self.budget_overflow_count,
            queries_total=len(self.queries)
        )
        return compute_reward(result, self.baseline_correct, self.config, self.global_step)

    @property
    def state(self) -> RecallState:
        return self._state

import os
import yaml
import numpy as np
from uuid import uuid4
from typing import Optional, List, Dict, Any, Union

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import RecallAction, RecallObservation, RecallState, FactDecision
except ImportError:
    from models import RecallAction, RecallObservation, RecallState, FactDecision

from .memory_backend import MemoryBackend
from .data_generator import DataGenerator, LevelConfig, Fact, Query, GroundTruth
from .rewards import compute_step_reward, compute_terminal_reward

class RecallEnvironment(Environment):
    """
    RECALL — An environment where the agent manages its own memory.
    """
    
    SUPPORTS_CONCURRENT_SESSIONS: bool = False  # Keep as False for now as per rules

    def __init__(self, config_dir: str = "training/configs"):
        """
        Initialize the Recall environment.
        Args:
            config_dir: Directory containing level_<N>.yaml files.
        """
        self.config_dir = config_dir
        self.data_generator = DataGenerator()
        self._state: Optional[RecallState] = None
        
        # Internal state for tracking
        self.config: Optional[LevelConfig] = None
        self.memory: Optional[MemoryBackend] = None
        self.facts: List[Fact] = []
        self.queries: List[Query] = []
        self.gt: Optional[GroundTruth] = None
        
        self.current_fact_idx = 0
        self.current_query_idx = 0
        self.phase = "ingest"
        self.last_reward = 0.0
        self.malformed_count = 0
        self.max_malformed = 3
        
        # Track for rewards
        self.new_correct = 0
        self.new_unknown_correct = 0
        self.storage_attempts = 0
        self.storage_rejected = 0
        
        # For retrieval results during step
        self.last_retrieval_results = None

    def _load_config(self, difficulty: int) -> LevelConfig:
        # Check if we are running in a directory where training/configs exists
        # If not, try to find it relative to this file
        config_path = os.path.join(self.config_dir, f"level_{difficulty}.yaml")
        if not os.path.exists(config_path):
            # Fallback for search
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../training/configs"))
            config_path = os.path.join(base_dir, f"level_{difficulty}.yaml")
            
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Curriculum config missing: {config_path}")
             
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            
        return LevelConfig(**data)

    def reset(self, difficulty: int = 1, seed: int = 0, **kwargs) -> RecallObservation:
        """
        Reset the environment for a new episode.
        """
        self.config = self._load_config(difficulty)
        rng = np.random.default_rng(seed)
        
        # Initialize data
        try:
            self.facts, self.queries, self.gt = self.data_generator.generate(self.config, rng)
        except NotImplementedError:
            # For now, if not implemented, we can't really run.
            # But we can mock it for L1 if needed for smoke tests.
            raise
            
        # Initialize Memory
        self.memory = MemoryBackend(
            budget=self.config.memory_budget,
            embedding_model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
            seed=seed
        )
        
        # Prefill memory
        if self.config.prefilled_memory_count > 0:
            prefill_items = self.data_generator.generate_prefill(self.config, rng)
            self.memory.prefill(prefill_items)
            
        self.phase = "ingest"
        self.current_fact_idx = 0
        self.current_query_idx = 0
        self.last_reward = 0.0
        self.malformed_count = 0
        self.new_correct = 0
        self.new_unknown_correct = 0
        self.storage_attempts = 0
        self.storage_rejected = 0
        self.last_retrieval_results = None
        
        self._state = RecallState(
            episode_id=str(uuid4()),
            step_count=0,
            difficulty=difficulty,
            seed=seed,
            phase=self.phase,
            facts_total=len(self.facts),
            facts_ingested=0,
            queries_total=len(self.queries),
            queries_answered=0,
            correct_answers=0,
            memory_used=len(self.memory.items),
            memory_budget=self.config.memory_budget,
            cumulative_reward=0.0,
            storage_decisions=[],
            failure_attribution=[]
        )
        
        return self._build_observation()

    def _build_observation(self) -> RecallObservation:
        instruction = self._get_instruction()
        
        current_batch = None
        if self.phase == "ingest":
            batch_end = min(self.current_fact_idx + self.config.batch_size, len(self.facts))
            batch = self.facts[self.current_fact_idx:batch_end]
            current_batch = [{"fact_id": f.fact_id, "text": f.text, "tags": f.tags} for f in batch]
            
        current_query = None
        if self.phase == "query" and self.current_query_idx < len(self.queries):
            current_query = self.queries[self.current_query_idx].text
            
        used, total = self.memory.usage()
        
        return RecallObservation(
            done=(self.phase == "done"),
            reward=self.last_reward,
            phase=self.phase,
            current_batch=current_batch,
            current_query=current_query,
            retrieval_results=self.last_retrieval_results,
            memory_anchors=self.memory.current_anchors(),
            memory_used=used,
            memory_budget=total,
            facts_remaining=len(self.facts) - self.current_fact_idx,
            queries_remaining=len(self.queries) - self.current_query_idx,
            queries_answered=self.current_query_idx,
            last_reward=self.last_reward,
            instruction=instruction
        )

    def _get_instruction(self) -> str:
        if self.phase == "ingest":
            hints = "\n".join(self.config.system_prompt_hints)
            return f"Ingestion Phase: Decide which facts to store. Budget: {self.memory.usage()[0]}/{self.memory.usage()[1]}.\n{hints}"
        elif self.phase == "query":
            return "Query Phase: Retrieve relevant facts and answer the question. Say UNKNOWN if not found."
        else:
            return "Episode Finished."

    def step(self, action: RecallAction) -> RecallObservation:
        self._state.step_count += 1
        self.new_correct = 0
        self.new_unknown_correct = 0
        self.storage_attempts = 0
        self.storage_rejected = 0
        self.last_retrieval_results = None
        
        action_was_valid = self._validate_action(action)
        
        if not action_was_valid:
            self.malformed_count += 1
            reward = compute_step_reward(action, False, 0, 0, 0, 0, self.config)
        else:
            self.malformed_count = 0  # Reset on valid action
            reward = self._process_action(action)
            
        self.last_reward = reward
        self._state.cumulative_reward += reward
        
        # Check termination
        if self.malformed_count >= self.max_malformed:
             self.phase = "done"
        
        if self.phase == "done":
             terminal_reward = compute_terminal_reward(self._state, self.config)
             self._state.cumulative_reward += terminal_reward
             self.last_reward += terminal_reward # Surface terminal reward in the last step?
             # Note: typically terminal reward is returned in the last step's observation
             
        # Update state fields
        self._state.phase = self.phase
        self._state.facts_ingested = self.current_fact_idx
        self._state.queries_answered = self.current_query_idx
        self._state.memory_used = len(self.memory.items)
        
        return self._build_observation()

    def _validate_action(self, action: RecallAction) -> bool:
        if self.phase == "ingest":
            if action.mode not in ["ingest", "delete"]:
                return False
            if action.mode == "ingest":
                if action.decisions is None:
                    return False
                # Check batch size
                batch_size = min(self.config.batch_size, len(self.facts) - self.current_fact_idx)
                if len(action.decisions) != batch_size:
                    return False
                # Check anchors for store decisions
                for d in action.decisions:
                    if d.decision == "store" and (not d.anchor or not d.anchor.strip()):
                        return False
            elif action.mode == "delete":
                if action.slot_id is None:
                    return False
                    
        elif self.phase == "query":
            if action.mode not in ["retrieve", "answer"]:
                return False
            if action.mode == "retrieve" and action.query is None:
                return False
            if action.mode == "answer" and action.answer_text is None:
                return False
                
        return True

    def _process_action(self, action: RecallAction) -> float:
        if action.mode == "ingest":
            for d in action.decisions:
                fact = next(f for f in self.facts if f.fact_id == d.fact_id)
                decision_info = {
                    "fact_id": d.fact_id,
                    "decision": d.decision,
                    "anchor": d.anchor,
                    "was_later_retrieved_and_correct": False,
                    "was_queried": False # Will be updated during query phase
                }
                
                if d.decision == "store":
                    self.storage_attempts += 1
                    slot_id = self.memory.store(d.anchor, fact.text, step=self._state.step_count)
                    if slot_id is None:
                        self.storage_rejected += 1
                        decision_info["status"] = "budget_full"
                    else:
                        decision_info["status"] = "stored"
                        decision_info["slot_id"] = slot_id
                else:
                    decision_info["status"] = "skipped"
                
                self._state.storage_decisions.append(decision_info)
                
            self.current_fact_idx += len(action.decisions)
            if self.current_fact_idx >= len(self.facts):
                self.phase = "query"
                if not self.queries:
                    self.phase = "done"
                    
        elif action.mode == "delete":
            self.memory.delete(action.slot_id)
            
        elif action.mode == "retrieve":
            self.last_retrieval_results = self.memory.retrieve(action.query, self.config.retrieval_k)
            # Track which stored items were retrieved for later reward
            retrieved_ids = {r["slot_id"] for r in self.last_retrieval_results}
            # For now we don't have a direct query link here, but we could track it.
            
        elif action.mode == "answer":
            query = self.queries[self.current_query_idx]
            is_correct = self._grade_answer(action.answer_text, query.expected_answer)
            
            if is_correct:
                if query.expected_answer == "UNKNOWN":
                    self.new_unknown_correct = 1
                else:
                    self.new_correct = 1
                self._state.correct_answers += 1
                
                # Mark facts as having contributed to a correct answer for shaping bonus
                # This is simplified: any fact retrieved before this answer that was relevant
                # gets the bonus.
                if self.last_retrieval_results:
                    for r in self.last_retrieval_results:
                        if r["slot_id"] is not None and r["slot_id"] in [item.slot_id for item in self.memory.items]:
                            # Find which fact it was
                            # This is a bit inefficient to scan everything, but works for MVP
                            for decision in self._state.storage_decisions:
                                if decision.get("slot_id") == r["slot_id"]:
                                    # If the fact was actually relevant to this specific query
                                    if decision["fact_id"] in query.relevant_fact_ids:
                                        decision["was_later_retrieved_and_correct"] = True
            
            # Log failure attribution
            failure = self._attribute_failure(action.answer_text, query, is_correct)
            self._state.failure_attribution.append({
                "query_id": query.query_id,
                "type": failure
            })
            
            # Mark facts as queried for skip bonus
            for fact_id in query.relevant_fact_ids:
                for decision in self._state.storage_decisions:
                    if decision["fact_id"] == fact_id:
                        decision["was_queried"] = True

            self.current_query_idx += 1
            if self.current_query_idx >= len(self.queries):
                self.phase = "done"
                
        return compute_step_reward(
            action, True, self.new_correct, self.new_unknown_correct,
            self.storage_attempts, self.storage_rejected, self.config
        )

    def _grade_answer(self, agent_answer: str, expected_answer: str) -> bool:
        if not agent_answer:
            return False
        # Exact match after normalization
        a = agent_answer.strip().lower().replace(".", "").replace(",", "")
        b = expected_answer.strip().lower().replace(".", "").replace(",", "")
        return a == b

    def _attribute_failure(self, agent_answer: str, query: Query, is_correct: bool) -> str:
        if is_correct:
            return "success"
        
        if query.expected_answer == "UNKNOWN":
            return "reasoning_failure" # Said something else when should have said UNKNOWN
            
        # Check if relevant facts were stored
        stored_ids = {d["fact_id"] for d in self._state.storage_decisions if d["decision"] == "store"}
        relevant_stored = [fid for fid in query.relevant_fact_ids if fid in stored_ids]
        
        if not relevant_stored:
            return "storage_failure"
            
        # Check if they were retrieved in the last retrieval (if any)
        if self.last_retrieval_results:
             retrieved_fact_ids = []
             for res in self.last_retrieval_results:
                  # Map slot_id back to fact_id
                  for d in self._state.storage_decisions:
                       if d.get("slot_id") == res["slot_id"]:
                            retrieved_fact_ids.append(d["fact_id"])
                            break
             
             if any(fid in retrieved_fact_ids for fid in relevant_stored):
                  return "reasoning_failure"
             else:
                  # If they were stored but not retrieved, check similarity
                  # This could be anchor_failure (bad match) or retrieval_failure (truncated)
                  return "anchor_failure" 
        else:
             return "retrieval_failure" # Answered without retrieving

    @property
    def state(self) -> RecallState:
        return self._state

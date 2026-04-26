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

from .memory_backend import MemoryBackend
from .data_generator import DataGenerator, LevelConfig, Fact, Query, GroundTruth
from .rewards import compute_reward, EpisodeResult


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
            
        mc = getattr(self.config, 'max_core_slots', self.config.memory_budget) if getattr(self.config, 'permanence_enabled', False) else self.config.memory_budget
        mw = getattr(self.config, 'max_working_slots', self.config.memory_budget) if getattr(self.config, 'permanence_enabled', False) else self.config.memory_budget
        self.memory = MemoryBackend(
            budget=self.config.memory_budget,
            embedding_model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
            seed=seed,
            retrieval_mode=getattr(self.config, 'retrieval_mode', 'hybrid'),
            max_core=mc,
            max_working=mw
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
        """Simulate FIFO behavior using real retrieval + grading pipeline.
        
        Stores first N facts (budget) with their text as anchors,
        then runs retrieval + grading identically to how the agent is scored.
        This ensures an honest apples-to-apples comparison.
        """
        fifo_mem = MemoryBackend(
            budget=self.config.memory_budget,
            embedding_model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
            seed=self._state.seed if self._state else 0,
            retrieval_mode=getattr(self.config, 'retrieval_mode', 'hybrid'),
        )
        
        # FIFO: store facts in order until budget is full
        for fact in self.facts:
            if len(fifo_mem.items) >= self.config.memory_budget:
                break
            fifo_mem.store(fact.text, fact.text, step=0)
        
        correct = 0
        for query in self.queries:
            if not query.relevant_fact_ids:
                # Negative/unknown query — FIFO answers UNKNOWN
                if query.expected_answer == "UNKNOWN":
                    correct += 1
                continue
            
            # Retrieve using the same cosine similarity as the agent
            results = fifo_mem.retrieve(query.text, self.config.retrieval_k)
            answer = "UNKNOWN"
            if results:
                answer = results[0]["content"]
            
            if self.data_generator.grade(answer, query.expected_answer):
                correct += 1
        
        return correct

    def _build_observation(self) -> RecallObservation:
        instruction = self._get_instruction()
        
        all_facts = None
        if self.phase == "ingest":
            all_facts = [{"fact_id": f.fact_id, "text": f.text, "tags": f.tags} for f in self.facts]
            
        current_query = None
        inferred_query_tag = None
        if self.phase == "query" and self.current_query_idx < len(self.queries):
            current_query = self.queries[self.current_query_idx].text
            q_lower = current_query.lower()
            if any(w in q_lower for w in ["when", "date", "scheduled", "before"]):
                inferred_query_tag = "temporal"
            elif any(w in q_lower for w in ["who", "architecture", "model", "which"]):
                inferred_query_tag = "identity"
            elif any(w in q_lower for w in ["relationship", "better than", "similar to", "compared"]):
                inferred_query_tag = "relational"
            elif any(w in q_lower for w in ["how", "steps", "fix for"]):
                inferred_query_tag = "procedural"
            else:
                inferred_query_tag = "factual"

        usage_dict = self.memory.usage()
        if isinstance(usage_dict, dict):
            used = usage_dict.get("total_used", 0)
            core_used = usage_dict.get("core_used", 0)
            core_tot = usage_dict.get("max_core", self.config.memory_budget)
            work_used = usage_dict.get("working_used", 0)
            work_tot = usage_dict.get("max_working", self.config.memory_budget)
            total = usage_dict.get("budget", self.config.memory_budget)
        else:
            used, total = usage_dict
            core_used, core_tot, work_used, work_tot = 0, 0, 0, 0
        
        memory_idx = None
        if getattr(self.config, 'overwrite_enabled', False) and self.phase == "ingest":
            memory_idx = []
            for item in self.memory.items:
                memory_idx.append({
                    "slot_id": str(item.slot_id),
                    "anchor": item.anchor,
                    "tag": item.tag,
                    "permanence": item.permanence
                })
        
        return RecallObservation(
            done=(self.phase == "done"),
            reward=self.last_reward,
            phase=self.phase,
            all_facts=all_facts,
            current_query=current_query,
            inferred_query_tag=inferred_query_tag,
            retrieval_results=self.last_retrieval_results,
            memory_anchors=self.memory.current_anchors(),
            memory_used=used,
            memory_budget=total,
            core_slots_used=core_used,
            core_slots_total=core_tot,
            working_slots_used=work_used,
            working_slots_total=work_tot,
            memory_index=memory_idx,
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
                # Auto-fill anchor from fact text if not provided
                anchor = d.anchor if d.anchor else fact.text
                decision_info = {"fact_id": d.fact_id, "decision": d.decision, "anchor": anchor}

                # F1 Tagging
                tag_to_store = "untagged"
                if getattr(self.config, 'tagging_enabled', False):
                    if d.tag is not None:
                        valid_tags = getattr(self.config, 'tag_vocabulary', [])
                        if valid_tags and d.tag in valid_tags:
                            tag_to_store = d.tag
                        else:
                            self.malformed_count += 1
                            tag_to_store = "untagged"
                            decision_info["status"] = "malformed_tag"
                
                if d.tag is not None:
                    decision_info["tag"] = d.tag

                # F2 Permanence
                permanence_to_store = "working"
                if getattr(self.config, 'permanence_enabled', False):
                    if d.permanence in ["core", "working"]:
                        permanence_to_store = d.permanence
                    elif d.permanence is not None:
                        self.malformed_count += 1
                        decision_info["status"] = "malformed_permanence"

                fact_emb = self.memory._get_embedding(fact.text)
                self.memory.check_and_reinforce(fact_emb)

                if d.decision == "store":
                    slot_id = self.memory.store(anchor, fact.text, step=self._state.step_count, tag=tag_to_store, permanence=permanence_to_store)
                    if slot_id is None:
                        self.budget_overflow_count += 1
                        decision_info["status"] = "budget_full"
                    else:
                        decision_info["status"] = "stored"
                        decision_info["slot_id"] = slot_id

                elif d.decision == "overwrite":
                    try:
                        t_id = int(d.target_id)
                    except (ValueError, TypeError):
                        t_id = -1
                    if t_id == -1 or not getattr(d, 'overwrite_anchor', None):
                        self.malformed_count += 1
                        decision_info["status"] = "malformed_overwrite"
                    else:
                        t_tag = d.tag if (getattr(self.config, 'tagging_enabled', False) and d.tag in getattr(self.config, 'tag_vocabulary', [])) else None
                        t_perm = d.permanence if (getattr(self.config, 'permanence_enabled', False) and d.permanence in ["core", "working"]) else None
                        
                        success = self.memory.overwrite(
                            slot_id=t_id,
                            anchor=d.overwrite_anchor,
                            content=fact.text,
                            step=self._state.step_count,
                            tag=t_tag,
                            permanence=t_perm
                        )
                        if not success:
                            self.malformed_count += 1
                            decision_info["status"] = "malformed_overwrite_failed"
                        else:
                            decision_info["status"] = "overwritten"
                            decision_info["slot_id"] = t_id
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
            tag_filter = getattr(action, "retrieve_tag_filter", None)
            self.last_retrieval_results = self.memory.retrieve(action.query, self.config.retrieval_k, tag_filter=tag_filter)
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
            failure_attrib = {"query_id": query.query_id, "correct": is_correct}
            if action.mode == "retrieve" or self.last_retrieval_results is not None:
                 failure_attrib["retrieval_tag_used"] = getattr(action, "retrieve_tag_filter", None) is not None
                 # simple heuristic for was_correct: if correct, the tag filter was correct
                 failure_attrib["retrieval_tag_was_correct"] = failure_attrib.get("retrieval_tag_used") and is_correct

            self._state.failure_attribution.append(failure_attrib)
            
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

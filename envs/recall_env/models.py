from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State

class FactDecision(BaseModel):
    """One per fact in the current ingestion batch."""
    fact_id: int
    decision: Literal["store", "skip", "overwrite"]
    anchor: Optional[str] = None  # required if decision == "store"
    tag: Optional[str] = None     # F1: tag for the memory
    permanence: Optional[str] = None # F2: "core" | "working"
    target_id: Optional[str] = None # F3: slot_id to overwrite
    overwrite_anchor: Optional[str] = None # F3: new anchor

class RecallAction(Action):
    """
    Action for the Recall environment (REVISED).
    """
    mode: Literal["ingest", "retrieve", "answer"]
    # for ingest mode (all decisions at once):
    decisions: Optional[List[FactDecision]] = None
    # for retrieve mode:
    query: Optional[str] = None
    retrieve_tag_filter: Optional[str] = None # F1
    # for answer mode:
    answer_text: Optional[str] = None  # may be the literal "UNKNOWN"

class RecallObservation(Observation):
    """
    Observation from the Recall environment (REVISED).
    """
    phase: Literal["ingest", "query", "done"]
    # Ingestion phase:
    all_facts: Optional[List[Dict]] = None         # full list shown once
    # Query phase:
    current_query: Optional[str] = None
    inferred_query_tag: Optional[str] = None # F1
    retrieval_results: Optional[List[Dict]] = None # populated only after retrieve action
    memory_index: Optional[List[Dict]] = None # F3
    # Always present:
    memory_anchors: List[str]
    memory_used: int
    memory_budget: int
    core_slots_used: int = 0      # F2
    core_slots_total: int = 0     # F2
    working_slots_used: int = 0   # F2
    working_slots_total: int = 0  # F2
    queries_remaining: int
    queries_answered: int
    last_reward: float
    instruction: str

class RecallState(State):
    """
    Full state of the Recall environment (REVISED).
    """
    difficulty: int
    seed: int
    phase: str
    facts_total: int
    queries_total: int
    queries_answered: int
    correct_answers: int
    memory_used: int
    memory_budget: int
    cumulative_reward: float
    # For debugging / metrics:
    storage_decisions: List[Dict] = Field(default_factory=list)  # per-fact log
    failure_attribution: List[Dict] = Field(default_factory=list)  # per-query log
    baseline_correct: int = 0              # FIFO baseline accuracy on this seed

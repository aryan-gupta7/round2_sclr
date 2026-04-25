from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State

class FactDecision(BaseModel):
    """One per fact in the current ingestion batch."""
    fact_id: int
    decision: Literal["store", "skip"]
    anchor: Optional[str] = None  # required if decision == "store"

class RecallAction(Action):
    """
    Action for the Recall environment (REVISED).
    """
    mode: Literal["ingest", "retrieve", "answer"]
    # for ingest mode (all decisions at once):
    decisions: Optional[List[FactDecision]] = None
    # for retrieve mode:
    query: Optional[str] = None
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
    retrieval_results: Optional[List[Dict]] = None # populated only after retrieve action
    # Always present:
    memory_anchors: List[str]
    memory_used: int
    memory_budget: int
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

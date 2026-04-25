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
    Action for the Recall environment.
    One Action class with a discriminated union over modes.
    """
    mode: Literal["ingest", "retrieve", "answer", "delete"]
    # for ingest mode:
    decisions: Optional[List[FactDecision]] = None
    # for retrieve mode:
    query: Optional[str] = None
    # for answer mode:
    answer_text: Optional[str] = None  # may be the literal "UNKNOWN"
    # for delete mode (only allowed if budget violation):
    slot_id: Optional[int] = None

class RecallObservation(Observation):
    """
    Observation from the Recall environment.
    """
    phase: Literal["ingest", "query", "done"]
    # During ingest phase:
    current_batch: Optional[List[Dict]] = None  # [{"fact_id": int, "text": str, "tags": list[str]}, ...]
    # During query phase:
    current_query: Optional[str] = None
    retrieval_results: Optional[List[Dict]] = None  # populated after a retrieve action
    # Always present:
    memory_anchors: List[str]                # CURRENT anchors only, not full content
    memory_used: int
    memory_budget: int
    facts_remaining: int
    queries_remaining: int
    queries_answered: int
    last_reward: float
    instruction: str                         # phase-appropriate instruction text for the LLM

class RecallState(State):
    """
    Full state of the Recall environment, used for evaluation and debugging.
    """
    difficulty: int
    seed: int
    phase: str
    facts_total: int
    facts_ingested: int
    queries_total: int
    queries_answered: int
    correct_answers: int
    memory_used: int
    memory_budget: int
    cumulative_reward: float
    # For debugging / metrics:
    storage_decisions: List[Dict] = Field(default_factory=list)  # per-fact: stored or not, anchor written, was-later-retrieved
    failure_attribution: List[Dict] = Field(default_factory=list)  # per-query: which failure mode (storage / anchor / retrieve / reasoning)

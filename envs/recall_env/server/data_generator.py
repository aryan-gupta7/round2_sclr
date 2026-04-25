import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

@dataclass
class LevelConfig:
    difficulty: int
    facts_total: int
    queries_total: int
    memory_budget: int
    batch_size: int                    # facts per ingestion step
    retrieval_k: int                   # top-k for retrieve action
    embedding_model: str
    embedding_dim: Optional[int] = None # projection dim; null = native dim
    prefilled_memory_count: int = 0
    distractor_rate: float = 0.0
    contradiction_rate: float = 0.0
    adversarial_tag_rate: float = 0.0
    explicit_importance_tags: bool = False   # facts may carry [IMPORTANT] markers
    query_distribution: Dict[str, float] = field(default_factory=dict)
    reward_shaping: Dict[str, float] = field(default_factory=dict)
    system_prompt_hints: List[str] = field(default_factory=list)

@dataclass
class Fact:
    fact_id: int
    text: str
    tags: List[str]            # e.g., ["[IMPORTANT]"], or empty
    is_distractor: bool        # ground truth — NOT shown to agent
    is_correction_of: Optional[int] = None  # if correcting earlier fact_id
    timestep: int              # ingestion order

@dataclass
class Query:
    query_id: int
    text: str
    expected_answer: str       # may be "UNKNOWN" for distractor-resistance queries
    query_type: str            # "specific" | "aggregation" | "contradiction" | "rationale" | "negative_recall" | "distractor_resistance"
    relevant_fact_ids: List[int]   # which fact(s) contain the answer; empty if UNKNOWN

@dataclass
class GroundTruth:
    queries: List[Query]
    fact_to_query_map: Dict[int, List[int]]   # fact_id -> [query_ids that depend on this fact]

class DataGenerator:
    def generate(self, config: LevelConfig, rng: np.random.Generator) -> Tuple[List[Fact], List[Query], GroundTruth]:
        """Return facts in ingestion order, queries in query order, and ground truth bundle."""
        raise NotImplementedError("Awaiting data design — see 08_DATA_GENERATION.md")

    def generate_prefill(self, config: LevelConfig, rng: np.random.Generator) -> List[Tuple[str, str]]:
        """Return (anchor, content) pairs to seed memory at episode start."""
        raise NotImplementedError("Awaiting data design — see 08_DATA_GENERATION.md")

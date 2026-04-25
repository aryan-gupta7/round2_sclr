from typing import Dict, Any
try:
    from ..models import RecallAction, RecallState
except ImportError:
    from models import RecallAction, RecallState
from .data_generator import LevelConfig

def compute_step_reward(
    action: RecallAction,
    action_was_valid: bool,
    new_correct: int,                     # how many correct answers this step (0 or 1 typically)
    new_unknown_correct: int,
    storage_attempts_during_step: int,
    storage_attempts_rejected: int,
    config: LevelConfig,
) -> float:
    r = 0.0
    if not action_was_valid:
        # Get penalty from config if present, default to -0.5
        penalty = config.reward_shaping.get("malformed_step_penalty", -0.5)
        r += penalty
        return r
    
    # Correct answers get +1.0 (constant across levels)
    r += new_correct * 1.0
    r += new_unknown_correct * 1.0
    
    # Budget overflow penalty
    if storage_attempts_rejected > 0:
        penalty = config.reward_shaping.get("budget_overflow_penalty", -0.2)
        r += storage_attempts_rejected * penalty
        
    return r

def compute_terminal_reward(
    state: RecallState,
    config: LevelConfig,
) -> float:
    """Episode-end reward: storage cost + shaping bonuses."""
    r = 0.0
    
    # Storage cost
    cost_per_fact = config.reward_shaping.get("per_fact_storage_cost", -0.05)
    r += state.memory_used * cost_per_fact
    
    # store_then_retrieved_bonus
    # This requires counting how many stored facts were later retrieved AND used in correct answer.
    # Note: RecallState.storage_decisions should track this.
    bonus_retrieved = config.reward_shaping.get("store_then_retrieved_bonus", 0.0)
    if bonus_retrieved > 0:
        count = 0
        for decision in state.storage_decisions:
            if decision.get("decision") == "store" and decision.get("was_later_retrieved_and_correct", False):
                count += 1
        r += count * bonus_retrieved
        
    # skip_then_never_queried_bonus
    bonus_skipped = config.reward_shaping.get("skip_then_never_queried_bonus", 0.0)
    if bonus_skipped > 0:
        count = 0
        for decision in state.storage_decisions:
            if decision.get("decision") == "skip" and not decision.get("was_queried", False):
                count += 1
        r += count * bonus_skipped
        
    return r

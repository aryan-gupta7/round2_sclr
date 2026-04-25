from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class EpisodeResult:
    correct_answers: int
    stored_then_retrieved_count: int
    memory_used: int
    malformed_count: int
    budget_overflow_count: int
    queries_total: int

def phase1_reward(episode_result: EpisodeResult, config: Any) -> float:
    """
    Phase 1 — Bootstrap (training_step < bootstrap_steps)
    Dense shaping ONLY at L1 and L2.
    """
    r = 0.0
    # Per-query correctness — primary signal even in bootstrap
    r += episode_result.correct_answers * 1.0
    
    # Mild shaping during bootstrap for L1/L2
    if config.difficulty <= 2:
        r += episode_result.stored_then_retrieved_count * 0.1
        r -= episode_result.memory_used * 0.02
        
    # Sharp penalty for malformed actions — must dominate
    r += episode_result.malformed_count * (-0.5)
    r += episode_result.budget_overflow_count * (-0.2)
    return r

def phase2_reward(episode_result: EpisodeResult, baseline_correct: int, config: Any) -> float:
    """
    Phase 2 — Binary baseline-comparison (training_step >= bootstrap_steps)
    """
    # Edge case: if agent never answered anything correctly, no reward
    if episode_result.correct_answers == 0:
        return 0.0
    # Edge case: malformed-action spam → still penalize
    if episode_result.malformed_count >= 3:
        return -1.0
        
    # Primary binary signal: did agent beat FIFO baseline accuracy?
    agent_acc = episode_result.correct_answers / episode_result.queries_total
    baseline_acc = baseline_correct / episode_result.queries_total
    
    if agent_acc > baseline_acc + 0.05:    # 5pp margin → clean win
        return 1.0
    elif agent_acc > baseline_acc:          # narrow win → partial
        return 0.3
    else:
        return 0.0

def compute_reward(
    episode_result: EpisodeResult,
    baseline_correct: int,
    config: Any,
    global_step: int,
) -> float:
    """
    Two-phase reward — bootstrap dense, then binary baseline-comparison.
    """
    bootstrap_steps = getattr(config, "bootstrap_steps", 0)
    if global_step < bootstrap_steps:
        return phase1_reward(episode_result, config)
    return phase2_reward(episode_result, baseline_correct, config)

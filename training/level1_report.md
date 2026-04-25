# RECALL Environment — Level 1 Final Report

## Agent Configuration

- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Adapter**: LoRA (4-bit, unsloth, r=16)
- **Algorithm**: GRPO (TRL)
- **Group Size**: 8
- **Hub ID**: `s1nn3rx69/recall-policy-l1`

## Final Training Metrics

- **Mean Reward**: _PENDING JOB COMPLETION_
- **Beat Baseline Rate**: _PENDING JOB COMPLETION_
- **Compute Cost**: ~$1.50 (T4 Medium for ~1.5 hours)

## Evaluation Results (Held-out seeds 0-19)

| Metric               | Trained Agent | FIFO Baseline |
| -------------------- | ------------- | ------------- |
| L1 Accuracy          | _PENDING_     | _PENDING_     |
| Distractor avoidance | _PENDING_     | _PENDING_     |

_Note: You can run `python training/eval.py --env-url https://s1nn3rx69-recall-env.hf.space` to immediately generate the exact evaluation comparison once the LoRA weights finish pushing to your hub!_

---

### Sample Trajectory (Qualitative)

_(Expected comparison after model convergence)_:

**User Query**: `What was the test accuracy for the 2048 expert sparse model experiment?`

**FIFO Baseline**:
_Fills memory with the first 8 facts, discarding the most recent facts._
**Retrieval**: Miss.
**Answer**: "UNKNOWN". Reward: 0.0

**Trained Agent**:
_Skips distractors (e.g. lab meeting notes) and reserves memory for [IMPORTANT] tagged metrics._
**Retrieval**: `[IMPORTANT] 2048 expert MoE achieved 92.4% test accuracy`
**Answer**: `{"action": "answer", "answer": "The test accuracy was 92.4%"}`
**Reward**: 1.0 (Beat Baseline)

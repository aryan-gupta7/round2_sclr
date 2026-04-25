# Evaluation & Metrics

> **Scope**: What to measure during training and at final eval. What plots become deliverables.
> **Implementation lives in**: `training/eval.py`, `training/plot_results.py`.

## Two kinds of evaluation

### Continuous eval (during training)

Light-weight, runs every N steps. Measures:
- Episode reward (mean over 5 fresh seeds at current level)
- Accuracy (mean over 5 fresh seeds)
- Regression on previously-trained levels (run 3 seeds at L1 and L2 even when training L3)

This catches catastrophic forgetting.

### Final eval (after each level completes)

Heavyweight, runs once. Measures:
- Accuracy on **20 fresh held-out seeds** (not seen during training)
- Per-query-type accuracy breakdown
- Failure mode attribution counts
- A few full episode transcripts saved as artifacts

The 20-seed eval is what goes in the README plots and what is compared against baselines.

## Live monitoring during training (W&B / wandb log)

Every training step, log:

| Metric | Why we track it |
|--------|----------------|
| `train/episode_reward_mean` | Primary learning signal |
| `train/episode_reward_std` | If too high → unstable; if zero → no exploration |
| `train/accuracy` | Fraction of queries answered correctly this episode |
| `train/memory_utilization` | Mean (used / budget) — should stabilize, not stay at 100% |
| `train/malformed_action_rate` | Should DROP rapidly. If stays high, action format is broken |
| `train/skip_rate` | Fraction of facts skipped — should rise as agent learns selectivity |
| `train/avg_anchor_length` | Sanity check on anchor authoring |
| `train/component_correct_answer` | Decomposed reward — how much from real answers |
| `train/component_storage_cost` | Decomposed reward — how much spent on storing |
| `train/component_shaping_bonus` | Decomposed reward — how much from bonuses |

Per eval pass, log:

| Metric | Why we track it |
|--------|----------------|
| `eval/accuracy_current_level` | Most important number |
| `eval/accuracy_L1`, `eval/accuracy_L2`, ... | Regression check |
| `eval/accuracy_by_query_type` | Diagnoses which skill is weakest |
| `eval/failure_mode_breakdown` | Which step is failing — storage / anchor / retrieve / reasoning |
| `eval/sample_trajectory` | Saved transcript for inspection |

## What "training is working" looks like

In the first 50–100 steps at L1:
- `train/malformed_action_rate` drops from ~50% to <5%
- `train/episode_reward_mean` climbs from near-zero to plausible positive
- `train/memory_utilization` settles below 100%

In the next 100–200 steps at L1:
- `eval/accuracy_current_level` reaches ≥90%
- `train/skip_rate` is small (L1 has no real skip pressure)

When advancing to L2:
- L1 accuracy drops 5–10% temporarily — this is fine
- L2 accuracy starts ~50% (random + tag heuristic baseline)
- After 200 steps, L2 accuracy should reach ≥70%

If at any point accuracy plateaus or decreases for 100+ consecutive steps, **stop**. Diagnose:
1. Is reward signal informative? (check component breakdown)
2. Is action format being learned? (check malformed rate)
3. Is the curriculum step too big? (split into sub-levels with intermediate values)

## What "training is failing" looks like — and what to try

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Malformed rate stays >20% | Prompt is unclear or JSON parsing too strict | Loosen parser, simplify prompt |
| Reward stays flat at zero | Episodes are terminating too fast (likely all malformed) | Same as above |
| Reward grows but accuracy doesn't | Agent is gaming a shaping bonus | Audit shaping; possibly disable a bonus |
| L1 succeeds, L2 fails immediately | Curriculum jump too big | Add an L1.5 with intermediate values |
| Memory utilization stuck at 100% | Agent never learned to skip | Increase `per_fact_storage_cost` |
| Memory utilization stuck at 0% | Agent over-skipping | Decrease `per_fact_storage_cost` or increase `correct_answer` reward |
| Accuracy on tag-bearing facts low | Agent ignoring tags | Make tags more salient in prompt instructions |

## Required plots (committed to `plots/` as PNGs)

Per the hackathon judging criteria, plots must be:
- Saved as committed PNGs (not Wandb-only links)
- Readable: labeled axes, units, captions
- Trained vs baseline on the same axes when relevant

### Plot 1: `training_curve.png`

X: training step, Y: reward / accuracy. One curve per level (color-coded). Vertical lines marking level transitions. Caption explains the curriculum.

### Plot 2: `eval_accuracy_comparison.png`

Grouped bar chart. X: level (L1, L2, L3). Y: accuracy. Bars: store-all, FIFO, LLM-judge, trained policy. Trained must visually dominate.

### Plot 3: `eval_reward_comparison.png`

Same structure as Plot 2 but Y: mean episode reward.

### Plot 4: `failure_modes.png`

Stacked bar showing what fraction of failures came from each mode (storage / anchor / retrieve / reasoning). One stack per (level, baseline). Shows that trained policy reduces specific failure modes (anchor especially).

### Plot 5: `memory_utilization_distribution.png`

Histogram of memory utilization at episode end across eval seeds. Trained policy should have a non-trivial distribution (not all at 100%, not all at 0%) — proves selectivity learned.

### Plot 6: `qualitative_example.png` or markdown

A side-by-side textual example: same fact stream, FIFO answers wrong, trained policy answers right. This is the README hero panel.

## What `plot_results.py` does

```python
def main(args):
    baseline_results = json.load(open("plots/baselines.json"))
    trained_results = json.load(open("plots/trained.json"))
    plot_training_curves(args.wandb_run_path, "plots/training_curve.png")
    plot_accuracy_comparison(baseline_results, trained_results, "plots/eval_accuracy_comparison.png")
    plot_reward_comparison(baseline_results, trained_results, "plots/eval_reward_comparison.png")
    plot_failure_modes(baseline_results, trained_results, "plots/failure_modes.png")
    plot_memory_utilization(trained_results, "plots/memory_utilization_distribution.png")
    generate_qualitative_example(args.eval_traces, "plots/qualitative_example.md")
```

All plots use a consistent style. Suggested: matplotlib with `seaborn-v0_8-darkgrid` style, font 11pt, `dpi=150`, transparent background optional.

## Eval seeds policy

- **Training seeds**: drawn from `range(1000, 100000)` per training step (effectively unbounded fresh seeds)
- **Held-out eval seeds**: `range(0, 20)` — these are NEVER used during training
- This avoids any train-eval contamination

## Sample trajectory artifacts

For each level, save 3 sample episode transcripts as `plots/sample_trajectory_L<n>_seed<s>.md`. Each contains:
- The full fact stream the agent saw
- The agent's per-fact decisions
- The retrieve queries it issued
- The answers it gave
- The ground truth
- Per-query failure attribution

These are gold for the demo — find the most cinematic one and quote it in the pitch.

## Final eval spec (what gets reported in README)

- 20 held-out seeds per level
- All 4 conditions (store_all, FIFO, LLM_judge, trained_policy) on the SAME 20 seeds
- Report: mean accuracy ± std, mean reward ± std, per-query-type accuracy
- Save raw numbers to `plots/final_results.json`

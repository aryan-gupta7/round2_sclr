# Baselines

> **Scope**: Three baselines + an evaluation harness that runs them all on identical seeds.
> **Implementation lives in**: `baselines/store_all.py`, `baselines/fifo.py`, `baselines/llm_judge.py`, `baselines/run_baselines.py`.

## Why baselines matter

20% of the hackathon score is "showing improvement in rewards". The ONLY way to demonstrate improvement is by comparison. Three baselines with reproducible numbers on the same eval seeds are the proof.

The trained policy must visibly beat all three. Specifically:
- Beat **store-everything** by surviving budget pressure
- Beat **FIFO** by storing important non-recent facts
- Beat **LLM-as-judge** by writing better anchors and learning task-specific importance

## Baseline 1: store-everything

The dumbest possible baseline. Stores every fact verbatim with the fact text as anchor. When budget is exhausted, all subsequent stores are rejected.

```python
class StoreAllBaseline:
    def act(self, observation):
        if observation.phase == "ingest":
            return RecallAction(
                mode="ingest",
                decisions=[
                    FactDecision(fact_id=f["fact_id"], decision="store", anchor=f["text"][:64])
                    for f in observation.current_batch
                ],
            )
        elif observation.phase == "query":
            if observation.retrieval_results is None:
                return RecallAction(mode="retrieve", query=observation.current_query)
            else:
                # Naive: concatenate top-1 result and use it as the answer
                if observation.retrieval_results:
                    return RecallAction(mode="answer", answer_text=observation.retrieval_results[0]["content"])
                else:
                    return RecallAction(mode="answer", answer_text="UNKNOWN")
```

**Expected behavior**: hits the budget cap on every level except L1. Most stores succeed early, late stores get rejected. Naive top-1 answering achieves low accuracy.

## Baseline 2: FIFO

Stores everything until full, then deletes oldest before storing new.

```python
class FIFOBaseline:
    def act(self, observation):
        if observation.phase == "ingest":
            decisions = []
            for f in observation.current_batch:
                if observation.memory_used >= observation.memory_budget:
                    # Need to delete oldest first; we'll emit a delete action separately
                    # Easiest impl: track an internal queue, emit deletes interleaved
                    ...
                decisions.append(FactDecision(fact_id=f["fact_id"], decision="store", anchor=f["text"][:64]))
            return RecallAction(mode="ingest", decisions=decisions)
        # ... query phase same as StoreAll
```

**Implementation note**: FIFO's "delete oldest" requires interleaving delete actions. Two ways to handle:
1. Emit a single `delete` action before each `ingest` step when budget is tight.
2. Internal queue tracking, emit deletes when needed.

Choose option 1 for simplicity. The env's `step()` accepts `mode="delete"` mid-episode.

**Expected behavior**: ~38% accuracy at L2, drops further at L3+ because important non-recent facts are evicted.

## Baseline 3: LLM-as-judge importance

This is the most important baseline. It is the **direct prior art** (Generative Agents 2023). The trained policy must beat this to demonstrate the value of RL.

```python
class LLMJudgeBaseline:
    def __init__(self, llm_client, threshold=0.5):
        self.llm = llm_client
        self.threshold = threshold

    def score_importance(self, fact_text):
        prompt = f"On a scale of 0 to 1, how important is this fact to remember for future questions? Respond with ONLY a number.\n\nFact: {fact_text}\nScore:"
        response = self.llm.generate(prompt, max_tokens=8)
        try:
            return float(response.strip())
        except ValueError:
            return 0.5

    def act(self, observation):
        if observation.phase == "ingest":
            decisions = []
            for f in observation.current_batch:
                score = self.score_importance(f["text"])
                if score >= self.threshold:
                    decisions.append(FactDecision(
                        fact_id=f["fact_id"], decision="store",
                        anchor=f["text"][:64]   # naive anchor — same as text
                    ))
                else:
                    decisions.append(FactDecision(fact_id=f["fact_id"], decision="skip"))
            return RecallAction(mode="ingest", decisions=decisions)
        # ... query phase: writes a query (naive: query = current_query verbatim)
```

**Critical detail**: this baseline uses the SAME base LLM as the trained policy (Qwen2.5-3B-Instruct). It demonstrates what an untrained LLM achieves by prompting alone. This is the apples-to-apples comparison.

**Naive anchor**: this baseline uses fact text as the anchor (no learned anchor authoring). This is the gap the trained policy must exploit.

**Expected behavior**: ~51% at L2/L3, plateaus at L4+ because it cannot adapt to the task-specific importance distribution from feedback.

## The eval harness — `run_baselines.py`

```python
def run_baseline(baseline, env, seeds, difficulty):
    rewards = []
    accuracies = []
    failure_attributions = []
    for seed in seeds:
        obs = env.reset(difficulty=difficulty, seed=seed)
        episode_reward = 0.0
        while not is_terminal(obs):
            action = baseline.act(obs)
            obs = env.step(action)
            episode_reward += obs.last_reward
        rewards.append(episode_reward)
        accuracies.append(env.state.correct_answers / env.state.queries_total)
        failure_attributions.append(env.state.failure_attribution)
    return {
        "mean_reward": np.mean(rewards),
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "failure_modes": aggregate_failure_modes(failure_attributions),
    }


if __name__ == "__main__":
    SEEDS = list(range(20))                # 20 episodes per condition
    LEVELS = [1, 2, 3]                     # L4, L5 added if time
    BASELINES = {
        "store_all": StoreAllBaseline(),
        "fifo": FIFOBaseline(),
        "llm_judge": LLMJudgeBaseline(qwen_client),
    }
    results = {}
    for level in LEVELS:
        for name, baseline in BASELINES.items():
            results[(level, name)] = run_baseline(baseline, env, SEEDS, level)
    save_results_table(results, "plots/baselines.json")
    plot_baseline_bar_chart(results, "plots/baselines.png")
```

## Required outputs from baselines

`baselines/run_baselines.py` produces:

1. `plots/baselines.json` — raw numbers per (level, baseline) cell
2. `plots/baselines_accuracy_bar.png` — grouped bar chart, x-axis = level, y-axis = accuracy, one bar per baseline
3. `plots/baselines_reward_bar.png` — same but for mean episode reward
4. `plots/failure_modes.png` — stacked bar showing what fraction of failures came from each failure mode (storage_failure, anchor_failure, retrieval_failure, reasoning_failure)

These plots become the "before training" panel in the README. After training, regenerate with the trained policy added as a fourth bar.

## What the trained policy must achieve to "win"

Quantitative bar (from `00_PROJECT_OVERVIEW.md`, restated):

| Level | Best baseline (target) | Trained policy (target) |
|-------|---------------------------|----------------------------|
| L1 | LLM-judge: ~80% | ≥95% |
| L2 | LLM-judge: ~55% | ≥75% |
| L3 | LLM-judge: ~50% | ≥70% |

If trained policy < best baseline at any level we trained on, the project's central claim fails. Investigate before continuing curriculum.

## Tests

`tests/test_baselines.py`:

1. Each baseline produces valid `RecallAction` for every observation
2. StoreAll fills budget then rejects further stores at L2+
3. FIFO never exceeds budget (always emits delete before exceeding)
4. LLMJudge respects threshold (low-importance facts skipped)
5. All baselines complete an L1 episode without crash

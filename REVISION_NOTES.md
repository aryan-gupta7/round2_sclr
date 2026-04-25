# REVISION_NOTES — 2026-04-25

> **Context**: Mid-project revision driven by deep research findings on OpenEnv and GRPO training regimes.
> **Files in this folder**: drop-in replacements for the originals. Coding agents should overwrite the prior versions in `/docs/`.

## What changed

| File | Change | Reason |
|------|--------|--------|
| `03_ENVIRONMENT_SPEC.md` | **MAJOR REWRITE** — collapsed to single-pass ingestion, removed delete action | GRPO turn-count regimes: prior 43 turns at L3 was past the 7-turn danger zone |
| `06_REWARD_DESIGN.md` | **MAJOR REWRITE** — two-phase reward (bootstrap dense → binary baseline-comparison), removed dense shaping bonuses | TRL Wordle/Sudoku results + GRPO PRM paper: dense shaping flattens group variance and weakens gradients |
| `08_DATA_GENERATION.md` | **NEW CONTENT** — was placeholder, now full spec | Domain locked, data design needed before training can start |
| `09_TRAINING_PIPELINE.md` | **MODERATE UPDATE** — adapted to new turn structure, two-phase reward, max_concurrent_envs increased | Cascade from env + reward changes |
| `13_HAIKU_PROMPTS.md` | **NEW FILE** — strict prompts for vocabulary generation | Required for data generator implementation |

## Files NOT changed

The following remain valid as previously written:
- `00_PROJECT_OVERVIEW.md` — pitch and positioning unchanged
- `01_REPOSITORY_STRUCTURE.md` — folder layout unchanged
- `02_AGENT_RULES.md` — rules unchanged
- `04_CURRICULUM.md` — 5-level structure unchanged; per-level YAML schemas need a `bootstrap_steps` field added (see `06_REWARD_DESIGN.md` revised)
- `05_MEMORY_BACKEND.md` — memory backend implementation unchanged
- `07_BASELINES.md` — three baselines unchanged; FIFO baseline now also runs at env reset for reward comparison (see `09_TRAINING_PIPELINE.md` revised)
- `10_EVALUATION.md` — metrics list extended (added `reward_std_within_group`); structure unchanged
- `11_DEPLOYMENT.md` — deployment process unchanged
- `12_OPEN_QUESTIONS.md` — Q1, Q2, Q3 resolved by revision; Q5, Q6, Q7 partially resolved

## Critical findings from deep research that drove these changes

1. **GRPO turn-count regimes**: 1-3 turns optimal, 3-7 turns workable, 7+ turns crashes. Prior design was 43 turns. Revised to 5-8.

2. **Binary > dense shaping for GRPO**: empirically and theoretically. Group-relative advantage requires within-group ranking variance, which dense shaping destroys.

3. **Per-session state isolation**: `num_generations=8` means 8 simultaneous WebSocket connections. Prior spec already had this right but `max_concurrent_envs` was set to 1. Updated to 8.

4. **Tool errors as free supervision**: not a structural change for us, but informs the design choice to allow optional retrieve action (lets agent self-correct via context when retrieval is weak).

## Action items for teammates / coding agents

In this order:

1. **Replace** the five revised files (`03_*`, `06_*`, `08_*`, `09_*`) in `/docs/` with the versions in this folder. Add `13_HAIKU_PROMPTS.md` (new).

2. **Update curriculum YAML files** (`training/configs/level_*.yaml`) to add `bootstrap_steps` field per the values in `06_REWARD_DESIGN.md` revised.

3. **Update `openenv.yaml`** to set `max_concurrent_envs: 8`.

4. **Move 12_OPEN_QUESTIONS.md** Q1, Q2, Q3 to the Resolved section with these decisions:
   - Q1: Solo PhD student doing transformer experiments (3-week project)
   - Q2: Templates with Haiku-generated vocabularies (approach a, hybrid)
   - Q3: Lexical mismatch = abbreviation/expansion (locked) + specific-to-categorical (locked)

5. **Run Haiku prompts** from `13_HAIKU_PROMPTS.md` to generate the 8 vocabulary files. Run `validate_vocab.py` after each. Commit to `envs/recall_env/server/vocab/`.

6. **Update CHANGELOGs** in any code folders touched by these revisions.

## What to flag if something looks off

If after these revisions, training smoke runs show:
- `reward_std_within_group` near zero → bootstrap not providing enough variance; extend bootstrap or simplify L1
- Malformed action rate stays >20% after 50 steps → JSON parser too strict OR prompt is unclear
- L1 succeeds but L2 fails to launch → curriculum jump too big; insert L1.5 with intermediate values

These are tunable. The structural revisions in this batch are the ones that needed urgent fixing.

# Agent Rules

> **Audience**: Coding agents (Cursor, Claude Code, etc.) and human teammates working alongside them.
> Read this before writing or editing any file.

## Hard rules

### R1. Read before writing
Before editing any file in a folder, **read the folder's `CHANGELOG.md`**. Before working on any subsystem, **read the corresponding doc in `/docs/`**. Do not infer scope from the code alone.

### R2. Update CHANGELOG.md after every change
Every code-bearing folder has a `CHANGELOG.md`. Append to it after each non-trivial change. Format:

```markdown
## YYYY-MM-DD — <agent or person>

### Changed
- `<file>`: <one-line description of what changed and why>

### Added
- `<file>`: <new file purpose>

### Notes
- <any gotchas, deferred items, or open questions raised>
```

Trivial = typo fixes, formatting. Non-trivial = anything affecting behavior or interface.

### R3. Do not invent scope
If a task is not in `00_PROJECT_OVERVIEW.md` "Locked" or in a numbered doc, **ask before adding**. Specifically these are out of scope for MVP:

- Graph relationships between memory items
- Multi-tier (hot/cold) memory
- Compression as a learned action
- Automatic clustering / summarization
- Streaming / async memory updates

If you find yourself wanting these, write the proposal in `12_OPEN_QUESTIONS.md` instead of implementing.

### R4. Defer to data design
Anything depending on the dataset structure, fact format, or query distribution is **deferred** until `08_DATA_GENERATION.md` is filled in by Suryansh. Stub these with `NotImplementedError("awaiting data spec — see 08_DATA_GENERATION.md")` rather than guessing.

### R5. OpenEnv compliance is non-negotiable
Read `01_REPOSITORY_STRUCTURE.md` before touching anything inside `envs/recall_env/`. The OpenEnv folder skeleton, manifest schema, dual-import pattern, and `create_app(...)` factory pattern are mandated. Breaking these breaks HF Spaces deployment.

### R6. One concern per file
Do not put memory backend logic in `recall_environment.py`. Do not put reward logic in `models.py`. Each file has a job listed in `01_REPOSITORY_STRUCTURE.md`. Respect it.

### R7. Tests with code
When you add functionality to `recall_environment.py`, `memory_backend.py`, `rewards.py`, or any baseline, add a corresponding test in `tests/`. Tests should run in <5 seconds each. They are smoke tests, not exhaustive.

### R8. No hidden state in module globals
Environments must keep state inside the `RecallEnvironment` instance. No module-level mutable state. OpenEnv's container model assumes per-session isolation.

### R9. No silent fallbacks
If a config is missing, **raise**. Do not default silently. Curriculum level must be explicit. Embedding model must be explicit. This prevents subtle training failures where the agent appears to work but is actually running the wrong setup.

### R10. Reproducibility
Every episode must be reproducible from `(difficulty, seed)`. The data generator, pre-filled memory, and query order all derive from these two values. No `random.random()` without a seeded `np.random.Generator`.

## Communication conventions

### Commit messages
Format: `<area>: <verb> <object>`

Examples:
- `env: implement step() ingestion phase`
- `rewards: add per-fact storage cost`
- `baselines: add FIFO baseline`
- `training: add Level 1 GRPO config`
- `docs: clarify anchor authoring spec`

### When unsure
Default to writing the question into `12_OPEN_QUESTIONS.md` and continuing on a different unblocked task. Do not block on Suryansh for things that can be parameterized.

### When you find a bug in a doc
Update the doc and note it in the doc folder's CHANGELOG (or in the relevant code folder's CHANGELOG if the doc-fix unblocked code).

## File-creation discipline

Do not create new top-level folders. Do not create config files outside `training/configs/`. Do not create new entry points outside `training/grpo_train.py`, `training/eval.py`, `baselines/run_baselines.py`. If you think you need one of these, raise it in `12_OPEN_QUESTIONS.md` first.

## What "done" means for any task

A task is done when:
1. Code is written and runs without errors
2. A smoke test exists and passes
3. The folder's `CHANGELOG.md` is updated
4. If the change affects an interface, the corresponding `/docs/` file is updated
5. Imports work both in-repo and in Docker (dual-import pattern where applicable)

## Anti-patterns to avoid

- **"I'll add tests later"** — no. Add a smoke test alongside the code.
- **"This config feels right"** — no. Either it's in a doc or you ask.
- **"I'll abstract this in case we need it"** — no. YAGNI. We have ~5 days.
- **"I'll add a fallback default"** — no. Raise on missing config (R9).
- **"Let me refactor this while I'm here"** — no. Refactors are separate commits and require a CHANGELOG entry explaining why.

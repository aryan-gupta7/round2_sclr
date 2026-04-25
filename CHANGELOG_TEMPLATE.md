# Changelog Template

> Copy this into every code-bearing folder as `CHANGELOG.md`.
> Do not delete entries. Append new ones at the top.

## How to use this file

Every non-trivial change to code in this folder gets an entry. Trivial = typos, formatting, comment edits. Non-trivial = anything that changes behavior or interface.

### Format

```markdown
## YYYY-MM-DD — <agent name or person>

### Changed
- `<file>`: <one-line description>

### Added
- `<file>`: <new file purpose>

### Removed
- `<file>`: <why removed>

### Notes
- <gotchas, deferred items, open questions raised>
```

### Why this matters

In a 5-day hackathon with multiple coding agents and human contributors, a single canonical log per folder is the only way to know what shipped, when, and why. PR descriptions get lost; Slack messages disappear. CHANGELOG.md doesn't.

### Bad entries (avoid)

```markdown
## 2026-04-25
Fixed stuff and made some changes.
```

This is useless. Anyone reading it later cannot figure out what happened.

### Good entries

```markdown
## 2026-04-25 — Claude Code

### Changed
- `recall_environment.py`: switched action validation to fail fast on malformed JSON instead of attempting recovery; affects malformed_step_penalty timing
- `models.py`: tightened `RecallAction.decisions` to require non-empty list when mode == "ingest"

### Added
- `tests/test_action_validation.py`: smoke tests for malformed action paths

### Notes
- Q8 in `12_OPEN_QUESTIONS.md` is now resolved as "strict parsing" — moved to Resolved.
- `parse_action()` in `training/grpo_train.py` will need to match this stricter validation; flagging in training/CHANGELOG.md too.
```

This tells a future reader exactly what shifted.

---

## (Existing entries below)

*(folder maintainer: append entries here as work happens)*

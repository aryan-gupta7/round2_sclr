# Changelog - Recall Environment

All notable changes to the Recall Environment will be documented in this file.

## 2026-04-25 — Antigravity (Vocab Integration)

### Added

- `envs/recall_env/server/vocab/`: Copied 8 Haiku-generated vocabulary files from `json_database/`.
- `envs/recall_env/server/data_generator.py`: **Full implementation** of template-based fact + query generation using vocabularies. All 7 fact categories (experiment, decision, paper, hypothesis, debug, correction, distractor) and 6 query types (specific, aggregation, rationale, negative, distractor_resistance, contradiction) implemented.

### Changed

- `envs/recall_env/server/data_generator.py`: No longer a stub. Now produces deterministic episodes from `(config, seed)` pairs with lexical mismatch built in.

## 2026-04-25 — Antigravity (REVISION)

### Changed

- `envs/recall_env/models.py`: Removed `delete` action, added `all_facts` to observation, added `baseline_correct` to state.
- `envs/recall_env/server/recall_env_environment.py`: Migrated to single-pass ingestion, added FIFO baseline simulation at reset, and integrated two-phase rewards.
- `envs/recall_env/server/rewards.py`: Implemented Phase 1 (Bootstrap) and Phase 2 (Binary comparison) reward system.
- `envs/recall_env/server/app.py`: Updated to use factory function and increased `max_concurrent_envs` to 8.
- `training/configs/level_*.yaml`: Added `bootstrap_steps` and adjusted fact/query counts for revised spec.
- `12_OPEN_QUESTIONS.md`: Resolved Q1, Q2, and Q3 based on the revised data generation spec.

## 2026-04-25 — Antigravity (Initial)

### Added

- `envs/recall_env/CHANGELOG.md`: Initial changelog.
- `envs/recall_env/server/CHANGELOG.md`: Initial server changelog.
- `envs/recall_env/models.py`: Pydantic models for action, observation, state.
- `envs/recall_env/server/memory_backend.py`: Vector store with random projection.
- `envs/recall_env/server/rewards.py`: Initial reward logic.
- `envs/recall_env/server/recall_env_environment.py`: Core environment.
- `envs/recall_env/server/app.py`: FastAPI app.
- `envs/recall_env/client.py`: Client.
- `training/configs/level_1.yaml`: L1 config.
- `tests/test_environment.py`: Smoke tests.

### Changed

- `envs/recall_env/pyproject.toml`: Added dependencies.
- `envs/recall_env/README.md`: Updated with RECALL details.
- `envs/recall_env/openenv.yaml`: Updated metadata.

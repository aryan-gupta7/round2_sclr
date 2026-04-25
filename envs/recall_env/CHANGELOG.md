# Changelog - Recall Environment

All notable changes to the Recall Environment will be documented in this file.

## 2026-04-25 — Antigravity

### Added

- `envs/recall_env/CHANGELOG.md`: Initial changelog for the environment folder.
- `envs/recall_env/server/CHANGELOG.md`: Initial changelog for the server folder.
- `envs/recall_env/models.py`: Implemented `FactDecision`, `RecallAction`, `RecallObservation`, and `RecallState`.
- `envs/recall_env/server/memory_backend.py`: Implemented `MemoryItem` and `MemoryBackend` with vector retrieval and random projection.
- `envs/recall_env/server/data_generator.py`: Implemented data generator interfaces with stubs as per 08_DATA_GENERATION.md.
- `envs/recall_env/server/rewards.py`: Implemented reward computation logic according to 06_REWARD_DESIGN.md.
- `envs/recall_env/server/recall_env_environment.py`: Implemented core environment logic for ingestion and query phases.
- `envs/recall_env/server/app.py`: Configured FastAPI app with `create_app` factory.
- `envs/recall_env/client.py`: Implemented `RecallEnv` client.
- `training/configs/level_1.yaml`: Initial configuration for curriculum training.
- `tests/test_environment.py`: Smoke tests for the environment.

### Changed

- `envs/recall_env/pyproject.toml`: Added necessary dependencies (`sentence-transformers`, `numpy`, `pyyaml`).
- `envs/recall_env/README.md`: Updated with RECALL-specific project details.
- `envs/recall_env/openenv.yaml`: Updated metadata and description.
- `envs/recall_env/server/requirements.txt`: Updated with project-specific dependencies.

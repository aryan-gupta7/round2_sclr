# Changelog - Recall Environment Server

All notable changes to the Recall Environment Server will be documented in this file.

## 2026-04-25 — Antigravity

### Added

- `envs/recall_env/server/memory_backend.py`: Vector store with random projection and mock fallback.
- `envs/recall_env/server/data_generator.py`: Stubs and interfaces for curriculum-aware data generation.
- `envs/recall_env/server/rewards.py`: Reward computation engine (step + terminal).
- `envs/recall_env/server/recall_env_environment.py`: Batch ingestion and sequential query phase management.
- `envs/recall_env/server/app.py`: FastAPI server setup with OpenEnv factory.
- `envs/recall_env/server/requirements.txt`: Pinned dependencies for runtime.

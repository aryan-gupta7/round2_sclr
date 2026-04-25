# RECALL Deployment Guide — HuggingFace Spaces

> **Complete step-by-step instructions** for deploying the RECALL environment to HuggingFace Spaces and running OpenEnv validation.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Verification](#2-local-verification)
3. [HuggingFace Setup](#3-huggingface-setup)
4. [Deployment via `openenv push`](#4-deployment-via-openenv-push)
5. [Manual Deployment (Alternative)](#5-manual-deployment-alternative)
6. [Post-Deployment Verification](#6-post-deployment-verification)
7. [Gradio Interface](#7-gradio-interface)
8. [Troubleshooting](#8-troubleshooting)
9. [Submission Checklist](#9-submission-checklist)

---

## 1. Prerequisites

### Tools

```bash
# HuggingFace CLI
pip install huggingface_hub[cli]
huggingface-cli login   # Paste your HF write token

# OpenEnv CLI (should already be installed)
pip install openenv-core[core]>=0.2.2

# Docker (optional, for local Docker testing)
docker --version  # Verify Docker is available
```

### Verify environment structure

```bash
ls envs/recall_env/
# Expected:
# __init__.py  client.py  gradio_app.py  models.py  openenv.yaml
# pyproject.toml  README.md  server/  uv.lock
```

---

## 2. Local Verification

### Step 2.1 — Import check

```bash
cd /path/to/sclr_round2
PYTHONPATH=src:envs python -c "
from envs.recall_env.server.recall_env_environment import RecallEnvironment
from envs.recall_env.models import RecallAction, RecallObservation, RecallState
print('✅ Import check passed')
print(f'  SUPPORTS_CONCURRENT_SESSIONS: {RecallEnvironment.SUPPORTS_CONCURRENT_SESSIONS}')
"
```

### Step 2.2 — Data generator test

```bash
cd envs/recall_env
python -m server.data_generator --difficulty 1 --seed 42 --print
# Should output 10 facts and 3 queries
```

### Step 2.3 — Server health check

```bash
# Terminal 1: Start server
cd /path/to/sclr_round2
PYTHONPATH=src:envs python -m uvicorn envs.recall_env.server.app:app --port 8000

# Terminal 2: Test health
curl -s http://localhost:8000/health
# Expected: {"status":"healthy"}

# Test schema endpoint
curl -s http://localhost:8000/schema | python -m json.tool | head -20

# Test metadata endpoint
curl -s http://localhost:8000/metadata | python -m json.tool
```

### Step 2.4 — OpenEnv validate (if available)

```bash
# With server running:
openenv validate --url http://localhost:8000

# OR build and validate:
cd envs/recall_env
openenv build
openenv validate --verbose
```

### Step 2.5 — Gradio interface test

```bash
cd envs/recall_env
python gradio_app.py
# Open http://localhost:7860 in browser
# Click "Generate Episode" → should render facts, queries, memory, rewards
```

### Step 2.6 — Docker build test (optional but recommended)

```bash
cd envs/recall_env
docker build -t recall-env -f server/Dockerfile .
docker run -p 8000:8000 recall-env
curl http://localhost:8000/health
```

---

## 3. HuggingFace Setup

### Step 3.1 — Login

```bash
huggingface-cli login
# Enter your HF write token (from https://huggingface.co/settings/tokens)
```

### Step 3.2 — Create the Space

```bash
# Option A: via CLI
huggingface-cli repo create recall-env --type space --space-sdk docker

# Option B: via web
# Go to https://huggingface.co/new-space
# Name: recall-env
# SDK: Docker
# Hardware: CPU Basic (Free)
```

---

## 4. Deployment via `openenv push`

This is the preferred method if the OpenEnv CLI supports it:

```bash
cd envs/recall_env

# Validate first
openenv validate

# Push to HuggingFace
openenv push --repo-id <your-username>/recall-env --enable-interface
```

The `--enable-interface` flag adds a Gradio web UI that judges can interact with.

---

## 5. Manual Deployment (Alternative)

If `openenv push` doesn't work or you prefer manual control:

### Step 5.1 — Clone the Space repo

```bash
git clone https://huggingface.co/spaces/<your-username>/recall-env
cd recall-env
```

### Step 5.2 — Copy environment files

```bash
# From the project root
cp -r envs/recall_env/* /path/to/recall-env/

# Also copy training configs (needed at runtime)
mkdir -p /path/to/recall-env/training/configs
cp training/configs/level_*.yaml /path/to/recall-env/training/configs/
```

### Step 5.3 — Prepare the Dockerfile

The Dockerfile at `server/Dockerfile` is already HF-compatible. If HF Spaces requires Dockerfile at the root:

```bash
cd /path/to/recall-env
cp server/Dockerfile ./Dockerfile
```

Or for a simpler setup that works directly on HF:

```dockerfile
FROM python:3.11-slim

WORKDIR /app/env

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy all env files
COPY . /app/env/

# Install Python deps
RUN pip install --no-cache-dir \
    openenv-core[core]>=0.2.2 \
    sentence-transformers \
    numpy \
    pyyaml \
    gradio>=4.0.0

# Pre-download model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

ENV PYTHONPATH="/app/env:$PYTHONPATH"

EXPOSE 8000 7860

# Run both FastAPI and Gradio (or just FastAPI)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 5.4 — Commit and push

```bash
cd /path/to/recall-env
git add .
git commit -m "Deploy RECALL environment"
git push
```

### Step 5.5 — Monitor build

- Go to `https://huggingface.co/spaces/<your-username>/recall-env`
- Click **"Logs"** tab to monitor the Docker build
- Wait for "Running on..." message (may take 5-10 minutes on first build)

---

## 6. Post-Deployment Verification

### Step 6.1 — Health check

```bash
curl -s https://<your-username>-recall-env.hf.space/health
# Expected: {"status":"healthy"}
```

### Step 6.2 — Functional test

```python
import asyncio
from envs.recall_env import RecallEnv, RecallAction
from envs.recall_env.models import FactDecision

async def main():
    async with RecallEnv(base_url="https://<your-username>-recall-env.hf.space") as env:
        obs = await env.reset(difficulty=1, seed=0)
        print(f"✅ Reset OK: phase={obs.phase}, facts={len(obs.all_facts)}")

        # Simple ingest: store everything
        decisions = [
            FactDecision(fact_id=f["fact_id"], decision="store", anchor=f["text"][:30])
            for f in obs.all_facts
        ]
        result = await env.step(RecallAction(mode="ingest", decisions=decisions))
        print(f"✅ Ingest OK: phase={result.observation.phase}")

        # Query phase
        obs = result.observation
        if obs.phase == "query":
            result = await env.step(RecallAction(mode="retrieve", query=obs.current_query))
            print(f"✅ Retrieve OK: {len(result.observation.retrieval_results or [])} results")

            result = await env.step(RecallAction(mode="answer", answer_text="UNKNOWN"))
            print(f"✅ Answer OK: reward={result.observation.last_reward}")

asyncio.run(main())
```

### Step 6.3 — OpenEnv validate against deployed Space

```bash
openenv validate --url https://<your-username>-recall-env.hf.space
```

---

## 7. Gradio Interface

The interactive Gradio interface is at `gradio_app.py`. It provides:

- **Episode Generation**: Select difficulty (L1-L3) and seed to generate episodes
- **Facts Stream**: Rich HTML table with category badges, importance tags, distractor markers
- **Query Viewer**: Cards for each query with type badges, expected answers, relevant fact IDs
- **Memory State**: Visual slot grid showing used vs available memory
- **Reward Engine**: Interactive simulator comparing agent accuracy vs FIFO baseline

### Running locally

```bash
cd envs/recall_env
python gradio_app.py
# Open http://localhost:7860
```

### Deploying alongside FastAPI

If you want both the API and Gradio on the same Space, mount the Gradio app:

```python
# In server/app.py, after creating the FastAPI app:
import gradio as gr
from gradio_app import demo

app = gr.mount_gradio_app(app, demo, path="/web")
```

---

## 8. Troubleshooting

### Import errors in Docker

- **Symptom**: `ModuleNotFoundError: No module named 'models'`
- **Fix**: Ensure `PYTHONPATH="/app/env"` is set in Docker and the dual-import pattern is used in all server files.

### Sentence-transformers download fails

- **Symptom**: First request hangs or crashes
- **Fix**: Pre-download in Dockerfile: `RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

### Config file not found

- **Symptom**: `FileNotFoundError: Curriculum config missing`
- **Fix**: Ensure `training/configs/level_*.yaml` files are copied into the HF Space repo.

### HF Space build timeout

- **Symptom**: Build exceeds 15-minute limit
- **Fix**: Use a smaller base image or pre-build dependencies in a separate layer.

### openenv validate fails

- **Symptom**: `Connection refused` or endpoint errors
- **Fix**: Ensure the server is running on port 8000 and responds to `/health`, `/reset`, `/step`, `/state`, `/metadata`, `/schema`.

---

## 9. Submission Checklist

- [ ] `openenv validate` passes locally
- [ ] All tests pass (`PYTHONPATH=src:envs python -m pytest tests/ -q`)
- [ ] `envs/recall_env/README.md` has HF frontmatter (`title`, `emoji`, `sdk: docker`)
- [ ] `envs/recall_env/openenv.yaml` has full metadata
- [ ] Docker build succeeds locally
- [ ] HF Space URL works and accepts `reset()` / `step()`
- [ ] Post-deployment smoke test passes
- [ ] Top-level `README.md` has:
  - [ ] Pitch paragraph
  - [ ] Link to deployed HF Space
  - [ ] Link to Colab training notebook
  - [ ] Embedded plots (or links)
  - [ ] Link to mini-blog / video / slides
- [ ] `plots/` has all PNGs committed
- [ ] Colab notebook runs end-to-end
- [ ] Video/blog is published and linked
- [ ] Final commit hash is submitted (no post-deadline pushes)

---

## Quick Reference Commands

```bash
# Import check
PYTHONPATH=src:envs python -c "from envs.recall_env.server.recall_env_environment import RecallEnvironment; print('OK')"

# Start server
PYTHONPATH=src:envs uvicorn envs.recall_env.server.app:app --port 8000

# Health check
curl http://localhost:8000/health

# Generate test episode
cd envs/recall_env && python -m server.data_generator --difficulty 2 --seed 0 --print

# Start Gradio UI
cd envs/recall_env && python gradio_app.py

# HF login + push
huggingface-cli login
cd envs/recall_env && openenv push --repo-id <username>/recall-env --enable-interface
```

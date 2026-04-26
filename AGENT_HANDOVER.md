# AGENT HANDOVER: RECALL Curriculum Training Maintenance & Monitoring

Welcome! You are taking over a pipeline handling a complex multi-stage curriculum training via Hugging Face Jobs and Spaces.

This document provides a complete crash course on the current state, how to monitor progress, the quirks of our infrastructure, and how to successfully intervene if something breaks.

---

## 1. Project Context & Infrastructure

We are training an agent (using `unsloth/Qwen2.5-7B-Instruct` 4-bit LoRA with GRPO) to selectively ingest and recall fact combinations across increasingly difficult sequences using a custom OpenEnv standard environment.

- **Environment Space (WebSockets/HTTP)**: Hosted at `https://s1nn3rx69-recall-env.hf.space`
- **GPU Compute Jobs**: Running on Hugging Face Jobs (`a10g-small` hardware format).

### Core Training File

The master orchestration file is `training/train_curriculum.py`. It is responsible for sequentially triggering environments L1 through L5.

For each level, it loads the respective weights from the Hub (ex: `s1nn3rx69/recall-policy-L(X-1)`), generates the specific difficulty prompts dynamically from the running Space environment, and pushes the newly adapted weights to the next level's Hub repository.

---

## 2. Viewing Current Training Progress

The single master job (`A10G-Small`) is continuously executing levels 1 through 5.

- **To check the current running job**:
  ```bash
  hf jobs ps --namespace s1nn3rx69
  ```
- **To stream the logs (replace with Job ID from ps)**:
  ```bash
  hf jobs logs <JOB_ID> 2>&1 | tail -100
  ```

In the logs, you will see periodic outputs containing:
`[STEP X] SIDE-BY-SIDE EVAL | Agent Acc: 42.0% vs FIFO Baseline Acc: 20.0%`
This metric is critical. It evaluates our dynamic GRPO batch against a dry-run FIFO Queue internal to the environment. The agent's accuracy relies entirely on structurally learning facts better than raw FIFO retention. The training loop ends a level when it exceeds the steps bounds.

---

## 3. Submitting/Restarting Jobs (If things break)

**NEVER use standard `hf jobs run` directly, and NEVER attempt standard `pip` without `uv`.**

If you need to start or restart the process, use `training/submit_unified_training.py`.

```bash
python training/submit_unified_training.py
```

This executes `run_job` via the Python SDK.

### Why this specific script over CLI?

1. **Base64 injection**: It sends the latest `train_curriculum.py` dynamically injected directly into a Python command list via `echo | base64 -d`. This avoids uploading via spaces which imposes harsh 3-hour rate limits per space API.
2. **`uv pip` override**: It installs dependencies via `uv pip install --python /opt/venv/bin/python`. This matches the specific `venv` expected inside the base `unsloth/unsloth` Docker image format.
3. **Specific Dependencies**:
   - `fastmcp`: The agent will otherwise silence-fail because `openenv-core` requires it transitively.
   - `huggingface-hub<1.0`: Crucial. If skipped, `pip` updates standard Transformers to conflicting bounds causing immediate process crashes.

---

## 4. Hardware and Framework Quirk Cheat Sheet (Do not Regress!)

If you modify `train_curriculum.py`, obey these parameters forcefully or it will cause obscure backend errors:

1. **Precision Binding**: `bf16=True` and `fp16=False` is set in `GRPOConfig`. Qwen2.5-7B native matrices are `bfloat16`! If you set `fp16=True`, Unsloth will crash inside the `new_init` intercept complaining about precision incompatibility. The hardware allocated is `A10G-Small` (Ampere class) which explicitly enables natively supporting `bf16`.
2. **VLLM v1 Engine Graph Clash**: `os.environ["VLLM_USE_V1"] = "0"` and `os.environ["TORCH_COMPILE_DISABLE"] = "1"`. Leave these alone. VLLM implicitly attempts compiling Deep Learning nodes across Dynamo and fails repeatedly when resolving `Node size_1 but it still had 1 users...` on Unsloth kernel patches.
3. **`.strip()` Safety Guards**: Any agent generation response to the environment needs to check for `None` before parsing inside `memory_backend.py`. Valid JSON LLM outputs format missing anchors as `None`, crashing `.strip()` natively inside Python, which will stall WebSockets silently causing training generation nodes to hang infinitely inside PyTorch ThreadPoolExecutors.

---

## 5. Your Next Steps

1. Check the statuses of `hf jobs`. Verify which Level is currently running.
2. Monitor the `[STEP X] SIDE-BY-SIDE EVAL` progression metrics.
3. At the completion of L4 and L5, the final repositories will be pushed! Determine from there if any additional downstream metric visualizations or fine-tuning runs are necessary based on the agent's baseline comparative metrics.

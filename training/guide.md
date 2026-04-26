Here's the complete runbook extracted from the chat — everything you need to rerun training clean and fast:

---

## Environment Setup

**HF Space (recall-env)**

- URL: `wss://s1nn3rx69-recall-env.hf.space` (also `https://` for HTTP calls)
- Push updates with: `cd envs/recall_env && openenv push --repo-id s1nn3rx69/recall-env --interface`
- `max_concurrent_envs` must be set to **64** in both `openenv.yaml` and `app.py` to handle GRPO group size of 8 without state collision

**Critical bug that was fixed — never regress this:**
`memory_backend.py` line ~62: when the LLM generates invalid JSON, `anchor` parses as `None`. The `.strip()` call on `None` crashed the Space silently, causing training to hang indefinitely. The fix is a None/empty-string guard before calling `.strip()`. Always verify this is in place before pushing.

---

## Job Submission — The Working Pattern

The final working command template (use this, not the `hf jobs run` CLI which has shifting flags):

```python
from huggingface_hub import run_job
import base64

token = open('/home/s1nn3r/.cache/huggingface/token').read().strip()
script_content = open('/path/to/train_curriculum.py', 'rb').read()
script_b64 = base64.b64encode(script_content).decode('ascii')

command = [
    'bash', '-c',
    f"echo '{script_b64}' | base64 -d > train_curriculum.py && "
    f"uv pip install --python /opt/venv/bin/python 'huggingface-hub<1.0' openenv-core fastmcp "
    f"git+https://huggingface.co/spaces/s1nn3rx69/recall-env && "
    f"python train_curriculum.py --env-url https://s1nn3rx69-recall-env.hf.space"
]

job = run_job(
    image='unsloth/unsloth:latest',
    command=command,
    secrets={'HF_TOKEN': token},
    flavor='a10g-small',      # NOT t4-medium — see CUDA error section below
    namespace='s1nn3rx69'
)
```

**Why base64 embed?** HF rate-limits `openenv push` for repo updates; embedding the script directly sidesteps that entirely.

---

## Model & LoRA Config

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (upgraded from 3B mid-session — 7B is the final confirmed working model; 3B was the original plan for T4)
- **Unsloth 4-bit loading:** `unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit`
- **LoRA:** `r=16, alpha=32`, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, dropout `0.05`
- **Dropout warning:** Unsloth prints `"Dropout = 0 is supported for fast patching. You are using dropout = 0.05"` — this is expected; it just means LoRA layers won't get the fastest kernel path, but it still works

---

## GRPOConfig — Critical Settings

```python
GRPOConfig(
    output_dir="./outputs/recall_l{N}",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_generations=8,           # GRPO group size
    max_prompt_length=4096,
    max_completion_length=2048,
    warmup_steps=10,
    logging_steps=1,
    save_steps=50,
    bf16=False,                  # T4 does NOT support bf16; a10g does
    fp16=True,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,
    report_to="none",            # NEVER trackio — version conflict (see below)
    push_to_hub=True,
    hub_model_id="s1nn3rx69/recall-policy-l{N}",
    remove_unused_columns=False,
)
```

---

## Dependency Install — Exact Order Matters

```bash
uv pip install --python /opt/venv/bin/python \
  'huggingface-hub<1.0' \
  openenv-core \
  fastmcp \
  git+https://huggingface.co/spaces/s1nn3rx69/recall-env
```

**Why `huggingface-hub<1.0`?** The Unsloth image ships an older version of `transformers` that requires it. Installing without the pin downgrades packages and breaks things.

**Why `fastmcp` explicitly?** `openenv-core` imports it transitively but doesn't install it by default — `ModuleNotFoundError: No module named 'fastmcp'` killed two jobs before this was caught.

**Why NOT `--no-deps`?** Using `--no-deps` skips `fastmcp` and other transitive deps. Never use it here.

---

## Errors Encountered + Root Causes + Fixes

| Error                                                                         | Root Cause                                                                                                                                           | Fix                                                                                                                 |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `ImportError: cannot import name 'Volume' from 'huggingface_hub'`             | `trackio` needs a newer `huggingface_hub` than the Unsloth base image has                                                                            | Set `report_to="none"` — metrics go to stdout                                                                       |
| `HfHubHTTPError: 401 Unauthorized` at trainer init                            | `push_to_hub=True` but `HF_TOKEN` not passed as a secret to the job                                                                                  | Pass `secrets={'HF_TOKEN': token}` in `run_job()`                                                                   |
| `ModuleNotFoundError: No module named 'fastmcp'`                              | `openenv-core` needs it but it's not auto-installed                                                                                                  | Add `fastmcp` explicitly to pip install                                                                             |
| `torch.AcceleratorError: CUDA error: illegal memory access` in `llm.sleep()`  | vLLM sleep/wake cycle is incompatible with T4 (compute capability < 8, no FA2 support)                                                               | **Switch hardware to `a10g-small`**; also add `enforce_eager=True` or disable vLLM sleep                            |
| `AttributeError: 'LoraModel' object has no attribute 'load_lora'` at L2 start | Curriculum index math bug: after removing L1 from schedule, `LEVEL_SCHEDULE[target_level - 1]` pointed to wrong level, loaded a non-existent adapter | Loop through `LEVEL_SCHEDULE` matching `args.target_level` explicitly, no index arithmetic                          |
| `NoneType has no attribute 'strip'` causing Space hang                        | LLM invalid JSON → `anchor=None` → crash in `memory_backend.py`                                                                                      | Guard: check `anchor is None or anchor == ""` before `.strip()`                                                     |
| Silent training freeze (no exception in logs)                                 | HF Space WebSocket dropped; `env.step()` blocked indefinitely                                                                                        | Wrap `_simulate_episode()` in `concurrent.futures.ThreadPoolExecutor` with `timeout=60.0`; return `-1.0` on timeout |

---

## Curriculum Logic

- **L1:** 250 steps, base `Qwen2.5-7B-Instruct`, pushes to `s1nn3rx69/recall-policy-l1`
- **L2:** 200 steps, loads `recall-policy-l1` as adapter base, pushes to `recall-policy-l2`
- **L3:** 250 steps, loads `recall-policy-l2`, pushes to `recall-policy-l3`

The `train_curriculum.py` orchestrator spawns each level as a subprocess with `--target-level N`. The level-matching must be done by **explicit loop match**, not array index math.

**Auto-resume:** If `./outputs/recall_l{N}/` contains checkpoints, pass `resume_from_checkpoint=True` to `trainer.train()`. It restores optimizer state, GRPO buffer, and LR schedule exactly.

---

## Two-Phase Reward Function

```python
bootstrap_steps = 100

if global_step < bootstrap_steps:
    # Phase 1: Dense shaping (L1/L2) — reward per correct answer, penalize memory use
    r = float(correct) - 0.02 * env_reward.get("memory_used", 0)
else:
    # Phase 2: Binary comparison vs FIFO baseline
    agent_acc = correct / 5        # L1 has 5 queries
    baseline_acc = baseline / 5
    if agent_acc > baseline_acc + 0.05:
        r = 1.0
    elif agent_acc > baseline_acc:
        r = 0.3
    else:
        r = 0.0
```

The FIFO baseline is **pre-computed during env reset** (dry-run) so comparison is always available.

---

## Speed Optimization Notes

- `vllm_gpu_memory_utilization=0.3` — leaves room for training; don't increase on a10g-small
- `gradient_accumulation_steps=8` with `batch_size=1` → effective batch of 8, matches `num_generations=8`
- **Hardware:** `a10g-small` is the minimum that avoids the CUDA illegal memory access. T4 fails due to compute capability < 8 (no FlashAttention 2, vLLM sleep mode incompatible)
- Eval callback fires every 25 steps; watch for `reward_std_within_group ≈ 0` — if all 8 completions get identical reward for 50+ steps, GRPO gradient is zero and you need to stop and diagnose diversity of outputs
- Metrics are stdout-only (`report_to="none"`); watch the HF job logs directly at `https://huggingface.co/jobs/s1nn3rx69/{job_id}`
- Use uv pip instead of pip for faster dependency installation

---

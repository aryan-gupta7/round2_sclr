from huggingface_hub import run_job
import base64
import os

token_path = '/home/s1nn3r/.cache/huggingface/token'
if not os.path.exists(token_path):
    raise Exception("Hugging Face token not found at ~/.cache/huggingface/token. Run `hf auth login` first.")

token = open(token_path).read().strip()
script_content = open('/home/s1nn3r/Documents/sclr_round2/training/train_curriculum.py', 'rb').read()
script_b64 = base64.b64encode(script_content).decode('ascii')

command = [
    'bash', '-c',
    f"echo '{script_b64}' | base64 -d > train_curriculum.py && "
    f"uv pip install --python /opt/venv/bin/python 'huggingface-hub<1.0' openenv-core fastmcp vllm "
    f"git+https://huggingface.co/spaces/s1nn3rx69/recall-env && "
    f"python train_curriculum.py --env-url https://s1nn3rx69-recall-env.hf.space"
]

print("Submitting training job...")
job = run_job(
    image='unsloth/unsloth:latest',
    command=command,
    secrets={'HF_TOKEN': token},
    flavor='a10g-small',
    namespace='s1nn3rx69'
)

print(f"Job launched successfully!")
print(f"Job ID: {job.uid}")
print(f"Watch logs at: https://huggingface.co/jobs/s1nn3rx69/{job.uid}")

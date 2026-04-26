import ast
import matplotlib.pyplot as plt
import re

import sys

if len(sys.argv) < 2:
    print("Usage: python plot_metrics.py <log_file>")
    sys.exit(1)

log_file = sys.argv[1]

metrics = []
eval_steps = []
eval_rewards = []

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith("{'loss':"):
            try:
                metrics.append(ast.literal_eval(line))
            except Exception as e:
                print(f"Failed to parse line: {line.strip()[:50]}... Error: {e}")
        elif "EVAL @ step" in line:
            m = re.search(r'EVAL @ step (\d+): reward_mean=([-\d.]+)', line)
            if m:
                eval_steps.append(int(m.group(1)))
                eval_rewards.append(float(m.group(2)))

epochs = [m.get('epoch', i) for i, m in enumerate(metrics)]
losses = [m.get('loss', 0) for m in metrics]
rewards = [m.get('reward', 0) for m in metrics]
kls = [m.get('kl', 0) for m in metrics]
completions_mean_length = [m.get('completions/mean_length', 0) for m in metrics]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(epochs, losses, label='Loss', color='red')
axs[0, 0].set_title('Loss over Epochs')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].grid(True)

axs[0, 1].plot(epochs, rewards, label='Train Reward', color='blue', alpha=0.6)
if eval_steps and eval_rewards:
    # Convert step to epoch if possible, here we assume 1 step = 1 item in metrics approximately or just plot by step if needed.
    # For simplicity, if we have 100 elements, epoch is likely step / 100. Let's just plot train reward.
    pass
axs[0, 1].set_title('Reward over Epochs')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Reward')
axs[0, 1].grid(True)

axs[1, 0].plot(epochs, kls, label='KL Divergence', color='green')
axs[1, 0].set_title('KL Divergence over Epochs')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('KL')
axs[1, 0].grid(True)

axs[1, 1].plot(epochs, completions_mean_length, label='Mean Completion Length', color='purple')
axs[1, 1].set_title('Mean Completion Length over Epochs')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Length')
axs[1, 1].grid(True)

plt.tight_layout()
import os
base_name = os.path.basename(log_file).replace('.txt', '')
output_path = f'/home/s1nn3r/Documents/sclr_round2/training/{base_name}_metrics_plot.png'
plt.savefig(output_path)
print(f"Plot saved successfully to {output_path}")

import sys
import re
import ast
import matplotlib.pyplot as plt

def parse_file(file_path):
    rewards = []
    losses = []
    steps = []
    kls = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("{'loss':"):
                try:
                    data = ast.literal_eval(line.strip())
                    rewards.append(data.get('reward', 0))
                    losses.append(data.get('loss', 0))
                    kls.append(data.get('kl', 0))
                    steps.append(len(steps) + 1)
                except Exception:
                    pass
    return steps, rewards, losses, kls

metrics = {}
for path in ['training/first_l2.txt', 'training/first_l3.txt']:
    metrics[path] = parse_file(f'/home/s1nn3r/Documents/sclr_round2/{path}')

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
for k, v in metrics.items():
    plt.plot(v[0], v[1], label=f'{k} Reward')
plt.ylabel('Mean Reward')
plt.legend()
plt.title('Training Metrics')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
for k, v in metrics.items():
    plt.plot(v[0], v[2], label=f'{k} Loss')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
for k, v in metrics.items():
    plt.plot(v[0], v[3], label=f'{k} KL Divergence')
plt.xlabel('Step')
plt.ylabel('KL')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/s1nn3r/Documents/sclr_round2/training/l2_l3_metrics.png')
print("Saved plots to /home/s1nn3r/Documents/sclr_round2/training/l2_l3_metrics.png")

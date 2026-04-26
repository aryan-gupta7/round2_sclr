"""Analyze L1 training logs from file — outputs compact summary only."""
import re, json, sys
import numpy as np

LOG_FILE = "training/full_l1_logs.txt"

rewards = []
reward_stds = []
frac_zero_std = []
kls = []
losses = []
eval_lines = []

with open(LOG_FILE) as f:
    for line in f:
        line = line.strip()
        
        # Capture HELD-OUT and EVAL lines verbatim
        if "HELD-OUT" in line or "BATCH STATS" in line or "EVAL @" in line or "L1 DONE" in line or "Reward error" in line:
            eval_lines.append(line)
            continue
        
        # Parse JSON-like log lines
        if "'rewards/recall_reward/mean'" in line:
            try:
                # Extract key metrics with regex
                r_mean = float(re.search(r"'rewards/recall_reward/mean': ([0-9e.\-]+)", line).group(1))
                r_std = float(re.search(r"'rewards/recall_reward/std': ([0-9e.\-]+)", line).group(1))
                fzs = float(re.search(r"'frac_reward_zero_std': ([0-9e.\-]+)", line).group(1))
                kl = float(re.search(r"'kl': ([0-9e.\-]+)", line).group(1))
                loss = float(re.search(r"'loss': ([0-9e.\-]+)", line).group(1))
                
                rewards.append(r_mean)
                reward_stds.append(r_std)
                frac_zero_std.append(fzs)
                kls.append(kl)
                losses.append(loss)
            except:
                pass

total_steps = len(rewards)
print(f"=" * 60)
print(f"  L1 TRAINING ANALYSIS — {total_steps} steps parsed")
print(f"=" * 60)

if total_steps == 0:
    print("No data found!")
    sys.exit(1)

# Bucketed summary
bucket_size = 25
print(f"\n{'Bucket':<12} {'Mean Reward':>12} {'Std(reward)':>12} {'KL':>10} {'ZeroStd%':>10} {'Loss':>10}")
print("-" * 68)
for i in range(0, total_steps, bucket_size):
    end = min(i + bucket_size, total_steps)
    r_slice = rewards[i:end]
    s_slice = reward_stds[i:end]
    k_slice = kls[i:end]
    z_slice = frac_zero_std[i:end]
    l_slice = losses[i:end]
    
    bucket_label = f"Steps {i}-{end-1}"
    print(f"{bucket_label:<12} {np.mean(r_slice):>12.4f} {np.mean(s_slice):>12.4f} {np.mean(k_slice):>10.5f} {np.mean(z_slice)*100:>9.1f}% {np.mean(l_slice):>10.5f}")

# Overall stats
print(f"\n{'='*60}")
print(f"  OVERALL STATISTICS")
print(f"{'='*60}")
print(f"  Total steps:         {total_steps}")
print(f"  Overall mean reward: {np.mean(rewards):.4f}")
print(f"  Overall std reward:  {np.std(rewards):.4f}")
print(f"  Min reward:          {np.min(rewards):.4f}")
print(f"  Max reward:          {np.max(rewards):.4f}")
print(f"  Median reward:       {np.median(rewards):.4f}")
print(f"  % steps with +reward:{sum(1 for r in rewards if r > 0)/total_steps*100:.1f}%")
print(f"  % zero-std steps:    {sum(1 for z in frac_zero_std if z > 0.5)/total_steps*100:.1f}%")
print(f"  Final 10-step avg:   {np.mean(rewards[-10:]):.4f}")

# Reward distribution
print(f"\n  REWARD DISTRIBUTION:")
bins = [(-1.1,-0.8), (-0.8,-0.5), (-0.5,-0.2), (-0.2,0.0), (0.0,0.3), (0.3,0.6), (0.6,1.0)]
for lo, hi in bins:
    count = sum(1 for r in rewards if lo < r <= hi)
    bar = "█" * int(count / total_steps * 50)
    print(f"  ({lo:+.1f},{hi:+.1f}]: {count:>4} ({count/total_steps*100:5.1f}%) {bar}")

# Trend line (simple linear regression)
x = np.arange(total_steps)
slope, intercept = np.polyfit(x, rewards, 1)
print(f"\n  LINEAR TREND: slope={slope:.5f}/step, intercept={intercept:.4f}")
print(f"  Projected step 250: {slope*250 + intercept:.4f}")

# EVAL and error lines
print(f"\n{'='*60}")
print(f"  EVAL CHECKPOINTS & ERRORS")
print(f"{'='*60}")
error_count = 0
for line in eval_lines:
    if "Reward error" in line:
        error_count += 1
    else:
        print(f"  {line}")
print(f"  Total 'Reward error' occurrences: {error_count}")

# Check if checkpoint was saved
print(f"\n{'='*60}")
print(f"  CHECKPOINT STATUS")
print(f"{'='*60}")
for line in eval_lines:
    if "DONE" in line or "Pushing" in line or "pushed" in line:
        print(f"  {line}")
if total_steps < 250:
    print(f"  Training still in progress ({total_steps}/250 steps)")
else:
    print(f"  Training complete!")

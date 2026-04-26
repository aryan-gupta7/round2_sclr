#!/usr/bin/env python3
"""
Comprehensive L1 Training Metrics Plotter
Parses l1_final.txt and generates:
  1. Standard 2x2 training metrics (Loss, Reward, KL, Completion Length)
  2. Agent vs Baseline comparison plots (Batch Accuracy, Moving Avg Accuracy)
  3. Eval reward over steps
"""

import ast
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

log_file = '/home/s1nn3r/Documents/sclr_round2/training/l1_final.txt'

# ── Parse training metrics ──────────────────────────────────────────────────
metrics = []
eval_steps = []
eval_rewards = []

# Baseline vs Agent from BATCH STATS
batch_steps = []
batch_agent_acc = []
batch_baseline_acc = []

# Moving average from MOVING AVG lines
mavg_steps = []
mavg_agent_acc = []
mavg_baseline_acc = []

# Held-out eval (FIFO baseline)
heldout_steps = []
heldout_fifo_acc = []

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        
        # Training step metrics
        if line.startswith("{'loss':"):
            try:
                metrics.append(ast.literal_eval(line))
            except Exception:
                pass
        
        # EVAL @ step
        elif "EVAL @ step" in line:
            m = re.search(r'EVAL @ step (\d+): reward_mean=([-\d.eE+]+|N/A)', line)
            if m and m.group(2) != 'N/A':
                eval_steps.append(int(m.group(1)))
                eval_rewards.append(float(m.group(2)))
        
        # BATCH STATS
        elif "BATCH STATS" in line:
            m = re.search(r'\[STEP (\d+)\] BATCH STATS \| Agent Acc: ([\d.]+)% vs Baseline: ([\d.]+)%', line)
            if m:
                batch_steps.append(int(m.group(1)))
                batch_agent_acc.append(float(m.group(2)))
                batch_baseline_acc.append(float(m.group(3)))
        
        # MOVING AVG
        elif "MOVING AVG" in line:
            m = re.search(r'\[MOVING AVG LAST \d+ STEPS\] Agent Acc: ([\d.]+)% \| Baseline Acc: ([\d.]+)%', line)
            if m:
                mavg_agent_acc.append(float(m.group(1)))
                mavg_baseline_acc.append(float(m.group(2)))
        
        # HELD-OUT EVAL
        elif "HELD-OUT EVAL" in line:
            m = re.search(r'\[STEP (\d+)\] HELD-OUT EVAL.*FIFO Acc: ([\d.]+)%', line)
            if m:
                heldout_steps.append(int(m.group(1)))
                heldout_fifo_acc.append(float(m.group(2)))

# Align moving avg steps with eval steps (they appear together)
# Trim to the shorter length to handle duplicates
min_len = min(len(eval_steps), len(mavg_agent_acc))
mavg_steps = eval_steps[:min_len]
mavg_agent_acc = mavg_agent_acc[:min_len]
mavg_baseline_acc = mavg_baseline_acc[:min_len]

# ── Extract metric arrays ───────────────────────────────────────────────────
steps = list(range(1, len(metrics) + 1))
losses = [m.get('loss', 0) for m in metrics]
rewards = [m.get('reward', 0) for m in metrics]
kls = [m.get('kl', 0) for m in metrics]
comp_lengths = [m.get('completions/mean_length', 0) for m in metrics]
grad_norms = [m.get('grad_norm', 0) for m in metrics]

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Standard Training Metrics (2x2)
# ═══════════════════════════════════════════════════════════════════════════
fig1, axs = plt.subplots(2, 2, figsize=(14, 9))
fig1.suptitle('L1 Training Metrics — 200 Steps (Qwen2.5-7B GRPO)', 
              fontsize=15, fontweight='bold', y=0.98)

# Loss
axs[0, 0].plot(steps, losses, color='#e74c3c', alpha=0.4, linewidth=0.8)
# Smoothed
window = 10
if len(losses) >= window:
    smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
    axs[0, 0].plot(range(window, len(losses)+1), smoothed_loss, color='#c0392b', linewidth=2, label=f'MA-{window}')
axs[0, 0].set_title('Loss', fontweight='bold')
axs[0, 0].set_xlabel('Step')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Reward
axs[0, 1].plot(steps, rewards, color='#3498db', alpha=0.4, linewidth=0.8)
if len(rewards) >= window:
    smoothed_reward = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axs[0, 1].plot(range(window, len(rewards)+1), smoothed_reward, color='#2980b9', linewidth=2, label=f'MA-{window}')
if eval_steps and eval_rewards:
    axs[0, 1].scatter(eval_steps, eval_rewards, color='#e67e22', s=40, zorder=5, label='Eval Reward', marker='D')
axs[0, 1].set_title('Reward', fontweight='bold')
axs[0, 1].set_xlabel('Step')
axs[0, 1].set_ylabel('Reward')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# KL Divergence
axs[1, 0].plot(steps, kls, color='#2ecc71', alpha=0.5, linewidth=0.8)
if len(kls) >= window:
    smoothed_kl = np.convolve(kls, np.ones(window)/window, mode='valid')
    axs[1, 0].plot(range(window, len(kls)+1), smoothed_kl, color='#27ae60', linewidth=2, label=f'MA-{window}')
axs[1, 0].set_title('KL Divergence', fontweight='bold')
axs[1, 0].set_xlabel('Step')
axs[1, 0].set_ylabel('KL')
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# Completion Length
axs[1, 1].plot(steps, comp_lengths, color='#9b59b6', alpha=0.5, linewidth=0.8)
if len(comp_lengths) >= window:
    smoothed_cl = np.convolve(comp_lengths, np.ones(window)/window, mode='valid')
    axs[1, 1].plot(range(window, len(comp_lengths)+1), smoothed_cl, color='#8e44ad', linewidth=2, label=f'MA-{window}')
axs[1, 1].set_title('Mean Completion Length', fontweight='bold')
axs[1, 1].set_xlabel('Step')
axs[1, 1].set_ylabel('Length (tokens)')
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

fig1.tight_layout(rect=[0, 0, 1, 0.96])
out1 = '/home/s1nn3r/Documents/sclr_round2/training/l1_final_training_metrics.png'
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {out1}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Agent vs Baseline Comparison (3 subplots)
# ═══════════════════════════════════════════════════════════════════════════
fig2, axs2 = plt.subplots(3, 1, figsize=(14, 12))
fig2.suptitle('L1 Agent vs FIFO Baseline — Accuracy Comparison', 
              fontsize=15, fontweight='bold', y=0.98)

# ── Panel 1: Batch Accuracy (Agent vs Baseline) ────────────────────────
if batch_steps:
    x = np.array(batch_steps)
    width = 3
    axs2[0].bar(x - width/2, batch_agent_acc, width, color='#3498db', alpha=0.8, label='Agent (Trained)')
    axs2[0].bar(x + width/2, batch_baseline_acc, width, color='#e74c3c', alpha=0.8, label='FIFO Baseline')
    axs2[0].set_title('Per-Checkpoint Batch Accuracy', fontweight='bold')
    axs2[0].set_xlabel('Training Step')
    axs2[0].set_ylabel('Accuracy (%)')
    axs2[0].set_ylim(0, 110)
    axs2[0].legend(loc='upper right')
    axs2[0].grid(True, alpha=0.3, axis='y')
    axs2[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random')

    # Annotate wins/losses
    for i, (s, ag, bl) in enumerate(zip(batch_steps, batch_agent_acc, batch_baseline_acc)):
        if ag > bl:
            axs2[0].annotate('▲', (s, ag + 2), ha='center', color='green', fontsize=10)
        elif bl > ag:
            axs2[0].annotate('▼', (s, max(ag, bl) + 2), ha='center', color='red', fontsize=10)

# ── Panel 2: Moving Average Accuracy ───────────────────────────────────
if mavg_steps:
    axs2[1].plot(mavg_steps, mavg_agent_acc, 'o-', color='#3498db', linewidth=2, markersize=6, label='Agent MA-10')
    axs2[1].plot(mavg_steps, mavg_baseline_acc, 's-', color='#e74c3c', linewidth=2, markersize=6, label='Baseline MA-10')
    axs2[1].fill_between(mavg_steps, mavg_agent_acc, mavg_baseline_acc,
                          where=[a > b for a, b in zip(mavg_agent_acc, mavg_baseline_acc)],
                          interpolate=True, alpha=0.15, color='green', label='Agent > Baseline')
    axs2[1].fill_between(mavg_steps, mavg_agent_acc, mavg_baseline_acc,
                          where=[a <= b for a, b in zip(mavg_agent_acc, mavg_baseline_acc)],
                          interpolate=True, alpha=0.15, color='red', label='Baseline > Agent')
    axs2[1].axhline(y=47.2, color='orange', linestyle=':', alpha=0.7, linewidth=1.5, label='FIFO Held-out (47.2%)')
    axs2[1].set_title('Moving Average Accuracy (10-Step Window)', fontweight='bold')
    axs2[1].set_xlabel('Training Step')
    axs2[1].set_ylabel('Accuracy (%)')
    axs2[1].set_ylim(0, 75)
    axs2[1].legend(loc='upper right', fontsize=9)
    axs2[1].grid(True, alpha=0.3)

# ── Panel 3: Eval Reward + Summary ────────────────────────────────────
if eval_steps and eval_rewards:
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in eval_rewards]
    axs2[2].bar(eval_steps, eval_rewards, width=6, color=colors, alpha=0.8, edgecolor='white')
    axs2[2].axhline(y=0, color='gray', linewidth=1)
    axs2[2].set_title('Eval Reward per Checkpoint', fontweight='bold')
    axs2[2].set_xlabel('Training Step')
    axs2[2].set_ylabel('Mean Reward')
    axs2[2].grid(True, alpha=0.3, axis='y')
    
    # Add summary stats
    mean_eval = np.mean(eval_rewards)
    max_eval = max(eval_rewards)
    max_step = eval_steps[eval_rewards.index(max_eval)]
    negative_count = sum(1 for r in eval_rewards if r <= 0)
    
    summary_text = (f"Mean Eval Reward: {mean_eval:.3f}\n"
                    f"Peak: {max_eval:.2f} @ step {max_step}\n"
                    f"Negative/Zero: {negative_count}/{len(eval_rewards)} checkpoints")
    axs2[2].text(0.02, 0.95, summary_text, transform=axs2[2].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

fig2.tight_layout(rect=[0, 0, 1, 0.96])
out2 = '/home/s1nn3r/Documents/sclr_round2/training/l1_final_agent_vs_baseline.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {out2}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════
fig3, axs3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('L1 Training Summary — Agent vs FIFO Baseline', 
              fontsize=15, fontweight='bold', y=1.02)

# ── Left: Agent wins vs losses over training ──────────
if batch_steps:
    agent_wins = sum(1 for a, b in zip(batch_agent_acc, batch_baseline_acc) if a > b)
    baseline_wins = sum(1 for a, b in zip(batch_agent_acc, batch_baseline_acc) if b > a)
    ties = sum(1 for a, b in zip(batch_agent_acc, batch_baseline_acc) if a == b)
    
    labels = ['Agent Wins', 'Baseline Wins', 'Ties']
    sizes = [agent_wins, baseline_wins, ties]
    colors_pie = ['#3498db', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0)
    
    wedges, texts, autotexts = axs3[0].pie(sizes, explode=explode, labels=labels, 
                                            colors=colors_pie, autopct='%1.0f%%',
                                            shadow=True, startangle=90, textprops={'fontsize': 12})
    axs3[0].set_title(f'Batch Eval Win Rate\n(out of {len(batch_steps)} checkpoints)', fontweight='bold')

# ── Right: Overall accuracy trends ───────────────────
if mavg_steps:
    mean_agent = np.mean(mavg_agent_acc)
    mean_baseline = np.mean(mavg_baseline_acc)
    
    categories = ['Agent\n(Trained)', 'FIFO\nBaseline', 'Held-out\nFIFO']
    values = [mean_agent, mean_baseline, 47.2]
    bar_colors = ['#3498db', '#e74c3c', '#e67e22']
    
    bars = axs3[1].bar(categories, values, color=bar_colors, alpha=0.85, edgecolor='white', width=0.5)
    axs3[1].set_title('Mean Accuracy Across Training', fontweight='bold')
    axs3[1].set_ylabel('Accuracy (%)')
    axs3[1].set_ylim(0, 70)
    axs3[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        axs3[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

fig3.tight_layout()
out3 = '/home/s1nn3r/Documents/sclr_round2/training/l1_final_summary.png'
fig3.savefig(out3, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {out3}")

# ── Print text summary ──────────────────────────────────────────────────
print("\n" + "="*70)
print("L1 TRAINING ANALYSIS SUMMARY")
print("="*70)
print(f"Total steps: {len(metrics)}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Mean reward (all steps): {np.mean(rewards):.3f}")
print(f"Mean eval reward: {np.mean(eval_rewards):.3f}")
print(f"Peak eval reward: {max(eval_rewards):.3f} @ step {eval_steps[eval_rewards.index(max(eval_rewards))]}")
print(f"KL range: [{min(kls):.6f}, {max(kls):.6f}]")
print()
print("AGENT vs BASELINE (Batch):")
print(f"  Agent wins: {agent_wins}/{len(batch_steps)} ({100*agent_wins/len(batch_steps):.0f}%)")
print(f"  Baseline wins: {baseline_wins}/{len(batch_steps)} ({100*baseline_wins/len(batch_steps):.0f}%)")
print(f"  Ties: {ties}/{len(batch_steps)}")
print()
print("AGENT vs BASELINE (Moving Avg):")
print(f"  Mean Agent Acc: {mean_agent:.1f}%")
print(f"  Mean Baseline Acc: {mean_baseline:.1f}%")
print(f"  FIFO Held-out Acc: 47.2% (constant)")
print(f"  Gap (Agent - Baseline): {mean_agent - mean_baseline:+.1f}%")
print()

# Where agent beats baseline in moving avg
wins_mavg = [(s, a, b) for s, a, b in zip(mavg_steps, mavg_agent_acc, mavg_baseline_acc) if a > b]
print(f"Steps where Agent MA > Baseline MA: {[w[0] for w in wins_mavg]}")
print(f"  → {len(wins_mavg)}/{len(mavg_steps)} ({100*len(wins_mavg)/len(mavg_steps):.0f}%)")
print("="*70)

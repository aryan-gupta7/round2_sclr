"""
RECALL Memory Environment — Interactive Gradio Interface
A visual, interactive demo for the RECALL environment.
Includes episode simulation, memory visualization, and reward tracking.
"""

import os
import sys
import json
import random
import numpy as np
import gradio as gr

# Ensure imports work from env root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from server.data_generator import DataGenerator, LevelConfig
    from server.memory_backend import MemoryBackend
    from server.rewards import compute_reward, EpisodeResult, phase1_reward, phase2_reward
except ImportError:
    from data_generator import DataGenerator, LevelConfig
    from memory_backend import MemoryBackend
    from rewards import compute_reward, EpisodeResult, phase1_reward, phase2_reward


# ---------------------------------------------------------------------------
# Shared state (per-session via Gradio State)
# ---------------------------------------------------------------------------

LEVEL_CONFIGS = {
    1: LevelConfig(difficulty=1, facts_total=10, queries_total=3, memory_budget=8,
                   batch_size=8, retrieval_k=5, embedding_model="test-fallback",
                   explicit_importance_tags=True, query_distribution={"specific": 1.0},
                   bootstrap_steps=100),
    2: LevelConfig(difficulty=2, facts_total=25, queries_total=5, memory_budget=20,
                   batch_size=25, retrieval_k=5, embedding_model="test-fallback",
                   distractor_rate=0.3, explicit_importance_tags=True,
                   query_distribution={"specific": 0.5, "distractor_resistance": 0.5},
                   bootstrap_steps=200),
    3: LevelConfig(difficulty=3, facts_total=50, queries_total=5, memory_budget=25,
                   batch_size=50, retrieval_k=5, embedding_model="test-fallback",
                   distractor_rate=0.4, prefilled_memory_count=15,
                   query_distribution={"specific": 0.5, "aggregation": 0.2, "rationale": 0.1, "distractor_resistance": 0.2},
                   bootstrap_steps=0),
}

gen = DataGenerator()


def generate_episode(difficulty, seed):
    """Generate a complete episode (facts + queries)."""
    config = LEVEL_CONFIGS.get(int(difficulty), LEVEL_CONFIGS[1])
    rng = np.random.default_rng(int(seed))
    facts, queries, gt = gen.generate(config, rng)
    return facts, queries, gt, config


# ---------------------------------------------------------------------------
# Interface helpers
# ---------------------------------------------------------------------------

def facts_to_html(facts):
    """Render facts as a rich HTML table."""
    cat_colors = {
        "experiment": "#2563eb",
        "decision": "#7c3aed",
        "paper": "#059669",
        "hypothesis": "#d97706",
        "debug": "#dc2626",
        "correction": "#f59e0b",
        "distractor": "#6b7280",
    }
    rows = []
    for f in facts:
        color = cat_colors.get(f.category, "#6b7280")
        badge = f'<span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;font-weight:600;">{f.category.upper()}</span>'
        dist_icon = "⚠️" if f.is_distractor else ""
        tags = ", ".join(f.tags) if f.tags else "—"
        rows.append(f"""
        <tr style="border-bottom:1px solid #333;">
            <td style="padding:8px;color:#a0a0a0;font-family:monospace;text-align:center;">{f.fact_id}</td>
            <td style="padding:8px;">{badge}</td>
            <td style="padding:8px;color:#e0e0e0;font-size:0.9em;line-height:1.5;">{f.text}</td>
            <td style="padding:8px;color:#a0a0a0;font-size:0.8em;">{tags}</td>
            <td style="padding:8px;text-align:center;">{dist_icon}</td>
        </tr>""")

    return f"""
    <div style="max-height:500px;overflow-y:auto;border-radius:12px;border:1px solid #333;">
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#1a1a2e;position:sticky;top:0;z-index:1;">
                    <th style="padding:10px;color:#888;font-size:0.8em;">ID</th>
                    <th style="padding:10px;color:#888;font-size:0.8em;">Category</th>
                    <th style="padding:10px;color:#888;font-size:0.8em;text-align:left;">Content</th>
                    <th style="padding:10px;color:#888;font-size:0.8em;">Tags</th>
                    <th style="padding:10px;color:#888;font-size:0.8em;">⚠️</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>"""


def queries_to_html(queries):
    """Render queries as rich HTML cards."""
    type_icons = {
        "specific": "🎯",
        "aggregation": "📊",
        "rationale": "💡",
        "negative": "❌",
        "distractor_resistance": "🛡️",
        "contradiction": "🔄",
    }
    cards = []
    for q in queries:
        icon = type_icons.get(q.query_type, "❓")
        cards.append(f"""
        <div style="background:#1a1a2e;border:1px solid #333;border-radius:12px;padding:16px;margin:8px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-size:1.2em;">{icon} <strong style="color:#e0e0e0;">Query {q.query_id}</strong></span>
                <span style="background:#333;color:#a0a0a0;padding:2px 10px;border-radius:12px;font-size:0.75em;">{q.query_type}</span>
            </div>
            <div style="color:#c4b5fd;font-size:1em;margin:8px 0;">"{q.text}"</div>
            <div style="display:flex;gap:12px;margin-top:8px;">
                <span style="color:#059669;font-size:0.85em;">✅ Expected: <code style="background:#0d3320;padding:2px 6px;border-radius:4px;">{q.expected_answer}</code></span>
                <span style="color:#888;font-size:0.85em;">📎 Facts: {q.relevant_fact_ids if q.relevant_fact_ids else "None (UNKNOWN)"}</span>
            </div>
        </div>""")
    return "".join(cards)


def memory_viz(facts, stored_ids, budget):
    """Render a memory visualization showing stored items vs budget."""
    used = len(stored_ids)
    pct = (used / budget * 100) if budget > 0 else 0
    bar_color = "#059669" if pct < 75 else ("#d97706" if pct < 95 else "#dc2626")

    slots = []
    for i in range(budget):
        if i < used:
            fact = next((f for f in facts if f.fact_id == stored_ids[i]), None)
            cat = fact.category if fact else "unknown"
            cat_colors = {"experiment": "#2563eb", "decision": "#7c3aed", "paper": "#059669",
                         "hypothesis": "#d97706", "debug": "#dc2626", "distractor": "#6b7280"}
            c = cat_colors.get(cat, "#444")
            slots.append(f'<div title="Slot {i}: {cat}" style="width:20px;height:20px;background:{c};border-radius:4px;border:1px solid #555;"></div>')
        else:
            slots.append('<div style="width:20px;height:20px;background:#1a1a2e;border-radius:4px;border:1px dashed #333;"></div>')

    return f"""
    <div style="padding:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span style="color:#e0e0e0;font-weight:600;">Memory Usage</span>
            <span style="color:{bar_color};font-weight:700;font-size:1.1em;">{used}/{budget} slots ({pct:.0f}%)</span>
        </div>
        <div style="background:#111;border-radius:8px;height:12px;overflow:hidden;margin-bottom:12px;">
            <div style="background:{bar_color};height:100%;width:{pct}%;transition:width 0.3s;border-radius:8px;"></div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:4px;">{"".join(slots)}</div>
    </div>"""


def reward_dashboard(correct, total, baseline, phase, step):
    """Show reward computation breakdown."""
    agent_acc = correct / total if total > 0 else 0
    baseline_acc = baseline / total if total > 0 else 0

    result = EpisodeResult(correct_answers=correct, stored_then_retrieved_count=0,
                          memory_used=0, malformed_count=0, budget_overflow_count=0, queries_total=total)

    if phase == "bootstrap":
        reward = phase1_reward(result, type("Cfg", (), {"difficulty": 1})())
        phase_label = "Phase 1 — Bootstrap (Dense)"
        phase_color = "#d97706"
    else:
        reward = phase2_reward(result, baseline, type("Cfg", (), {"difficulty": 3})())
        phase_label = "Phase 2 — Binary Comparison"
        phase_color = "#7c3aed"

    diff = agent_acc - baseline_acc
    diff_color = "#059669" if diff > 0 else "#dc2626"
    badge = "WON ✅" if diff > 0.05 else ("NARROW WIN ⚡" if diff > 0 else "LOST ❌")

    return f"""
    <div style="background:#0f0f23;border:1px solid #333;border-radius:16px;padding:20px;">
        <div style="text-align:center;margin-bottom:16px;">
            <span style="background:{phase_color};color:white;padding:4px 16px;border-radius:20px;font-size:0.85em;font-weight:600;">{phase_label}</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;">
            <div style="text-align:center;background:#1a1a2e;padding:16px;border-radius:12px;">
                <div style="font-size:2em;font-weight:700;color:#2563eb;">{agent_acc:.0%}</div>
                <div style="color:#888;font-size:0.8em;margin-top:4px;">Agent Accuracy</div>
            </div>
            <div style="text-align:center;background:#1a1a2e;padding:16px;border-radius:12px;">
                <div style="font-size:2em;font-weight:700;color:#888;">{baseline_acc:.0%}</div>
                <div style="color:#888;font-size:0.8em;margin-top:4px;">FIFO Baseline</div>
            </div>
            <div style="text-align:center;background:#1a1a2e;padding:16px;border-radius:12px;">
                <div style="font-size:2em;font-weight:700;color:{diff_color};">{diff:+.0%}</div>
                <div style="color:#888;font-size:0.8em;margin-top:4px;">Margin ({badge})</div>
            </div>
        </div>
        <div style="text-align:center;background:#1a1a2e;padding:16px;border-radius:12px;">
            <div style="font-size:2.5em;font-weight:800;color:{'#059669' if reward > 0 else '#dc2626' if reward < 0 else '#888'};">{reward:+.2f}</div>
            <div style="color:#888;font-size:0.85em;margin-top:4px;">Final Reward Signal</div>
        </div>
    </div>"""


def episode_stats_html(facts, queries, config):
    """Summary stats card."""
    n_dist = sum(1 for f in facts if f.is_distractor)
    n_real = len(facts) - n_dist
    cats = {}
    for f in facts:
        cats[f.category] = cats.get(f.category, 0) + 1

    cat_bars = ""
    cat_colors = {"experiment": "#2563eb", "decision": "#7c3aed", "paper": "#059669",
                 "hypothesis": "#d97706", "debug": "#dc2626", "distractor": "#6b7280", "correction": "#f59e0b"}
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        pct = cnt / len(facts) * 100
        c = cat_colors.get(cat, "#444")
        cat_bars += f"""
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
            <span style="width:100px;text-align:right;color:#a0a0a0;font-size:0.8em;">{cat}</span>
            <div style="flex:1;background:#1a1a2e;border-radius:4px;height:16px;overflow:hidden;">
                <div style="background:{c};width:{pct}%;height:100%;border-radius:4px;"></div>
            </div>
            <span style="color:#a0a0a0;font-size:0.8em;width:30px;">{cnt}</span>
        </div>"""

    q_types = {}
    for q in queries:
        q_types[q.query_type] = q_types.get(q.query_type, 0) + 1

    return f"""
    <div style="background:#0f0f23;border:1px solid #333;border-radius:16px;padding:20px;">
        <h3 style="color:#e0e0e0;margin-top:0;">📊 Episode Statistics</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:16px;">
            <div style="text-align:center;background:#1a1a2e;padding:12px;border-radius:8px;">
                <div style="font-size:1.5em;font-weight:700;color:#2563eb;">{len(facts)}</div>
                <div style="color:#888;font-size:0.75em;">Total Facts</div>
            </div>
            <div style="text-align:center;background:#1a1a2e;padding:12px;border-radius:8px;">
                <div style="font-size:1.5em;font-weight:700;color:#059669;">{n_real}</div>
                <div style="color:#888;font-size:0.75em;">Relevant</div>
            </div>
            <div style="text-align:center;background:#1a1a2e;padding:12px;border-radius:8px;">
                <div style="font-size:1.5em;font-weight:700;color:#6b7280;">{n_dist}</div>
                <div style="color:#888;font-size:0.75em;">Distractors</div>
            </div>
            <div style="text-align:center;background:#1a1a2e;padding:12px;border-radius:8px;">
                <div style="font-size:1.5em;font-weight:700;color:#7c3aed;">{len(queries)}</div>
                <div style="color:#888;font-size:0.75em;">Queries</div>
            </div>
        </div>
        <h4 style="color:#a0a0a0;margin-bottom:8px;">Fact Category Distribution</h4>
        {cat_bars}
        <h4 style="color:#a0a0a0;margin-top:16px;margin-bottom:4px;">Query Types</h4>
        <div style="color:#c4b5fd;font-size:0.85em;">{", ".join(f"{k}: {v}" for k, v in q_types.items())}</div>
    </div>"""


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def on_generate(difficulty, seed):
    facts, queries, gt, config = generate_episode(difficulty, seed)
    facts_html = facts_to_html(facts)
    queries_html = queries_to_html(queries)
    stats_html = episode_stats_html(facts, queries, config)
    # FIFO baseline simulation
    memory_ids = []
    for fact in facts:
        if len(memory_ids) >= config.memory_budget:
            memory_ids.pop(0)
        memory_ids.append(fact.fact_id)
    baseline_correct = 0
    for q in queries:
        if not q.relevant_fact_ids:
            if q.expected_answer == "UNKNOWN":
                baseline_correct += 1
            continue
        if any(fid in set(memory_ids) for fid in q.relevant_fact_ids):
            baseline_correct += 1

    mem_html = memory_viz(facts, memory_ids, config.memory_budget)
    reward_html = reward_dashboard(baseline_correct, len(queries), baseline_correct, "binary", 0)

    return facts_html, queries_html, stats_html, mem_html, reward_html


def on_simulate(difficulty, seed, correct_override, phase):
    facts, queries, gt, config = generate_episode(difficulty, seed)
    memory_ids = []
    for fact in facts:
        if len(memory_ids) >= config.memory_budget:
            memory_ids.pop(0)
        memory_ids.append(fact.fact_id)
    baseline_correct = 0
    for q in queries:
        if not q.relevant_fact_ids:
            if q.expected_answer == "UNKNOWN":
                baseline_correct += 1
            continue
        if any(fid in set(memory_ids) for fid in q.relevant_fact_ids):
            baseline_correct += 1

    correct = int(correct_override)
    reward_html = reward_dashboard(correct, len(queries), baseline_correct, phase, 0)
    return reward_html


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

custom_css = """
.gradio-container { max-width: 1400px !important; }
.dark { background: #0a0a14 !important; }
"""

with gr.Blocks(
    title="RECALL — Memory Environment",
) as demo:
    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px 0;">
        <h1 style="font-size:2.5em;margin:0;background:linear-gradient(135deg,#7c3aed,#2563eb,#059669);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🧠 RECALL — Memory Environment
        </h1>
        <p style="color:#888;font-size:1.1em;margin-top:8px;">
            An OpenEnv RL environment for training memory management policies
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<h3 style="color:#c4b5fd;margin-bottom:4px;">⚙️ Episode Configuration</h3>')
            difficulty = gr.Slider(minimum=1, maximum=3, step=1, value=1, label="Difficulty Level")
            seed = gr.Number(value=42, label="Random Seed", precision=0)
            generate_btn = gr.Button("🎲 Generate Episode", variant="primary", size="lg")

        with gr.Column(scale=3):
            stats_display = gr.HTML(label="Episode Statistics")

    with gr.Tabs() as tabs:
        with gr.TabItem("📝 Facts Stream", id=0):
            facts_display = gr.HTML(label="Facts")

        with gr.TabItem("❓ Queries", id=1):
            queries_display = gr.HTML(label="Queries")

        with gr.TabItem("💾 Memory State", id=2):
            memory_display = gr.HTML(label="Memory")

        with gr.TabItem("🏆 Reward Engine", id=3):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<h3 style="color:#c4b5fd;">Simulate Agent Performance</h3>')
                    correct_slider = gr.Slider(minimum=0, maximum=7, step=1, value=2,
                                              label="Agent Correct Answers")
                    phase_select = gr.Radio(["bootstrap", "binary"], value="binary",
                                          label="Reward Phase")
                    sim_btn = gr.Button("⚡ Compute Reward", variant="secondary")
                with gr.Column(scale=2):
                    reward_display = gr.HTML(label="Reward Dashboard")

    gr.HTML("""
    <div style="text-align:center;padding:20px;color:#555;font-size:0.85em;border-top:1px solid #222;margin-top:24px;">
        <strong>RECALL</strong> · OpenEnv Hackathon 2026 · 
        <a href="https://github.com/meta-pytorch/OpenEnv" style="color:#7c3aed;">OpenEnv Framework</a> ·
        Write-side memory management via RL
    </div>
    """)

    # Event bindings
    generate_btn.click(
        on_generate,
        inputs=[difficulty, seed],
        outputs=[facts_display, queries_display, stats_display, memory_display, reward_display]
    )

    sim_btn.click(
        on_simulate,
        inputs=[difficulty, seed, correct_slider, phase_select],
        outputs=[reward_display]
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=custom_css,
    )

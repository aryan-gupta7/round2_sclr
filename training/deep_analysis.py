"""
Deep analysis: Run episodes through the environment with different strategies
to understand exactly what the model sees, what retrieval returns, and where
the bottleneck is. Saves output to training/deep_analysis_report.txt
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recall_env.client import RecallEnv
from recall_env.models import RecallAction, FactDecision

ENV_URL = "https://s1nn3rx69-recall-env.hf.space"
REPORT = []

def log(msg=""):
    REPORT.append(msg)
    print(msg)

def run_episode(seed, difficulty, strategy_name, make_decisions_fn):
    """Run a full episode with a given decision strategy and trace everything."""
    log(f"\n{'='*80}")
    log(f"  EPISODE: seed={seed}, difficulty={difficulty}, strategy={strategy_name}")
    log(f"{'='*80}")
    
    with RecallEnv(base_url=ENV_URL).sync() as env:
        res = env.reset(difficulty=difficulty, seed=seed)
        obs = res.observation
        
        # Phase 1: Show all facts
        log(f"\n  FACTS ({len(obs.all_facts)} total, budget={obs.memory_budget}):")
        for f in obs.all_facts:
            tags = f.get('tags', [])
            imp = "[IMPORTANT]" if 'important' in tags else "           "
            text = f['text'][:100]
            log(f"    [{f['fact_id']:2d}] {imp} {text}")
        
        # Make decisions
        decisions = make_decisions_fn(obs)
        
        log(f"\n  DECISIONS ({strategy_name}):")
        stored_count = 0
        for d in decisions:
            if d.decision == "store":
                stored_count += 1
                anchor_preview = (d.anchor or "")[:60]
                log(f"    [{d.fact_id:2d}] STORE  anchor='{anchor_preview}'")
            else:
                log(f"    [{d.fact_id:2d}] SKIP")
        log(f"  Total stored: {stored_count}/{obs.memory_budget}")
        
        # Submit ingest
        res = env.step(RecallAction(mode="ingest", decisions=decisions))
        obs = res.observation
        
        # Phase 2: Query loop
        query_num = 0
        correct = 0
        log(f"\n  QUERIES:")
        while obs.phase == "query":
            query = obs.current_query
            log(f"\n  Q{query_num}: '{query}'")
            
            # Retrieve
            res = env.step(RecallAction(mode="retrieve", query=query))
            obs = res.observation
            
            if obs.retrieval_results and len(obs.retrieval_results) > 0:
                top = obs.retrieval_results[0]
                sim = top.get("similarity", "?")
                anchor = top.get("anchor", "?")
                content = top.get("content", "UNKNOWN")
                log(f"    RETRIEVAL: sim={sim:.3f}, anchor='{anchor}'")
                log(f"    CONTENT: '{content[:120]}'")
                answer = content
            else:
                log(f"    RETRIEVAL: NO RESULTS")
                answer = "UNKNOWN"
            
            # Show all retrieval results for first query
            if query_num == 0 and obs.retrieval_results:
                log(f"    ALL RETRIEVAL RESULTS ({len(obs.retrieval_results)}):")
                for i, r in enumerate(obs.retrieval_results[:5]):
                    log(f"      [{i}] sim={r.get('similarity','?'):.3f} anchor='{r.get('anchor','?')}'")
            
            # Answer
            res = env.step(RecallAction(mode="answer", answer_text=answer))
            obs = res.observation
            
            # Check step reward
            log(f"    ANSWER SENT: '{answer[:80]}'")
            
            query_num += 1
        
        # Final state
        state = env.state()
        log(f"\n  FINAL STATE:")
        log(f"    correct_answers: {state.correct_answers}/{state.queries_total}")
        log(f"    cumulative_reward: {state.cumulative_reward:.4f}")
        log(f"    baseline_correct (FIFO): {state.baseline_correct}")
        log(f"    memory_used: {state.memory_used}/{obs.memory_budget if hasattr(obs,'memory_budget') else '?'}")
        
        return state.correct_answers, state.queries_total, state.cumulative_reward, state.baseline_correct


def strategy_fifo(obs):
    """Store first N facts in order (naive FIFO)."""
    decisions = []
    stored = 0
    for f in obs.all_facts:
        if stored < obs.memory_budget:
            decisions.append(FactDecision(
                fact_id=f['fact_id'], decision="store", 
                anchor=f['text'][:50]
            ))
            stored += 1
        else:
            decisions.append(FactDecision(fact_id=f['fact_id'], decision="skip"))
    return decisions

def strategy_important_first(obs):
    """Store [IMPORTANT] tagged facts first, then fill remaining with others."""
    important = [f for f in obs.all_facts if 'important' in f.get('tags', [])]
    others = [f for f in obs.all_facts if 'important' not in f.get('tags', [])]
    
    stored = 0
    store_ids = set()
    for f in important:
        if stored < obs.memory_budget:
            store_ids.add(f['fact_id'])
            stored += 1
    for f in others:
        if stored < obs.memory_budget:
            store_ids.add(f['fact_id'])
            stored += 1
    
    decisions = []
    for f in obs.all_facts:
        if f['fact_id'] in store_ids:
            decisions.append(FactDecision(
                fact_id=f['fact_id'], decision="store",
                anchor=f['text'][:50]
            ))
        else:
            decisions.append(FactDecision(fact_id=f['fact_id'], decision="skip"))
    return decisions

def strategy_smart_anchor(obs):
    """Store [IMPORTANT] first with FULL TEXT as anchor (not truncated)."""
    important = [f for f in obs.all_facts if 'important' in f.get('tags', [])]
    others = [f for f in obs.all_facts if 'important' not in f.get('tags', [])]
    
    stored = 0
    store_ids = set()
    for f in important:
        if stored < obs.memory_budget:
            store_ids.add(f['fact_id'])
            stored += 1
    for f in others:
        if stored < obs.memory_budget:
            store_ids.add(f['fact_id'])
            stored += 1
    
    decisions = []
    for f in obs.all_facts:
        if f['fact_id'] in store_ids:
            # Use FULL text as anchor, not truncated
            decisions.append(FactDecision(
                fact_id=f['fact_id'], decision="store",
                anchor=f['text']  # FULL text
            ))
        else:
            decisions.append(FactDecision(fact_id=f['fact_id'], decision="skip"))
    return decisions

def strategy_keyword_anchor(obs):
    """Store important first, anchor = key numerical values from the fact."""
    important = [f for f in obs.all_facts if 'important' in f.get('tags', [])]
    others = [f for f in obs.all_facts if 'important' not in f.get('tags', [])]
    
    stored = 0
    store_ids = set()
    for f in important:
        if stored < obs.memory_budget:
            store_ids.add(f['fact_id'])
            stored += 1
    for f in others:
        if stored < obs.memory_budget:
            store_ids.add(f['fact_id'])
            stored += 1
    
    import re
    decisions = []
    for f in obs.all_facts:
        if f['fact_id'] in store_ids:
            # Extract key terms: numbers, model names, metrics
            text = f['text']
            # Use a summary-like anchor: first sentence + numbers
            nums = re.findall(r'[\d.]+[eE]?[-+]?\d*', text)
            anchor = text[:80] + " " + " ".join(nums[:5])
            decisions.append(FactDecision(
                fact_id=f['fact_id'], decision="store",
                anchor=anchor
            ))
        else:
            decisions.append(FactDecision(fact_id=f['fact_id'], decision="skip"))
    return decisions


# ============================================================
# Main analysis
# ============================================================

log("=" * 80)
log("  DEEP ENVIRONMENT ANALYSIS")
log("  Testing 4 strategies across 5 seeds at difficulty=1")
log("=" * 80)

strategies = [
    ("FIFO (first-50-chars anchor)", strategy_fifo),
    ("Important-first (first-50-chars)", strategy_important_first),
    ("Important-first (FULL-TEXT anchor)", strategy_smart_anchor),
    ("Important-first (keyword anchor)", strategy_keyword_anchor),
]

results = {name: [] for name, _ in strategies}

test_seeds = [0, 1, 2, 1000, 1001]

for seed in test_seeds:
    for strat_name, strat_fn in strategies:
        try:
            correct, total, reward, baseline = run_episode(seed, 1, strat_name, strat_fn)
            results[strat_name].append({
                "seed": seed, "correct": correct, "total": total, 
                "reward": reward, "baseline": baseline
            })
        except Exception as e:
            log(f"\n  ERROR: {strat_name} seed={seed}: {e}")
            results[strat_name].append({
                "seed": seed, "correct": 0, "total": 3, 
                "reward": -1, "baseline": 0
            })

# Summary table
log(f"\n\n{'='*80}")
log(f"  STRATEGY COMPARISON SUMMARY")
log(f"{'='*80}")
log(f"\n{'Strategy':<40} {'Avg Correct':>12} {'Avg Reward':>12} {'Accuracy':>10}")
log("-" * 76)

for strat_name, _ in strategies:
    data = results[strat_name]
    avg_correct = sum(d['correct'] for d in data) / len(data)
    avg_total = sum(d['total'] for d in data) / len(data)
    avg_reward = sum(d['reward'] for d in data) / len(data)
    accuracy = avg_correct / avg_total * 100 if avg_total > 0 else 0
    log(f"{strat_name:<40} {avg_correct:>10.2f}/{avg_total:.0f} {avg_reward:>12.4f} {accuracy:>9.1f}%")

# Per-seed breakdown
log(f"\n{'Seed':<8}", end="")
for strat_name, _ in strategies:
    short = strat_name[:20]
    log(f" {short:>22}", end="")
log()
log("-" * (8 + 23 * len(strategies)))
for seed in test_seeds:
    log(f"{seed:<8}", end="")
    for strat_name, _ in strategies:
        data = [d for d in results[strat_name] if d['seed'] == seed][0]
        log(f" {data['correct']}/{data['total']} r={data['reward']:+.2f}", end="")
    log()

# Save report
with open("training/deep_analysis_report.txt", "w") as f:
    f.write("\n".join(REPORT))
log(f"\nReport saved to training/deep_analysis_report.txt")

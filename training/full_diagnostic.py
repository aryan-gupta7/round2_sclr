"""
Full diagnostic: probe the live environment end-to-end.
Runs LOCALLY using the environment server code directly (no GPU needed).
Shows facts, queries, expected answers, retrieval results, grading, and rewards.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs", "recall_env"))

import numpy as np
from server.data_generator import DataGenerator, LevelConfig, grade, normalize
from server.memory_backend import MemoryBackend
from server.rewards import compute_reward, EpisodeResult, phase1_reward, phase2_reward

import yaml

def load_config(difficulty):
    path = os.path.join(os.path.dirname(__file__), "configs", f"level_{difficulty}.yaml")
    with open(path) as f:
        return LevelConfig(**yaml.safe_load(f))


def run_episode_analysis(seed, difficulty=1, verbose=True):
    """Run a full episode and show every detail."""
    config = load_config(difficulty)
    rng = np.random.default_rng(seed)
    gen = DataGenerator()
    facts, queries, gt = gen.generate(config, rng)

    if verbose:
        print(f"\n{'='*80}")
        print(f"  SEED={seed}  DIFFICULTY={difficulty}  FACTS={len(facts)}  QUERIES={len(queries)}  BUDGET={config.memory_budget}")
        print(f"{'='*80}")

        # Show all facts
        print(f"\n--- FACTS ({len(facts)}) ---")
        for f in facts:
            tag = "[DIST]" if f.is_distractor else f"[{f.category[:4].upper()}]"
            imp = " [IMPORTANT]" if "[IMPORTANT]" in f.text else ""
            print(f"  fact_id={f.fact_id:2d} {tag:7s}{imp} {f.text[:120]}")
            if f.meta:
                key_meta = {k: v for k, v in f.meta.items() if k in ('arch_full', 'metric_full', 'result', 'hp_full', 'hp_value', 'choice', 'alternative', 'rationale')}
                if key_meta:
                    print(f"           meta: {key_meta}")

        # Show all queries with expected answers
        print(f"\n--- QUERIES ({len(queries)}) ---")
        for q in queries:
            print(f"  Q{q.query_id}: [{q.query_type}] \"{q.text}\"")
            print(f"       expected_answer: \"{q.expected_answer}\"")
            print(f"       relevant_fact_ids: {q.relevant_fact_ids}")
            # Show which fact(s) are relevant
            for fid in q.relevant_fact_ids:
                fact = next((f for f in facts if f.fact_id == fid), None)
                if fact:
                    print(f"       → fact {fid}: \"{fact.text[:100]}...\"")

    # =========================================================================
    # STRATEGY 1: FIFO (store first N with text[:60] as anchor)
    # =========================================================================
    fifo_mem = MemoryBackend(
        budget=config.memory_budget,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        seed=seed
    )
    for fact in facts:
        if len(fifo_mem.items) >= config.memory_budget:
            break
        fifo_mem.store(fact.text[:60], fact.text, step=0)

    # =========================================================================
    # STRATEGY 2: "Store all" (what the model does — stores 10, budget=8)
    # =========================================================================
    store_all_mem = MemoryBackend(
        budget=config.memory_budget,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        seed=seed
    )
    for fact in facts:
        # Model gives short keyword anchors
        anchor = fact.text[:40]  # rough model-like anchor
        store_all_mem.store(anchor, fact.text, step=0)

    # =========================================================================
    # STRATEGY 3: Oracle — store only queried facts with perfect anchors
    # =========================================================================
    oracle_mem = MemoryBackend(
        budget=config.memory_budget,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        seed=seed
    )
    queried_fact_ids = set()
    for q in queries:
        for fid in q.relevant_fact_ids:
            queried_fact_ids.add(fid)
    for fact in facts:
        if fact.fact_id in queried_fact_ids:
            # Oracle: use query text as anchor for perfect retrieval
            relevant_queries = [q for q in queries if fact.fact_id in q.relevant_fact_ids]
            anchor = relevant_queries[0].text if relevant_queries else fact.text[:60]
            oracle_mem.store(anchor, fact.text, step=0)

    if verbose:
        print(f"\n--- MEMORY STATES ---")
        print(f"  FIFO:      {len(fifo_mem.items)}/{config.memory_budget} slots used")
        print(f"  Store-all: {len(store_all_mem.items)}/{config.memory_budget} slots used")
        print(f"  Oracle:    {len(oracle_mem.items)}/{config.memory_budget} slots used (only stores queried facts)")

    # =========================================================================
    # EVALUATE each strategy on each query
    # =========================================================================
    strategies = {
        "FIFO": fifo_mem,
        "Store-All": store_all_mem,
        "Oracle": oracle_mem,
    }

    results = {}
    for strat_name, mem in strategies.items():
        correct = 0
        strat_details = []
        for q in queries:
            retrieval = mem.retrieve(q.text, config.retrieval_k)

            answer = "UNKNOWN"
            if retrieval:
                answer = retrieval[0]["content"]

            is_correct = gen.grade(answer, q.expected_answer)
            if is_correct:
                correct += 1

            detail = {
                "query": q.text,
                "expected": q.expected_answer,
                "answer_sent": answer[:80] if answer != "UNKNOWN" else "UNKNOWN",
                "correct": is_correct,
                "top_sim": retrieval[0]["similarity"] if retrieval else 0,
                "top_anchor": retrieval[0]["anchor"][:50] if retrieval else "N/A",
            }
            strat_details.append(detail)

        results[strat_name] = {"correct": correct, "total": len(queries), "details": strat_details}

    if verbose:
        print(f"\n--- RETRIEVAL + GRADING ANALYSIS ---")
        for q_idx, q in enumerate(queries):
            print(f"\n  Q{q.query_id}: \"{q.text}\"")
            print(f"       Expected: \"{q.expected_answer}\"")
            for strat_name in strategies:
                d = results[strat_name]["details"][q_idx]
                grade_str = "✅" if d["correct"] else "❌"
                print(f"       {strat_name:10s}: sim={d['top_sim']:.3f} anchor='{d['top_anchor']}' → {grade_str}")

                # Debug the grading
                if not d["correct"] and d["answer_sent"] != "UNKNOWN":
                    norm_exp = normalize(q.expected_answer)
                    norm_pred = normalize(d["answer_sent"])
                    print(f"                 grade debug: norm_exp='{norm_exp}' in norm_pred='{norm_pred[:60]}' → {norm_exp in norm_pred}")

    # =========================================================================
    # COMPUTE REWARDS for each strategy
    # =========================================================================
    if verbose:
        print(f"\n--- REWARDS ---")
        # Phase 1 (bootstrap, step < 100)
        fifo_baseline_correct = results["FIFO"]["correct"]
        for strat_name in strategies:
            correct = results[strat_name]["correct"]
            total = len(queries)
            stored = len(strategies[strat_name].items)

            ep = EpisodeResult(
                correct_answers=correct,
                stored_then_retrieved_count=0,
                memory_used=stored,
                malformed_count=0,
                budget_overflow_count=max(0, 10 - config.memory_budget) if strat_name != "Oracle" else 0,
                queries_total=total,
            )
            p1 = phase1_reward(ep, config)
            p2 = phase2_reward(ep, fifo_baseline_correct, config)
            print(f"  {strat_name:10s}: correct={correct}/{total}  phase1_reward={p1:.2f}  phase2_reward={p2:.2f}")

    return results


def check_grading_is_possible(n_seeds=20, difficulty=1):
    """Check across many seeds: can the answer EVER be found via retrieval+grading?"""
    print(f"\n{'='*80}")
    print(f"  GRADING FEASIBILITY CHECK ({n_seeds} seeds, difficulty={difficulty})")
    print(f"{'='*80}")

    config = load_config(difficulty)
    gen = DataGenerator()

    fifo_pass = 0
    oracle_pass = 0
    total_queries = 0
    query_type_stats = {}

    for seed in range(1000, 1000 + n_seeds):
        rng = np.random.default_rng(seed)
        facts, queries, gt = gen.generate(config, rng)

        # FIFO strategy
        fifo_mem = MemoryBackend(budget=config.memory_budget, embedding_model=config.embedding_model,
                                  embedding_dim=config.embedding_dim, seed=seed)
        for fact in facts:
            if len(fifo_mem.items) >= config.memory_budget:
                break
            fifo_mem.store(fact.text[:60], fact.text, step=0)

        # Oracle strategy
        oracle_mem = MemoryBackend(budget=config.memory_budget, embedding_model=config.embedding_model,
                                    embedding_dim=config.embedding_dim, seed=seed)
        queried_fact_ids = set()
        for q in queries:
            for fid in q.relevant_fact_ids:
                queried_fact_ids.add(fid)
        for fact in facts:
            if fact.fact_id in queried_fact_ids:
                relevant_queries = [q for q in queries if fact.fact_id in q.relevant_fact_ids]
                anchor = relevant_queries[0].text if relevant_queries else fact.text[:60]
                oracle_mem.store(anchor, fact.text, step=0)

        for q in queries:
            total_queries += 1
            qtype = q.query_type
            if qtype not in query_type_stats:
                query_type_stats[qtype] = {"fifo": 0, "oracle": 0, "total": 0}
            query_type_stats[qtype]["total"] += 1

            # FIFO
            r = fifo_mem.retrieve(q.text, config.retrieval_k)
            answer = r[0]["content"] if r else "UNKNOWN"
            if gen.grade(answer, q.expected_answer):
                fifo_pass += 1
                query_type_stats[qtype]["fifo"] += 1

            # Oracle
            r = oracle_mem.retrieve(q.text, config.retrieval_k)
            answer = r[0]["content"] if r else "UNKNOWN"
            if gen.grade(answer, q.expected_answer):
                oracle_pass += 1
                query_type_stats[qtype]["oracle"] += 1

    print(f"\n  Total queries: {total_queries}")
    print(f"  FIFO accuracy:   {fifo_pass}/{total_queries} = {100*fifo_pass/total_queries:.1f}%")
    print(f"  Oracle accuracy: {oracle_pass}/{total_queries} = {100*oracle_pass/total_queries:.1f}%")
    print(f"\n  Per query type:")
    for qtype, stats in sorted(query_type_stats.items()):
        print(f"    {qtype:20s}: FIFO={stats['fifo']}/{stats['total']} ({100*stats['fifo']/max(1,stats['total']):.0f}%)  "
              f"Oracle={stats['oracle']}/{stats['total']} ({100*stats['oracle']/max(1,stats['total']):.0f}%)")

    if oracle_pass / total_queries < 0.5:
        print(f"\n  🔴 CRITICAL: Even Oracle strategy gets < 50% correct!")
        print(f"     This means the environment itself is too hard to learn from.")
        print(f"     The cosine similarity between query-text-anchors and queries is too low,")
        print(f"     OR the grading function doesn't accept the retrieved fact text as a valid answer.")
        print(f"     Training CANNOT succeed until this is fixed.")
    else:
        print(f"\n  🟢 Oracle accuracy is {100*oracle_pass/total_queries:.0f}% — environment is learnable.")


def check_what_answer_looks_like(n_seeds=5, difficulty=1):
    """Show exactly what 'expected_answer' vs 'retrieved content' looks like for grading."""
    print(f"\n{'='*80}")
    print(f"  ANSWER vs RETRIEVED CONTENT COMPARISON")
    print(f"{'='*80}")

    config = load_config(difficulty)
    gen = DataGenerator()

    for seed in range(1000, 1000 + n_seeds):
        rng = np.random.default_rng(seed)
        facts, queries, gt = gen.generate(config, rng)

        # Oracle with query-text anchors
        oracle_mem = MemoryBackend(budget=config.memory_budget, embedding_model=config.embedding_model,
                                    embedding_dim=config.embedding_dim, seed=seed)
        queried_fact_ids = set()
        for q in queries:
            for fid in q.relevant_fact_ids:
                queried_fact_ids.add(fid)
        for fact in facts:
            if fact.fact_id in queried_fact_ids:
                relevant_queries = [q for q in queries if fact.fact_id in q.relevant_fact_ids]
                anchor = relevant_queries[0].text if relevant_queries else fact.text[:60]
                oracle_mem.store(anchor, fact.text, step=0)

        print(f"\n  Seed {seed}:")
        for q in queries:
            r = oracle_mem.retrieve(q.text, 1)
            content = r[0]["content"] if r else "NOTHING RETRIEVED"
            sim = r[0]["similarity"] if r else 0
            expected = q.expected_answer
            is_correct = gen.grade(content, expected)

            norm_exp = normalize(expected)
            norm_content = normalize(content)
            containment = norm_exp in norm_content

            grade_str = "✅" if is_correct else "❌"
            print(f"    Q: \"{q.text[:70]}\"")
            print(f"       type={q.query_type} sim={sim:.3f}")
            print(f"       expected:  \"{expected}\"")
            print(f"       retrieved: \"{content[:100]}...\"")
            print(f"       normalize(expected)='{norm_exp}'  in normalize(retrieved)={containment}")
            print(f"       grade={grade_str}")
            print()


if __name__ == "__main__":
    # 1. Detailed breakdown of 3 seeds
    for seed in [1000, 1001, 1002]:
        run_episode_analysis(seed, difficulty=1, verbose=True)

    # 2. Grading feasibility across 20 seeds
    check_grading_is_possible(n_seeds=20, difficulty=1)

    # 3. Show exact answer vs content comparison
    check_what_answer_looks_like(n_seeds=5, difficulty=1)

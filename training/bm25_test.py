"""Quick feasibility test with the new BM25+full-names config."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs", "recall_env"))

import numpy as np
from server.data_generator import DataGenerator, LevelConfig, grade, normalize
from server.memory_backend import MemoryBackend
import yaml

def load_config(difficulty):
    path = os.path.join(os.path.dirname(__file__), "configs", f"level_{difficulty}.yaml")
    with open(path) as f:
        return LevelConfig(**yaml.safe_load(f))

def run_feasibility(n_seeds=20, difficulty=1):
    config = load_config(difficulty)
    gen = DataGenerator()
    retrieval_mode = getattr(config, 'retrieval_mode', 'hybrid')

    fifo_pass = 0
    oracle_pass = 0
    total_queries = 0

    for seed in range(1000, 1000 + n_seeds):
        rng = np.random.default_rng(seed)
        facts, queries, gt = gen.generate(config, rng)

        # FIFO: store first N with text[:60] as anchor
        fifo_mem = MemoryBackend(budget=config.memory_budget,
                                  embedding_model=config.embedding_model,
                                  embedding_dim=config.embedding_dim,
                                  seed=seed, retrieval_mode=retrieval_mode)
        for fact in facts:
            if len(fifo_mem.items) >= config.memory_budget:
                break
            fifo_mem.store(fact.text[:60], fact.text, step=0)

        # Oracle: store only queried facts with query-text anchors  
        oracle_mem = MemoryBackend(budget=config.memory_budget,
                                    embedding_model=config.embedding_model,
                                    embedding_dim=config.embedding_dim,
                                    seed=seed, retrieval_mode=retrieval_mode)
        queried_fact_ids = set()
        for q in queries:
            for fid in q.relevant_fact_ids:
                queried_fact_ids.add(fid)
        for fact in facts:
            if fact.fact_id in queried_fact_ids:
                relevant_queries = [q for q in queries if fact.fact_id in q.relevant_fact_ids]
                anchor = relevant_queries[0].text if relevant_queries else fact.text[:60]
                oracle_mem.store(anchor, fact.text, step=0)

        # "Smart FIFO": store first N but use full-name anchor (what model could learn)
        smart_mem = MemoryBackend(budget=config.memory_budget,
                                   embedding_model=config.embedding_model,
                                   embedding_dim=config.embedding_dim,
                                   seed=seed, retrieval_mode=retrieval_mode)
        for fact in facts:
            if len(smart_mem.items) >= config.memory_budget:
                break
            # Use category-aware anchor: arch_full + metric_full + result
            if fact.category == "experiment" and fact.meta:
                anchor = f"{fact.meta.get('arch_full','')} {fact.meta.get('metric_full','')} {fact.meta.get('result','')}"
            else:
                anchor = fact.text[:60]
            smart_mem.store(anchor, fact.text, step=0)

        for q in queries:
            total_queries += 1
            # FIFO
            r = fifo_mem.retrieve(q.text, config.retrieval_k)
            if r and gen.grade(r[0]["content"], q.expected_answer):
                fifo_pass += 1
            # Oracle
            r = oracle_mem.retrieve(q.text, config.retrieval_k)
            if r and gen.grade(r[0]["content"], q.expected_answer):
                oracle_pass += 1

    print(f"\n{'='*60}")
    print(f"  FEASIBILITY: {n_seeds} seeds, difficulty={difficulty}")
    print(f"  Retrieval mode: {retrieval_mode}")
    print(f"  Budget: {config.memory_budget}/{config.facts_total}")
    print(f"{'='*60}")
    print(f"  Total queries: {total_queries}")
    print(f"  FIFO accuracy:   {fifo_pass}/{total_queries} = {100*fifo_pass/total_queries:.1f}%")
    print(f"  Oracle accuracy: {oracle_pass}/{total_queries} = {100*oracle_pass/total_queries:.1f}%")
    print(f"  Gap (learnable):  {100*(oracle_pass - fifo_pass)/total_queries:.1f}pp")

    if fifo_pass > 0:
        print(f"\n  🟢 FIFO baseline gets {100*fifo_pass/total_queries:.0f}% — nonzero baseline means reward signal works!")
    else:
        print(f"\n  🔴 FIFO baseline gets 0% — model can't learn from this.")

    if oracle_pass / total_queries > 0.7:
        print(f"  🟢 Oracle gets {100*oracle_pass/total_queries:.0f}% — environment is learnable!")
    else:
        print(f"  🔴 Oracle < 70% — retrieval mechanism still broken.")

    if (oracle_pass - fifo_pass) / total_queries > 0.2:
        print(f"  🟢 20pp+ gap between FIFO and Oracle — room for the model to learn!")
    else:
        print(f"  🔴 Gap too small — model can't differentiate from FIFO.")

# Also show a detailed seed example
def show_one_seed(seed=1000, difficulty=1):
    config = load_config(difficulty)
    gen = DataGenerator()
    retrieval_mode = getattr(config, 'retrieval_mode', 'hybrid')
    rng = np.random.default_rng(seed)
    facts, queries, gt = gen.generate(config, rng)

    print(f"\n{'='*60}")
    print(f"  DETAILED: seed={seed}")
    print(f"{'='*60}")

    # Show facts
    print(f"\nFacts ({len(facts)}):")
    for f in facts:
        tag = "[DIST]" if f.is_distractor else f"[{f.category[:4].upper()}]"
        print(f"  {f.fact_id:2d} {tag:7s} {f.text[:100]}")

    # FIFO with BM25
    fifo_mem = MemoryBackend(budget=config.memory_budget,
                              embedding_model=config.embedding_model,
                              embedding_dim=config.embedding_dim,
                              seed=seed, retrieval_mode=retrieval_mode)
    for fact in facts:
        if len(fifo_mem.items) >= config.memory_budget:
            break
        fifo_mem.store(fact.text[:60], fact.text, step=0)

    print(f"\nStored (FIFO, budget={config.memory_budget}):")
    for item in fifo_mem.items:
        print(f"  slot={item.slot_id} anchor='{item.anchor}'")

    print(f"\nQueries + BM25 retrieval:")
    correct = 0
    for q in queries:
        r = fifo_mem.retrieve(q.text, 1)
        top = r[0] if r else None
        is_correct = gen.grade(top["content"], q.expected_answer) if top else False
        if is_correct:
            correct += 1
        grade_str = "✅" if is_correct else "❌"
        print(f"  Q: \"{q.text[:70]}\"")
        print(f"     expected: \"{q.expected_answer}\"")
        if top:
            print(f"     top: score={top['similarity']:.3f} anchor='{top['anchor'][:60]}'")
            print(f"     content: \"{top['content'][:80]}\"")
        print(f"     {grade_str}")
    print(f"\n  Score: {correct}/{len(queries)}")


if __name__ == "__main__":
    show_one_seed(seed=1000, difficulty=1)
    show_one_seed(seed=1001, difficulty=1)
    run_feasibility(n_seeds=20, difficulty=1)

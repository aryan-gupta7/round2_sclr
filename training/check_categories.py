"""Check fact categories and [IMPORTANT] tag distribution."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.recall_env.server.data_generator import DataGenerator, LevelConfig

config = LevelConfig(
    difficulty=1, facts_total=10, queries_total=3, memory_budget=5,
    batch_size=8, retrieval_k=5,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_mode="bm25", explicit_importance_tags=True,
    query_distribution={"specific": 1.0},
    reward_shaping={}, system_prompt_hints=[],
)

dg = DataGenerator()

for seed in [0, 1, 2, 3, 4, 42]:
    rng = np.random.default_rng(seed)
    facts, queries, gt = dg.generate(config, rng)
    
    exp_count = sum(1 for f in facts if f.category == "experiment")
    imp_count = sum(1 for f in facts if "important" in f.tags)
    
    print(f"Seed {seed}: {exp_count} experiment, {imp_count} [IMPORTANT], budget={config.memory_budget}")
    
    for f in facts:
        tag = "[IMP]" if "important" in f.tags else "     "
        print(f"  {tag} [{f.fact_id}] cat={f.category:12s} {f.text[:80]}")
    
    print(f"  Queries target fact IDs: {[q.relevant_fact_ids for q in queries]}")
    print()

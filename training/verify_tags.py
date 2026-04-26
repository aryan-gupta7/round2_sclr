"""Verify that [IMPORTANT] tags = experiment facts = queryable facts.
Avoids loading SentenceTransformer which causes OOM on this machine."""
import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # suppress warning

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Mock SentenceTransformer to avoid OOM
import unittest.mock as mock
sys.modules['sentence_transformers'] = mock.MagicMock()

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

print("Testing [IMPORTANT] = experiment = queryable alignment:")
print("=" * 70)

all_pass = True
for seed in range(20):
    rng = np.random.default_rng(seed)
    facts, queries, gt = dg.generate(config, rng)
    
    exp_facts = [f for f in facts if f.category == "experiment"]
    imp_facts = [f for f in facts if "important" in f.tags]
    exp_ids = {f.fact_id for f in exp_facts}
    imp_ids = {f.fact_id for f in imp_facts}
    query_target_ids = set()
    for q in queries:
        query_target_ids.update(q.relevant_fact_ids)
    
    all_targets_important = query_target_ids.issubset(imp_ids)
    imp_eq_exp = imp_ids == exp_ids
    n_imp = len(imp_facts)
    
    ok = imp_eq_exp and all_targets_important and n_imp == 5
    if not ok:
        all_pass = False
    status = "✓" if ok else "✗"
    print(f"  {status} Seed {seed}: exp={len(exp_facts)} imp={n_imp} | imp==exp:{imp_eq_exp} | queries⊆imp:{all_targets_important}")

print()
if all_pass:
    print("ALL PASS ✓ — Model can achieve 100% by storing all [IMPORTANT] facts.")
else:
    print("SOME FAILED ✗ — Need to investigate mismatches.")

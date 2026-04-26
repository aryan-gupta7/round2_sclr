"""End-to-end local simulation: store [IMPORTANT] facts → retrieve → grade."""
import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import unittest.mock as mock
sys.modules['sentence_transformers'] = mock.MagicMock()

from envs.recall_env.server.data_generator import DataGenerator, LevelConfig, grade
from envs.recall_env.server.memory_backend import MemoryBackend

config = LevelConfig(
    difficulty=1, facts_total=10, queries_total=3, memory_budget=5,
    batch_size=8, retrieval_k=5,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_mode="bm25", explicit_importance_tags=True,
    query_distribution={"specific": 1.0},
    reward_shaping={}, system_prompt_hints=[],
)

dg = DataGenerator()

print("Full episode simulation: store [IMPORTANT] → retrieve → grade")
print("=" * 70)

total_correct = 0
total_queries = 0

for seed in range(20):
    rng = np.random.default_rng(seed)
    facts, queries, gt = dg.generate(config, rng)
    
    # Create BM25 memory
    mem = MemoryBackend(budget=5, embedding_model="", retrieval_mode="bm25", seed=seed)
    
    # Store only [IMPORTANT] facts (= experiment facts)
    for f in facts:
        if "important" in f.tags:
            mem.store(f.text, f.text, step=0)
    
    # Run retrieval + grading for each query
    correct = 0
    for q in queries:
        results = mem.retrieve(q.text, top_k=5)
        if results:
            answer = results[0]["content"]
        else:
            answer = "UNKNOWN"
        
        is_correct = grade(answer, q.expected_answer)
        if is_correct:
            correct += 1
        else:
            # Show failure details
            print(f"  Seed {seed} MISS: Q='{q.text[:60]}...'")
            print(f"    Expected answer: {q.expected_answer}")
            if results:
                print(f"    Top retrieved: sim={results[0]['similarity']:.3f} '{results[0]['anchor'][:60]}...'")
            print()
    
    total_correct += correct
    total_queries += len(queries)
    acc = correct / max(1, len(queries))
    status = "✓" if acc == 1.0 else f"✗ ({correct}/{len(queries)})"
    print(f"  Seed {seed}: {status}")

print()
print(f"Overall: {total_correct}/{total_queries} = {total_correct/total_queries:.1%}")

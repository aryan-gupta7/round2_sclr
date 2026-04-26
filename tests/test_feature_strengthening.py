import numpy as np
from envs.recall_env.server.memory_backend import MemoryBackend

def test_identical_facts_reinforce():
    # Use embedding mode so cosine sim makes sense
    backend = MemoryBackend(budget=10, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128, retrieval_mode="embedding")
    backend.store("anchor text", "fact content")
    initial_strength = backend.items[0].strength
    incoming_emb = backend._get_embedding("anchor text")
    backend.check_and_reinforce(incoming_emb)
    assert backend.items[0].strength > initial_strength

def test_dissimilar_facts_dont_reinforce():
    backend = MemoryBackend(budget=10, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128, retrieval_mode="embedding")
    backend.store("chemistry periodic table elements", "fact A")
    backend.check_and_reinforce(backend._get_embedding("football match score goals"))
    assert backend.items[0].reinforcement_count == 0

def test_strength_capped_at_3():
    backend = MemoryBackend(budget=10, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128, retrieval_mode="embedding")
    backend.store("anchor", "fact")
    emb = backend._get_embedding("anchor")
    for _ in range(20):
        backend.check_and_reinforce(emb)
    assert backend.items[0].strength <= 3.0

def test_strength_affects_retrieval_order():
    backend = MemoryBackend(budget=10, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128, retrieval_mode="embedding")
    backend.store("machine learning gradient descent", "fact A")
    backend.store("machine learning optimization", "fact B")
    emb_a = backend._get_embedding("machine learning gradient descent")
    for _ in range(3):
        backend.check_and_reinforce(emb_a)
    results = backend.retrieve("machine learning training", top_k=2)
    assert results[0]["anchor"] == "machine learning gradient descent"

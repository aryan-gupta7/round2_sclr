from envs.recall_env.server.memory_backend import MemoryBackend
from envs.recall_env.server.recall_env_environment import RecallEnvironment

def test_tag_filter_restricts_retrieval():
    backend = MemoryBackend(budget=10, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128)
    backend.store("curie polonium discovery", "fact A", tag="identity")
    backend.store("meeting scheduled tuesday", "fact B", tag="temporal")
    results = backend.retrieve("when is the meeting", top_k=5, tag_filter="temporal")
    assert all(r["tag"] == "temporal" for r in results)
    assert results[0]["anchor"] == "meeting scheduled tuesday"

def test_tag_filter_empty_falls_back():
    backend = MemoryBackend(budget=10, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128)
    backend.store("some identity fact", "content", tag="identity")
    results = backend.retrieve("when something happened", top_k=3, tag_filter="temporal")
    assert len(results) > 0  # fallback worked

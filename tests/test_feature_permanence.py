from envs.recall_env.server.memory_backend import MemoryBackend

def test_core_slots_are_permanent():
    backend = MemoryBackend(budget=30, max_core=3, max_working=27, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128)
    for i in range(3):
        backend.store(f"core fact {i}", f"content {i}", permanence="core")
    # Fill working slots completely — core should not be evicted
    for i in range(30):
        backend.store(f"working fact {i}", f"content {i}", permanence="working")
    core_items = [item for item in backend.items if item.permanence == "core"]
    assert len(core_items) == 3

def test_core_overflow_rejected():
    backend = MemoryBackend(budget=30, max_core=2, max_working=28, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128)
    backend.store("core 1", "c1", permanence="core")
    backend.store("core 2", "c2", permanence="core")
    result = backend.store("core 3", "c3", permanence="core")  # should be rejected
    assert result is None

def test_working_evicts_oldest():
    backend = MemoryBackend(budget=10, max_core=0, max_working=3, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128)
    backend.store("fact 1", "c1", permanence="working")
    backend.store("fact 2", "c2", permanence="working")
    backend.store("fact 3", "c3", permanence="working")
    backend.store("fact 4", "c4", permanence="working")  # should evict "fact 1"
    anchors = [item.anchor for item in backend.items]
    assert "fact 1" not in anchors
    assert "fact 4" in anchors

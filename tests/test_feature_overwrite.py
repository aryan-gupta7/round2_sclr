import pytest
from envs.recall_env.server.memory_backend import MemoryBackend

def test_overwrite_success():
    backend = MemoryBackend(budget=10, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128)
    slot1 = backend.store("old anchor", "old content")
    assert slot1 is not None

    success = backend.overwrite(slot1, "new anchor", "new content")
    assert success is True
    assert backend.items[0].anchor == "new anchor"
    assert backend.items[0].content == "new content"
    assert "new anchor" in backend.current_anchors()
    assert "old anchor" not in backend.current_anchors()

def test_overwrite_invalid_id():
    backend = MemoryBackend(budget=10, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128)
    success = backend.overwrite(999, "anchor", "content")
    assert success is False

def test_overwrite_with_permanence_change():
    backend = MemoryBackend(budget=10, max_core=1, max_working=9, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=128)
    # Fill core
    backend.store("core anchor", "core content", permanence="core")
    slot = backend.store("work anchor", "work content", permanence="working")
    
    # Try overwrite working -> core, should fail because core budget=1 is FULL
    success = backend.overwrite(slot, "new anchor", "new content", permanence="core")
    assert success is False
    assert backend.items[1].anchor == "work anchor"  # undisturbed

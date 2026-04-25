# Memory Backend

> **Scope**: The data structure and retrieval logic powering memory. Single-tier, vector-indexed.
> **Implementation lives in**: `envs/recall_env/server/memory_backend.py`.

## Core data structure

```python
@dataclass
class MemoryItem:
    slot_id: int
    anchor: str                    # the agent-authored retrieval anchor
    content: str                   # the original fact text (verbatim)
    anchor_embedding: np.ndarray   # cached embedding of anchor
    stored_at_step: int            # for analysis only, NOT used in retrieval
    is_prefilled: bool             # True if seeded by reset(), False if agent-stored
```

Memory is a fixed-capacity list of `MemoryItem`s.

```python
class MemoryBackend:
    def __init__(self, budget: int, embedding_model: str, embedding_dim: int):
        self.budget = budget
        self.items: list[MemoryItem] = []
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = embedding_dim
        # Optional projection if embedding_dim < native dim:
        self.projection = None  # see "Dimensionality reduction" below
```

## Operations (full API)

### `store(anchor: str, content: str) -> int | None`

Adds a new item if budget allows. Returns the new slot_id, or `None` if rejected.

- Anchor is embedded once at storage time and cached.
- If `len(items) >= budget`: rejected, returns None. The environment's `step()` is responsible for surfacing this as a budget-overflow penalty.
- Anchor must be non-empty after stripping. Empty anchors → rejected.

### `delete(slot_id: int) -> bool`

Removes the item with that slot_id. Returns `True` on success, `False` if slot_id not found.

### `retrieve(query: str, top_k: int) -> list[dict]`

1. Embed query with same model used for anchors.
2. Compute cosine similarity against all stored items' anchor embeddings.
3. Return top-k as `[{"slot_id": int, "anchor": str, "content": str, "similarity": float}, ...]` sorted descending.

If memory is empty, return `[]`. Do not raise.

### `prefill(items: list[tuple[str, str]]) -> None`

Seed the memory with `(anchor, content)` pairs at episode reset. Each prefilled item:
- Has `is_prefilled=True`
- Counts against the budget the same as agent-stored items
- Is fully retrievable

Prefilling more items than the budget allows is a config error → raise.

### `current_anchors() -> list[str]`

Return the list of anchors only (not content). Used by observations to show the agent what's already in memory without exploding token cost.

### `usage() -> tuple[int, int]`

Returns `(used_slots, total_budget)`.

## Embedding model choice

Default: `sentence-transformers/all-MiniLM-L6-v2`. 384 native dims.

Why this model:
- Small (~80 MB) — fits in HF Spaces memory budget
- Fast on CPU — episodes don't need GPU for embedding
- Well-supported, no licensing issues
- Output is normalized, so cosine = dot product

Configurable via the `embedding_model` field in level YAML. If a teammate experiments with `BAAI/bge-small-en-v1.5` or similar, document the choice in `envs/recall_env/CHANGELOG.md`.

## Dimensionality reduction

The level config specifies `embedding_dim`. If lower than the model's native dim, apply a fixed random projection (deterministic, seeded):

```python
def _build_projection(self, native_dim: int, target_dim: int, seed: int = 0):
    if target_dim >= native_dim:
        return None  # no projection needed
    rng = np.random.default_rng(seed)
    # Gaussian random projection, scaled
    matrix = rng.standard_normal((native_dim, target_dim)) / np.sqrt(target_dim)
    return matrix
```

Why random projection (not PCA):
- PCA requires fitting on a corpus; we don't want corpus-dependent retrieval quality
- Johnson-Lindenstrauss guarantees similarity preservation in expectation
- Reproducible from seed

**Important**: the same projection matrix is used for both anchors and queries within an episode. They must live in the same space.

## Why low-dim embeddings make the problem harder (and why that's the point)

At 128-dim or 64-dim, semantically distinct facts cluster more tightly. Naive anchors get confused. The agent must write disambiguating anchors to succeed.

This is also the **mobile / edge deployment** story for the pitch: RECALL is designed for memory-constrained agent deployments where you cannot afford 768-dim embeddings × thousands of items.

## Pre-fill design (from L2 onwards)

Pre-filled memory simulates the realistic case where the agent inherits context from prior sessions. The data generator (`08_DATA_GENERATION.md` — TBD) is responsible for producing the prefill items. The memory backend just consumes them.

### Pre-fill composition (target ratios — to be tuned)

| Type | L2 | L3 | L4 | L5 |
|------|----|----|----|----|
| Relevant-to-upcoming-queries | 30% | 30% | 30% | 25% |
| Distractor (similar-looking but unqueried) | 50% | 50% | 40% | 50% |
| Outdated-version (will be contradicted later) | 20% | 20% | 30% | 25% |

These ratios are **not locked** — they need empirical tuning. Start here, adjust based on smoke runs.

## Retrieval semantics (subtle but important)

- Cosine similarity is over **anchor** embeddings only. The content is NOT embedded. This forces the agent to put the matching signal into the anchor.
- Top-k is fixed at `retrieval_k` from level config. No threshold-based filtering. Even very weak matches are returned in the top-k. The agent may then choose to answer "UNKNOWN" if results look irrelevant.
- Prefilled items and agent-stored items are indistinguishable in retrieval results. No special weighting.

## What the memory backend does NOT do

- No graph edges between items (deferred future work)
- No item-level decay or aging (deferred)
- No clustering or summarization (deferred)
- No automatic deduplication on store (the agent is responsible for checking via observation.memory_anchors)

## Tests

`tests/test_memory_backend.py` must cover:

1. Store + retrieve round-trip
2. Budget enforcement (store rejected when full)
3. Delete creates a free slot
4. Empty retrieve returns `[]`
5. Prefill counts against budget
6. Cosine similarity ordering of results
7. Random projection determinism (same seed → same projection)
8. Anchor encoding equivalence (same anchor text → same embedding)

Each test runs in <1 second.

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

@dataclass
class MemoryItem:
    slot_id: int
    anchor: str                    # the agent-authored retrieval anchor
    content: str                   # the original fact text (verbatim)
    anchor_embedding: np.ndarray   # cached embedding of anchor
    stored_at_step: int            # for analysis only, NOT used in retrieval
    is_prefilled: bool             # True if seeded by reset(), False if agent-stored

class MemoryBackend:
    def __init__(self, budget: int, embedding_model: str, embedding_dim: Optional[int] = None, seed: int = 0):
        self.budget = budget
        self.items: List[MemoryItem] = []
        if SentenceTransformer:
            self.embedder = SentenceTransformer(embedding_model)
            # Determine native dimension
            dummy_emb = self.embedder.encode(["dummy"])[0]
            self.native_dim = dummy_emb.shape[0]
        else:
            self.embedder = None
            self.native_dim = 384 # Default for MiniLM
        
        self.embedding_dim = embedding_dim or self.native_dim
        self.projection = self._build_projection(self.native_dim, self.embedding_dim, seed)

    def _build_projection(self, native_dim: int, target_dim: int, seed: int):
        if target_dim >= native_dim:
            return None  # no projection needed
        rng = np.random.default_rng(seed)
        # Gaussian random projection, scaled
        matrix = rng.standard_normal((native_dim, target_dim)) / np.sqrt(target_dim)
        return matrix

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.embedder:
            emb = self.embedder.encode([text])[0]
        else:
            # Deterministic dummy embedding for testing
            rng = np.random.default_rng(len(text))
            emb = rng.standard_normal(self.native_dim)
            
        if self.projection is not None:
            emb = emb @ self.projection
        # Normalize for cosine similarity
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm
        return emb

    def store(self, anchor: str, content: str, step: int = 0, is_prefilled: bool = False) -> Optional[int]:
        if len(self.items) >= self.budget:
            return None
        
        anchor = anchor.strip()
        if not anchor:
            return None
        
        # Use next available slot_id (can be gap if we support delete)
        used_ids = {item.slot_id for item in self.items}
        slot_id = 0
        while slot_id in used_ids:
            slot_id += 1
            
        emb = self._get_embedding(anchor)
        
        item = MemoryItem(
            slot_id=slot_id,
            anchor=anchor,
            content=content,
            anchor_embedding=emb,
            stored_at_step=step,
            is_prefilled=is_prefilled
        )
        self.items.append(item)
        return slot_id

    def delete(self, slot_id: int) -> bool:
        for i, item in enumerate(self.items):
            if item.slot_id == slot_id:
                self.items.pop(i)
                return True
        return False

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        if not self.items:
            return []
        
        query_emb = self._get_embedding(query)
        
        results = []
        for item in self.items:
            similarity = np.dot(query_emb, item.anchor_embedding)
            results.append({
                "slot_id": item.slot_id,
                "anchor": item.anchor,
                "content": item.content,
                "similarity": float(similarity)
            })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def prefill(self, items: List[Tuple[str, str]]) -> None:
        if len(self.items) + len(items) > self.budget:
            raise ValueError(f"Prefill items ({len(items)}) exceed remaining budget ({self.budget - len(self.items)})")
        
        for anchor, content in items:
            self.store(anchor, content, step=0, is_prefilled=True)

    def current_anchors(self) -> List[str]:
        return [item.anchor for item in self.items]

    def usage(self) -> Tuple[int, int]:
        return len(self.items), self.budget

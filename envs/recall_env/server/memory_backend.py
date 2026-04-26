import re
import math
import numpy as np
from collections import Counter
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
    anchor_tokens: Counter         # tokenized anchor for BM25 scoring
    stored_at_step: int            # for analysis only, NOT used in retrieval
    is_prefilled: bool             # True if seeded by reset(), False if agent-stored


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\.\-]", " ", text)
    tokens = text.split()
    # Remove very short tokens and stopwords
    stopwords = {"the", "a", "an", "is", "was", "were", "are", "of", "for",
                 "in", "on", "at", "to", "by", "with", "and", "or", "not",
                 "it", "we", "did", "do", "how", "what", "which", "that",
                 "this", "has", "had", "have", "been"}
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


class MemoryBackend:
    def __init__(self, budget: int, embedding_model: str,
                 embedding_dim: Optional[int] = None, seed: int = 0,
                 retrieval_mode: str = "hybrid"):
        """
        retrieval_mode:
            "embedding" - pure cosine similarity (original behavior)
            "bm25"      - pure BM25 keyword matching
            "hybrid"    - 0.5 * bm25_score + 0.5 * cosine_sim (default)
        """
        self.budget = budget
        self.items: List[MemoryItem] = []
        self.retrieval_mode = retrieval_mode

        if SentenceTransformer and retrieval_mode != "bm25":
            self.embedder = SentenceTransformer(embedding_model)
            dummy_emb = self.embedder.encode(["dummy"])[0]
            self.native_dim = dummy_emb.shape[0]
        else:
            self.embedder = None
            self.native_dim = 384

        self.embedding_dim = embedding_dim or self.native_dim
        self.projection = self._build_projection(self.native_dim, self.embedding_dim, seed)

        # BM25 parameters
        self.k1 = 1.5
        self.b = 0.75

    def _build_projection(self, native_dim: int, target_dim: int, seed: int):
        if target_dim >= native_dim:
            return None
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((native_dim, target_dim)) / np.sqrt(target_dim)
        return matrix

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.embedder:
            emb = self.embedder.encode([text])[0]
        else:
            rng = np.random.default_rng(len(text))
            emb = rng.standard_normal(self.native_dim)

        if self.projection is not None:
            emb = emb @ self.projection
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm
        return emb

    def _bm25_score(self, query_tokens: List[str], doc_tokens: Counter, avg_dl: float) -> float:
        """Compute BM25 score for a single document against a query."""
        score = 0.0
        dl = sum(doc_tokens.values())
        N = len(self.items)
        if N == 0:
            return 0.0

        for term in query_tokens:
            # Term frequency in document
            tf = doc_tokens.get(term, 0)
            if tf == 0:
                continue

            # Document frequency (how many docs contain this term)
            df = sum(1 for item in self.items if term in item.anchor_tokens)
            # IDF
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            # BM25 term score
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / max(avg_dl, 1)))
            score += idf * tf_norm

        return score

    def store(self, anchor: str, content: str, step: int = 0, is_prefilled: bool = False) -> Optional[int]:
        if len(self.items) >= self.budget:
            return None

        if not anchor:
            return None
        anchor = anchor.strip()
        if not anchor:
            return None

        used_ids = {item.slot_id for item in self.items}
        slot_id = 0
        while slot_id in used_ids:
            slot_id += 1

        emb = self._get_embedding(anchor) if self.retrieval_mode != "bm25" else np.zeros(1)
        tokens = Counter(_tokenize(anchor))

        item = MemoryItem(
            slot_id=slot_id,
            anchor=anchor,
            content=content,
            anchor_embedding=emb,
            anchor_tokens=tokens,
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

        results = []

        # Compute BM25 scores
        if self.retrieval_mode in ("bm25", "hybrid"):
            query_tokens = _tokenize(query)
            avg_dl = np.mean([sum(item.anchor_tokens.values()) for item in self.items])
            bm25_scores = []
            for item in self.items:
                score = self._bm25_score(query_tokens, item.anchor_tokens, avg_dl)
                bm25_scores.append(score)
            # Normalize BM25 to [0, 1]
            max_bm25 = max(bm25_scores) if bm25_scores else 1.0
            if max_bm25 > 0:
                bm25_scores = [s / max_bm25 for s in bm25_scores]
        else:
            bm25_scores = [0.0] * len(self.items)

        # Compute embedding scores
        if self.retrieval_mode in ("embedding", "hybrid") and self.embedder:
            query_emb = self._get_embedding(query)
            emb_scores = [float(np.dot(query_emb, item.anchor_embedding)) for item in self.items]
            # Normalize to [0, 1] (cosine sim is already in [-1, 1], shift to [0, 1])
            emb_scores = [(s + 1) / 2 for s in emb_scores]
        else:
            emb_scores = [0.0] * len(self.items)

        # Combine scores
        for i, item in enumerate(self.items):
            if self.retrieval_mode == "bm25":
                combined = bm25_scores[i]
            elif self.retrieval_mode == "embedding":
                combined = emb_scores[i]
            else:  # hybrid
                combined = 0.6 * bm25_scores[i] + 0.4 * emb_scores[i]

            results.append({
                "slot_id": item.slot_id,
                "anchor": item.anchor,
                "content": item.content,
                "similarity": combined
            })

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

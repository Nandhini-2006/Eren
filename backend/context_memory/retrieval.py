"""
Dense Vector Retrieval.

Provides two vectorisation strategies:
1. **TF-IDF** (deterministic, no external dependencies) — builds a
   vocabulary from the corpus and represents each document as a sparse
   TF-IDF vector.  This is a *lexical* vectorisation, not a semantic
   embedding.
2. An optional hook for a locally available sentence-transformer model.
   When unavailable, TF-IDF is used as the default.

Cosine similarity is used for ranking.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple, Optional, Dict

from .memory_store import Memory


# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the is was were be been being have has had do does did "
    "will would shall should may might can could am are at by for "
    "from in into of on to with and but or nor not so yet".split()
)


def tokenise(text: str) -> List[str]:
    """Lower-case, strip punctuation, remove stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


# ---------------------------------------------------------------------------
# TF-IDF Vectoriser  (deterministic, stdlib-only)
# ---------------------------------------------------------------------------

class TfidfVectoriser:
    """Minimal TF-IDF vectoriser.

    This is a *lexical* vectorisation approach: two texts are similar
    only if they share surface-level tokens.  It does **not** capture
    semantic similarity between paraphrases.
    """

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self._fitted = False

    def fit(self, documents: List[str]) -> "TfidfVectoriser":
        """Build vocabulary and compute IDF from a list of documents."""
        if not documents:
            raise ValueError("Cannot fit on an empty document list")

        # Build vocabulary
        token_sets = [set(tokenise(doc)) for doc in documents]
        all_tokens = sorted({t for ts in token_sets for t in ts})
        self.vocab = {t: i for i, t in enumerate(all_tokens)}

        # IDF: log(N / (1 + df))  with add-one smoothing
        n = len(documents)
        df = Counter(t for ts in token_sets for t in ts)
        self.idf = [math.log(n / (1 + df.get(tok, 0))) for tok in all_tokens]
        self._fitted = True
        return self

    def transform(self, text: str) -> List[float]:
        """Return the TF-IDF vector for a single document."""
        if not self._fitted:
            raise RuntimeError("Vectoriser has not been fitted")
        tokens = tokenise(text)
        tf = Counter(tokens)
        vec = [0.0] * len(self.vocab)
        for tok, idx in self.vocab.items():
            if tok in tf:
                vec[idx] = (1 + math.log(tf[tok])) * self.idf[idx]
        return vec

    def transform_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.transform(t) for t in texts]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0 for zero vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Retrieve memories by cosine similarity of TF-IDF vectors.

    Because TF-IDF is lexical, results will favour term overlap rather
    than semantic meaning.  For genuine semantic retrieval, replace the
    vectoriser with a sentence-transformer model.
    """

    def __init__(self, vectoriser: Optional[TfidfVectoriser] = None) -> None:
        self.vectoriser = vectoriser or TfidfVectoriser()
        self._memories: List[Memory] = []

    def index(self, memories: List[Memory]) -> None:
        """Fit the vectoriser on the memory texts and compute embeddings."""
        if not memories:
            raise ValueError("Cannot index an empty memory list")
        self._memories = list(memories)
        texts = [m.text for m in self._memories]
        self.vectoriser.fit(texts)
        for mem in self._memories:
            mem.embedding = self.vectoriser.transform(mem.text)

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[Memory, float]]:
        """Return the top-k memories ranked by cosine similarity to *query_text*.

        Returns a list of (Memory, score) tuples in descending order.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self._memories:
            return []
        q_vec = self.vectoriser.transform(query_text)
        scored = []
        for mem in self._memories:
            if mem.embedding is not None:
                sim = cosine_similarity(q_vec, mem.embedding)
                scored.append((mem, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

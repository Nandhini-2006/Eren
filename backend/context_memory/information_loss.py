"""
Information-Loss Estimation.

Estimates how much important information is lost when memories are
summarised, compressed, or removed from the context.

Three complementary estimators:

1. **Token-level loss**: fraction of unique informative tokens lost.
2. **Embedding-level loss**: cosine distance between the embedding
   of the original text and the compressed/summarised text.
3. **Coverage loss**: fraction of original memories whose content
   is not represented in the reduced set.
"""

from __future__ import annotations

import math
from typing import List, Optional, Dict

from .retrieval import tokenise, cosine_similarity, TfidfVectoriser
from .memory_store import Memory


def token_information_loss(
    original_texts: List[str],
    retained_texts: List[str],
) -> float:
    """Fraction of unique informative tokens in *original_texts* that
    are absent from *retained_texts*.

    Returns a value in [0, 1].  0.0 = no loss; 1.0 = total loss.
    """
    original_tokens = set()
    for text in original_texts:
        original_tokens.update(tokenise(text))

    if not original_tokens:
        return 0.0

    retained_tokens = set()
    for text in retained_texts:
        retained_tokens.update(tokenise(text))

    lost = original_tokens - retained_tokens
    return len(lost) / len(original_tokens)


def embedding_information_loss(
    original_text: str,
    compressed_text: str,
    vectoriser: TfidfVectoriser,
) -> float:
    """Cosine distance between the original and compressed text embeddings.

    Returns a value in [0, 1].  0.0 = identical; 1.0 = orthogonal.
    """
    vec_orig = vectoriser.transform(original_text)
    vec_comp = vectoriser.transform(compressed_text)
    sim = cosine_similarity(vec_orig, vec_comp)
    return 1.0 - max(0.0, sim)


def coverage_loss(
    original_memories: List[Memory],
    retained_memories: List[Memory],
    similarity_threshold: float = 0.3,
) -> float:
    """Fraction of original memories not adequately represented in the
    retained set.

    An original memory is 'covered' if at least one retained memory
    has cosine similarity >= *similarity_threshold* to it.

    Returns a value in [0, 1].
    """
    if not original_memories:
        return 0.0

    covered = 0
    for orig in original_memories:
        if orig.embedding is None:
            continue
        is_covered = False
        for ret in retained_memories:
            if ret.embedding is None:
                continue
            sim = cosine_similarity(orig.embedding, ret.embedding)  # type: ignore
            if sim >= similarity_threshold:
                is_covered = True
                break
        if is_covered:
            covered += 1

    total_with_embeddings = sum(1 for m in original_memories if m.embedding is not None)
    if total_with_embeddings == 0:
        return 0.0
    return 1.0 - covered / total_with_embeddings


def full_information_loss_report(
    original_memories: List[Memory],
    retained_memories: List[Memory],
    vectoriser: Optional[TfidfVectoriser] = None,
    similarity_threshold: float = 0.3,
) -> Dict[str, float]:
    """Generate a comprehensive information-loss report.

    Returns
    -------
    dict with keys:
        - 'token_loss': fraction of unique tokens lost
        - 'coverage_loss': fraction of originals not covered
        - 'embedding_loss': average embedding distance (if vectoriser provided)
        - 'memory_reduction': 1 - (retained / original)
        - 'token_reduction': 1 - (retained_tokens / original_tokens)
    """
    orig_texts = [m.text for m in original_memories]
    ret_texts = [m.text for m in retained_memories]

    t_loss = token_information_loss(orig_texts, ret_texts)
    c_loss = coverage_loss(original_memories, retained_memories, similarity_threshold)

    # Embedding loss (average over originals)
    e_loss = 0.0
    if vectoriser is not None and orig_texts:
        combined_retained = " ".join(ret_texts) if ret_texts else ""
        losses = []
        for text in orig_texts:
            losses.append(embedding_information_loss(text, combined_retained, vectoriser))
        e_loss = sum(losses) / len(losses)

    # Memory count reduction
    n_orig = len(original_memories)
    n_ret = len(retained_memories)
    mem_reduction = 1.0 - (n_ret / n_orig) if n_orig > 0 else 0.0

    # Token reduction
    orig_tokens = sum(m.token_count() for m in original_memories)
    ret_tokens = sum(m.token_count() for m in retained_memories)
    tok_reduction = 1.0 - (ret_tokens / orig_tokens) if orig_tokens > 0 else 0.0

    return {
        "token_loss": round(t_loss, 4),
        "coverage_loss": round(c_loss, 4),
        "embedding_loss": round(e_loss, 4),
        "memory_reduction": round(mem_reduction, 4),
        "token_reduction": round(tok_reduction, 4),
    }

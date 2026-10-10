"""
Memory-Dependence Alignment Error.

Calculates the difference between the user's requested memory-dependence
level and the measured dependence of the response on retrieved memories.

The user specifies a preference on a [0, 1] scale:
    0.0 = rely entirely on the model's internal knowledge
    1.0 = rely entirely on retrieved memories

The measured dependence is estimated by:
    - Token overlap between the response and the retrieved memories.
    - Embedding similarity between the response and the memory set.

The alignment error is the absolute difference between the requested
and measured dependence levels.
"""

from __future__ import annotations

import math
from typing import List, Optional, Dict

from .retrieval import tokenise, cosine_similarity, TfidfVectoriser
from .memory_store import Memory


def measure_token_dependence(
    response_text: str,
    memories: List[Memory],
) -> float:
    """Estimate memory dependence via token overlap.

    Returns the fraction of response tokens that also appear in the
    concatenated memory texts.  Value in [0, 1].
    """
    response_tokens = tokenise(response_text)
    if not response_tokens:
        return 0.0

    memory_tokens = set()
    for m in memories:
        memory_tokens.update(tokenise(m.text))

    overlap = sum(1 for t in response_tokens if t in memory_tokens)
    return overlap / len(response_tokens)


def measure_embedding_dependence(
    response_text: str,
    memories: List[Memory],
    vectoriser: Optional[TfidfVectoriser] = None,
) -> float:
    """Estimate memory dependence via embedding similarity.

    Computes the mean cosine similarity between the response embedding
    and each memory embedding.  Requires that memories have embeddings
    and a fitted vectoriser is provided.

    Returns a value in [0, 1] (cosine similarity is non-negative for
    TF-IDF vectors).
    """
    if vectoriser is None:
        return 0.0

    response_vec = vectoriser.transform(response_text)
    embedded = [m for m in memories if m.embedding is not None]
    if not embedded:
        return 0.0

    similarities = [
        cosine_similarity(response_vec, m.embedding)  # type: ignore
        for m in embedded
    ]
    return sum(similarities) / len(similarities)


def alignment_error(
    user_preference: float,
    measured_dependence: float,
) -> float:
    """Absolute alignment error between preference and measurement.

    Parameters
    ----------
    user_preference : float
        User's desired memory-dependence level in [0, 1].
    measured_dependence : float
        Estimated actual memory dependence in [0, 1].

    Returns
    -------
    float
        |user_preference - measured_dependence|
    """
    if not (0.0 <= user_preference <= 1.0):
        raise ValueError(f"user_preference must be in [0,1], got {user_preference}")
    if not (0.0 <= measured_dependence <= 1.0):
        raise ValueError(
            f"measured_dependence must be in [0,1], got {measured_dependence}"
        )
    return abs(user_preference - measured_dependence)


def full_alignment_report(
    response_text: str,
    memories: List[Memory],
    user_preference: float,
    vectoriser: Optional[TfidfVectoriser] = None,
) -> Dict[str, float]:
    """Generate a complete alignment report.

    Returns
    -------
    dict with keys:
        - 'user_preference': the requested dependence level
        - 'token_dependence': measured via token overlap
        - 'embedding_dependence': measured via embedding similarity
        - 'combined_dependence': average of token and embedding
        - 'alignment_error': |user_preference - combined_dependence|
    """
    token_dep = measure_token_dependence(response_text, memories)
    embed_dep = measure_embedding_dependence(response_text, memories, vectoriser)

    combined = (token_dep + embed_dep) / 2.0
    error = alignment_error(user_preference, min(combined, 1.0))

    return {
        "user_preference": round(user_preference, 4),
        "token_dependence": round(token_dep, 4),
        "embedding_dependence": round(embed_dep, 4),
        "combined_dependence": round(combined, 4),
        "alignment_error": round(error, 4),
    }

"""
Confidence-Based Validation.

Estimates and validates confidence in retrieved evidence and
memory-supported results.

Important: the confidence scores produced here are **heuristic
estimates**, not calibrated probabilities.  They indicate relative
strength of evidence but should not be interpreted as the probability
that a claim is true.

Metrics computed:
- **Evidence density**: fraction of query terms covered by retrieved memories.
- **Agreement score**: mean pairwise similarity among retrieved memories
  (high agreement suggests consistent evidence).
- **Coverage score**: proportion of retrieved memories that exceed a
  relevance threshold.
- **Composite confidence**: a weighted combination of the above, clearly
  labelled as a heuristic.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict

from .retrieval import tokenise, cosine_similarity
from .memory_store import Memory


def evidence_density(query: str, memories: List[Memory]) -> float:
    """Fraction of query tokens that appear in at least one memory.

    Returns a value in [0, 1].  A value of 1.0 means every query term
    is covered; 0.0 means no coverage.
    """
    query_tokens = set(tokenise(query))
    if not query_tokens:
        return 0.0

    memory_tokens = set()
    for m in memories:
        memory_tokens.update(tokenise(m.text))

    covered = query_tokens & memory_tokens
    return len(covered) / len(query_tokens)


def agreement_score(memories: List[Memory]) -> float:
    """Mean pairwise cosine similarity among memory embeddings.

    Higher values indicate the memories are mutually consistent
    (or redundant).  Returns 0.0 if fewer than 2 memories have
    embeddings.
    """
    embedded = [m for m in memories if m.embedding is not None]
    if len(embedded) < 2:
        return 0.0

    total_sim = 0.0
    count = 0
    for i in range(len(embedded)):
        for j in range(i + 1, len(embedded)):
            total_sim += cosine_similarity(
                embedded[i].embedding, embedded[j].embedding  # type: ignore
            )
            count += 1

    return total_sim / count if count > 0 else 0.0


def coverage_score(
    query_embedding: List[float],
    memories: List[Memory],
    threshold: float = 0.1,
) -> float:
    """Fraction of memories whose relevance to the query exceeds *threshold*.

    This measures how many retrieved memories are actually relevant
    rather than noise.
    """
    if not memories:
        return 0.0

    relevant = 0
    for m in memories:
        if m.embedding is not None:
            sim = cosine_similarity(query_embedding, m.embedding)
            if sim >= threshold:
                relevant += 1

    return relevant / len(memories)


def composite_confidence(
    query: str,
    query_embedding: List[float],
    memories: List[Memory],
    weights: Optional[Dict[str, float]] = None,
    relevance_threshold: float = 0.1,
) -> Dict[str, float]:
    """Compute a composite heuristic confidence estimate.

    Parameters
    ----------
    query : str
        The query text.
    query_embedding : list of float
        The query embedding vector.
    memories : list of Memory
        Retrieved memories.
    weights : dict, optional
        Weights for 'density', 'agreement', 'coverage'.
        Defaults to equal weights.
    relevance_threshold : float
        Threshold for the coverage metric.

    Returns
    -------
    dict with keys:
        - 'evidence_density': float
        - 'agreement_score': float
        - 'coverage_score': float
        - 'composite': float (heuristic, NOT a calibrated probability)
        - 'interpretation': str (human-readable description)
    """
    if weights is None:
        weights = {"density": 1.0, "agreement": 1.0, "coverage": 1.0}

    ed = evidence_density(query, memories)
    ag = agreement_score(memories)
    cs = coverage_score(query_embedding, memories, relevance_threshold)

    total_weight = sum(weights.values())
    if total_weight == 0:
        composite = 0.0
    else:
        composite = (
            weights.get("density", 1.0) * ed
            + weights.get("agreement", 1.0) * ag
            + weights.get("coverage", 1.0) * cs
        ) / total_weight

    # Interpretation brackets
    if composite >= 0.7:
        interpretation = "High heuristic confidence — strong evidence support"
    elif composite >= 0.4:
        interpretation = "Moderate heuristic confidence — partial evidence support"
    else:
        interpretation = "Low heuristic confidence — weak or insufficient evidence"

    return {
        "evidence_density": round(ed, 4),
        "agreement_score": round(ag, 4),
        "coverage_score": round(cs, 4),
        "composite": round(composite, 4),
        "interpretation": interpretation,
    }

r"""
Maximal Marginal Relevance (MMR).

Selects items that are both relevant to the query and diverse with
respect to items already selected, reducing redundancy.

Reference:
    Carbonell, J., & Goldstein, J. (1998).  The use of MMR, diversity-
    based reranking for reordering documents and producing summaries.
    SIGIR.

Formula:
    MMR = argmax_{d in R \ S} [ lambda * sim(d, q) - (1-lambda) * max_{s in S} sim(d, s) ]
"""


from __future__ import annotations

from typing import List, Tuple, Optional

from .retrieval import cosine_similarity
from .memory_store import Memory


def mmr_select(
    query_embedding: List[float],
    candidates: List[Memory],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> List[Memory]:
    """Select memories using Maximal Marginal Relevance.

    Parameters
    ----------
    query_embedding : list of float
        The embedding of the query.
    candidates : list of Memory
        Candidate memories (must have non-None embeddings).
    top_k : int
        Number of items to select.
    lambda_param : float
        Trade-off between relevance (1.0) and diversity (0.0).
        Default is 0.5 (equal weight).

    Returns
    -------
    list of Memory
        Selected memories in order of selection.

    Raises
    ------
    ValueError
        If lambda_param is not in [0, 1] or top_k < 1.
    """
    if not (0.0 <= lambda_param <= 1.0):
        raise ValueError(f"lambda_param must be in [0, 1], got {lambda_param}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    # Filter out candidates without embeddings
    valid = [c for c in candidates if c.embedding is not None]
    if not valid:
        return []

    selected: List[Memory] = []
    remaining = list(valid)

    for _ in range(min(top_k, len(valid))):
        best_score: float = float("-inf")
        best_idx: int = 0

        for i, cand in enumerate(remaining):
            assert cand.embedding is not None  # guaranteed by filter above

            relevance = cosine_similarity(cand.embedding, query_embedding)

            # Maximum similarity to any already-selected item
            if selected:
                max_sim = max(
                    cosine_similarity(cand.embedding, s.embedding)  # type: ignore
                    for s in selected
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def mmr_scores(
    query_embedding: List[float],
    candidates: List[Memory],
    lambda_param: float = 0.5,
) -> List[Tuple[Memory, float]]:
    """Return MMR scores for all candidates (without greedy selection).

    This evaluates each candidate independently against an *empty*
    selected set, which simplifies to λ * sim(d, q).  Useful for
    inspecting raw relevance scores before iterative selection.
    """
    results = []
    for cand in candidates:
        if cand.embedding is None:
            continue
        score = lambda_param * cosine_similarity(cand.embedding, query_embedding)
        results.append((cand, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

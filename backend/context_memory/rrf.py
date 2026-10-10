"""
Reciprocal Rank Fusion (RRF).

Combines rankings from multiple retrieval systems (e.g., keyword-based
and vector-based) into a single fused ranking.

Reference:
    Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
    Reciprocal rank fusion outperforms condorcet and individual rank
    learning methods.  SIGIR.

Formula:
    rrf_score(d) = sum_over_rankers( 1 / (k + rank_i(d)) )

where *k* is a constant (default 60) that dampens the impact of high
rankings.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, TypeVar

T = TypeVar("T")


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Fuse multiple rankings using Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings : list of list of (item_id, score)
        Each inner list is a ranking from a different retrieval system.
        Items are identified by *item_id* (string).  The *score* from
        each ranker is ignored; only the positional rank matters.
    k : int
        Damping constant (default 60 per the original paper).

    Returns
    -------
    list of (item_id, rrf_score) sorted by descending rrf_score.

    Raises
    ------
    ValueError
        If *rankings* is empty or *k* < 0.
    """
    if not rankings:
        raise ValueError("At least one ranking must be provided")
    if k < 0:
        raise ValueError("k must be >= 0")

    rrf_scores: Dict[str, float] = {}

    for ranking in rankings:
        for rank_position, (item_id, _score) in enumerate(ranking, start=1):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank_position)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def rrf_from_memory_rankings(
    rankings: List[List[Tuple[Any, float]]],
    id_attr: str = "id",
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Convenience wrapper that extracts memory ids before fusing.

    Each ranking is a list of (Memory, score) tuples.  The Memory
    objects are identified by the attribute named *id_attr*.
    """
    converted = []
    for ranking in rankings:
        converted.append([(getattr(item, id_attr), score) for item, score in ranking])
    return reciprocal_rank_fusion(converted, k=k)

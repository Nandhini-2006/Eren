"""
Lexicographic Multi-Criteria Ranking.

Ranks memories using a strict priority ordering of criteria.  Unlike a
weighted sum, this approach compares items on the highest-priority
criterion first; ties are broken by the next criterion, and so on.

Default priority order: relevance > importance > recency > access_frequency.

This mirrors how a human might evaluate memories: first check relevance
to the current task, then importance, then how recent the memory is,
and finally how frequently it has been used.
"""

from __future__ import annotations

from typing import List, Tuple, Callable, Optional
from .memory_store import Memory
from .retrieval import cosine_similarity


def _default_criteria(
    query_embedding: Optional[List[float]] = None,
    now: Optional[float] = None,
) -> List[Tuple[str, Callable[[Memory], float]]]:
    """Return the default criteria in priority order.

    Each criterion is a (name, scorer) pair.  The scorer maps a Memory
    to a float value where **higher is better**.
    """

    def relevance(m: Memory) -> float:
        if query_embedding is None or m.embedding is None:
            return 0.0
        return cosine_similarity(m.embedding, query_embedding)

    def importance(m: Memory) -> float:
        return m.importance

    def recency(m: Memory) -> float:
        return m.recency_score(now)

    def access_frequency(m: Memory) -> float:
        return float(m.access_count)

    return [
        ("relevance", relevance),
        ("importance", importance),
        ("recency", recency),
        ("access_frequency", access_frequency),
    ]


def lexicographic_rank(
    memories: List[Memory],
    criteria: Optional[List[Tuple[str, Callable[[Memory], float]]]] = None,
    query_embedding: Optional[List[float]] = None,
    now: Optional[float] = None,
    tolerance: float = 1e-6,
) -> List[Tuple[Memory, List[Tuple[str, float]]]]:
    """Rank memories by lexicographic multi-criteria ordering.

    Parameters
    ----------
    memories : list of Memory
        Candidate memories to rank.
    criteria : list of (name, scorer), optional
        Custom criteria in priority order.  If None, the default criteria
        (relevance, importance, recency, access_frequency) are used.
    query_embedding : list of float, optional
        Embedding of the query (used by the default relevance criterion).
    now : float, optional
        Current timestamp for recency scoring.
    tolerance : float
        Two scores are considered tied if they differ by less than this.

    Returns
    -------
    list of (Memory, scores)
        Memories sorted in descending order of the lexicographic ranking.
        Each *scores* entry is a list of (criterion_name, value) pairs.
    """
    if criteria is None:
        criteria = _default_criteria(query_embedding, now)

    # Pre-compute all scores
    scored: List[Tuple[Memory, List[Tuple[str, float]]]] = []
    for mem in memories:
        mem_scores = [(name, scorer(mem)) for name, scorer in criteria]
        scored.append((mem, mem_scores))

    # Lexicographic sort: compare on criterion 0, then 1, etc.
    # Python's sort is stable, so we sort in reverse-priority order.
    for criterion_idx in range(len(criteria) - 1, -1, -1):
        scored.sort(
            key=lambda item: item[1][criterion_idx][1],
            reverse=True,
        )

    return scored


def format_ranking(
    ranked: List[Tuple[Memory, List[Tuple[str, float]]]],
    max_text_len: int = 50,
) -> str:
    """Format a ranked list as a human-readable table."""
    lines = ["Rank | ID                | " + " | ".join(
        f"{name:>12}" for name, _ in ranked[0][1]
    ) if ranked else ""]
    lines.append("-" * len(lines[0]) if lines[0] else "")
    for rank, (mem, scores) in enumerate(ranked, 1):
        score_str = " | ".join(f"{v:>12.4f}" for _, v in scores)
        lines.append(f"{rank:>4} | {mem.id:<17} | {score_str}")
    return "\n".join(lines)

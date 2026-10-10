"""
Evaluation Metrics.

Provides functions for measuring retrieval quality, redundancy,
token efficiency, and selection quality across the pipeline.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Dict, Set, Optional
from contextlib import contextmanager

from .memory_store import Memory
from .retrieval import cosine_similarity


# -------------------------------------------------------------------
# Precision and recall
# -------------------------------------------------------------------

def retrieval_precision(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """Precision = |retrieved ∩ relevant| / |retrieved|."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(retrieved_ids)


def retrieval_recall(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """Recall = |retrieved ∩ relevant| / |relevant|."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(relevant_ids)


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# -------------------------------------------------------------------
# Redundancy
# -------------------------------------------------------------------

def redundancy_rate(memories: List[Memory], threshold: float = 0.8) -> float:
    """Fraction of memory pairs whose cosine similarity exceeds *threshold*.

    A high redundancy rate indicates that the selected set contains
    many near-duplicate memories.
    """
    embedded = [m for m in memories if m.embedding is not None]
    n = len(embedded)
    if n < 2:
        return 0.0

    total_pairs = n * (n - 1) / 2
    redundant_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(
                embedded[i].embedding, embedded[j].embedding  # type: ignore
            )
            if sim >= threshold:
                redundant_pairs += 1

    return redundant_pairs / total_pairs


# -------------------------------------------------------------------
# Token budget
# -------------------------------------------------------------------

def token_budget_utilisation(
    selected_memories: List[Memory],
    token_budget: int,
) -> Dict[str, float]:
    """Compute token budget metrics.

    Returns
    -------
    dict with keys:
        - 'total_tokens': total tokens in selected memories
        - 'budget': the budget
        - 'utilisation': total_tokens / budget
        - 'within_budget': bool
    """
    total = sum(m.token_count() for m in selected_memories)
    return {
        "total_tokens": total,
        "budget": token_budget,
        "utilisation": round(total / token_budget, 4) if token_budget > 0 else 0.0,
        "within_budget": total <= token_budget,
    }


# -------------------------------------------------------------------
# Selection quality (composite)
# -------------------------------------------------------------------

def selection_quality_report(
    selected: List[Memory],
    all_candidates: List[Memory],
    relevant_ids: Set[str],
    token_budget: int,
) -> Dict[str, float]:
    """Comprehensive selection-quality report."""
    selected_ids = [m.id for m in selected]

    prec = retrieval_precision(selected_ids, relevant_ids)
    rec = retrieval_recall(selected_ids, relevant_ids)
    f1 = f1_score(prec, rec)
    red = redundancy_rate(selected)
    budget = token_budget_utilisation(selected, token_budget)

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "redundancy_rate": round(red, 4),
        **budget,
    }


# -------------------------------------------------------------------
# Timing utility
# -------------------------------------------------------------------

class Timer:
    """Simple context-manager timer."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self._end = time.perf_counter()
        self.elapsed = self._end - self._start

    def __repr__(self) -> str:
        return f"Timer({self.label}: {self.elapsed:.4f}s)"

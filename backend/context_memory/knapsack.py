"""
0/1 Knapsack Optimization for Memory Selection.

Selects a subset of candidate memories that maximises total utility
while respecting a configurable token budget.

This uses the classic dynamic-programming approach for the 0/1 knapsack
problem with integer weights.  Each memory has a *weight* (token count)
and a *value* (utility score from ranking).

Time complexity: O(n * W) where n = number of candidates, W = token budget.
Space complexity: O(n * W), with an optional rolling-array variant.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Callable
from .memory_store import Memory


def knapsack_select(
    candidates: List[Memory],
    token_budget: int,
    value_fn: Optional[Callable[[Memory], float]] = None,
) -> Tuple[List[Memory], int, float]:
    """Select memories under a token budget using 0/1 knapsack DP.

    Parameters
    ----------
    candidates : list of Memory
        Available memories to choose from.
    token_budget : int
        Maximum total tokens allowed (must be > 0).
    value_fn : callable, optional
        Maps a Memory to a numeric value.  Higher is better.
        Defaults to ``Memory.importance``.

    Returns
    -------
    (selected, total_tokens, total_value) : tuple
        - selected: list of chosen Memory objects
        - total_tokens: actual token count of the selection
        - total_value: sum of values of the selection

    Raises
    ------
    ValueError
        If token_budget <= 0.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be > 0, got {token_budget}")

    if value_fn is None:
        value_fn = lambda m: m.importance

    n = len(candidates)
    if n == 0:
        return [], 0, 0.0

    # Weights and values
    weights = [c.token_count() for c in candidates]
    values = [value_fn(c) for c in candidates]

    # DP table: dp[i][w] = best value using first i items with capacity w
    W = token_budget
    dp: List[List[float]] = [[0.0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i = weights[i - 1]
        v_i = values[i - 1]
        for w in range(W + 1):
            # Don't take item i
            dp[i][w] = dp[i - 1][w]
            # Take item i if it fits
            if w_i <= w:
                candidate_val = dp[i - 1][w - w_i] + v_i
                if candidate_val > dp[i][w]:
                    dp[i][w] = candidate_val

    # Backtrack to find selected items
    selected_indices: List[int] = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_indices.append(i - 1)
            w -= weights[i - 1]

    selected_indices.reverse()
    selected = [candidates[i] for i in selected_indices]
    total_tokens = sum(weights[i] for i in selected_indices)
    total_value = sum(values[i] for i in selected_indices)

    return selected, total_tokens, total_value


def greedy_knapsack(
    candidates: List[Memory],
    token_budget: int,
    value_fn: Optional[Callable[[Memory], float]] = None,
) -> Tuple[List[Memory], int, float]:
    """Greedy knapsack approximation (value-density ordering).

    Faster than DP for large budgets but not guaranteed optimal.
    Useful as a comparison baseline.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be > 0, got {token_budget}")

    if value_fn is None:
        value_fn = lambda m: m.importance

    # Sort by value-density (value / weight) descending
    scored = []
    for c in candidates:
        w = c.token_count()
        v = value_fn(c)
        density = v / w if w > 0 else 0.0
        scored.append((c, w, v, density))
    scored.sort(key=lambda x: x[3], reverse=True)

    selected: List[Memory] = []
    total_tokens = 0
    total_value = 0.0
    for c, w, v, _ in scored:
        if total_tokens + w <= token_budget:
            selected.append(c)
            total_tokens += w
            total_value += v

    return selected, total_tokens, total_value

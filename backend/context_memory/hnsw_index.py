"""
HNSW (Hierarchical Navigable Small World) Index.

A genuine implementation of the HNSW algorithm for approximate nearest-
neighbour search.  This is a from-scratch implementation in pure Python
with no external dependencies.

Reference:
    Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust
    approximate nearest neighbor search using Hierarchical Navigable
    Small World graphs.  IEEE TPAMI 42(4), 824-836.

Limitations:
    - Pure-Python performance is adequate for demonstration but not for
      production workloads with millions of vectors.  For production use,
      consider `hnswlib` or `faiss`.
    - This implementation uses Euclidean distance.  Cosine distance can
      be approximated by L2-normalising vectors before insertion.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional

from .retrieval import euclidean_distance


@dataclass
class _HNSWNode:
    """Internal node representation."""
    node_id: int
    vector: List[float]
    level: int
    neighbours: Dict[int, List[int]] = field(default_factory=dict)
    # neighbours[layer] → list of node_ids


class HNSWIndex:
    """Hierarchical Navigable Small World graph for approximate nearest-neighbour search.

    Parameters
    ----------
    m : int
        Number of bidirectional links per node per layer (default 16).
    ef_construction : int
        Size of the dynamic candidate list during construction (default 200).
    ml : float
        Level generation factor.  1/ln(m) is the recommended value.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        ml: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        if m < 2:
            raise ValueError("m must be >= 2")
        if ef_construction < 1:
            raise ValueError("ef_construction must be >= 1")

        self.m = m
        self.m_max0 = m * 2            # max neighbours at layer 0
        self.ef_construction = ef_construction
        self.ml = ml if ml is not None else 1.0 / math.log(m)
        self._rng = random.Random(seed)

        self._nodes: Dict[int, _HNSWNode] = {}
        self._entry_point: Optional[int] = None
        self._max_level: int = -1
        self._next_id: int = 0
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, vector: List[float], node_id: Optional[int] = None) -> int:
        """Insert a vector into the index. Returns the assigned node id."""
        if self._dim is None:
            self._dim = len(vector)
        elif len(vector) != self._dim:
            raise ValueError(
                f"Dimension mismatch: expected {self._dim}, got {len(vector)}"
            )

        nid = node_id if node_id is not None else self._next_id
        self._next_id = max(self._next_id, nid + 1)

        level = self._random_level()
        node = _HNSWNode(node_id=nid, vector=vector, level=level)
        for lyr in range(level + 1):
            node.neighbours[lyr] = []
        self._nodes[nid] = node

        if self._entry_point is None:
            self._entry_point = nid
            self._max_level = level
            return nid

        ep = self._entry_point

        # Traverse from top layer down to level+1 using greedy search
        for lyr in range(self._max_level, level, -1):
            ep = self._search_layer_greedy(vector, ep, lyr)

        # For layers from min(level, max_level) down to 0, do ef_construction search
        for lyr in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(vector, ep, self.ef_construction, lyr)
            m_max = self.m_max0 if lyr == 0 else self.m
            neighbours = self._select_neighbours(vector, candidates, m_max)

            node.neighbours[lyr] = [n for n, _ in neighbours]

            # Create bidirectional links
            for nbr_id, _ in neighbours:
                nbr_node = self._nodes[nbr_id]
                if lyr not in nbr_node.neighbours:
                    nbr_node.neighbours[lyr] = []
                nbr_node.neighbours[lyr].append(nid)

                # Shrink if over capacity
                if len(nbr_node.neighbours[lyr]) > m_max:
                    nbr_candidates = [
                        (cid, euclidean_distance(nbr_node.vector, self._nodes[cid].vector))
                        for cid in nbr_node.neighbours[lyr]
                    ]
                    nbr_node.neighbours[lyr] = [
                        c for c, _ in self._select_neighbours(
                            nbr_node.vector, nbr_candidates, m_max
                        )
                    ]

            if candidates:
                ep = min(candidates, key=lambda x: x[1])[0]

        if level > self._max_level:
            self._entry_point = nid
            self._max_level = level

        return nid

    def add_batch(self, vectors: List[List[float]]) -> List[int]:
        """Insert multiple vectors. Returns list of assigned node ids."""
        return [self.add(v) for v in vectors]

    def search(self, query: List[float], k: int = 5, ef: int = 50) -> List[Tuple[int, float]]:
        """Search for the *k* approximate nearest neighbours.

        Parameters
        ----------
        query : list of float
            Query vector.
        k : int
            Number of neighbours to return.
        ef : int
            Size of the dynamic candidate list (higher → more accurate but slower).

        Returns
        -------
        list of (node_id, distance) tuples sorted by ascending distance.
        """
        if self._entry_point is None:
            return []
        if self._dim is not None and len(query) != self._dim:
            raise ValueError(
                f"Dimension mismatch: expected {self._dim}, got {len(query)}"
            )

        ep = self._entry_point

        # Greedy search from top to layer 1
        for lyr in range(self._max_level, 0, -1):
            ep = self._search_layer_greedy(query, ep, lyr)

        # ef-search at layer 0
        candidates = self._search_layer(query, ep, max(ef, k), 0)
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    @property
    def size(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """Generate a random level using an exponential distribution."""
        r = self._rng.random()
        if r == 0:
            r = 1e-10
        return int(-math.log(r) * self.ml)

    def _search_layer_greedy(self, query: List[float], entry: int, layer: int) -> int:
        """Greedy walk toward the nearest node in *layer* starting from *entry*."""
        current = entry
        current_dist = euclidean_distance(query, self._nodes[current].vector)
        improved = True
        while improved:
            improved = False
            nbrs = self._nodes[current].neighbours.get(layer, [])
            for nbr_id in nbrs:
                d = euclidean_distance(query, self._nodes[nbr_id].vector)
                if d < current_dist:
                    current = nbr_id
                    current_dist = d
                    improved = True
        return current

    def _search_layer(
        self,
        query: List[float],
        entry: int,
        ef: int,
        layer: int,
    ) -> List[Tuple[int, float]]:
        """Beam search in a single layer. Returns up to *ef* (node_id, dist) pairs."""
        visited: Set[int] = {entry}
        entry_dist = euclidean_distance(query, self._nodes[entry].vector)
        candidates: List[Tuple[float, int]] = [(entry_dist, entry)]  # min-heap proxy
        results: List[Tuple[float, int]] = [(entry_dist, entry)]

        while candidates:
            # Pop nearest candidate
            candidates.sort()
            c_dist, c_id = candidates.pop(0)

            # Farthest in results
            results.sort()
            if results and c_dist > results[-1][0]:
                break

            for nbr_id in self._nodes[c_id].neighbours.get(layer, []):
                if nbr_id in visited:
                    continue
                visited.add(nbr_id)
                d = euclidean_distance(query, self._nodes[nbr_id].vector)

                results.sort()
                if len(results) < ef or d < results[-1][0]:
                    candidates.append((d, nbr_id))
                    results.append((d, nbr_id))
                    if len(results) > ef:
                        results.sort()
                        results.pop()

        return [(nid, d) for d, nid in results]

    def _select_neighbours(
        self,
        query_vec: List[float],
        candidates: List[Tuple[int, float]],
        m: int,
    ) -> List[Tuple[int, float]]:
        """Select up to *m* neighbours from candidates (simple strategy)."""
        candidates_sorted = sorted(candidates, key=lambda x: x[1])
        return candidates_sorted[:m]

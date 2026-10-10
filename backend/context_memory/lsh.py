"""
Locality-Sensitive Hashing (LSH) for Near-Duplicate Detection.

Uses banded MinHash signatures to hash memories into buckets.  Two
memories that share a bucket are *candidate* near-duplicates and can
be verified with a full similarity check.

The LSH scheme divides a MinHash signature of length *num_hashes*
into *bands* bands of *rows* rows each.  Two documents are candidates
if they agree on all rows in at least one band.

Probability of becoming a candidate ≈ 1 - (1 - s^rows)^bands,
where s is the true Jaccard similarity.

Reference:
    Leskovec, J., Rajaraman, A., & Ullman, J. D. (2014).
    Mining of Massive Datasets, ch. 3.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Set
from .minhash import MinHash


class LSHIndex:
    """Locality-Sensitive Hashing index for candidate-pair detection.

    Parameters
    ----------
    num_hashes : int
        Total length of MinHash signatures.
    bands : int
        Number of bands to divide the signature into.
    shingle_size : int
        Token shingle size for MinHash.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_hashes: int = 128,
        bands: int = 16,
        shingle_size: int = 2,
        seed: int = 42,
    ) -> None:
        if num_hashes < bands:
            raise ValueError("num_hashes must be >= bands")
        if num_hashes % bands != 0:
            raise ValueError("num_hashes must be divisible by bands")

        self.num_hashes = num_hashes
        self.bands = bands
        self.rows = num_hashes // bands
        self.minhash = MinHash(num_hashes=num_hashes, shingle_size=shingle_size, seed=seed)

        # buckets[band_index][(hash_of_band_slice)] → set of item ids
        self._buckets: List[Dict[int, Set[str]]] = [
            {} for _ in range(self.bands)
        ]
        self._signatures: Dict[str, List[int]] = {}

    def insert(self, item_id: str, text: str) -> None:
        """Compute the MinHash signature and insert into band buckets."""
        sig = self.minhash.signature(text)
        self._signatures[item_id] = sig

        for band_idx in range(self.bands):
            start = band_idx * self.rows
            band_slice = tuple(sig[start : start + self.rows])
            bucket_key = hash(band_slice)
            if bucket_key not in self._buckets[band_idx]:
                self._buckets[band_idx][bucket_key] = set()
            self._buckets[band_idx][bucket_key].add(item_id)

    def query_candidates(self, item_id: str) -> Set[str]:
        """Return candidate near-duplicates for a previously inserted item.

        A candidate is any item that shares at least one bucket with
        *item_id*.  False positives are expected; verify with a full
        similarity check.
        """
        if item_id not in self._signatures:
            raise KeyError(f"Item '{item_id}' not in index")

        sig = self._signatures[item_id]
        candidates: Set[str] = set()

        for band_idx in range(self.bands):
            start = band_idx * self.rows
            band_slice = tuple(sig[start : start + self.rows])
            bucket_key = hash(band_slice)
            bucket = self._buckets[band_idx].get(bucket_key, set())
            candidates.update(bucket)

        candidates.discard(item_id)  # Remove self
        return candidates

    def query_text(self, text: str) -> Set[str]:
        """Find candidate matches for an arbitrary text (not in the index)."""
        sig = self.minhash.signature(text)
        candidates: Set[str] = set()

        for band_idx in range(self.bands):
            start = band_idx * self.rows
            band_slice = tuple(sig[start : start + self.rows])
            bucket_key = hash(band_slice)
            bucket = self._buckets[band_idx].get(bucket_key, set())
            candidates.update(bucket)

        return candidates

    def find_all_candidate_pairs(self) -> List[Tuple[str, str]]:
        """Return all candidate near-duplicate pairs in the index."""
        pairs: Set[Tuple[str, str]] = set()
        for band_idx in range(self.bands):
            for bucket in self._buckets[band_idx].values():
                items = sorted(bucket)
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pairs.add((items[i], items[j]))
        return sorted(pairs)

    @property
    def size(self) -> int:
        return len(self._signatures)

    def candidate_pair_threshold(self) -> float:
        """Approximate Jaccard threshold at which two items have ~50% chance
        of becoming candidates.

        Threshold ≈ (1/bands)^(1/rows)
        """
        return (1.0 / self.bands) ** (1.0 / self.rows)

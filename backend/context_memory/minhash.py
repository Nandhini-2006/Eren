"""
MinHash for Set Similarity Estimation.

Estimates Jaccard similarity between memory texts using token-level
k-shingles and multiple independent hash functions.

The Jaccard similarity J(A, B) = |A ∩ B| / |A ∪ B| is approximated
by the fraction of hash functions for which the minimum hash values
agree.  Accuracy improves with more hash functions (num_hashes).

Reference:
    Broder, A. Z. (1997). On the resemblance and containment of
    documents.  Compression and Complexity of Sequences.
"""

from __future__ import annotations

import hashlib
import struct
from typing import List, Set, Tuple, Optional

from .retrieval import tokenise


_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


class MinHash:
    """MinHash signature for approximate Jaccard similarity.

    Parameters
    ----------
    num_hashes : int
        Number of independent hash functions (signature length).
    shingle_size : int
        Size of token shingles (n-grams).
    seed : int
        Base seed for generating hash function parameters.
    """

    def __init__(
        self,
        num_hashes: int = 128,
        shingle_size: int = 2,
        seed: int = 42,
    ) -> None:
        if num_hashes < 1:
            raise ValueError("num_hashes must be >= 1")
        if shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")

        self.num_hashes = num_hashes
        self.shingle_size = shingle_size

        # Generate hash function parameters: h(x) = (a*x + b) mod p mod max_hash
        import random
        rng = random.Random(seed)
        self._a = [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_hashes)]
        self._b = [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_hashes)]

    def shingles(self, text: str) -> Set[str]:
        """Extract token-level k-shingles from text."""
        tokens = tokenise(text)
        if len(tokens) < self.shingle_size:
            return set(tokens) if tokens else set()
        return {
            " ".join(tokens[i : i + self.shingle_size])
            for i in range(len(tokens) - self.shingle_size + 1)
        }

    def _hash_shingle(self, shingle: str) -> int:
        """Hash a shingle string to a 32-bit integer."""
        h = hashlib.md5(shingle.encode("utf-8")).digest()
        return struct.unpack("<I", h[:4])[0]

    def signature(self, text: str) -> List[int]:
        """Compute the MinHash signature for a text.

        Returns a list of *num_hashes* minimum hash values.
        """
        shingle_set = self.shingles(text)
        if not shingle_set:
            return [_MAX_HASH] * self.num_hashes

        # Hash each shingle to an integer
        hashed_shingles = [self._hash_shingle(s) for s in shingle_set]

        sig = []
        for i in range(self.num_hashes):
            a, b = self._a[i], self._b[i]
            min_val = min(
                ((a * h + b) % _MERSENNE_PRIME) & _MAX_HASH
                for h in hashed_shingles
            )
            sig.append(min_val)
        return sig

    def estimate_jaccard(self, sig_a: List[int], sig_b: List[int]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if len(sig_a) != len(sig_b):
            raise ValueError("Signature lengths must match")
        if not sig_a:
            return 0.0
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute estimated Jaccard similarity between two texts."""
        return self.estimate_jaccard(
            self.signature(text_a),
            self.signature(text_b),
        )


def find_near_duplicates(
    texts: List[str],
    threshold: float = 0.5,
    num_hashes: int = 128,
    shingle_size: int = 2,
) -> List[Tuple[int, int, float]]:
    """Find pairs of near-duplicate texts by MinHash Jaccard estimation.

    Returns a list of (index_a, index_b, estimated_jaccard) tuples
    where estimated Jaccard >= threshold.
    """
    mh = MinHash(num_hashes=num_hashes, shingle_size=shingle_size)
    sigs = [mh.signature(t) for t in texts]

    pairs: List[Tuple[int, int, float]] = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            sim = mh.estimate_jaccard(sigs[i], sigs[j])
            if sim >= threshold:
                pairs.append((i, j, sim))

    return pairs

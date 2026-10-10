"""
Memory Store: Core data structures for storing and managing agent memories.

Provides the Memory dataclass and MemoryStore for in-memory storage with
persistence to JSON files. Each memory tracks text content, metadata
(importance, access count, timestamps), and an optional embedding vector.
"""

from __future__ import annotations

import json
import time
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Memory:
    """A single unit of agent memory with metadata for ranking and selection."""

    id: str
    text: str
    importance: float = 0.5          # User-assigned importance in [0, 1]
    created_at: float = 0.0          # Unix timestamp
    last_accessed: float = 0.0       # Unix timestamp of last retrieval
    access_count: int = 0            # How many times this memory was retrieved
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # Dense vector, set by retrieval
    version: int = 1                 # Incremented on update
    previous_text: Optional[str] = None  # For delta encoding

    def token_count(self) -> int:
        """Estimate token count as ceil(word_count * 1.3).

        This is a deterministic heuristic that approximates sub-word
        tokenisation without requiring an external tokeniser.
        """
        words = len(self.text.split())
        return max(1, math.ceil(words * 1.3))

    def recency_score(self, now: Optional[float] = None) -> float:
        """Exponential-decay recency score in [0, 1].

        Half-life is 86400 seconds (one day).  A memory accessed one
        day ago receives a score of 0.5; older memories decay toward 0.
        """
        if now is None:
            now = time.time()
        age = max(0.0, now - self.last_accessed)
        half_life = 86400.0  # seconds
        return math.exp(-0.693147 * age / half_life)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        d = asdict(self)
        # Embedding may be large; include it only if present
        if d["embedding"] is None:
            del d["embedding"]
        if d["previous_text"] is None:
            del d["previous_text"]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Memory":
        """Reconstruct a Memory from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MemoryStore:
    """In-memory collection of Memory objects with basic CRUD and persistence."""

    def __init__(self) -> None:
        self._memories: Dict[str, Memory] = {}

    # ---- CRUD ----

    def add(self, memory: Memory) -> None:
        """Add or replace a memory in the store."""
        if not isinstance(memory, Memory):
            raise TypeError(f"Expected Memory, got {type(memory).__name__}")
        if not memory.id:
            raise ValueError("Memory must have a non-empty id")
        self._memories[memory.id] = memory

    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by id, returning None if absent."""
        return self._memories.get(memory_id)

    def remove(self, memory_id: str) -> bool:
        """Remove a memory by id. Returns True if it existed."""
        return self._memories.pop(memory_id, None) is not None

    def all(self) -> List[Memory]:
        """Return all memories as a list (order not guaranteed)."""
        return list(self._memories.values())

    def size(self) -> int:
        return len(self._memories)

    def update_text(self, memory_id: str, new_text: str) -> Memory:
        """Update the text of an existing memory, incrementing its version.

        The old text is stored in ``previous_text`` so that delta encoding
        can reconstruct history.
        """
        mem = self._memories.get(memory_id)
        if mem is None:
            raise KeyError(f"Memory '{memory_id}' not found")
        mem.previous_text = mem.text
        mem.text = new_text
        mem.version += 1
        return mem

    # ---- Persistence ----

    def save(self, path: str) -> None:
        """Persist all memories to a JSON file."""
        data = [m.to_dict() for m in self._memories.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load memories from a JSON file, replacing current contents."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._memories.clear()
        for d in data:
            mem = Memory.from_dict(d)
            self._memories[mem.id] = mem

    def __repr__(self) -> str:
        return f"MemoryStore(size={self.size()})"

"""
Delta Encoding for Memory Version History.

Stores differences between successive versions of a memory text rather
than repeating unchanged content.  This reduces storage when memories
are updated frequently with small changes.

The implementation uses a line-level diff (via Python's ``difflib``)
and encodes operations as a compact list of (op, data) tuples.

Supported operations:
- ``("keep", n)``: keep the next *n* lines unchanged.
- ``("delete", n)``: delete the next *n* lines from the old version.
- ``("insert", lines)``: insert new lines.
"""

from __future__ import annotations

import difflib
from typing import List, Tuple, Union

# Operation types for the delta
Op = Tuple[str, Union[int, List[str]]]


def compute_delta(old_text: str, new_text: str) -> List[Op]:
    """Compute a delta between *old_text* and *new_text*.

    Returns a list of operations that, when applied to *old_text*,
    reconstruct *new_text*.
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    ops: List[Op] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ops.append(("keep", i2 - i1))
        elif tag == "delete":
            ops.append(("delete", i2 - i1))
        elif tag == "insert":
            ops.append(("insert", new_lines[j1:j2]))
        elif tag == "replace":
            ops.append(("delete", i2 - i1))
            ops.append(("insert", new_lines[j1:j2]))

    return ops


def apply_delta(old_text: str, delta: List[Op]) -> str:
    """Apply a delta to *old_text* to reconstruct the new text.

    Raises
    ------
    ValueError
        If the delta operations are inconsistent with *old_text*.
    """
    old_lines = old_text.splitlines(keepends=True)
    result: List[str] = []
    cursor = 0

    for op, data in delta:
        if op == "keep":
            n = data
            if not isinstance(n, int):
                raise ValueError(f"'keep' expects int, got {type(n)}")
            if cursor + n > len(old_lines):
                raise ValueError(
                    f"'keep {n}' at cursor {cursor} exceeds old text ({len(old_lines)} lines)"
                )
            result.extend(old_lines[cursor : cursor + n])
            cursor += n
        elif op == "delete":
            n = data
            if not isinstance(n, int):
                raise ValueError(f"'delete' expects int, got {type(n)}")
            if cursor + n > len(old_lines):
                raise ValueError(
                    f"'delete {n}' at cursor {cursor} exceeds old text ({len(old_lines)} lines)"
                )
            cursor += n  # Skip deleted lines
        elif op == "insert":
            if not isinstance(data, list):
                raise ValueError(f"'insert' expects list, got {type(data)}")
            result.extend(data)
        else:
            raise ValueError(f"Unknown operation: {op}")

    return "".join(result)


def delta_size(delta: List[Op]) -> int:
    """Estimate the storage size of a delta in characters.

    This counts the characters in inserted lines plus a small
    overhead for keep/delete operations.
    """
    size = 0
    for op, data in delta:
        if op == "insert":
            size += sum(len(line) for line in data)
        else:
            size += 8  # small overhead per op
    return size


def compression_ratio(old_text: str, new_text: str) -> float:
    """Compute the compression ratio: delta_size / len(new_text).

    A ratio < 1.0 indicates that delta encoding is more space-efficient
    than storing the full new text.  Returns 0.0 if new_text is empty.
    """
    if not new_text:
        return 0.0
    delta = compute_delta(old_text, new_text)
    return delta_size(delta) / len(new_text)


class DeltaHistory:
    """Maintains a version history using delta encoding.

    Stores the initial text and a chain of deltas.  Any version can
    be reconstructed by applying deltas sequentially.
    """

    def __init__(self, initial_text: str) -> None:
        self._initial = initial_text
        self._deltas: List[List[Op]] = []

    def update(self, new_text: str) -> List[Op]:
        """Record a new version by computing and storing the delta."""
        current = self.reconstruct(self.version_count - 1)
        delta = compute_delta(current, new_text)
        self._deltas.append(delta)
        return delta

    def reconstruct(self, version: int) -> str:
        """Reconstruct the text at a given version (0 = initial)."""
        if version < 0 or version > len(self._deltas):
            raise ValueError(
                f"Version {version} out of range [0, {len(self._deltas)}]"
            )
        text = self._initial
        for i in range(version):
            text = apply_delta(text, self._deltas[i])
        return text

    @property
    def version_count(self) -> int:
        """Number of versions (initial + number of updates)."""
        return 1 + len(self._deltas)

    def total_delta_size(self) -> int:
        """Total storage used by all deltas (excludes initial text)."""
        return sum(delta_size(d) for d in self._deltas)

    def full_storage_size(self) -> int:
        """Storage that would be needed if every version were stored in full."""
        total = len(self._initial)
        for i in range(1, self.version_count):
            total += len(self.reconstruct(i))
        return total

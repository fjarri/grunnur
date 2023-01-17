"""
A class for a list that keeps itself sorted.
Allows for fast element-based insertions/removals and element finding.
"""

from bisect import bisect_left, bisect_right
from typing import (
    Union,
    Generic,
    Iterable,
    Callable,
    Any,
    Optional,
    Sequence,
    TypeVar,
    Iterator,
    overload,
)
import sys


_T = TypeVar("_T")


class SortedList(Sequence[_T]):
    def __init__(self, iterable: Iterable[_T], key: Callable[[_T], int]):
        decorated = sorted((key(item), item) for item in iterable)
        self._keys = [k for k, item in decorated]
        self._items = [item for k, item in decorated]
        self._key = key

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, idx: int) -> _T:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[_T]:
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[_T, Sequence[_T]]:
        return self._items[idx]

    def __iter__(self) -> Iterator[_T]:
        return iter(self._items)

    def index(self, value: _T, start: int = 0, stop: int = sys.maxsize) -> int:
        """
        Finds the position of an item.
        Raises ``ValueError`` if not found.
        """
        k = self._key(value)
        i = bisect_left(self._keys, k, lo=max(start, 0), hi=min(stop, len(self._keys)))
        j = bisect_right(self._keys, k, lo=max(start, 0), hi=min(stop, len(self._keys)))
        return self._items[i:j].index(value) + i

    def insert(self, value: _T) -> None:
        """
        Inserts a new item.
        If equal keys are found, inserts on the left.
        """
        k = self._key(value)
        i = bisect_left(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, value)

    def remove(self, value: _T) -> None:
        """
        Removes the first occurence of an item.
        Raises ``ValueError`` if not found.
        """
        i = self.index(value)
        del self._keys[i]
        del self._items[i]

    def argfind_ge(self, key: int) -> int:
        """
        Returns the index of the first item with ``key >= k``.
        Raises ``ValueError`` if not found.
        """
        i = bisect_left(self._keys, key)
        if i != len(self):
            return i
        raise ValueError(f"No item found with key at or above: {key}")

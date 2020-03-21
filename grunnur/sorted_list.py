"""
A class for a list that keeps itself sorted.
Allows for fast element-based insertions/removals and element finding.
"""

from bisect import bisect_left, bisect_right
from typing import Iterable, Callable, Any, Optional


class SortedList:

    def __init__(self, iterable: Iterable=(), key: Optional[Callable[[Any], Any]]=None):
        key = (lambda x: x) if key is None else key
        decorated = sorted((key(item), item) for item in iterable)
        self._keys = [k for k, item in decorated]
        self._items = [item for k, item in decorated]
        self._key = key

    def clear(self):
        self.__init__(key=self._key)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def index(self, item) -> int:
        """
        Finds the position of an item.
        Raises ``ValueError`` if not found.
        """
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return self._items[i:j].index(item) + i

    def insert(self, item):
        """
        Inserts a new item.
        If equal keys are found, inserts on the left.
        """
        k = self._key(item)
        i = bisect_left(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def remove(self, item):
        """
        Removes the first occurence of an item.
        Raises ``ValueError`` if not found.
        """
        i = self.index(item)
        del self._keys[i]
        del self._items[i]

    def argfind_ge(self, k) -> int:
        """
        Returns the index of the first item with ``key >= k``.
        Raises ``ValueError`` if not found.
        """
        i = bisect_left(self._keys, k)
        if i != len(self):
            return i
        raise ValueError('No item found with key at or above: %r' % (k,))

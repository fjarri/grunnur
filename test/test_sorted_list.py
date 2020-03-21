import pytest

from grunnur.sorted_list import SortedList


def test_basics():
    s = SortedList([2, 4, 6, 1, 3, 9])
    assert len(s) == 6
    assert s[3] == 4
    assert list(s) == [1, 2, 3, 4, 6, 9]

    s.clear()
    assert len(s) == 0
    assert list(s) == []


def test_index():
    s = SortedList([2, 4, 6, 1, 3, 9])
    assert s.index(4) == 3
    with pytest.raises(ValueError):
        s.index(0)


def test_insert():
    s = SortedList([2, 4, 6, 1, 3, 9])
    s.insert(5)
    assert list(s) == [1, 2, 3, 4, 5, 6, 9]

    # Check that if there are several elements with the same key,
    # the new item is inserted on the left.
    s = SortedList([2, 4, 6, 1, 5, 9], key=lambda x: x//2)
    s.insert(5)
    assert list(s) == [1, 2, 5, 4, 5, 6, 9]


def test_remove():
    s = SortedList([2, 4, 6, 1, 3, 9])
    s.remove(3)
    assert list(s) == [1, 2, 4, 6, 9]


def test_argfind_ge():
    s = SortedList([2, 4, 6, 1, 3, 9])
    assert s.argfind_ge(8) == 5
    with pytest.raises(ValueError):
        s.argfind_ge(10)

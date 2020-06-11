import pytest

from grunnur.utils import (
    all_same, all_different, wrap_in_tuple, min_blocks, log2, bounding_power_of_2, prod,
    string_matches_masks, normalize_object_sequence)


def test_all_same():
    assert all_same([1, 1, 3]) == False
    assert all_same([1, 1, 1]) == True


def test_all_different():
    assert all_different([1, 1, 3]) == False
    assert all_different([1, 2, 3]) == True


def test_wrap_in_tuple():
    with pytest.raises(ValueError):
        wrap_in_tuple(None)
    assert wrap_in_tuple(1) == (1,)
    assert wrap_in_tuple([1]) == (1,)
    assert wrap_in_tuple([1, 2]) == (1, 2)


def test_min_blocls():
    assert min_blocks(19, 10) == 2
    assert min_blocks(20, 10) == 2
    assert min_blocks(21, 10) == 3
    with pytest.raises(ValueError):
        assert min_blocks(0, 10)


def test_log2():
    assert log2(1) == 0
    assert log2(2) == 1
    assert log2(3) == 1
    assert log2(4) == 2


def test_bounding_power_of_2():
    assert bounding_power_of_2(1) == 1
    assert bounding_power_of_2(2) == 2
    assert bounding_power_of_2(3) == 4
    assert bounding_power_of_2(4) == 4
    assert bounding_power_of_2(5) == 8


def test_prod():
    assert prod([]) == 1
    assert prod([1, 2, 3]) == 6


def test_string_matches_masks():
    assert string_matches_masks("foo")
    assert string_matches_masks("foo", include_masks=['fo', 'ba'])
    assert not string_matches_masks("foo", include_masks=['ff', 'ba'])
    assert not string_matches_masks("foo", exclude_masks=['fo', 'ba'])
    assert string_matches_masks("foo", exclude_masks=['ff', 'ba'])


def test_normalize_object_sequence():
    assert normalize_object_sequence(1, int) == (1,)
    assert normalize_object_sequence([1], int) == (1,)
    assert normalize_object_sequence([1, 2], int) == (1, 2)

    # Empty sequence
    with pytest.raises(ValueError):
        normalize_object_sequence([], str)

    # Some of the objects are equal
    with pytest.raises(ValueError):
        normalize_object_sequence([1, 2, 1], int)

    # Wrong type
    with pytest.raises(TypeError):
        normalize_object_sequence(['1', 2], str)

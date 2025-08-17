import re

import pytest

from grunnur._utils import (
    all_different,
    all_same,
    bounding_power_of_2,
    find_local_size,
    get_launch_size,
    log2,
    max_factor,
    min_blocks,
    normalize_object_sequence,
    prod,
    string_matches_masks,
    update_dict,
)


def test_all_same() -> None:
    assert not all_same([1, 1, 3])
    assert all_same([1, 1, 1])


def test_all_different() -> None:
    assert not all_different([1, 1, 3])
    assert all_different([1, 2, 3])


def test_min_blocls() -> None:
    assert min_blocks(19, 10) == 2
    assert min_blocks(20, 10) == 2
    assert min_blocks(21, 10) == 3
    with pytest.raises(ValueError, match="The length must be positive"):
        assert min_blocks(0, 10)


def test_log2() -> None:
    assert log2(1) == 0
    assert log2(2) == 1
    assert log2(3) == 1
    assert log2(4) == 2


def test_bounding_power_of_2() -> None:
    assert bounding_power_of_2(1) == 1
    assert bounding_power_of_2(2) == 2
    assert bounding_power_of_2(3) == 4
    assert bounding_power_of_2(4) == 4
    assert bounding_power_of_2(5) == 8


def test_prod() -> None:
    assert prod([]) == 1
    assert prod([1, 2, 3]) == 6


def test_string_matches_masks() -> None:
    assert string_matches_masks("foo")
    assert string_matches_masks("foo", include_masks=["fo", "ba"])
    assert not string_matches_masks("foo", include_masks=["ff", "ba"])
    assert not string_matches_masks("foo", exclude_masks=["fo", "ba"])
    assert string_matches_masks("foo", exclude_masks=["ff", "ba"])


def test_normalize_object_sequence() -> None:
    assert normalize_object_sequence([1], int) == (1,)
    assert normalize_object_sequence([1, 2], int) == (1, 2)

    # Empty sequence
    with pytest.raises(
        ValueError, match="The iterable of base objects for the context cannot be empty"
    ):
        normalize_object_sequence([], str)

    # Some of the objects are equal
    with pytest.raises(ValueError, match="All base objects must be different"):
        normalize_object_sequence([1, 2, 1], int)

    # Wrong type
    with pytest.raises(
        TypeError,
        match=re.escape(
            "The iterable must contain only subclasses of <class 'str'>, "
            "got [<class 'str'>, <class 'int'>]"
        ),
    ):
        normalize_object_sequence(["1", 2], str)


def test_max_factor() -> None:
    assert max_factor(3, 6) == 3
    assert max_factor(6, 6) == 6
    assert max_factor(37 * 37, 1024) == 37
    assert max_factor(512 * 37, 1024) == 37 * 16
    assert max_factor(1024 * 37, 1024) == 1024
    assert max_factor(521, 512) == 1


def test_find_local_size() -> None:
    assert find_local_size((10 * 127, 3 * 127, 300), (64, 64, 32), 64) == (10, 3, 2)
    assert find_local_size((10 * 127, 3 * 127, 300), (1, 1, 1), 1) == (1, 1, 1)


def test_get_launch_size() -> None:
    with pytest.raises(ValueError, match="Global size has too many dimensions"):
        get_launch_size((64, 64, 64), 64, (100, 100, 100, 100))

    with pytest.raises(ValueError, match="Global/local work sizes have differing dimensions"):
        get_launch_size((64, 64, 64), 64, (100, 100, 100), (10, 10))

    with pytest.raises(
        ValueError, match="Global sizes must be multiples of corresponding local sizes"
    ):
        get_launch_size((64, 64, 64), 64, (100, 100, 100), (10, 10, 15))

    assert get_launch_size((64, 64, 32), 64, (10 * 127, 3 * 127, 300)) == (
        (127, 127, 150),
        (10, 3, 2),
    )
    assert get_launch_size((64, 64, 32), 64, (100, 200, 300), (5, 10, 1)) == (
        (20, 20, 300),
        (5, 10, 1),
    )


def test_update_dict() -> None:
    assert update_dict({"1": 2, "2": 3}, {"3": 4}) == {"1": 2, "2": 3, "3": 4}
    with pytest.raises(
        ValueError, match="Cannot add an item '2' - it already exists in the old dictionary"
    ):
        update_dict({"1": 2, "2": 3}, {"2": 4})
    with pytest.raises(ValueError, match="Custom error message for '2'"):
        update_dict({"1": 2, "2": 3}, {"2": 4}, error_msg="Custom error message for '{name}'")

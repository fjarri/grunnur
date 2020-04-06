import collections
from functools import reduce
import os.path
from typing import Tuple, Iterable
import re


def all_same(seq: Iterable) -> bool:
    seq = list(seq)
    return all(elem == seq[0] for elem in seq[1:])


def all_different(seq: Iterable) -> bool:
    seq = list(seq)
    return len(seq) == len(set(seq))


def wrap_in_tuple(seq_or_elem) -> Tuple:
    """
    If ``seq_or_elem`` is a sequence, converts it to a ``tuple``,
    otherwise returns a tuple with a single element ``seq_or_elem``.
    If ``seq_or_elem`` is ``None``, raises ``ValueError``.
    """
    if seq_or_elem is None:
        raise ValueError("The argument must not be None")
    elif isinstance(seq_or_elem, collections.abc.Iterable):
        return tuple(seq_or_elem)
    else:
        return (seq_or_elem,)


def min_blocks(length: int, block: int) -> int:
    """
    Returns minimum number of blocks with length ``block``
    necessary to cover the array with non-zero length ``length``.
    """
    if length <= 0:
        raise ValueError("The length must be positive")
    return (length - 1) // block + 1


def log2(num: int) -> int:
    """
    Integer-valued logarigthm with base 2.
    If ``n`` is not a power of 2, the result is rounded to the smallest number.
    """
    return num.bit_length() - 1


def bounding_power_of_2(num):
    """
    Returns the minimal number of the form ``2**m`` such that it is greater or equal to ``n``.
    """
    if num == 1:
        return 1
    else:
        return 2 ** (log2(num - 1) + 1)


def prod(seq):
    # Integer product. `numpy.prod` returns a float when given an empty sequence.
    return reduce(lambda x, y: x * y, seq, 1)


def name_matches_masks(name, include_masks=None, exclude_masks=None):

    if include_masks is None:
        include_masks = []
    if exclude_masks is None:
        exclude_masks = []

    if len(include_masks) > 0:
        for include_mask in include_masks:
            if re.search(include_mask, name):
                break
        else:
            return False

    if len(exclude_masks) > 0:
        for exclude_mask in exclude_masks:
            if re.search(exclude_mask, name):
                return False

    return True


def normalize_base_objects(objs, expected_cls):

    objs = wrap_in_tuple(objs)

    elems = list(objs) # in case it is a generic iterable
    if len(elems) == 0:
        raise ValueError("The iterable of base objects for the context cannot be empty")

    if not all_different(elems):
        raise ValueError("All base objects must be different")

    types = [type(elem) for elem in elems]
    if not all(issubclass(tp, expected_cls) for tp in types):
        raise ValueError(f"The iterable must contain only subclasses of {expected_cls}, got {types}")

    return objs

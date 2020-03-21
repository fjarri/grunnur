import collections
from functools import reduce
import os.path
from typing import Tuple


def all_same(seq):
    seq = list(seq)
    return all(elem == seq[0] for elem in seq[1:])


def all_different(seq):
    seq = list(seq)
    return len(seq) == len(set(seq))


def wrap_in_tuple(seq_or_elem) -> Tuple:
    """
    If ``seq_or_elem`` is a sequence, converts it to a ``tuple``,
    otherwise returns a tuple with a single element ``seq_or_elem``.
    """
    if seq_or_elem is None:
        return tuple()
    elif isinstance(seq_or_elem, str):
        return (seq_or_elem,)
    elif isinstance(seq_or_elem, collections.abc.Iterable):
        return tuple(seq_or_elem)
    else:
        return (seq_or_elem,)


def min_blocks(length, block):
    """
    Returns minimum number of blocks with length ``block``
    necessary to cover the array with non-zero length ``length``.
    """
    return (length - 1) // block + 1


def log2(num: int):
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

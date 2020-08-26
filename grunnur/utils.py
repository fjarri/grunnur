import collections
from functools import reduce
from typing import Iterable, Optional, Tuple, TypeVar, Type, Sequence, Mapping
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
    if isinstance(seq_or_elem, collections.abc.Iterable):
        return tuple(seq_or_elem)
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


def bounding_power_of_2(num: int) -> int:
    """
    Returns the minimal number of the form ``2**m`` such that it is greater or equal to ``n``.
    """
    if num == 1:
        return 1
    return 2 ** (log2(num - 1) + 1)


def prod(seq: Iterable):
    # Integer product. `numpy.prod` returns a float when given an empty sequence.
    return reduce(lambda x, y: x * y, seq, 1)


def string_matches_masks(
        s: str,
        include_masks: Optional[Sequence[str]]=None,
        exclude_masks: Optional[Sequence[str]]=None) -> bool:
    """
    Returns ``True`` if:
    1) ``s`` matches with at least one of regexps from ``include_masks`` (if there are any), and
    2) ``s`` doesn't match any regexp from ``exclude_masks``.
    """

    if include_masks is None:
        include_masks = []
    if exclude_masks is None:
        exclude_masks = []

    if len(include_masks) > 0:
        for include_mask in include_masks:
            if re.search(include_mask, s):
                break
        else:
            return False

    if len(exclude_masks) > 0:
        for exclude_mask in exclude_masks:
            if re.search(exclude_mask, s):
                return False

    return True


_T = TypeVar('_T')


def normalize_object_sequence(objs, expected_cls: Type[_T]) -> Tuple[_T, ...]:
    """
    For a sequence of objects, or a single object, checks that:
    1) the sequence is non-empty;
    2) all objects are different; and
    3) every object is an instance of ``expected_cls``,
    raising a ``ValueError`` otherwise.
    Returns a tuple of objects (1-tuple, if there was a single object).
    """

    objs = wrap_in_tuple(objs)

    if len(objs) == 0:
        raise ValueError("The iterable of base objects for the context cannot be empty")

    if not all_different(objs):
        raise ValueError("All base objects must be different")

    types = [type(obj) for obj in objs]
    if not all(issubclass(tp, expected_cls) for tp in types):
        raise TypeError(f"The iterable must contain only subclasses of {expected_cls}, got {types}")

    return objs


def max_factor(x: int, y: int) -> int:
    """
    Find the maximum `d` such that `x % d == 0` and `d <= y`.
    """
    if x <= y:
        return x

    result = 1
    for d in range(2, min(int(x**0.5), y) + 1):
        inv_d = x // d
        if inv_d * d == x:
            if inv_d <= y:
                return inv_d
            result = d

    return result


def find_local_size(
        global_size: Sequence[int],
        max_local_sizes: Sequence[int],
        max_total_local_size: int) -> Tuple[int, ...]:
    """
    Mimics the OpenCL local size finding algorithm.
    Returns the tuple of the same length as ``global_size``, with every element
    being a factor of the corresponding element of ``global_size``.
    Neither of the elements of ``local_size`` are greater then the corresponding element
    of ``max_local_sizes``, and their product is not greater than ``max_total_local_size``.
    """
    local_size = []
    for gs, mls in zip(global_size, max_local_sizes):
        d = max_factor(gs, min(mls, max_total_local_size))
        max_total_local_size //= d
        local_size.append(d)

    return tuple(local_size)


def get_launch_size(
        max_local_sizes: Tuple[int, ...],
        max_total_local_size: int,
        global_size: Tuple[int, ...],
        local_size: Optional[Tuple[int, ...]]=None) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Constructs the grid and block tuples to launch a CUDA kernel
    based on the provided global and local sizes.
    """

    if len(global_size) > len(max_local_sizes):
        raise ValueError("Global size has too many dimensions")

    if local_size is not None:
        if len(local_size) != len(global_size):
            raise ValueError("Global/local work sizes have differing dimensions")
        for gs, ls in zip(global_size, local_size):
            if gs % ls != 0:
                raise ValueError("Global sizes must be multiples of corresponding local sizes")
    else:
        local_size = find_local_size(global_size, max_local_sizes, max_total_local_size)

    grid_size = tuple(gs // ls for gs, ls in zip(global_size, local_size))
    return grid_size, local_size


_UPDATE_ERROR_TEMPLATE = "Cannot add an item '{name}' - it already exists in the old dictionary"

def update_dict(d: Mapping, new_d: Mapping, error_msg: str=_UPDATE_ERROR_TEMPLATE) -> dict:
    res = dict(d)
    for name, value in new_d.items():
        if name in d:
            raise ValueError(error_msg.format(name=name))
        res[name] = value
    return res

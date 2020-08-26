from __future__ import annotations

from collections import Counter
import itertools
from math import floor, ceil, sqrt
from typing import List, Dict, Iterable, Optional, Tuple, Generator, Sequence

from .template import Template
from .modules import Module, Snippet
from .utils import min_blocks, wrap_in_tuple, prod


TEMPLATE = Template.from_associated_file(__file__)


def factorize(num: int) -> List[int]:
    step = lambda x: 1 + (x<<2) - ((x>>1)<<1)
    maxq = int(floor(sqrt(num)))
    d = 1
    q = 2 if num % 2 == 0 else 3
    while q <= maxq and num % q != 0:
        q = step(d)
        d += 1
    return [q] + factorize(num // q) if q <= maxq else [num]


class PrimeFactors:
    """
    Contains a natural number's decomposition into prime factors.
    """

    def __init__(self, factors: Dict[int, int]):
        self.factors = factors

    @classmethod
    def decompose(cls, num: int) -> PrimeFactors:
        factors_list = factorize(num)
        factors = Counter(factors_list)
        return cls(dict(factors))

    def get_value(self) -> int:
        res = 1
        for pwr, exp in self.factors.items():
            res *= pwr ** exp
        return res

    def get_arrays(self) -> Tuple[List[int], List[int]]:
        bases = list(self.factors.keys())
        exponents = [self.factors[base] for base in bases]
        return bases, exponents

    def div_by(self, other: PrimeFactors) -> PrimeFactors:
        # assumes that `self` is a multiple of `other`
        factors = dict(self.factors)
        for o_pwr, o_exp in other.factors.items():
            factors[o_pwr] -= o_exp
            assert factors[o_pwr] >= 0 # sanity check
            if factors[o_pwr] == 0:
                del factors[o_pwr]
        return PrimeFactors(factors)

    def __eq__(self, other) -> bool:
        return self.factors == other.factors


def _get_decompositions(
        num_factors: PrimeFactors, parts: int) -> Generator[Tuple[int, ...], None, None]:
    """
    Helper recursive function for ``get_decompositions()``.
    Iterates over all possible decompositions of ``num_factors`` into ``parts`` factors.
    """
    if parts == 1:
        yield (num_factors.get_value(),)
        return

    bases, exponents = num_factors.get_arrays()
    for sub_exps in itertools.product(*[range(exp, -1, -1) for exp in exponents]):
        part_factors = PrimeFactors(
            dict(((pwr,sub_exp) for pwr, sub_exp in zip(bases, sub_exps) if sub_exp > 0)))
        part = part_factors.get_value()
        remainder = num_factors.div_by(part_factors)
        for decomp in _get_decompositions(remainder, parts - 1):
            yield (part,) + decomp


def get_decompositions(num: int, parts: int) -> Generator[Tuple[int, ...], None, None]:
    """
    Iterates over all possible decompositions of ``num`` into ``parts`` factors.
    """
    num_factors = PrimeFactors.decompose(num)
    return _get_decompositions(num_factors, parts)


def find_local_size_decomposition(
        global_size: Tuple[int, ...], flat_local_size: int, threshold: float=0.05) \
        -> Tuple[int, ...]:
    """
    Returns a tuple of the same size as ``global_size``,
    with the product equal to ``flat_local_size``,
    and minimal difference between ``product(global_size)``
    and ``product(min_blocks(gs, ls) * ls for gs, ls in zip(global_size, local_size))``
    (i.e. tries to minimize the amount of empty threads).
    """
    flat_global_size = prod(global_size)
    if flat_local_size >= flat_global_size:
        return global_size

    threads_num = prod(global_size)

    best_ratio = None
    best_local_size = None

    for local_size in get_decompositions(flat_local_size, len(global_size)):
        bounding_global_size = tuple(
            ls * min_blocks(gs, ls) for gs, ls in zip(global_size, local_size))
        empty_threads = prod(bounding_global_size) - threads_num
        ratio = empty_threads / threads_num

        # Stopping iteration early, because there may be a lot of elements to iterate over,
        # and we do not need the perfect solution.
        if ratio < threshold:
            return local_size

        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_local_size = local_size

    # This looks like the above loop can finish without setting `best_local_size`,
    # but providing flat_local_size <= product(global_size),
    # there is at least one decomposition (flat_local_size, 1, 1, ...).

    assert best_local_size is not None # sanity check to catch a possible bug early

    return best_local_size


def _group_dimensions(
        vdim: int, virtual_shape: Tuple[int, ...],
        adim: int, available_shape: Tuple[int, ...]) \
        -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """
    ``vdim`` and ``adim`` are used for the absolute addressing of dimensions during recursive calls.
    """
    if len(virtual_shape) == 0:
        return [], []

    vdim_group = 1 # number of currently grouped virtual dimensions
    adim_group = 1 # number of currently grouped available dimensions

    while True:
        # If we have more elements in the virtual group than there is in the available group,
        # extend the available group by one dimension.
        if prod(virtual_shape[:vdim_group]) > prod(available_shape[:adim_group]):
            adim_group += 1
            continue

        # If the remaining available dimensions cannot accommodate the remaining virtual dimensions,
        # we try to fit one more virtual dimension in the virtual group.
        if prod(virtual_shape[vdim_group:]) > prod(available_shape[adim_group:]):
            vdim_group += 1
            continue

        # If we are here, it means that:
        # 1) the current available group can accommodate the current virtual group;
        # 2) the remaining available dimensions can accommodate the remaining virtual dimensions.
        # This means we can make a recursive call now.

        # Attach any following trivial virtual dimensions (of size 1) to this group
        # This will help to avoid unassigned trivial dimensions with no real dimensions left.
        while vdim_group < len(virtual_shape) and virtual_shape[vdim_group] == 1:
            vdim_group += 1

        v_res = tuple(range(vdim, vdim + vdim_group))
        a_res = tuple(range(adim, adim + adim_group))

        v_remainder, a_remainder = _group_dimensions(
            vdim + vdim_group, virtual_shape[vdim_group:],
            adim + adim_group, available_shape[adim_group:])
        return [v_res] + v_remainder, [a_res] + a_remainder


def group_dimensions(
        virtual_shape: Tuple[int, ...],
        available_shape: Tuple[int, ...]) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """
    Determines which available dimensions the virtual dimensions can be embedded into.
    Prefers using the maximum number of available dimensions, since in that case
    less divisions will be needed to calculate virtual indices based on the real ones.

    Returns two lists, one of tuples with indices of grouped virtual dimensions, the other
    one of tuples with indices of corresponding group of available dimensions,
    such that for any group of virtual dimensions, the total number of elements they cover
    does not exceed the number of elements covered by the
    corresponding group of available dimensions.

    Dimensions are grouped in order, so tuples in both lists, if concatenated,
    give `(0 ... len(virtual_shape)-1)` and `(0 ... n)`, where `n < len(available_shape`.
    """
    assert prod(virtual_shape) <= prod(available_shape)
    return _group_dimensions(0, virtual_shape, 0, available_shape)


def find_bounding_shape(virtual_size: int, available_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Finds a tuple of the same length as ``available_shape``, with every element
    not greater than the corresponding element of ``available_shape``,
    and product not lower than ``virtual_size`` (trying to minimize that difference).
    """

    # TODO: in most cases it is possible to find such a tuple that `prod(result) == virtual_size`,
    # but the current algorithm does not gurantee it. Finding such a tuple would
    # eliminate some empty threads.

    assert virtual_size <= prod(available_shape)

    free_size = virtual_size
    free_dims = set(range(len(available_shape)))
    bounding_shape = [0] * len(available_shape)

    # The loop terminates, since `virtual_size` is guaranteed to fit into `available_shape`
    # (worst case scenario, the result will be just `available_shape` itself).
    while True:
        guess = ceil(free_size**(1/len(free_dims)))
        fixed_size = 1
        for fdim in list(free_dims):
            if guess > available_shape[fdim]:
                fixed_size *= available_shape[fdim]
                free_dims.remove(fdim)
                bounding_shape[fdim] = available_shape[fdim]
            else:
                bounding_shape[fdim] = guess

        if fixed_size == 1:
            break

        free_size = min_blocks(free_size, fixed_size)

    return tuple(bounding_shape)


class ShapeGroups:

    def __init__(self, virtual_shape, available_shape):
        # A mapping from a dimension in the virtual shape to a tuple of dimensions
        # in the real shape it uses (and possibly shares with other virtual dimensions).
        self.real_dims: Dict[int, Tuple[int, ...]] = {}

        # A mapping from a dimension in the virtual shape to a tuple of strides
        # used to get a flat index in the group of real dimensions it uses.
        self.real_strides: Dict[int, Tuple[int, ...]] = {}

        # A mapping from a dimension in the virtual shape to the stride that is used to extract it
        # from the flat index obtained from the corresponding group of real dimensions.
        self.virtual_strides: Dict[int, int] = {}

        # A mapping from a dimension in the virtual shape to the major dimension
        # (the one with the largest stride) in the group of virtual dimensions it belongs to
        # (the group includes all virtual dimensions using a certain subset of real dimensions).
        self.major_vdims: Dict[int, int] = {}

        # The actual shape used to enqueue the kernel.
        self.bounding_shape: Tuple[int, ...] = tuple()

        # A list of tuples `(threshold, stride_info)` used for skipping unused threads.
        # `stride_info` is a list of 2-tuples `(real_dim, stride)` used to construct
        # a flat index from several real dimensions, and then compare it with the threshold.
        self.skip_thresholds: List[Tuple[int, List[Tuple[int, int]]]] = []

        v_groups, a_groups = group_dimensions(virtual_shape, available_shape)

        for v_group, a_group in zip(v_groups, a_groups):
            virtual_subshape = virtual_shape[v_group[0]:v_group[-1]+1]
            virtual_subsize = prod(virtual_subshape)

            bounding_subshape = find_bounding_shape(
                virtual_subsize,
                available_shape[a_group[0]:a_group[-1]+1])

            self.bounding_shape += bounding_subshape

            if virtual_subsize < prod(bounding_subshape):
                strides = [(adim, prod(bounding_subshape[:i])) for i, adim in enumerate(a_group)]
                self.skip_thresholds.append((virtual_subsize, strides))

            for vdim in v_group:
                self.real_dims[vdim] = a_group
                self.real_strides[vdim] = tuple(
                    prod(self.bounding_shape[a_group[0]:adim]) for adim in a_group)
                self.virtual_strides[vdim] = prod(virtual_shape[v_group[0]:vdim])

                # The major virtual dimension (the one that does not require
                # modulus operation when extracting its index from the flat index)
                # is the last non-trivial one (not of size 1).
                # Modulus will not be optimized away by the compiler,
                # but we know that all threads outside of the virtual group will be
                # filtered out by VIRTUAL_SKIP_THREADS.
                for major_vdim in range(len(v_group) - 1, -1, -1):
                    if virtual_shape[v_group[major_vdim]] > 1:
                        break

                self.major_vdims[vdim] = v_group[major_vdim]


class VsizeModules:
    """
    A collection of modules passed to :py:class:`grunnur.StaticKernel`.
    Should be used instead of regular group/thread id functions.
    """

    local_id: Module
    """
    Provides the function ``VSIZE_T ${local_id}(int dim)``
    returning the local id of the current thread.
    """

    local_size: Module
    """
    Provides the function ``VSIZE_T ${local_size}(int dim)``
    returning the size of the current group.
    """

    group_id: Module
    """
    Provides the function ``VSIZE_T ${group_id}(int dim)``
    returning the group id of the current thread.
    """

    num_groups: Module
    """
    Provides the function ``VSIZE_T ${num_groups}(int dim)``
    returning the number of groups in dimension  ``dim``.
    """

    global_id: Module
    """
    Provides the function ``VSIZE_T ${global_id}(int dim)``
    returning the global id of the current thread.
    """

    global_size: Module
    """
    Provides the function ``VSIZE_T ${global_size}(int dim)``
    returning the global size along dimension ``dim``."""

    global_flat_id: Module
    """
    Provides the function ``VSIZE_T ${global_flat_id}()``
    returning the global id of the current thread with all dimensions flattened.
    """

    global_flat_size: Module
    """
    Provides the function ``VSIZE_T ${global_flat_size}()``.
    returning the global size of with all dimensions flattened.
    """

    begin: Module
    """
    Provides the statement ``${begin}`` that should be used
    at the start of a static kernel function.
    """

    def __init__(
            self,
            local_id,
            local_size,
            group_id,
            num_groups,
            global_id,
            global_size,
            global_flat_id,
            global_flat_size,
            begin):

        self.local_id = local_id
        self.local_size = local_size
        self.group_id = group_id
        self.num_groups = num_groups
        self.global_id = global_id
        self.global_size = global_size
        self.global_flat_id = global_flat_id
        self.global_flat_size = global_flat_size
        self.begin = begin

    def __process_modules__(self, process):
        return VsizeModules(
            local_id=process(self.local_id),
            local_size=process(self.local_size),
            group_id=process(self.group_id),
            num_groups=process(self.num_groups),
            global_id=process(self.global_id),
            global_size=process(self.global_size),
            global_flat_id=process(self.global_flat_id),
            global_flat_size=process(self.global_flat_size),
            begin=process(self.begin))

    @classmethod
    def from_shape_data(
            cls,
            virtual_global_size,
            virtual_local_size,
            bounding_global_size,
            virtual_grid_size,
            local_groups,
            grid_groups):

        local_id_mod = Module(
            TEMPLATE.get_def('local_id'),
            render_globals=dict(
                virtual_local_size=virtual_local_size,
                local_groups=local_groups))

        local_size_mod = Module(
            TEMPLATE.get_def('local_size'),
            render_globals=dict(
                virtual_local_size=virtual_local_size))

        group_id_mod = Module(
            TEMPLATE.get_def('group_id'),
            render_globals=dict(
                virtual_grid_size=virtual_grid_size,
                grid_groups=grid_groups))

        num_groups_mod = Module(
            TEMPLATE.get_def('num_groups'),
            render_globals=dict(
                virtual_grid_size=virtual_grid_size))

        global_id_mod = Module(
            TEMPLATE.get_def('global_id'),
            render_globals=dict(
                local_id_mod=local_id_mod,
                group_id_mod=group_id_mod,
                local_size_mod=local_size_mod))

        global_size_mod = Module(
            TEMPLATE.get_def('global_size'),
            render_globals=dict(
                virtual_global_size=virtual_global_size))

        global_flat_id_mod = Module(
            TEMPLATE.get_def('global_flat_id'),
            render_globals=dict(
                virtual_global_size=virtual_global_size,
                global_id_mod=global_id_mod,
                prod=prod))

        global_flat_size_mod = Module(
            TEMPLATE.get_def('global_flat_size'),
            render_globals=dict(
                global_size_mod=global_size_mod,
                virtual_global_size=virtual_global_size))

        skip_local_threads_mod = Module(
            TEMPLATE.get_def('skip_local_threads'),
            render_globals=dict(
                local_groups=local_groups))

        skip_groups_mod = Module(
            TEMPLATE.get_def('skip_groups'),
            render_globals=dict(
                grid_groups=grid_groups))

        skip_global_threads_mod = Module(
            TEMPLATE.get_def('skip_global_threads'),
            render_globals=dict(
                virtual_global_size=virtual_global_size,
                bounding_global_size=bounding_global_size,
                global_id_mod=global_id_mod))

        begin_static_kernel = Snippet(
            TEMPLATE.get_def('begin_static_kernel'),
            render_globals=dict(
                skip_local_threads_mod=skip_local_threads_mod,
                skip_groups_mod=skip_groups_mod,
                skip_global_threads_mod=skip_global_threads_mod))

        return cls(
            local_id=local_id_mod,
            local_size=local_size_mod,
            group_id=group_id_mod,
            num_groups=num_groups_mod,
            global_id=global_id_mod,
            global_size=global_size_mod,
            global_flat_id=global_flat_id_mod,
            global_flat_size=global_flat_size_mod,
            begin=begin_static_kernel)


class VirtualSizeError(Exception):
    """
    Raised when a virtual size cannot be found due to device limitations.
    """
    pass


class VirtualSizes:

    def __init__(
            self,
            max_total_local_size: int,
            max_local_sizes: Iterable[int],
            max_num_groups: Iterable[int],
            local_size_multiple: int,
            virtual_global_size: Sequence[int],
            virtual_local_size: Optional[Sequence[int]]=None):

        virtual_global_size = wrap_in_tuple(virtual_global_size)
        if virtual_local_size is not None:
            virtual_local_size = wrap_in_tuple(virtual_local_size)
            if len(virtual_local_size) != len(virtual_global_size):
                raise ValueError(
                    "Global size and local size must have the same number of dimensions")

        # Since the device uses column-major ordering of sizes, while we get
        # row-major ordered shapes, we invert our shapes
        # to facilitate internal handling.
        virtual_global_size = tuple(reversed(virtual_global_size))
        if virtual_local_size is not None:
            virtual_local_size = tuple(reversed(virtual_local_size))

        # In device parameters `max_total_local_size` is >= any of `max_local_sizes`,
        # but it can be overridden to get a kernel that uses less resources.
        max_local_sizes = [min(max_total_local_size, mls) for mls in max_local_sizes]

        assert max_total_local_size <= prod(max_local_sizes) # sanity check

        if virtual_local_size is None:
            # FIXME: we can obtain better results by taking occupancy into account here,
            # but for now we will assume that the more threads, the better.
            flat_global_size = prod(virtual_global_size)

            if flat_global_size < max_total_local_size:
                flat_local_size = flat_global_size
            else:
                # A sanity check - it would be very strange if a device had a local size multiple
                # so big you can't actually launch that many threads.
                assert max_total_local_size >= local_size_multiple
                flat_local_size = (
                    local_size_multiple * (max_total_local_size // local_size_multiple))

            # product(virtual_local_size) == flat_local_size <= max_total_local_size
            # Note: it's ok if local size elements are greater
            # than the corresponding global size elements as long as it minimizes the total
            # number of skipped threads.
            virtual_local_size = find_local_size_decomposition(virtual_global_size, flat_local_size)
        else:
            if prod(virtual_local_size) > max_total_local_size:
                raise VirtualSizeError(
                    f"Requested local size is greater than the maximum {max_total_local_size}")

        # Global and local sizes supported by CUDA or OpenCL restricted number of dimensions,
        # which may have limited size, so we need to pack our multidimensional sizes.

        virtual_grid_size = tuple(
            min_blocks(gs, ls) for gs, ls in zip(virtual_global_size, virtual_local_size))
        bounding_global_size = tuple(
            grs * ls for grs, ls in zip(virtual_grid_size, virtual_local_size))

        if prod(virtual_grid_size) > prod(max_num_groups):
            # Report the bounding size in reversed form so that it matches the provided
            # virtual global size.
            raise VirtualSizeError(
                f"Bounding global size {tuple(reversed(bounding_global_size))} is too large")

        local_groups = ShapeGroups(virtual_local_size, max_local_sizes)
        grid_groups = ShapeGroups(virtual_grid_size, max_num_groups)

        # These can be different lenghts because of expansion into multiple dimensions
        # find_bounding_shape() does.
        real_local_size = local_groups.bounding_shape
        real_grid_size = grid_groups.bounding_shape

        diff = len(real_local_size) - len(real_grid_size)
        real_local_size = real_local_size + (1,) * (-diff)
        real_grid_size = real_grid_size + (1,) * diff

        # This function will be used to translate between internal column-major vdims
        # and user-supplied row-major vdims.

        vsize_modules = VsizeModules.from_shape_data(
            virtual_local_size=virtual_local_size,
            virtual_global_size=virtual_global_size,
            bounding_global_size=bounding_global_size,
            virtual_grid_size=virtual_grid_size,
            local_groups=local_groups,
            grid_groups=grid_groups)

        # For testing purposes
        # (Note that these will have column-major order, same as real_global/local_size)
        self._virtual_local_size = virtual_local_size
        self._virtual_global_size = virtual_global_size
        self._bounding_global_size = bounding_global_size
        self._virtual_grid_size = virtual_grid_size

        self.real_local_size = real_local_size
        self.real_global_size = tuple(gs * ls for gs, ls in zip(real_grid_size, real_local_size))
        self.vsize_modules = vsize_modules

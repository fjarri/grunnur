import pytest

from grunnur._utils import min_blocks, prod
from grunnur._vsize import (
    PrimeFactors,
    ShapeGroups,
    VirtualSizeError,
    VirtualSizes,
    factorize,
    find_bounding_shape,
    find_local_size_decomposition,
    get_decompositions,
    group_dimensions,
)


def test_factorize() -> None:
    assert factorize(2**10) == [2] * 10
    assert factorize(2**3 * 3**2 * 7 * 13 * 251) == [2, 2, 2, 3, 3, 7, 13, 251]


def test_prime_factors() -> None:
    val = 2**3 * 3**2 * 7 * 13 * 251
    pf = PrimeFactors.decompose(val)

    assert pf.get_value() == val
    bases, exponents = pf.get_arrays()
    assert (list(bases), list(exponents)) == ([2, 3, 7, 13, 251], [3, 2, 1, 1, 1])

    val2 = 2**2 * 3 * 251
    pf2 = PrimeFactors.decompose(val2)

    assert pf.div_by(pf2) == PrimeFactors.decompose(val // val2)


def test_get_decompositions() -> None:
    val = 2**3 * 3**2 * 7 * 13
    parts = 3
    decomps = list(get_decompositions(val, parts))

    assert len({tuple(d) for d in decomps}) == len(decomps)
    assert all(prod(decomp) == val for decomp in decomps)
    assert all(len(decomp) == parts for decomp in decomps)


def test_find_local_size_decomposition() -> None:
    # Main pathway: the algorithm can find a local size
    # with less than `threshold` ratio of empty threads
    global_size = [128, 200, 300]
    flat_local_size = 1024
    threshold = 0.05
    local_size = find_local_size_decomposition(global_size, flat_local_size, threshold=threshold)

    full_global_size = [
        min_blocks(gs, ls) * ls for gs, ls in zip(global_size, local_size, strict=True)
    ]

    assert len(local_size) == len(global_size)
    assert prod(local_size) == flat_local_size
    assert (prod(full_global_size) - prod(global_size)) / prod(global_size) <= threshold

    # Now the algorithm can't find a local size with less than `threshold` empty threads,
    # and just returns the best one
    global_size = [100, 200, 300]
    flat_local_size = 1024
    threshold = 0.05
    local_size = find_local_size_decomposition(global_size, flat_local_size, threshold=threshold)

    assert len(local_size) == len(global_size)
    assert prod(local_size) == flat_local_size

    # Shortcut - if `product(global_size) == flat_local_size`, just return the global size
    global_size = [10, 10, 10]
    local_size = find_local_size_decomposition(global_size, 1000)
    assert local_size == global_size

    with pytest.raises(
        ValueError, match="The given local size does not fit in the given global size"
    ):
        find_local_size_decomposition([10, 10, 10], 1024)


def test_group_dimensions() -> None:
    # more virtual dimensions than available dimensions
    vgroups, agroups = group_dimensions((8, 8, 8, 8), (32, 32, 32))
    assert vgroups == [[0], [1, 2, 3]]
    assert agroups == [[0], [1, 2]]

    # less virtual dimensions than available dimensions
    vgroups, agroups = group_dimensions((8, 8), (32, 32, 32))
    assert vgroups == [[0], [1]]
    assert agroups == [[0], [1]]

    # some trivial dimensions (of size 1)
    vgroups, agroups = group_dimensions((8, 1, 1, 8, 1), (32, 32, 32))
    assert vgroups == [[0, 1, 2], [3, 4]]
    assert agroups == [[0], [1]]


def test_find_bounding_shape() -> None:
    assert find_bounding_shape(100, (8, 16)) == [8, 13]
    assert find_bounding_shape(120, (11, 11)) == [11, 11]


def test_shape_groups() -> None:
    sg = ShapeGroups((8, 8, 8, 8), (32, 32, 32))
    assert sg.real_dims == {0: [0], 1: [1, 2], 2: [1, 2], 3: [1, 2]}
    assert sg.real_strides == {0: [1], 1: [1, 23], 2: [1, 23], 3: [1, 23]}
    assert sg.virtual_strides == {0: 1, 1: 1, 2: 8, 3: 64}
    assert sg.major_vdims == {0: 0, 1: 3, 2: 3, 3: 3}
    assert sg.bounding_shape == [8, 23, 23]
    assert sg.skip_thresholds == [(512, [(1, 1), (2, 23)])]

    # A test to cover the case of all virtual dimensions being 1,
    # to test for a corner case in a search for the major vdim.
    sg = ShapeGroups((1, 1, 1), (32,))
    assert sg.real_dims == {0: [0], 1: [0], 2: [0]}
    assert sg.real_strides == {0: [1], 1: [1], 2: [1]}
    assert sg.virtual_strides == {0: 1, 1: 1, 2: 1}
    assert sg.major_vdims == {0: 0, 1: 0, 2: 0}
    assert sg.bounding_shape == [1]
    assert sg.skip_thresholds == []


def test_virtual_sizes() -> None:
    vs = VirtualSizes(
        max_total_local_size=1024,
        max_local_sizes=(1024, 1024, 64),
        max_num_groups=(2**31 - 1, 65536, 65536),
        local_size_multiple=16,
        virtual_global_size=(8, 16, 32, 60),
        virtual_local_size=None,
    )

    assert vs._virtual_local_size == [4, 32, 8, 1]
    assert vs._virtual_global_size == [60, 32, 16, 8]
    assert vs._bounding_global_size == [60, 32, 16, 8]
    assert vs._virtual_grid_size == [15, 1, 2, 8]
    assert vs.real_local_size == (4, 32, 8)
    assert vs.real_global_size == (60, 64, 64)

    # Test the case of small total global size (< max_total_local_size)
    vs = VirtualSizes(
        max_total_local_size=1024,
        max_local_sizes=(1024, 1024, 64),
        max_num_groups=(2**31 - 1, 65536, 65536),
        local_size_multiple=16,
        virtual_global_size=(8, 16),
        virtual_local_size=None,
    )

    assert vs._virtual_local_size == [16, 8]
    assert vs._virtual_global_size == [16, 8]
    assert vs._bounding_global_size == [16, 8]
    assert vs._virtual_grid_size == [1, 1]
    assert vs.real_local_size == (16, 8)
    assert vs.real_global_size == (16, 8)

    # Test the case of specified virtual_local_size
    vs = VirtualSizes(
        max_total_local_size=1024,
        max_local_sizes=(1024, 1024, 64),
        max_num_groups=(2**31 - 1, 65536, 65536),
        local_size_multiple=16,
        virtual_global_size=(123, 345),
        virtual_local_size=(13, 15),
    )

    assert vs._virtual_local_size == [15, 13]
    assert vs._virtual_global_size == [345, 123]
    assert vs._bounding_global_size == [345, 130]
    assert vs._virtual_grid_size == [23, 10]
    assert vs.real_local_size == (15, 13)
    assert vs.real_global_size == (345, 130)


def test_vsize_errors() -> None:
    with pytest.raises(
        ValueError, match="Global size and local size must have the same number of dimensions"
    ):
        VirtualSizes(
            max_total_local_size=1024,
            max_local_sizes=(1024, 1024, 64),
            max_num_groups=(2**31 - 1, 65536, 65536),
            local_size_multiple=16,
            # different number of dimensions in global and local sizes
            virtual_global_size=(8, 16, 32, 60),
            virtual_local_size=(4, 4, 4),
        )

    with pytest.raises(VirtualSizeError):
        VirtualSizes(
            max_total_local_size=1024,
            max_local_sizes=(1024, 1024, 64),
            max_num_groups=(2**31 - 1, 65536, 65536),
            local_size_multiple=16,
            virtual_global_size=(8, 16, 32, 60),
            # Requested total local size >= max_total_local_size
            virtual_local_size=(8, 8, 8, 8),
        )

    with pytest.raises(VirtualSizeError):
        VirtualSizes(
            max_total_local_size=1024,
            max_local_sizes=(1024, 1024, 64),
            max_num_groups=(4, 4, 4),
            local_size_multiple=16,
            # resulting number of groups is greater than total defined by `prod(max_num_groups)`
            virtual_global_size=(8, 16, 32, 60),
            virtual_local_size=(4, 4, 4, 4),
        )

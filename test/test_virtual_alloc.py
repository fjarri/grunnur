from collections import Counter
import weakref

import numpy
import pytest

from grunnur.adapter_base import APIID
from grunnur import Queue, MultiQueue, Program, Array, Context, Buffer
import grunnur.dtypes as dtypes
from grunnur.virtual_alloc import extract_dependencies, TrivialManager, ZeroOffsetManager


def mock_fill(buf, val):
    buf.kernel_arg._set(numpy.ones(buf.size, numpy.uint8) * numpy.uint8(val))


def mock_get(buf):
    res = numpy.empty(buf.size, numpy.uint8)
    buf.kernel_arg._get(res)
    return res


def allocate_test_set(virtual_alloc, allocate_callable):

    # Allocate virtual buffers with dependencies

    buffers_metadata = [
        (0, 2, []),
        (1, 5, [0]),
        (2, 5, [0]),
        (3, 10, [1, 2]),
        (4, 8, [1, 3]),
        (5, 5, []),
        (6, 3, [3]),
        (7, 10, [2, 5]),
        (8, 5, [1, 6]),
        (9, 20, [2, 4, 8]),
        ]

    buffers = {}
    for name, size, deps in buffers_metadata:
        allocator = virtual_alloc.allocator([buffers[d] for d in deps])
        buffers[name] = allocate_callable(allocator, size)

    return buffers_metadata, buffers


@pytest.mark.parametrize('pack', [False, True], ids=['no_pack', 'pack'])
def test_contract(context, valloc_cls, pack):

    dtype = numpy.int32

    program = Program(
        context,
        """
        KERNEL void fill(GLOBAL_MEM ${ctype} *dest, ${ctype} val)
        {
            const SIZE_T i = get_global_id(0);
            dest[i] = val;
        }
        """,
        render_globals=dict(ctype=dtypes.ctype(dtype)))
    fill = program.kernel.fill

    queue = Queue(context)
    virtual_alloc = valloc_cls(context)

    buffers_metadata, arrays = allocate_test_set(
        virtual_alloc,
        # Bump size to make sure buffer alignment doesn't hide any out-of-bounds access
        lambda allocator, size: Array.empty(context, size * 100, dtype, allocator=allocator))
    dependencies = {id_: deps for id_, _, deps in buffers_metadata}

    if pack:
        virtual_alloc.pack(queue)

    # Clear all arrays
    for name in sorted(arrays.keys()):
        fill(queue, arrays[name].shape, None, arrays[name], dtype(0))

    for i, name in enumerate(sorted(arrays.keys())):
        val = dtype(i + 1)
        fill(queue, arrays[name].shape, None, arrays[name], val)
        # According to the virtual allocator contract, the allocated buffer
        # will not intersect with the buffers from the specified dependencies.
        # So we're filling the buffer and checking that the dependencies did not change.
        for dep in dependencies[name]:
            assert (arrays[dep].get(queue) != val).all()


@pytest.mark.parametrize('pack', [False, True], ids=['no_pack', 'pack'])
def test_contract_mocked(mock_backend_pycuda, mock_context_pycuda, valloc_cls, pack):

    # Using PyCUDA backend here because it tracks the allocations.

    context = mock_context_pycuda
    queue = Queue(context)
    virtual_alloc = valloc_cls(context)

    buffers_metadata, buffers = allocate_test_set(
        virtual_alloc, lambda allocator, size: allocator(size, 0))

    for name, size, deps in buffers_metadata:
        # Virtual buffer size should be exactly as requested
        assert buffers[name].size == size
        # The real buffer behind the virtual buffer may be larger
        # (note that _size is only present in mocked DeviceAllocation)
        assert buffers[name].kernel_arg._size >= size

    if pack:
        virtual_alloc.pack(queue)

    # Clear all buffers
    for name, _, _ in buffers_metadata:
        mock_fill(buffers[name], -1)

    for i, metadata in enumerate(buffers_metadata):
        name, size, deps = metadata
        mock_fill(buffers[name], i)
        # According to the virtual allocator contract, the allocated buffer
        # will not intersect with the buffers from the specified dependencies.
        # So we're filling the buffer and checking that the dependencies did not change.
        for dep in deps:
            assert (mock_get(buffers[dep]) != i).all()

    # Check that after deleting virtual buffers all the real buffers are freed as well
    del buffers
    assert mock_backend_pycuda.allocation_count() == 0


def test_extract_dependencies(mock_context):

    queue = Queue(mock_context)
    virtual_alloc = TrivialManager(mock_context).allocator()

    vbuf = virtual_alloc(100, 0)
    varr = Array.empty(mock_context, 100, numpy.int32, allocator=virtual_alloc)

    assert extract_dependencies(vbuf) == {vbuf._buffer_adapter._id}
    assert extract_dependencies(varr) == {varr.data._buffer_adapter._id}
    assert extract_dependencies([vbuf, varr]) == {vbuf._buffer_adapter._id, varr.data._buffer_adapter._id}

    class DependencyHolder:
        __virtual_allocations__ = [vbuf, varr]

    assert extract_dependencies(DependencyHolder()) == {vbuf._buffer_adapter._id, varr.data._buffer_adapter._id}

    # An object not having any dependencies
    assert extract_dependencies(123) == set()


def check_statistics(buffers_metadata, stats):
    virtual_sizes = [size for _, size, _ in buffers_metadata]
    assert stats.virtual_size_total == sum(virtual_sizes)
    assert stats.virtual_num == len(virtual_sizes)
    assert stats.virtual_sizes == dict(Counter(virtual_sizes))

    assert stats.real_size_total <= sum(virtual_sizes)
    assert stats.real_num <= len(virtual_sizes)


def test_statistics(mock_context, valloc_cls):

    context = mock_context
    queue = Queue(context)
    virtual_alloc = valloc_cls(context)

    buffers_metadata, buffers = allocate_test_set(
        virtual_alloc, lambda allocator, size: allocator(size, 0))

    stats = virtual_alloc.statistics()
    check_statistics(buffers_metadata, stats)

    virtual_alloc.pack(queue)

    stats = virtual_alloc.statistics()
    check_statistics(buffers_metadata, stats)

    s = str(stats)
    assert str(stats.real_size_total) in s
    assert str(stats.real_num) in s
    assert str(stats.virtual_size_total) in s
    assert str(stats.virtual_num) in s


def test_non_existent_dependencies(mock_context, valloc_cls):
    context = mock_context
    virtual_alloc = valloc_cls(context)
    with pytest.raises(ValueError, match="12345"):
        virtual_alloc._allocate_virtual(100, {12345}, 0)


def test_virtual_buffer(mock_4_device_context_pyopencl):

    # Using an OpenCL mock context here because it keeps track of buffer migrations

    context = mock_4_device_context_pyopencl
    mqueue = MultiQueue(context)
    virtual_alloc = TrivialManager(context)

    allocator = virtual_alloc.allocator()
    vbuf = allocator(100, 0)

    assert vbuf.size == 100
    assert isinstance(vbuf.kernel_arg._buffer, bytes)
    assert vbuf.context is context
    with pytest.raises(NotImplementedError):
        vbuf.get_sub_region(0, 50)
    assert vbuf.offset == 0

    arr = numpy.arange(100).astype(numpy.uint8)
    vbuf.set(mqueue.queues[0], arr)
    res = numpy.empty_like(arr)
    vbuf.get(mqueue.queues[0], res)
    assert (arr == res).all()

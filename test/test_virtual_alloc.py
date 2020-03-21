from collections import Counter
import weakref

import numpy
import pytest

import grunnur.dtypes as dtypes
from grunnur.virtual_alloc import extract_dependencies, TrivialManager, ZeroOffsetManager


valloc_cls_fixture = pytest.mark.parametrize(
    'valloc_cls',
    [TrivialManager, ZeroOffsetManager],
    ids=['trivial', 'zero_offset'])


def mock_fill(buf, val):
    buf.backend_buffer[:buf.size] = val


def mock_get(buf):
    return buf.backend_buffer[:buf.size]


class MockBuffer:

    def __init__(self, context, id_, size):
        self.id = id_
        self.size = size
        self.context = context
        self.backend_buffer = numpy.empty(size, numpy.int64)


class MockContext:

    def __init__(self):
        self._allocation_id = 0
        self.allocations = weakref.WeakValueDictionary()

    def allocate(self, size):
        buf = MockBuffer(self, self._allocation_id, size)
        self.allocations[self._allocation_id] = buf
        self._allocation_id += 1
        return buf

    def make_queue(self):
        return MockQueue(self)


class MockQueue:

    def __init__(self, context):
        self.context = context

    def synchronize(self):
        pass


def test_extract_dependencies(monkeypatch):

    class MockVirtualBuffer:
        def __init__(self, id_):
            self._id = id_

    class MockArray:
        def __init__(self, id_):
            self.data = MockVirtualBuffer(id_)

    monkeypatch.setattr('grunnur.virtual_alloc.VirtualBuffer', MockVirtualBuffer)
    monkeypatch.setattr('grunnur.virtual_alloc.Array', MockArray)

    assert extract_dependencies(MockVirtualBuffer(123)) == {123}
    assert extract_dependencies(MockArray(123)) == {123}

    arrays = [MockArray(123), MockArray(123), MockArray(456)]
    assert extract_dependencies(arrays) == {123, 456}

    class DependencyHolder:
        __virtual_allocations__ = [MockArray(5), MockArray(6), MockArray(7)]

    assert extract_dependencies(DependencyHolder()) == {5, 6, 7}

    # An object not having any dependencies
    assert extract_dependencies(123) == set()


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


@valloc_cls_fixture
@pytest.mark.parametrize('pack', [False, True], ids=['no_pack', 'pack'])
def test_contract(valloc_cls, pack):

    context = MockContext()
    queue = context.make_queue()
    virtual_alloc = valloc_cls(queue)

    buffers_metadata, buffers = allocate_test_set(
        virtual_alloc, lambda allocator, size: allocator(size))

    for name, size, deps in buffers_metadata:
        # Virtual buffer size should be exactly as requested
        assert buffers[name].size == size
        # The real buffer behind the virtual buffer may be larger
        assert buffers[name].backend_buffer.size >= size

    if pack:
        virtual_alloc.pack()

    # Clear all buffers
    for name, _, _ in buffers_metadata:
        mock_fill(buffers[name], 0)

    for i, metadata in enumerate(buffers_metadata):
        name, size, deps = metadata
        val = i + 1
        mock_fill(buffers[name], val)
        # According to the virtual allocator contract, the allocated buffer
        # will not intersect with the buffers from the specified dependencies.
        # So we're filling the buffer and checking that the dependencies did not change.
        for dep in deps:
            assert (mock_get(buffers[dep]) != val).all()

    # Check that after deleting virtual buffers all the real buffers are freed as well
    del buffers
    assert len(context.allocations) == 0


@valloc_cls_fixture
@pytest.mark.parametrize('pack', [False, True], ids=['no_pack', 'pack'])
def test_contract_on_device(context, valloc_cls, pack):

    dtype = numpy.int32

    program = context.compile(
    """
    KERNEL void fill(GLOBAL_MEM ${ctype} *dest, ${ctype} val)
    {
        const SIZE_T i = get_global_id(0);
        dest[i] = val;
    }
    """, render_globals=dict(ctype=dtypes.ctype(dtype)))
    fill = program.fill

    queue = context.make_queue()
    virtual_alloc = valloc_cls(queue)

    buffers_metadata, arrays = allocate_test_set(
        virtual_alloc,
        # Bump size to make sure buffer alignment doesn't hide any out-of-bounds access
        lambda allocator, size: context.empty_array(queue, size * 100, dtype, allocator=allocator))
    dependencies = {id_: deps for id_, _, deps in buffers_metadata}

    if pack:
        virtual_alloc.pack()

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
            assert (arrays[dep].get() != val).all()


def check_statistics(buffers_metadata, stats):
    virtual_sizes = [size for _, size, _ in buffers_metadata]
    assert stats.virtual_size_total == sum(virtual_sizes)
    assert stats.virtual_num == len(virtual_sizes)
    assert stats.virtual_sizes == dict(Counter(virtual_sizes))

    assert stats.real_size_total <= sum(virtual_sizes)
    assert stats.real_num <= len(virtual_sizes)


@valloc_cls_fixture
def test_statistics(valloc_cls):

    context = MockContext()
    queue = context.make_queue()
    virtual_alloc = valloc_cls(queue)

    buffers_metadata, buffers = allocate_test_set(
        virtual_alloc, lambda allocator, size: allocator(size))

    stats = virtual_alloc.statistics()
    check_statistics(buffers_metadata, stats)

    virtual_alloc.pack()

    stats = virtual_alloc.statistics()
    check_statistics(buffers_metadata, stats)

    s = str(stats)
    assert str(stats.real_size_total) in s
    assert str(stats.real_num) in s
    assert str(stats.virtual_size_total) in s
    assert str(stats.virtual_num) in s


@valloc_cls_fixture
def test_non_existent_dependencies(valloc_cls):
    context = MockContext()
    queue = context.make_queue()
    virtual_alloc = valloc_cls(queue)
    with pytest.raises(ValueError, match="12345"):
        virtual_alloc._allocate_virtual(100, {12345})


def test_virtual_buffer():
    context = MockContext()
    queue = context.make_queue()
    virtual_alloc = TrivialManager(queue)

    allocator = virtual_alloc.allocator()
    vbuf = allocator(100)

    assert vbuf.size == 100
    assert isinstance(vbuf.backend_buffer, numpy.ndarray)
    assert vbuf.context is context
    with pytest.raises(NotImplementedError):
        vbuf.get_sub_region(0, 50)
    assert vbuf.offset == 0


@valloc_cls_fixture
def test_continuous_pack(valloc_cls):
    context = MockContext()
    queue = context.make_queue()
    virtual_alloc_ref = valloc_cls(queue, pack_on_alloc=False, pack_on_free=False)
    virtual_alloc = valloc_cls(queue, pack_on_alloc=True, pack_on_free=True)

    _, buffers_ref = allocate_test_set(
        virtual_alloc_ref, lambda allocator, size: allocator(size))

    _, buffers = allocate_test_set(
        virtual_alloc, lambda allocator, size: allocator(size))

    virtual_alloc_ref.pack()
    assert virtual_alloc_ref.statistics() == virtual_alloc.statistics()

    for id_ in (2, 4, 6, 8):
        del buffers_ref[id_]
        del buffers[id_]

    virtual_alloc_ref.pack()
    assert virtual_alloc_ref.statistics() == virtual_alloc.statistics()

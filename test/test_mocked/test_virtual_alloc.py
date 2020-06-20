from collections import Counter
import weakref

import numpy
import pytest

from grunnur.adapter_base import APIID
from grunnur import Queue, Program, Array, Context, Buffer
import grunnur.dtypes as dtypes
from grunnur.virtual_alloc import extract_dependencies, TrivialManager, ZeroOffsetManager


def mock_fill(buf, val):
    buf.kernel_arg[:buf.size] = val


def mock_get(buf):
    return buf.kernel_arg[:buf.size]


class MockAPIAdapter:
    id = APIID("mock")


class MockPlatformAdapter:
    api_adapter = MockAPIAdapter()
    name = "mock"
    platform_idx = 0


class MockDeviceAdapter:
    platform_adapter = MockPlatformAdapter()
    name = "mock"
    device_idx = 0


class MockBufferAdapter:

    def __init__(self, context, id_, size):
        self.id = id_
        self.size = size
        self.context = context
        self.kernel_arg = numpy.empty(size, numpy.uint8)
        self.device_idx = None

    def set(self, queue_adapter, device_idx, host_array, async_=False):
        self.kernel_arg[:] = host_array[:self.size]

    def get(self, queue_adapter, device_idx, host_array, async_=False):
        host_array[:self.size] = self.kernel_arg

    def migrate(self, queue_adapter, device_idx):
        self.device_idx = device_idx


class MockContextAdapter:

    def __init__(self):
        self._allocation_id = 0
        self.allocations = weakref.WeakValueDictionary()
        self.device_adapters = [MockDeviceAdapter()]

    def allocate(self, size):
        buf = MockBufferAdapter(self, self._allocation_id, size)
        self.allocations[self._allocation_id] = buf
        self._allocation_id += 1
        return buf

    def make_queue_adapter(self, device_idxs):
        return MockQueueAdapter(self)


class MockQueueAdapter:

    def __init__(self, context_adapter):
        self.context_adapter = context_adapter
        self.device_adapters = {
            device_idx: context_adapter.device_adapters[device_idx]
            for device_idx in range(len(context_adapter.device_adapters))}

    def synchronize(self):
        pass


def test_extract_dependencies(monkeypatch):

    class MockVirtualBufferAdapter:
        def __init__(self, id_):
            self._id = id_

    class MockArray:
        def __init__(self, id_):
            self.data = Buffer(None, MockVirtualBufferAdapter(id_))

    monkeypatch.setattr('grunnur.virtual_alloc.VirtualBufferAdapter', MockVirtualBufferAdapter)
    monkeypatch.setattr('grunnur.virtual_alloc.Array', MockArray)

    assert extract_dependencies(MockVirtualBufferAdapter(123)) == {123}
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


@pytest.mark.parametrize('pack', [False, True], ids=['no_pack', 'pack'])
def test_contract(valloc_cls, pack):

    context = Context(MockContextAdapter())
    queue = Queue.from_device_idxs(context)
    virtual_alloc = valloc_cls(queue)

    buffers_metadata, buffers = allocate_test_set(
        virtual_alloc, lambda allocator, size: allocator(size))

    for name, size, deps in buffers_metadata:
        # Virtual buffer size should be exactly as requested
        assert buffers[name].size == size
        # The real buffer behind the virtual buffer may be larger
        assert buffers[name].kernel_arg.size >= size

    if pack:
        virtual_alloc.pack()

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
    assert len(context._context_adapter.allocations) == 0


def check_statistics(buffers_metadata, stats):
    virtual_sizes = [size for _, size, _ in buffers_metadata]
    assert stats.virtual_size_total == sum(virtual_sizes)
    assert stats.virtual_num == len(virtual_sizes)
    assert stats.virtual_sizes == dict(Counter(virtual_sizes))

    assert stats.real_size_total <= sum(virtual_sizes)
    assert stats.real_num <= len(virtual_sizes)


def test_statistics(valloc_cls):

    context = Context(MockContextAdapter())
    queue = Queue.from_device_idxs(context)
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


def test_non_existent_dependencies(valloc_cls):
    context = Context(MockContextAdapter())
    queue = Queue.from_device_idxs(context)
    virtual_alloc = valloc_cls(queue)
    with pytest.raises(ValueError, match="12345"):
        virtual_alloc._allocate_virtual(100, {12345})


def test_virtual_buffer():
    context = Context(MockContextAdapter())
    queue = Queue.from_device_idxs(context)
    virtual_alloc = TrivialManager(queue)

    allocator = virtual_alloc.allocator()
    vbuf = allocator(100)

    assert vbuf.size == 100
    assert isinstance(vbuf.kernel_arg, numpy.ndarray)
    assert vbuf.context is context
    with pytest.raises(NotImplementedError):
        vbuf.get_sub_region(0, 50)
    assert vbuf.offset == 0

    arr = numpy.arange(100).astype(numpy.uint8)
    vbuf.set(queue, arr)
    res = numpy.empty_like(arr)
    vbuf.get(queue, res)
    assert (arr == res).all()

    assert vbuf._buffer_adapter._real_buffer_adapter.device_idx is None
    vbuf.migrate(queue, 0)
    assert vbuf._buffer_adapter._real_buffer_adapter.device_idx == 0


def test_continuous_pack(valloc_cls):
    context = Context(MockContextAdapter())
    queue = Queue.from_device_idxs(context)
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

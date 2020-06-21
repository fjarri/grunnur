import numpy

from grunnur import Buffer, Queue


def test_allocate(context):
    length = 100
    dtype = numpy.dtype('int32')
    size = length * dtype.itemsize

    arr = numpy.arange(length).astype(dtype)

    buf = Buffer.allocate(context, size)
    assert buf.size == size
    assert buf.offset == 0

    queue = Queue.from_device_idxs(context)
    buf.set(queue, arr)

    res = numpy.empty_like(arr)
    buf.get(queue, res)

    assert (res == arr).all()


def test_migrate(multi_device_context):
    context = multi_device_context

    length = 100
    dtype = numpy.dtype('int32')
    size = length * dtype.itemsize

    arr = numpy.arange(length).astype(dtype)

    buf = Buffer.allocate(context, size)
    assert buf.size == size
    assert buf.offset == 0

    queue0 = Queue.from_device_idxs(context, [0])
    queue1 = Queue.from_device_idxs(context, [1])

    res = numpy.empty_like(arr)

    buf.bind(0)
    buf.set(queue0, arr)
    queue0.synchronize()

    buf1 = buf.get_sub_region(0, size)
    buf1.bind(1)
    buf1.migrate(queue1)

    buf1.get(queue1, res)
    queue1.synchronize()
    assert (res == arr).all()

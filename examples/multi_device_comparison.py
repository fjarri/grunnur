import time
import numpy
from grunnur import opencl_api as api
from grunnur import Context, Queue, Program, Array, MultiDevice


src = """
KERNEL void sum(GLOBAL_MEM unsigned long *a, int pwr)
{
    unsigned long m = 4294967231;
    int gid = get_global_id(0);

    int idx = gid;
    unsigned long x = a[idx];
    unsigned long res = x;
    for (int i = 1; i < pwr; i++)
    {
        res *= x;
        res %= m;
    }
    a[idx] = res;
}
"""


def calc_ref(x, pwr):
    m = numpy.uint64(2**32-65)
    res = x.copy()
    for i in range(1, pwr):
        res *= x
        res %= m
        #res += 1
    return res


def test_single_device(device_idx, full_len, benchmark=False):
    pwr = 50

    a = numpy.arange(full_len).astype(numpy.uint64)

    context = Context.from_devices([api.platforms[0].devices[device_idx]])
    queue = Queue.on_all_devices(context)

    program = Program(context, src)
    a_dev = Array.from_host(queue, a)

    gs = full_len
    ls = None

    queue.synchronize()
    t1 = time.time()
    program.kernel.sum(queue, gs, ls, a_dev, numpy.int32(pwr))
    queue.synchronize()
    t2 = time.time()
    print(f"Single device time (device {device_idx}):", t2 - t1)

    a_res = a_dev.get()

    if not benchmark:
        a_ref = calc_ref(a, pwr)
        assert (a_ref == a_res).all()


def test_multi_device(device_idxs, full_len, benchmark=False):

    pwr = 50

    a = numpy.arange(full_len).astype(numpy.uint64)

    context = Context.from_devices([api.platforms[0].devices[device_idx] for device_idx in device_idxs])
    queue = Queue.on_all_devices(context)

    program = Program(context, src)
    a_dev = Array.from_host(queue, a)

    a_dev_1 = a_dev.single_device_view(0)[:full_len//2]
    a_dev_2 = a_dev.single_device_view(1)[full_len//2:]

    gs = full_len // 2
    ls = None

    queue.synchronize()
    t1 = time.time()
    program.kernel.sum(queue, gs, ls, MultiDevice(a_dev_1, a_dev_2), numpy.int32(pwr), device_idxs=[0, 1])
    queue.synchronize()
    t2 = time.time()
    print(f"Multidevice time (devices {device_idxs}):", t2 - t1)

    a_res = a_dev.get()

    if not benchmark:
        a_ref = calc_ref(a, pwr)
        assert (a_ref == a_res).all()

test_single_device(1, 2**20)
test_single_device(2, 2**20)
test_multi_device([1, 2], 2**20)

test_single_device(1, 2**24, benchmark=True)
test_single_device(2, 2**24, benchmark=True)
test_multi_device([1, 2], 2**24, benchmark=True)

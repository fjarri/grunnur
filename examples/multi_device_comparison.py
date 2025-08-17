"""
An example comparing performance of executing a kernel on one device
versus multiple devices simultaneously.
"""

# ruff: noqa: T201, S101

import time
from collections.abc import Sequence
from typing import Any

import numpy
from numpy.typing import NDArray

from grunnur import API, Array, Context, MultiArray, MultiQueue, Program, Queue

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


def calc_ref(x: NDArray[Any], pwr: int) -> NDArray[Any]:
    """Reference function for the kernel."""
    m = numpy.uint64(2**32 - 65)
    res = x.copy()
    for _ in range(1, pwr):
        res *= x
        res %= m
    return res


def run_single_device(device_idx: int, full_len: int, *, benchmark: bool = False) -> None:
    """Runs the kernel on a single device."""
    api = API.opencl()

    pwr = 50

    a = numpy.arange(full_len).astype(numpy.uint64)

    context = Context.from_devices([api.platforms[0].devices[device_idx]])
    queue = Queue(context.device)

    program = Program([context.device], src)
    a_dev = Array.from_host(queue, a)

    queue.synchronize()
    t1 = time.time()
    program.kernel.sum(queue, [full_len], None, a_dev, numpy.int32(pwr))
    queue.synchronize()
    t2 = time.time()
    print(f"Single device time (device {device_idx}):", t2 - t1)

    a_res = a_dev.get(queue)

    if not benchmark:
        a_ref = calc_ref(a, pwr)
        assert (a_ref == a_res).all()


def run_multi_device(device_idxs: Sequence[int], full_len: int, *, benchmark: bool = False) -> None:
    """Runs the kernel on a multiple devices."""
    api = API.opencl()

    pwr = 50

    a = numpy.arange(full_len).astype(numpy.uint64)

    context = Context.from_devices(
        [api.platforms[0].devices[device_idx] for device_idx in device_idxs]
    )
    mqueue = MultiQueue.on_devices(context.devices)

    program = Program(context.devices, src)
    a_dev = MultiArray.from_host(mqueue, a)

    mqueue.synchronize()
    t1 = time.time()
    program.kernel.sum(mqueue, a_dev.shapes, None, a_dev, numpy.int32(pwr))
    mqueue.synchronize()
    t2 = time.time()
    print(f"Multidevice time (devices {device_idxs}):", t2 - t1)

    a_res = a_dev.get(mqueue)

    if not benchmark:
        a_ref = calc_ref(a, pwr)
        assert (a_ref == a_res).all()


run_single_device(0, 2**20)
run_single_device(1, 2**20)
run_multi_device([0, 1], 2**20)

run_single_device(0, 2**24, benchmark=True)
run_single_device(1, 2**24, benchmark=True)
run_multi_device([0, 1], 2**24, benchmark=True)

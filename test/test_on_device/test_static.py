import pytest
import numpy

from grunnur import StaticKernel, Queue, Array
from grunnur import CUDA_API_ID, OPENCL_API_ID


src = """
KERNEL void multiply(GLOBAL_MEM int *dest, GLOBAL_MEM int *a, GLOBAL_MEM int *b)
{
    ${static.begin};
    const int i = ${static.global_id}(0);
    const int j = ${static.global_id}(1);
    const int idx = ${static.global_flat_id}();
    dest[idx] = a[i] * b[j];
}
"""


def test_compile_static(context):

    a = numpy.arange(11).astype(numpy.int32)
    b = numpy.arange(15).astype(numpy.int32)
    ref = numpy.outer(a, b)

    queue = Queue.from_device_idxs(context)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(queue, (11, 15), numpy.int32)

    multiply = StaticKernel(context, src, 'multiply', (11, 15))
    multiply(queue, res_dev, a_dev, b_dev)

    res = res_dev.get()

    assert (res == ref).all()


def test_compile_static_multi_device(multi_device_context):

    context = multi_device_context

    a = numpy.arange(22).astype(numpy.int32)
    b = numpy.arange(15).astype(numpy.int32)
    ref = numpy.outer(a, b)

    queue = Queue.from_device_idxs(context, device_idxs=[0, 1])

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(queue, (22, 15), numpy.int32)

    a_dev_1 = a_dev.single_device_view(0)[:11]
    a_dev_2 = a_dev.single_device_view(1)[11:]

    b_dev_1 = b_dev.single_device_view(0)[:]
    b_dev_2 = b_dev.single_device_view(0)[:]

    res_dev_1 = res_dev.single_device_view(0)[:11,:]
    res_dev_2 = res_dev.single_device_view(1)[11:,:]

    multiply = StaticKernel(context, src, 'multiply', (11, 15), device_idxs=[0, 1])
    multiply(queue, [res_dev_1, res_dev_2], [a_dev_1, a_dev_2], [b_dev_1, b_dev_2])

    res = res_dev.get()
    correct_result = (res == ref).all()

    device_names = [device.name for device in queue.devices.values()]
    expected_to_fail = (
        context.api.id == OPENCL_API_ID and
        'Apple' in context.platform.name and
        any('GeForce' in name for name in device_names) and
        not all('GeForce' in name for name in device_names))

    if expected_to_fail:
        if correct_result:
            raise Exception("This test was expected to fail on this configuration.")
        else:
            pytest.xfail(
                "Multi-device OpenCL contexts on an Apple platform with one device being a GeForce "
                "don't work correctly (the kernel invocation on GeForce is ignored).")

    assert correct_result

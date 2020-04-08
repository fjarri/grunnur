import numpy

import pytest

from grunnur import CUDA_API_ID, OPENCL_API_ID, Program, Queue, Array
from grunnur.template import Template


src_opencl = """
__kernel void multiply(__global int *dest, __global int *a, __global int *b)
{
    const int i = get_global_id(0);
    dest[i] = a[i] * b[i];
}
"""

src_cuda = """
extern "C" __global__ void multiply(int *dest, int *a, int *b)
{
    const int i = threadIdx.x;
    dest[i] = a[i] * b[i];
}
"""

src_generic = """
KERNEL void multiply(GLOBAL_MEM int *dest, GLOBAL_MEM int *a, GLOBAL_MEM int *b)
{
    const int i = get_global_id(0);
    dest[i] = a[i] * b[i];
}
"""


@pytest.mark.parametrize('no_prelude', [False, True], ids=["with-prelude", "no-prelude"])
def test_compile(context, no_prelude):

    if no_prelude:
        src = src_cuda if context.api.id == CUDA_API_ID else src_opencl
    else:
        src = src_generic

    program = Program(context, src, no_prelude=no_prelude)
    a = numpy.arange(16).astype(numpy.int32)
    b = numpy.arange(16).astype(numpy.int32) + 1
    ref = a * b

    queue = Queue.from_device_nums(context)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(queue, 16, numpy.int32)

    program.multiply(queue, 16, None, res_dev, a_dev, b_dev)

    res = res_dev.get()

    assert (res == ref).all()


def test_compile_multi_device(multi_device_context):

    context = multi_device_context

    program = Program(context, src_generic)
    a = numpy.arange(16).astype(numpy.int32)
    b = numpy.arange(16).astype(numpy.int32) + 1
    ref = a * b

    queue = Queue.from_device_nums(context, device_nums=[0, 1])

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)
    res_dev = Array.empty(queue, 16, numpy.int32)

    a_dev_1 = a_dev.single_device_view(0)[:8]
    a_dev_2 = a_dev.single_device_view(1)[8:]

    b_dev_1 = b_dev.single_device_view(0)[:8]
    b_dev_2 = b_dev.single_device_view(1)[8:]

    res_dev_1 = res_dev.single_device_view(0)[:8]
    res_dev_2 = res_dev.single_device_view(1)[8:]

    program.multiply(
        queue, 8, None, [res_dev_1, res_dev_2], [a_dev_1, a_dev_2], [b_dev_1, b_dev_2])

    queue.synchronize()

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


src_constant_mem = """
KERNEL void copy_from_cm(
    GLOBAL_MEM int *dest
#ifdef GRUNNUR_OPENCL_API
    , CONSTANT_MEM_ARG int *cm1
    , CONSTANT_MEM_ARG int *cm2
#endif
    )
{
    const int i = get_global_id(0);
    dest[i] = cm1[i] + cm2[i];
}
"""


def test_constant_memory(context):

    cm1 = numpy.arange(16).astype(numpy.int32)
    cm2 = numpy.arange(16).astype(numpy.int32) * 2

    queue = Queue.from_device_nums(context)

    cm1_dev = Array.from_host(queue, cm1)
    cm2_dev = Array.from_host(queue, cm2)
    res_dev = Array.empty(queue, 16, numpy.int32)

    if context.api.id == CUDA_API_ID:
        program = Program(context, src_constant_mem, constant_arrays=dict(cm1=cm1, cm2=cm2))
        program.set_constant_array('cm1', cm1_dev) # setting from a device array
        program.set_constant_array('cm2', cm2, queue=queue) # setting from a host array
        program.copy_from_cm(queue, 16, None, res_dev)
    else:
        program = Program(context, src_constant_mem)
        program.copy_from_cm(queue, 16, None, res_dev, cm1_dev, cm2_dev)

    res = res_dev.get()

    assert (res == cm1 + cm2).all()

import pytest
import numpy

from grunnur import StaticKernel, Queue, Array, MultiDevice
from grunnur import cuda_api_id, opencl_api_id

from ..mock_base import MockKernel, MockDefTemplate, MockDefTemplate
from ..test_on_device.test_program import _test_constant_memory


SRC = """
KERNEL void multiply(GLOBAL_MEM int *dest, GLOBAL_MEM int *a, GLOBAL_MEM int *b)
{
    ${static.begin};
    const int i = ${static.global_id}(0);
    const int j = ${static.global_id}(1);
    const int idx = ${static.global_flat_id}();
    dest[idx] = a[i] * b[j];
}
"""


def _test_compile_static(context, is_mocked):

    if is_mocked:
        kernel = MockKernel('multiply', [None, None, None], max_total_local_sizes={0: 1024})
        src = MockDefTemplate(kernels=[kernel])
    else:
        src = SRC

    a = numpy.arange(11).astype(numpy.int32)
    b = numpy.arange(15).astype(numpy.int32)
    ref = numpy.outer(a, b)

    queue = Queue.on_all_devices(context)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(queue, (11, 15), numpy.int32)

    multiply = StaticKernel(queue, src, 'multiply', (11, 15))
    multiply(res_dev, a_dev, b_dev)

    res = res_dev.get()

    if not is_mocked:
        assert (res == ref).all()


def test_compile_static(context):
    _test_compile_static(context=context, is_mocked=False)


def _test_compile_static_multi_device(context, is_mocked):

    if is_mocked:
        kernel = MockKernel(
            'multiply', [None, None, None], max_total_local_sizes={0: 1024, 1: 512})
        src = MockDefTemplate(kernels=[kernel])
    else:
        src = SRC

    a = numpy.arange(22).astype(numpy.int32)
    b = numpy.arange(15).astype(numpy.int32)
    ref = numpy.outer(a, b)

    queue = Queue.on_device_idxs(context, device_idxs=[0, 1])

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(queue, (22, 15), numpy.int32)

    a_dev_1 = a_dev.single_device_view(0)[:11]
    a_dev_2 = a_dev.single_device_view(1)[11:]

    b_dev_1 = b_dev.single_device_view(0)[:]
    b_dev_2 = b_dev.single_device_view(0)[:]

    res_dev_1 = res_dev.single_device_view(0)[:11,:]
    res_dev_2 = res_dev.single_device_view(1)[11:,:]

    multiply = StaticKernel(queue, src, 'multiply', (11, 15), device_idxs=[0, 1])
    multiply(
        MultiDevice(res_dev_1, res_dev_2),
        MultiDevice(a_dev_1, a_dev_2),
        MultiDevice(b_dev_1, b_dev_2))

    res = res_dev.get()

    if is_mocked:
        correct_result = True
        expected_to_fail = False
    else:
        correct_result = (res == ref).all()
        device_names = [device.name for device in queue.devices.values()]
        expected_to_fail = (
            context.api.id == opencl_api_id() and
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


def test_compile_static_multi_device(multi_device_context):
    _test_compile_static_multi_device(context=multi_device_context, is_mocked=False)


def test_constant_memory(context):
    _test_constant_memory(context=context, is_mocked=False, is_static=True)

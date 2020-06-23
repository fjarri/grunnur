import numpy
import pytest

from grunnur import API, Context, Queue, Array, Program, CompilationError, CUDA_API_ID, StaticKernel

from ..mock_base import MockDefTemplate, MockKernel
from ..test_on_device.test_program import (
    _test_compile,
    _test_constant_memory,
    _test_compilation_error,
    _test_compile_multi_device,
    _test_keep,
    )


@pytest.mark.parametrize('no_prelude', [False, True], ids=["with-prelude", "no-prelude"])
def test_compile(mock_context, no_prelude):
    _test_compile(
        context=mock_context,
        no_prelude=no_prelude,
        is_mocked=True)


def test_constant_memory(mock_context):
    _test_constant_memory(context=mock_context, is_mocked=True, is_static=False)


def test_compilation_error(mock_context, capsys):
    _test_compilation_error(context=mock_context, capsys=capsys, is_mocked=True)


def test_compile_multi_device(mock_4_device_context):
    _test_compile_multi_device(
        context=mock_4_device_context,
        device_idxs=[2, 1],
        is_mocked=True)


def test_wrong_device_idxs(mock_4_device_context):
    src = MockDefTemplate(kernels=[MockKernel('multiply', [None])])

    context = mock_4_device_context
    program = Program(context, src, device_idxs=[0, 1])
    queue = Queue.from_device_idxs(context, device_idxs=[2, 1])
    res_dev = Array.empty(queue, 16, numpy.int32)

    # Using all the queue's devices (1, 2)
    with pytest.raises(ValueError, match="This kernel's program was not compiled for devices"):
        program.multiply(queue, 8, None, res_dev)

    # Explicit device_idxs
    with pytest.raises(ValueError, match="This kernel's program was not compiled for devices"):
        program.multiply(queue, 8, None, res_dev, device_idxs=[0, 2])


def test_set_constant_array_errors(mock_4_device_context, mock_backend):

    context = mock_4_device_context

    api = API.from_api_id(mock_backend.api_id)
    other_context = Context.from_criteria(api)
    other_queue = Queue.from_device_idxs(other_context)
    other_context.deactivate()

    cm1 = numpy.arange(16).astype(numpy.int32)
    src = MockDefTemplate(kernels=[
        MockKernel(
            'kernel', [], max_total_local_sizes={0: 1024, 1: 1024, 2: 1024, 3: 1024})],
            constant_mem={'cm1': cm1.size * cm1.dtype.itemsize})
    queue = Queue.from_device_idxs(context)

    if context.api.id == CUDA_API_ID:
        program = Program(context, src, constant_arrays=dict(cm1=cm1))

        with pytest.raises(
                ValueError,
                match="The provided queue must belong to the same context as this program uses"):
            program.set_constant_array(other_queue, 'cm1', cm1)

        with pytest.raises(TypeError, match="Unsupported array type"):
            program.set_constant_array(queue, 'cm1', [1])

        with pytest.raises(ValueError, match="Incorrect size of the constant buffer;"):
            program.set_constant_array(queue, 'cm1', cm1[:8])

        with pytest.raises(TypeError, match="Unknown constant array metadata type"):
            program = Program(context, src, constant_arrays=dict(cm1=1), device_idxs=[0, 1, 2])

        program = Program(context, src, constant_arrays=dict(cm1=cm1), device_idxs=[0, 1, 2])
        queue3 = Queue.from_device_idxs(context, device_idxs=[3])

        with pytest.raises(
                ValueError,
                match="The provided queue must include the device this program uses"):
            program.set_constant_array(queue3, 'cm1', cm1)

    else:
        with pytest.raises(ValueError, match="Compile-time constant arrays are only supported by CUDA API"):
            program = Program(context, src, constant_arrays=dict(cm1=cm1))

        program = Program(context, src)
        with pytest.raises(ValueError, match="Constant arrays are only supported for CUDA API"):
            program.set_constant_array(queue, 'cm1', cm1)

        with pytest.raises(ValueError, match="Compile-time constant arrays are only supported by CUDA API"):
            sk = StaticKernel(context, src, 'kernel', 1024, constant_arrays=dict(cm1=cm1))

        sk = StaticKernel(context, src, 'kernel', 1024)
        with pytest.raises(ValueError, match="Constant arrays are only supported for CUDA API"):
            sk.set_constant_array(queue, 'cm1', cm1)


def test_max_total_local_sizes(mock_backend):
    mock_backend.add_devices(["Device1", "Device2 - tag", "Device3 - tag", "Device4"])
    api = API.from_api_id(mock_backend.api_id)
    context = Context.from_criteria(api, devices_num=2, device_include_masks=["tag"])

    # Providing max_total_local_sizes for all possible devices to make sure
    # only the ones corresponding to the context will get picked up
    kernel = MockKernel('kernel', max_total_local_sizes={0: 64, 1: 1024, 2: 512, 3: 128})

    src = MockDefTemplate(kernels=[kernel])
    program = Program(context, src)

    # The indices here correspond to the devices in the context, not in the platform
    assert program.kernel.max_total_local_sizes == {0: 1024, 1: 512}


def test_keep(mock_context, capsys):
    _test_keep(context=mock_context, capsys=capsys, is_mocked=True)

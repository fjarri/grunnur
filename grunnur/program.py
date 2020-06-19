from __future__ import annotations

from math import log10
from typing import NamedTuple, Tuple

import numpy

from .adapter_base import AdapterCompilationError, KernelAdapter
from .modules import render_with_modules
from .utils import wrap_in_tuple
from .array import Array
from .buffer import Buffer
from .api import CUDA_API_ID


class MultiDevice:
    def __init__(self, *args):
        self.values = args


class CompilationError(RuntimeError):

    def __init__(self, backend_exception):
        super().__init__(str(backend_exception))
        self.backend_exception = backend_exception


def _set_constant_array(
        queue: Queue, program_adapter, name: str, arr: Union[Array, Buffer, numpy.ndarray]):
    """
    Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
    """
    device_idx = program_adapter._device_idx
    queue_adapter = queue._queue_adapter

    if isinstance(arr, Array):
        # TODO: temporary check; arrays shouldn't have built-in queues
        assert queue is arr._queue

    if queue_adapter.context_adapter is not program_adapter.context_adapter:
        raise ValueError(
            "The provided queue must belong to the same context as this program uses")
    if device_idx not in queue.devices:
        raise ValueError(
            f"The provided queue must include the device this program uses ({device_idx})")

    if isinstance(arr, Array):
        constant_data = arr.data._buffer_adapter
    elif isinstance(arr, Buffer):
        constant_data = arr._buffer_adapter
    elif isinstance(arr, numpy.ndarray):
        constant_data = arr
    else:
        raise TypeError(f"Unsupported array type: {type(arr)}")

    program_adapter.set_constant_buffer(queue_adapter, name, constant_data)



class SingleDeviceProgram:
    """
    A program compiled for a single device.
    """

    def __init__(
            self,
            context: Context,
            device_idx: int,
            template_src: str,
            no_prelude: bool=False,
            fast_math: bool=False,
            render_args: Union[Tuple, List]=[],
            render_globals: Dict={},
            **kwds):
        """
        Renders and compiles the given template on a single device.

        :param context:
        :param device_idx: the number of the device to compile on.
        :param template_src: see :py:meth:`compile`.
        :param no_prelude: see :py:meth:`compile`.
        :param fast_math: see :py:meth:`compile`.
        :param render_args: see :py:meth:`compile`.
        :param render_globals: see :py:meth:`compile`.
        :param kwds: additional parameters for compilation, see :py:func:`compile`.
        """
        self.context = context
        self._device_idx = device_idx

        src = render_with_modules(
            template_src, render_args=render_args, render_globals=render_globals)

        context_adapter = context._context_adapter

        if no_prelude:
            prelude = ""
        else:
            prelude = context_adapter.render_prelude(fast_math=fast_math)

        try:
            self._sd_program_adapter = context_adapter.compile_single_device(
                device_idx, prelude, src, fast_math=fast_math, **kwds)
        except AdapterCompilationError as e:
            print(f"Failed to compile on device {device_idx} ({context.devices[device_idx]})")

            lines = e.source.split("\n")
            max_num_len = int(log10(len(lines))) + 1
            for i, l in enumerate(lines):
                print(str(i+1).rjust(max_num_len) + ": " + l)

            raise CompilationError(e.backend_exception)

        self.source = self._sd_program_adapter.source

    def get_kernel_adapter(self, kernel_name: str) -> KernelAdapter:
        """
        Returns a :py:class:`SingleDeviceKernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        return getattr(self._sd_program_adapter, kernel_name)

    def set_constant_array(
            self, queue: Queue, name: str, arr: Union[Array, Buffer, numpy.ndarray]):
        """
        Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
        """
        _set_constant_array(queue, self._sd_program_adapter, name, arr)


class Program:

    def __init__(
            self,
            context: Context,
            template_src: str,
            device_idxs: Optional[Iterable[int]]=None,
            no_prelude: bool=False,
            fast_math: bool=False,
            render_args: Union[List, Tuple]=[],
            render_globals: Dict={},
            compiler_options: Iterable[str]=[],
            keep: bool=False,
            constant_arrays: Mapping[str, Tuple[int, numpy.dtype]]={}):
        """
        Compiles the given source on multiple devices.

        :param context:
        :param template_src: a string with the source code, or a Mako template source to render.
        :param device_idxs: a list of device numbers to compile on.
        :param no_prelude: do not add prelude to the rendered source.
        :param fast_math: compile using fast (but less accurate) math functions.
        :param render_args: a list of positional args to pass to the template.
        :param render_globals: a dictionary of globals to pass to the template.
        :param compiler_options: a list of options to pass to the backend compiler.
        :param keep: keep the intermediate files in a temporary directory.
        :param constant_arrays: (**CUDA only**) a dictionary ``name: (size, dtype)``
            of global constant arrays to be declared in the program.
        """
        if device_idxs is None:
            device_idxs = range(len(context.devices))
        else:
            device_idxs = sorted(device_idxs)

        if context.api.id != CUDA_API_ID and len(constant_arrays) > 0:
            raise ValueError("Compile-time constant arrays are only supported by CUDA API")

        sd_programs = {}
        sources = {}
        for device_idx in device_idxs:
            sd_program = SingleDeviceProgram(
                context,
                device_idx, template_src,
                no_prelude=no_prelude,
                fast_math=fast_math,
                render_args=render_args,
                render_globals=render_globals,
                compiler_options=compiler_options,
                keep=keep,
                constant_arrays=constant_arrays)
            sd_programs[device_idx] = sd_program
            sources[device_idx] = sd_program.source

        self._sd_programs = sd_programs
        self.sources = sources
        self.context = context

    def __getattr__(self, kernel_name: str) -> Kernel:
        """
        Returns a :py:class:`Kernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        sd_kernel_adapters = {
            device_idx: sd_program.get_kernel_adapter(kernel_name)
            for device_idx, sd_program in self._sd_programs.items()}
        return Kernel(self, sd_kernel_adapters)

    def set_constant_array(
            self, queue: Queue, name: str, arr: Union[Array, numpy.ndarray]):
        """
        Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
        """
        if self.context.api.id != CUDA_API_ID:
            raise ValueError("Constant arrays are only supported for CUDA API")
        for sd_program in self._sd_programs.values():
            sd_program.set_constant_array(queue, name, arr)


def process_arg(arg):
    if isinstance(arg, Array):
        return arg.data.kernel_arg
    elif isinstance(arg, Buffer):
        return arg.kernel_arg
    else:
        return arg


def _call_kernels(
            queue: Queue,
            sd_kernel_adapters,
            global_size: Union[int, Iterable[int]],
            local_size: Union[int, Iterable[int], None],
            *args,
            device_idxs: Optional[Iterable[int]]=None,
            **kwds):

    if device_idxs is None:
        device_idxs = queue.device_idxs

    if not all(device_idx in sd_kernel_adapters for device_idx in device_idxs):
        missing_dev_nums = [
            str(device_idx) for device_idx in device_idxs if device_idx not in sd_kernel_adapters]
        raise ValueError(
            f"This kernel's program was not compiled for devices {', '.join(missing_dev_nums)}")

    ret_vals = []
    for i, device_idx in enumerate(device_idxs):
        kernel_args = [arg.values[i] if isinstance(arg, MultiDevice) else arg for arg in args]
        kernel_args = [process_arg(arg) for arg in kernel_args]

        kernel_ls = local_size.values[i] if isinstance(local_size, MultiDevice) else local_size
        kernel_gs = global_size.values[i] if isinstance(global_size, MultiDevice) else global_size

        if kernel_ls is not None:
            kernel_ls = wrap_in_tuple(kernel_ls)
        kernel_gs = wrap_in_tuple(kernel_gs)

        ret_val = sd_kernel_adapters[device_idx](
            queue._queue_adapter, kernel_gs, kernel_ls, *kernel_args, **kwds)
        ret_vals.append(ret_val)

    return ret_vals


class Kernel:
    """
    A kernel compiled for multiple devices.
    """

    def __init__(self, program: Program, sd_kernel_adapters: Dict[int, KernelAdapter]):
        self._program = program
        self._device_idxs = set(sd_kernel_adapters)
        self._sd_kernel_adapters = sd_kernel_adapters

    @property
    def max_total_local_sizes(self) -> Dict[int, int]:
        """
        The maximum possible number of threads in a block (CUDA)/work items in a work group (OpenCL)
        for this kernel.
        """
        return {
            device_idx: sd_kernel_adapter.max_total_local_size
            for device_idx, sd_kernel_adapter in self._sd_kernel_adapters.items()}

    def __call__(
            self,
            queue: Queue,
            global_size: Union[int, Iterable[int]],
            local_size: Union[int, Iterable[int], None],
            *args,
            device_idxs: Optional[Iterable[int]]=None,
            **kwds):
        """
        Enqueues the kernel on the chosen devices.

        :param queue: the multi-device queue to use.
        :param global_size: the total number of threads (CUDA)/work items (OpenCL) in each dimension
            (column-major). Note that there may be a maximum size in each dimension as well
            as the maximum number of dimensions. See :py:class:`DeviceParameters` for details.
        :param local_size: the number of threads in a block (CUDA)/work items in a
            work group (OpenCL) in each dimension (column-major).
            If ``None``, it will be chosen automatically.
        :param args: kernel arguments. Can be: :py:class:`Array` objects,
            :py:class:`Buffer` objects, ``numpy`` scalars.
        :param device_idxs: the devices to enqueue the kernel on
            (*in the context, not in the queue*). Must be a subset of the devices of the ``queue``.
            If ``None``, all the ``queue``'s devices are used.
            Note that the used devices must be among the ones the parent :py:class:`Program`
            was compiled for.
        :param kwds: backend-specific keyword parameters.
        """
        return _call_kernels(
            queue, self._sd_kernel_adapters, global_size, local_size, *args,
            device_idxs=device_idxs, **kwds)

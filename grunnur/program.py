from __future__ import annotations

from math import log10
from typing import (
    Tuple, Union, List, Dict, Optional, Iterable,
    Mapping, Generic, TypeVar, Callable, Sequence)
import weakref

import numpy

from .adapter_base import AdapterCompilationError, KernelAdapter, BufferAdapter, ProgramAdapter
from .modules import render_with_modules
from .utils import wrap_in_tuple, update_dict
from .array import Array
from .buffer import Buffer
from .queue import Queue
from .context import Context
from .api import cuda_api_id
from .template import DefTemplate
from .modules import Snippet


_T = TypeVar('_T', covariant=True)


class MultiDevice(Generic[_T]):
    """
    A wrapper for a sequence of arguments where each should be passed to a separate device.
    """
    def __init__(self, *args: _T):
        self.values: Tuple[_T, ...] = args


class CompilationError(RuntimeError):

    def __init__(self, backend_exception):
        super().__init__(str(backend_exception))
        self.backend_exception = backend_exception


def _set_constant_array(
        queue: Queue, program_adapter: ProgramAdapter, name: str, arr: Union[Array, Buffer, numpy.ndarray]):
    """
    Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
    """
    device_idx = program_adapter._device_idx
    queue_adapter = queue._queue_adapter

    if isinstance(arr, Array):
        assert queue is arr._queue

    if queue_adapter._context_adapter is not program_adapter._context_adapter:
        raise ValueError(
            "The provided queue must belong to the same context as this program uses")
    if device_idx not in queue.devices:
        raise ValueError(
            f"The provided queue must include the device this program uses ({device_idx})")

    constant_data: Union[BufferAdapter, numpy.ndarray]

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
            template_src: Union[str, Callable[..., str], DefTemplate, Snippet],
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

        render_globals = update_dict(
            render_globals, dict(device_params=context.devices[device_idx].params),
            error_msg="'device_params' is a reserved global name and cannot be used")

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
    """
    A compiled program on device(s).
    """

    context: Context
    """The context this program was compiled for."""

    sources: Dict[int, str]
    """Source files used for each device."""

    kernel: KernelHub
    """An object whose attributes are :py:class:`~grunnur.program.Kernel` objects with the corresponding names."""

    def __init__(
            self,
            context: Context,
            template_src: Union[str, Callable[..., str], DefTemplate, Snippet],
            device_idxs: Optional[Sequence[int]]=None,
            no_prelude: bool=False,
            fast_math: bool=False,
            render_args: Union[List, Tuple]=[],
            render_globals: Dict={},
            compiler_options: Iterable[str]=[],
            keep: bool=False,
            constant_arrays: Mapping[str, Tuple[int, numpy.dtype]]={}):
        """
        :param context: context to compile the program on.
        :param template_src: a string with the source code, or a Mako template source to render.
        :param device_idxs: a list of device numbers to compile on.
            If ``None``, compile on all context's devices.
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

        if context.api.id != cuda_api_id() and len(constant_arrays) > 0:
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

        self.kernel = KernelHub(self)

    def set_constant_array(
            self, queue: Queue, name: str, arr: Union[Array, numpy.ndarray]):
        """
        Uploads a constant array to the context's devices (**CUDA only**).

        :param queue: the queue to use for the transfer.
        :param name: the name of the constant array symbol in the code.
        :param arr: either a device or a host array.
        """
        if self.context.api.id != cuda_api_id():
            raise ValueError("Constant arrays are only supported for CUDA API")
        for sd_program in self._sd_programs.values():
            sd_program.set_constant_array(queue, name, arr)


class KernelHub:
    """
    An object providing access to the host program's kernels.
    """

    def __init__(self, program: Program):
        self._program_ref = weakref.ref(program)

    def __getattr__(self, kernel_name: str) -> Kernel:
        """
        Returns a :py:class:`~grunnur.program.Kernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        program = self._program_ref()
        sd_kernel_adapters = {
            device_idx: sd_program.get_kernel_adapter(kernel_name)
            for device_idx, sd_program in program._sd_programs.items()}
        return Kernel(program, sd_kernel_adapters)


def process_arg(arg):
    if isinstance(arg, Array):
        return arg.data.kernel_arg
    if isinstance(arg, Buffer):
        return arg.kernel_arg
    return arg


class PreparedKernel:
    """
    A kernel specialized for execution on a given queue
    with all possible preparations and checks performed.
    """

    def __init__(
            self,
            sd_kernel_adapters: Dict[int, KernelAdapter],
            queue: Queue,
            device_idxs: Tuple[int],
            global_sizes: Union[int, Sequence[int], MultiDevice[Union[int, Sequence[int]]]],
            local_sizes: Union[int, Sequence[int], MultiDevice[Union[int, Sequence[int]]], None],
            hold_reference=None):

        # If this object can be used by itself (e.g. when created from `Kernel.prepare()`),
        # this attribute will hold thre reference to the original `Kernel`.
        # On the other hand, in `StaticKernel` the object is used internally,
        # and holding a reference to the parent `StaticKernel` here will result in a reference cycle.
        # So `StaticKernel` will just pass `None`.
        self._hold_reference = hold_reference

        if not all(device_idx in sd_kernel_adapters for device_idx in device_idxs):
            missing_dev_nums = [
                str(device_idx) for device_idx in device_idxs if device_idx not in sd_kernel_adapters]
            raise ValueError(
                f"This kernel's program was not compiled for devices {', '.join(missing_dev_nums)}")

        self._prepared_kernel_adapters = {}
        self._device_idxs = device_idxs

        for i, device_idx in enumerate(device_idxs):
            _kernel_ls = local_sizes.values[i] if isinstance(local_sizes, MultiDevice) else local_sizes
            _kernel_gs = global_sizes.values[i] if isinstance(global_sizes, MultiDevice) else global_sizes

            kernel_ls = wrap_in_tuple(_kernel_ls) if _kernel_ls is not None else _kernel_ls
            kernel_gs = wrap_in_tuple(_kernel_gs)

            pkernel = sd_kernel_adapters[device_idx].prepare(
                queue._queue_adapter, kernel_gs, kernel_ls)

            self._prepared_kernel_adapters[device_idx] = pkernel

    def __call__(self, *args, device_idxs: Iterable[int]=None, **kwds):
        """
        Enqueues the kernel on the chosen devices.

        :param args: kernel arguments. Can be: :py:class:`~grunnur.Array` objects,
            :py:class:`~grunnur.Buffer` objects, ``numpy`` scalars.
        :param device_idxs: the devices to enqueue the kernel on
            (*in the context, not in the queue*). Must be a subset of the devices of the ``queue``
            used for preparation.
            If ``None``, all the ``queue``'s devices are used.
            Note that the used devices must be among the ones the parent
            :py:class:`~grunnur.Program` was compiled for.
        :param kwds: backend-specific keyword parameters.
        :returns: a list of ``Event`` objects for enqueued kernels in case of PyOpenCL.
        """

        if device_idxs is None:
            device_idxs = self._device_idxs

        ret_vals = []
        for i, device_idx in enumerate(device_idxs):
            kernel_args = [arg.values[i] if isinstance(arg, MultiDevice) else arg for arg in args]
            kernel_args = [process_arg(arg) for arg in kernel_args]

            pkernel = self._prepared_kernel_adapters[device_idx]
            ret_val = pkernel(*kernel_args, **kwds)
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

    def prepare(
            self,
            queue: Queue,
            global_size: Union[int, Sequence[int], MultiDevice[Union[int, Sequence[int]]]],
            local_size: Union[int, Sequence[int], MultiDevice[Union[int, Sequence[int]]], None]
            ) -> PreparedKernel:
        """
        Prepares the kernel for execution.

        :param queue: the multi-device queue to use.
        :param global_size: the total number of threads (CUDA)/work items (OpenCL) in each dimension
            (column-major). Note that there may be a maximum size in each dimension as well
            as the maximum number of dimensions. See :py:class:`DeviceParameters` for details.
        :param local_size: the number of threads in a block (CUDA)/work items in a
            work group (OpenCL) in each dimension (column-major).
            If ``None``, it will be chosen automatically.
        """
        return PreparedKernel(
            self._sd_kernel_adapters, queue, queue.device_idxs, global_size, local_size,
            hold_reference=self)

    def __call__(
            self,
            queue: Queue,
            global_size: Union[int, Sequence[int], MultiDevice[Union[int, Sequence[int]]]],
            local_size: Union[int, Sequence[int], MultiDevice[Union[int, Sequence[int]]], None],
            *args,
            **kwds):
        """
        A shortcut for :py:meth:`Kernel.prepare` and subsequent :py:meth:`PreparedKernel.__call__`.
        See their doc entries for details.
        """
        pkernel = self.prepare(queue, global_size, local_size)
        return pkernel(*args, **kwds)

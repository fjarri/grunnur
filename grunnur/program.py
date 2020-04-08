from __future__ import annotations

from math import log10

import numpy

from .modules import render_with_modules
from .utils import wrap_in_tuple
from .array import Array
from .buffer import Buffer
from .api import CUDA_API_ID


def process_arg(arg):
    if isinstance(arg, Array):
        return arg.data.kernel_arg
    elif isinstance(arg, (list, tuple)):
        return [process_arg(arg_part) for arg_part in arg]
    else:
        return arg


class SingleDeviceProgram:
    """
    A program compiled for a single device.
    """

    def __init__(
            self,
            context: Context,
            device_num: int,
            template_src: str,
            no_prelude: bool=False,
            fast_math: bool=False,
            render_args: Union[Tuple, List]=[],
            render_globals: Dict={},
            **kwds):
        """
        Renders and compiles the given template on a single device.

        :param context:
        :param device_num: the number of the device to compile on.
        :param template_src: see :py:meth:`compile`.
        :param no_prelude: see :py:meth:`compile`.
        :param fast_math: see :py:meth:`compile`.
        :param render_args: see :py:meth:`compile`.
        :param render_globals: see :py:meth:`compile`.
        :param kwds: additional parameters for compilation, see :py:func:`compile`.
        """
        self.context = context
        self._device_num = device_num

        src = render_with_modules(
            template_src, render_args=render_args, render_globals=render_globals)

        context_adapter = context._context_adapter

        if no_prelude:
            prelude = ""
        else:
            prelude = context_adapter.render_prelude(fast_math=fast_math)

        try:
            self._sd_program_adapter = context_adapter.compile_single_device(
                device_num, prelude, src, fast_math=fast_math, **kwds)
        except context_adapter.compile_error_class:
            print(f"Failed to compile on device {device_num} ({context.devices[device_num]})")

            lines = src.split("\n")
            max_num_len = int(log10(len(lines))) + 1
            for i, l in enumerate(lines):
                print(str(i+1).rjust(max_num_len) + ": " + l)

            raise

    def __getattr__(self, kernel_name: str) -> SingleDeviceKernel:
        """
        Returns a :py:class:`SingleDeviceKernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        return getattr(self._sd_program_adapter, kernel_name)

    def set_constant_array(
            self, name: str, arr: Union[Array, numpy.ndarray], queue: Optional[Queue]=None):
        """
        Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
        """
        if isinstance(arr, Array):
            # TODO: check that array is contiguous, offset is 0 and there's no padding in the end
            if queue is None:
                queue = arr._queue
            else:
                raise ValueError("Cannot use a custom queue when setting a constant from Array")

        if queue is not None:
            if queue.context is not self.context:
                raise ValueError(
                    "The provided queue must belong to the same context as this program uses")
            if self._device_num not in queue.devices:
                raise ValueError(
                    f"The provided queue must include the device this program uses ({self._device_num})")

        if isinstance(arr, Array):
            constant_data = arr.data._buffer_adapter
        elif isinstance(arr, Buffer):
            constant_data = arr._buffer_adapter
        elif isinstance(arr, numpy.ndarray):
            constant_data = arr
        else:
            raise TypeError(f"Uunsupported array type: {type(arr)}")

        if queue is not None:
            queue_adapter = queue._queue_adapter
        else:
            queue_adapter = None

        self._sd_program_adapter.set_constant_buffer(name, constant_data, queue=queue_adapter)


class Program:

    def __init__(
            self,
            context: Context,
            template_src: str,
            device_nums: Optional[Iterable[int]]=None,
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
        :param device_nums: a list of device numbers to compile on.
        :param no_prelude: do not add prelude to the rendered source.
        :param fast_math: compile using fast (but less accurate) math functions.
        :param render_args: a list of positional args to pass to the template.
        :param render_globals: a dictionary of globals to pass to the template.
        :param compiler_options: a list of options to pass to the backend compiler.
        :param keep: keep the intermediate files in a temporary directory.
        :param constant_arrays: (**CUDA only**) a dictionary ``name: (size, dtype)``
            of global constant arrays to be declared in the program.
        """
        if device_nums is None:
            device_nums = range(len(context.devices))
        else:
            device_nums = sorted(device_nums)

        sd_programs = {}
        for device_num in device_nums:
            sd_program = SingleDeviceProgram(
                context,
                device_num, template_src,
                no_prelude=no_prelude,
                fast_math=fast_math,
                render_args=render_args,
                render_globals=render_globals,
                compiler_options=compiler_options,
                keep=keep,
                constant_arrays=constant_arrays)
            sd_programs[device_num] = sd_program

        self._sd_programs = sd_programs
        self.context = context

    def __getattr__(self, kernel_name: str) -> Kernel:
        """
        Returns a :py:class:`Kernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        sd_kernels = {
            device_num: getattr(sd_program, kernel_name)
            for device_num, sd_program in self._sd_programs.items()}
        return Kernel(self, sd_kernels)

    def set_constant_array(
            self, name: str, arr: Union[Array, numpy.ndarray], queue: Optional[Queue]=None):
        """
        Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
        """
        if self.context.api.id != CUDA_API_ID:
            raise ValueError("Constant arrays are only supported for CUDA API")
        for sd_program in self._sd_programs.values():
            sd_program.set_constant_array(name, arr, queue=queue)


class Kernel:
    """
    A kernel compiled for multiple devices.
    """

    def __init__(self, program: Program, sd_kernels: Dict[int, SingleDeviceKernel]):
        self._program = program
        self._device_nums = set(sd_kernels)
        self._sd_kernels = sd_kernels

    def __call__(
            self,
            queue: Queue,
            global_size: Union[int, Iterable[int]],
            local_size: Union[int, Iterable[int], None],
            *args,
            device_nums: Optional[Iterable[int]]=None,
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
        :param device_nums: the devices to enqueue the kernel on
            (*in the context, not in the queue*). Must be a subset of the devices of the ``queue``.
            If ``None``, all the ``queue``'s devices are used.
            Note that the used devices must be among the ones the parent :py:class:`Program`
            was compiled for.
        :param kwds: backend-specific keyword parameters.
        """

        if local_size is not None:
            local_size = wrap_in_tuple(local_size)
        global_size = wrap_in_tuple(global_size)

        args = process_arg(args)

        if device_nums is None:
            device_nums = list(queue.devices)
        else:
            device_nums = list(device_nums)

        # TODO: speed this up. Probably shouldn't create sets on every kernel call.
        if not set(device_nums).issubset(self._device_nums):
            missing_dev_nums = [
                str(device_num) for device_num in device_nums if device_num not in self._device_nums]
            raise ValueError(
                f"This kernel's program was not compiled for devices {', '.join(missing_dev_nums)}")

        ret_vals = []
        for i, device_num in enumerate(device_nums):
            kernel_args = [arg[i] if isinstance(arg, (list, tuple)) else arg for arg in args]
            ret_val = self._sd_kernels[device_num](
                queue._queue_adapter, global_size, local_size, *kernel_args, **kwds)
            ret_vals.append(ret_val)

        return ret_vals


class SingleDeviceKernel:
    """
    A kernel compiled for a single device.
    """
    def __init__(self, sd_kernel_adapter):
        self._sd_kernel_adapter = sd_kernel_adapter

    @property
    def max_total_local_size(self) -> int:
        """
        The maximum possible number of threads in a block (CUDA)/work items in a work group (OpenCL)
        for this kernel.
        """
        return self._sd_kernel_adapter.max_total_local_size

    def __call__(
            self,
            queue: Queue,
            global_size: Union[int, Iterable[int]],
            local_size: Union[int, Iterable[int], None],
            *args,
            **kwds):
        """
        Enqueues the kernel on a single device.

        :param queue: see :py:meth:`Kernel.__call__`.
        :param global_size: see :py:meth:`Kernel.__call__`.
        :param local_size: see :py:meth:`Kernel.__call__`.
        :param args: see :py:meth:`Kernel.__call__`.
        :param kwds: see :py:meth:`Kernel.__call__`.
        """
        if local_size is not None:
            local_size = wrap_in_tuple(local_size)
        global_size = wrap_in_tuple(global_size)

        args = process_arg(args)

        assert self._device_num in queue.devices

        return self._sd_kernel_adapter(global_size, local_size, *args, **kwds)

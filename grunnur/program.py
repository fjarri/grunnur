from __future__ import annotations

from math import log10
from typing import (
    Any,
    Tuple,
    Union,
    List,
    Dict,
    Optional,
    Iterable,
    Mapping,
    Generic,
    TypeVar,
    Callable,
    Sequence,
    cast,
)
import weakref

import numpy

from .device import Device
from .adapter_base import (
    AdapterCompilationError,
    KernelAdapter,
    BufferAdapter,
    ProgramAdapter,
)
from .modules import render_with_modules
from .utils import update_dict
from .array_metadata import ArrayMetadataLike
from .array import Array, MultiArray
from .buffer import Buffer
from .queue import Queue, MultiQueue
from .context import Context, BoundDevice, BoundMultiDevice
from .api import cuda_api_id
from .template import DefTemplate
from .modules import Snippet


class CompilationError(RuntimeError):
    def __init__(self, backend_exception: Exception):
        super().__init__(str(backend_exception))
        self.backend_exception = backend_exception


def _check_set_constant_array(queue: Queue, program_devices: BoundMultiDevice) -> None:
    if queue.device.context != program_devices.context:
        raise ValueError("The provided queue must belong to the same context as this program uses")
    if queue.device not in program_devices:
        raise ValueError(
            f"The program was not compiled for the device this queue uses ({queue.device})"
        )


def _set_constant_array(
    queue: Queue,
    program_adapter: ProgramAdapter,
    name: str,
    arr: Union[Array, Buffer, "numpy.ndarray[Any, numpy.dtype[Any]]"],
) -> None:
    """
    Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
    """
    queue_adapter = queue._queue_adapter

    constant_data: Union[BufferAdapter, "numpy.ndarray[Any, numpy.dtype[Any]]"]

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

    device: BoundDevice

    source: str

    def __init__(
        self,
        device: BoundDevice,
        template_src: Union[str, Callable[..., str], DefTemplate, Snippet],
        no_prelude: bool = False,
        fast_math: bool = False,
        render_args: Sequence[Any] = [],
        render_globals: Mapping[str, Any] = {},
        constant_arrays: Optional[Mapping[str, ArrayMetadataLike]] = None,
        keep: bool = False,
        compiler_options: Optional[Sequence[str]] = None,
    ):
        """
        Renders and compiles the given template on a single device.

        :param device:
        :param template_src: see :py:meth:`compile`.
        :param no_prelude: see :py:meth:`compile`.
        :param fast_math: see :py:meth:`compile`.
        :param render_args: see :py:meth:`compile`.
        :param render_globals: see :py:meth:`compile`.
        :param kwds: additional parameters for compilation, see :py:func:`compile`.
        """
        if device.context.api.id != cuda_api_id() and constant_arrays and len(constant_arrays) > 0:
            raise ValueError("Compile-time constant arrays are only supported for CUDA API")

        self.device = device

        render_globals = update_dict(
            render_globals,
            dict(device_params=device.params),
            error_msg="'device_params' is a reserved global name and cannot be used",
        )

        src = render_with_modules(
            template_src, render_args=render_args, render_globals=render_globals
        )

        context_adapter = device.context._context_adapter

        if no_prelude:
            prelude = ""
        else:
            prelude = context_adapter.render_prelude(fast_math=fast_math)

        try:
            self._sd_program_adapter = context_adapter.compile_single_device(
                device._device_adapter,
                prelude,
                src,
                fast_math=fast_math,
                constant_arrays=constant_arrays,
                keep=keep,
                compiler_options=compiler_options,
            )
        except AdapterCompilationError as e:
            print(f"Failed to compile on {device}")

            lines = e.source.split("\n")
            max_num_len = int(log10(len(lines))) + 1
            for i, l in enumerate(lines):
                print(str(i + 1).rjust(max_num_len) + ": " + l)

            raise CompilationError(e.backend_exception)

        self.source = self._sd_program_adapter.source

    def get_kernel_adapter(self, kernel_name: str) -> KernelAdapter:
        """
        Returns a :py:class:`SingleDeviceKernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        return cast(KernelAdapter, getattr(self._sd_program_adapter, kernel_name))

    def set_constant_array(
        self,
        queue: Queue,
        name: str,
        arr: Union[Array, Buffer, "numpy.ndarray[Any, numpy.dtype[Any]]"],
    ) -> None:
        """
        Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
        """
        _set_constant_array(queue, self._sd_program_adapter, name, arr)


class Program:
    """
    A compiled program on device(s).
    """

    devices: BoundMultiDevice
    """The devices on which this program was compiled."""

    sources: Dict[BoundDevice, str]
    """Source files used for each device."""

    kernel: "KernelHub"
    """An object whose attributes are :py:class:`~grunnur.program.Kernel` objects with the corresponding names."""

    def __init__(
        self,
        devices: Sequence[BoundDevice],
        template_src: Union[str, Callable[..., str], DefTemplate, Snippet],
        no_prelude: bool = False,
        fast_math: bool = False,
        render_args: Sequence[Any] = (),
        render_globals: Mapping[str, Any] = {},
        compiler_options: Optional[Sequence[str]] = None,
        keep: bool = False,
        constant_arrays: Optional[Mapping[str, ArrayMetadataLike]] = None,
    ):
        """
        :param devices: a single- or a multi-device object on which to compile this program.
        :param template_src: a string with the source code, or a Mako template source to render.
        :param no_prelude: do not add prelude to the rendered source.
        :param fast_math: compile using fast (but less accurate) math functions.
        :param render_args: a list of positional args to pass to the template.
        :param render_globals: a dictionary of globals to pass to the template.
        :param compiler_options: a list of options to pass to the backend compiler.
        :param keep: keep the intermediate files in a temporary directory.
        :param constant_arrays: (**CUDA only**) a dictionary ``name: (size, dtype)``
            of global constant arrays to be declared in the program.
        """
        sd_programs = {}
        sources = {}

        multi_device = BoundMultiDevice.from_bound_devices(devices)

        for device in multi_device:
            sd_program = SingleDeviceProgram(
                device,
                template_src,
                no_prelude=no_prelude,
                fast_math=fast_math,
                render_args=render_args,
                render_globals=render_globals,
                compiler_options=compiler_options,
                keep=keep,
                constant_arrays=constant_arrays,
            )
            sd_programs[device] = sd_program
            sources[device] = sd_program.source

        self._sd_programs = sd_programs
        self.sources = sources
        self.devices = multi_device

        # TODO: create dynamically, in case someone wants to hold a reference to it and
        # discard this Program object
        self.kernel = KernelHub(self)

    def set_constant_array(
        self, queue: Queue, name: str, arr: Union[Array, "numpy.ndarray[Any, numpy.dtype[Any]]"]
    ) -> None:
        """
        Uploads a constant array to the context's devices (**CUDA only**).

        :param queue: the queue to use for the transfer.
        :param name: the name of the constant array symbol in the code.
        :param arr: either a device or a host array.
        """
        _check_set_constant_array(queue, self.devices)
        self._sd_programs[queue.device].set_constant_array(queue, name, arr)


class KernelHub:
    """
    An object providing access to the host program's kernels.
    """

    def __init__(self, program: Program):
        self._program_ref = weakref.proxy(program)

    def __getattr__(self, kernel_name: str) -> "Kernel":
        """
        Returns a :py:class:`~grunnur.program.Kernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        program = self._program_ref
        sd_kernel_adapters = {
            device: sd_program.get_kernel_adapter(kernel_name)
            for device, sd_program in program._sd_programs.items()
        }
        return Kernel(program, sd_kernel_adapters)


def extract_arg(
    arg: Union[
        Mapping[BoundDevice, Union[Array, Buffer, numpy.generic]],
        MultiArray,
        Array,
        Buffer,
        numpy.generic,
    ],
    device: BoundDevice,
) -> Union[BufferAdapter, numpy.generic]:

    single_device_arg: Union[Array, Buffer, numpy.generic]
    if isinstance(arg, Mapping):
        single_device_arg = arg[device]
    elif isinstance(arg, MultiArray):
        single_device_arg = arg.subarrays[device]
    else:
        single_device_arg = arg

    if isinstance(single_device_arg, Array):
        return single_device_arg.data._buffer_adapter
    elif isinstance(single_device_arg, Buffer):
        return single_device_arg._buffer_adapter
    else:
        return single_device_arg


class PreparedKernel:
    """
    A kernel specialized for execution on a set of devices
    with all possible preparations and checks performed.
    """

    def __init__(
        self,
        devices: BoundMultiDevice,
        sd_kernel_adapters: Mapping[BoundDevice, KernelAdapter],
        global_sizes: Mapping[BoundDevice, Sequence[int]],
        local_sizes: Mapping[BoundDevice, Optional[Sequence[int]]],
        hold_reference: Optional["Kernel"] = None,
    ):

        # If this object can be used by itself (e.g. when created from `Kernel.prepare()`),
        # this attribute will hold thre reference to the original `Kernel`.
        # On the other hand, in `StaticKernel` the object is used internally,
        # and holding a reference to the parent `StaticKernel` here will result in a reference cycle.
        # So `StaticKernel` will just pass `None`.
        self._hold_reference = hold_reference

        self._prepared_kernel_adapters = {}

        for device in sd_kernel_adapters:
            kernel_ls = local_sizes[device]
            kernel_gs = global_sizes[device]
            pkernel = sd_kernel_adapters[device].prepare(kernel_gs, kernel_ls)

            self._prepared_kernel_adapters[device] = pkernel

        self._devices = devices

    def __call__(
        self,
        queue: Union[Queue, MultiQueue],
        *args: Union[MultiArray, Array, Buffer, numpy.generic],
        local_mem: int = 0,
    ) -> Any:
        """
        Enqueues the kernel on the devices in the given queue.
        The kernel must have been prepared for all of these devices.

        If an argument is a :py:class:`~grunnur.Array` or :py:class:`~grunnur.Buffer` object,
        it must belong to the device on which the kernel is being executed
        (so ``queue`` must only have one device).

        If an argument is a :py:class:`~grunnur.MultiArray`, it should have subarrays
        on all the devices from the given ``queue``.

        If an argument is a ``numpy`` scalar, it will be passed to the kernel directly.

        If an argument is a integer-keyed ``dict``, its values corresponding to the
        device indices the kernel is executed on will be passed as kernel arguments.

        :param args: kernel arguments.
        :param kwds: backend-specific keyword parameters.
        :returns: a list of ``Event`` objects for enqueued kernels in case of PyOpenCL.
        """
        if isinstance(queue, Queue):
            queue = MultiQueue([queue])

        # Technically this would be caught by `issubset()`, but it'll help to provide
        # a more specific error to the user.
        if queue.devices.context != self._devices.context:
            raise ValueError("The provided queue must belong to the same context this program uses")

        if not queue.devices.issubset(self._devices):
            raise ValueError(
                f"Requested execution on devices {queue.devices}; "
                f"only compiled for {self._devices}"
            )

        ret_vals = []
        for device in queue.devices:
            kernel_args = [extract_arg(arg, device) for arg in args]

            single_queue = queue.queues[device]

            pkernel = self._prepared_kernel_adapters[device]
            ret_val = pkernel(single_queue._queue_adapter, *kernel_args, local_mem=0)
            ret_vals.append(ret_val)

        return ret_vals


def normalize_sizes(
    devices: Sequence[BoundDevice],
    global_size: Union[Sequence[int], Mapping[BoundDevice, Sequence[int]]],
    local_size: Union[Sequence[int], None, Mapping[BoundDevice, Optional[Sequence[int]]]] = None,
) -> Tuple[
    BoundMultiDevice,
    Dict[BoundDevice, Tuple[int, ...]],
    Dict[BoundDevice, Optional[Tuple[int, ...]]],
]:
    if not isinstance(global_size, Mapping):
        global_size = {device: global_size for device in devices}

    if not isinstance(local_size, Mapping):
        local_size = {device: local_size for device in devices}

    normalized_global_size = {device: tuple(gs) for device, gs in global_size.items()}
    normalized_local_size = {
        device: tuple(ls) if ls is not None else None for device, ls in local_size.items()
    }

    if normalized_global_size.keys() != normalized_local_size.keys():
        raise ValueError(
            "Mismatched device sets for global and local sizes: "
            f"local sizes have {list(normalized_local_size.keys())}, "
            f"global sizes have {list(normalized_global_size.keys())}"
        )

    devices_subset = BoundMultiDevice.from_bound_devices(
        [device for device in devices if device in normalized_global_size]
    )

    return devices_subset, normalized_global_size, normalized_local_size


class Kernel:
    """
    A kernel compiled for multiple devices.
    """

    def __init__(self, program: Program, sd_kernel_adapters: Dict[BoundDevice, KernelAdapter]):
        self._program = program
        self._sd_kernel_adapters = sd_kernel_adapters

    @property
    def max_total_local_sizes(self) -> Dict[BoundDevice, int]:
        """
        The maximum possible number of threads in a block (CUDA)/work items in a work group (OpenCL)
        for this kernel.
        """
        return {
            device: sd_kernel_adapter.max_total_local_size
            for device, sd_kernel_adapter in self._sd_kernel_adapters.items()
        }

    def prepare(
        self,
        global_size: Union[Sequence[int], Mapping[BoundDevice, Sequence[int]]],
        local_size: Union[
            Sequence[int], None, Mapping[BoundDevice, Optional[Sequence[int]]]
        ] = None,
    ) -> "PreparedKernel":
        """
        Prepares the kernel for execution.

        If ``local_size`` or ``global_size`` are integer, they will be treated as 1-tuples.

        One can pass specific global and local sizes for each device
        using dictionaries keyed with device indices.
        This achieves another purpose: the kernel will only be prepared for those devices,
        and not for all devices available in the context.

        :param global_size: the total number of threads (CUDA)/work items (OpenCL) in each dimension
            (column-major). Note that there may be a maximum size in each dimension as well
            as the maximum number of dimensions. See :py:class:`~grunnur.adapter_base.DeviceParameters`
            for details.
        :param local_size: the number of threads in a block (CUDA)/work items in a
            work group (OpenCL) in each dimension (column-major).
            If ``None``, it will be chosen automatically.
        """
        multi_device, n_global_size, n_local_size = normalize_sizes(
            self._program.devices, global_size, local_size
        )

        # Filter out only the kernel adapters mentioned in global/local_size
        sd_kernel_adapters = {device: self._sd_kernel_adapters[device] for device in multi_device}

        return PreparedKernel(
            multi_device, sd_kernel_adapters, n_global_size, n_local_size, hold_reference=self
        )

    def __call__(
        self,
        queue: Union[Queue, MultiQueue],
        global_size: Union[Sequence[int], Mapping[BoundDevice, Sequence[int]]],
        local_size: Union[
            Sequence[int], None, Mapping[BoundDevice, Optional[Sequence[int]]]
        ] = None,
        *args: Union[MultiArray, Array, Buffer, numpy.generic],
        local_mem: int = 0,
    ) -> Any:
        """
        A shortcut for :py:meth:`Kernel.prepare` and subsequent :py:meth:`PreparedKernel.__call__`.
        See their doc entries for details.
        """
        pkernel = self.prepare(global_size, local_size)
        return pkernel(queue, *args, local_mem=local_mem)

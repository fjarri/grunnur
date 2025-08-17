from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from math import log10
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import numpy

from ._adapter_base import (
    AdapterCompilationError,
    BufferAdapter,
    KernelAdapter,
    ProgramAdapter,
)
from ._api import cuda_api_id
from ._array import Array, MultiArray
from ._buffer import Buffer
from ._context import BoundDevice, BoundMultiDevice, Context
from ._device import Device
from ._modules import Snippet, render_with_modules
from ._queue import MultiQueue, Queue
from ._utils import update_dict

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from ._array_metadata import AsArrayMetadata
    from ._template import DefTemplate


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
    arr: Array | Buffer | NDArray[Any],
) -> None:
    """Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context."""
    queue_adapter = queue._queue_adapter  # noqa: SLF001

    constant_data: BufferAdapter | NDArray[Any]

    if isinstance(arr, Array):
        constant_data = arr.data._buffer_adapter  # noqa: SLF001
    elif isinstance(arr, Buffer):
        constant_data = arr._buffer_adapter  # noqa: SLF001
    elif isinstance(arr, numpy.ndarray):
        constant_data = arr
    else:
        raise TypeError(f"Unsupported array type: {type(arr)}")

    program_adapter.set_constant_buffer(queue_adapter, name, constant_data)


class SingleDeviceProgram:
    """A program compiled for a single device."""

    device: BoundDevice

    source: str

    def __init__(
        self,
        device: BoundDevice,
        template_src: str | Callable[..., str] | DefTemplate | Snippet,
        *,
        no_prelude: bool = False,
        fast_math: bool = False,
        render_args: Sequence[Any] = [],
        render_globals: Mapping[str, Any] = {},
        constant_arrays: Mapping[str, AsArrayMetadata] = {},
        keep: bool = False,
        compiler_options: Iterable[str] = [],
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

        context_adapter = device.context._context_adapter  # noqa: SLF001

        prelude = "" if no_prelude else context_adapter.render_prelude(fast_math=fast_math)

        constant_arrays_metadata = {
            name: array.as_array_metadata() for name, array in constant_arrays.items()
        }

        try:
            self._sd_program_adapter = context_adapter.compile_single_device(
                device._device_adapter,  # noqa: SLF001
                prelude,
                src,
                fast_math=fast_math,
                constant_arrays=constant_arrays_metadata,
                keep=keep,
                compiler_options=compiler_options,
            )
        except AdapterCompilationError as exc:
            print(f"Failed to compile on {device}")  # noqa: T201

            lines = exc.source.split("\n")
            max_num_len = int(log10(len(lines))) + 1
            for line_num, line in enumerate(lines):
                print(str(line_num + 1).rjust(max_num_len) + ": " + line)  # noqa: T201

            raise CompilationError(exc.backend_exception) from exc

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
        arr: Array | Buffer | NDArray[Any],
    ) -> None:
        """Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context."""
        _set_constant_array(queue, self._sd_program_adapter, name, arr)


class Program:
    """A compiled program on device(s)."""

    devices: BoundMultiDevice
    """The devices on which this program was compiled."""

    sources: dict[BoundDevice, str]
    """Source files used for each device."""

    kernel: KernelHub
    """
    An object whose attributes are :py:class:`~grunnur._program.Kernel` objects
    with the corresponding names.
    """

    def __init__(
        self,
        devices: Sequence[BoundDevice],
        template_src: str | Callable[..., str] | DefTemplate | Snippet,
        *,
        no_prelude: bool = False,
        fast_math: bool = False,
        render_args: Sequence[Any] = (),
        render_globals: Mapping[str, Any] = {},
        compiler_options: Sequence[str] = [],
        keep: bool = False,
        constant_arrays: Mapping[str, AsArrayMetadata] = {},
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
        self, queue: Queue, name: str, arr: Array | Buffer | NDArray[Any]
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
    """An object providing access to the host program's kernels."""

    def __init__(self, program: Program):
        self._program_ref = weakref.proxy(program)

    def __getattr__(self, kernel_name: str) -> Kernel:
        """
        Returns a :py:class:`~grunnur._program.Kernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        program = self._program_ref
        sd_kernel_adapters = {
            device: sd_program.get_kernel_adapter(kernel_name)
            for device, sd_program in program._sd_programs.items()  # noqa: SLF001
        }
        return Kernel(program, sd_kernel_adapters)


def extract_arg(
    arg: Mapping[BoundDevice, Array | Buffer | numpy.generic]
    | MultiArray
    | Array
    | Buffer
    | numpy.generic,
    device: BoundDevice,
) -> BufferAdapter | numpy.generic:
    single_device_arg: Array | Buffer | numpy.generic
    if isinstance(arg, Mapping):
        single_device_arg = arg[device]
    elif isinstance(arg, MultiArray):
        single_device_arg = arg.subarrays[device]
    else:
        single_device_arg = arg

    if isinstance(single_device_arg, Array):
        return single_device_arg.data._buffer_adapter  # noqa: SLF001
    if isinstance(single_device_arg, Buffer):
        return single_device_arg._buffer_adapter  # noqa: SLF001
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
        local_sizes: Mapping[BoundDevice, Sequence[int] | None],
        hold_reference: Kernel | None = None,
    ):
        # If this object can be used by itself (e.g. when created from `Kernel.prepare()`),
        # this attribute will hold thre reference to the original `Kernel`.
        # On the other hand, in `StaticKernel` the object is used internally,
        # and holding a reference to the parent `StaticKernel` here
        # will result in a reference cycle. So `StaticKernel` will just pass `None`.
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
        queue: Queue | MultiQueue,
        *args: Mapping[BoundDevice, Array | Buffer | numpy.generic]
        | MultiArray
        | Array
        | Buffer
        | numpy.generic,
        cu_dynamic_local_mem: int = 0,
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

        :param cu_dynamic_local_mem: **CUDA only.** The size of dynamically allocated local
            (shared in CUDA terms) memory, in bytes. That is, the size of
            ``extern __shared__`` arrays in CUDA kernels.
        :param args: kernel arguments.
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
                f"Requested execution on devices {queue.devices}; only compiled for {self._devices}"
            )

        ret_vals = []
        for device in queue.devices:
            kernel_args = [extract_arg(arg, device) for arg in args]

            single_queue = queue.queues[device]

            pkernel = self._prepared_kernel_adapters[device]
            ret_val = pkernel(
                single_queue._queue_adapter,  # noqa: SLF001
                *kernel_args,
                cu_dynamic_local_mem=cu_dynamic_local_mem,
            )
            ret_vals.append(ret_val)

        return ret_vals


def normalize_sizes(
    devices: Sequence[BoundDevice],
    global_size: Sequence[int] | Mapping[BoundDevice, Sequence[int]],
    local_size: Sequence[int] | None | Mapping[BoundDevice, Sequence[int] | None] = None,
) -> tuple[
    BoundMultiDevice,
    dict[BoundDevice, tuple[int, ...]],
    dict[BoundDevice, tuple[int, ...] | None],
]:
    if not isinstance(global_size, Mapping):
        global_size = dict.fromkeys(devices, global_size)

    if not isinstance(local_size, Mapping):
        local_size = dict.fromkeys(devices, local_size)

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
    """A kernel compiled for multiple devices."""

    def __init__(self, program: Program, sd_kernel_adapters: dict[BoundDevice, KernelAdapter]):
        self._program = program
        self._sd_kernel_adapters = sd_kernel_adapters

    @property
    def max_total_local_sizes(self) -> dict[BoundDevice, int]:
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
        global_size: Sequence[int] | Mapping[BoundDevice, Sequence[int]],
        local_size: Sequence[int] | None | Mapping[BoundDevice, Sequence[int] | None] = None,
    ) -> PreparedKernel:
        """
        Prepares the kernel for execution.

        If ``local_size`` or ``global_size`` are integer, they will be treated as 1-tuples.

        One can pass specific global and local sizes for each device
        using dictionaries keyed with device indices.
        This achieves another purpose: the kernel will only be prepared for those devices,
        and not for all devices available in the context.

        :param global_size: the total number of threads (CUDA)/work items (OpenCL) in each dimension
            (column-major). Note that there may be a maximum size in each dimension as well
            as the maximum number of dimensions.
            See :py:class:`~grunnur.DeviceParameters` for details.
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
        queue: Queue | MultiQueue,
        global_size: Sequence[int] | Mapping[BoundDevice, Sequence[int]],
        local_size: Sequence[int] | None | Mapping[BoundDevice, Sequence[int] | None] = None,
        *args: Mapping[BoundDevice, Array | Buffer | numpy.generic]
        | MultiArray
        | Array
        | Buffer
        | numpy.generic,
        cu_dynamic_local_mem: int = 0,
    ) -> Any:
        """
        A shortcut for :py:meth:`Kernel.prepare` and subsequent :py:meth:`PreparedKernel.__call__`.
        See their doc entries for details.
        """
        pkernel = self.prepare(global_size, local_size)
        return pkernel(queue, *args, cu_dynamic_local_mem=cu_dynamic_local_mem)

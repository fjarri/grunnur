from __future__ import annotations

import functools
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, cast

# Skipping coverage count - can't test properly
try:  # pragma: no cover
    import pyopencl
except ImportError:  # pragma: no cover
    pyopencl = None  # type: ignore[assignment]

from . import dtypes
from ._adapter_base import (
    APIID,
    AdapterCompilationError,
    APIAdapter,
    APIAdapterFactory,
    BufferAdapter,
    ContextAdapter,
    DeviceAdapter,
    DeviceParameters,
    DeviceType,
    KernelAdapter,
    PlatformAdapter,
    PreparedKernelAdapter,
    ProgramAdapter,
    QueueAdapter,
)
from ._template import Template
from ._utils import normalize_object_sequence

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    import numpy
    from numpy.typing import NDArray

    from ._array_metadata import ArrayMetadata


_API_ID = APIID("opencl")
_TEMPLATE = Template.from_associated_file(__file__)
_PRELUDE = _TEMPLATE.get_def("prelude")


class OclAPIAdapterFactory(APIAdapterFactory):
    @property
    def api_id(self) -> APIID:
        return _API_ID

    @property
    def available(self) -> bool:
        return pyopencl is not None

    def make_api_adapter(self) -> OclAPIAdapter:
        if not self.available:
            raise ImportError(
                "OpenCL API is not operational. Check if PyOpenCL is installed correctly."
            )

        return OclAPIAdapter()


class OclAPIAdapter(APIAdapter):
    @property
    def id(self) -> APIID:
        return _API_ID

    @property
    def platform_count(self) -> int:
        return len(pyopencl.get_platforms())

    def get_platform_adapters(self) -> tuple[OclPlatformAdapter, ...]:
        return tuple(
            OclPlatformAdapter(self, platform, platform_idx)
            for platform_idx, platform in enumerate(pyopencl.get_platforms())
        )

    def isa_backend_device(self, obj: Any) -> bool:
        return isinstance(obj, pyopencl.Device)

    def isa_backend_platform(self, obj: Any) -> bool:
        return isinstance(obj, pyopencl.Platform)

    def isa_backend_context(self, obj: Any) -> bool:
        return isinstance(obj, pyopencl.Context)

    def make_device_adapter(self, pyopencl_device: pyopencl.Device) -> OclDeviceAdapter:
        return OclDeviceAdapter.from_pyopencl_device(pyopencl_device)

    def make_platform_adapter(self, pyopencl_platform: pyopencl.Platform) -> OclPlatformAdapter:
        return OclPlatformAdapter.from_pyopencl_platform(pyopencl_platform)

    def make_context_adapter_from_device_adapters(
        self, device_adapters: Sequence[DeviceAdapter]
    ) -> OclContextAdapter:
        ocl_device_adapters = normalize_object_sequence(device_adapters, OclDeviceAdapter)
        return OclContextAdapter.from_device_adapters(ocl_device_adapters)

    def make_context_adapter_from_backend_contexts(
        self,
        pyopencl_contexts: Sequence[Any],
        *,
        take_ownership: bool,  # noqa: ARG002
    ) -> OclContextAdapter:
        if len(pyopencl_contexts) > 1:
            raise ValueError("Cannot make one OpenCL context out of several contexts")
        return OclContextAdapter.from_pyopencl_context(pyopencl_contexts[0])


class OclPlatformAdapter(PlatformAdapter):
    @classmethod
    def from_pyopencl_platform(cls, pyopencl_platform: pyopencl.Platform) -> OclPlatformAdapter:
        """Creates this object from a PyOpenCL ``Platform`` object."""
        api_adapter = OclAPIAdapter()
        platform_adapters = api_adapter.get_platform_adapters()
        for platform_idx, platform_adapter in enumerate(platform_adapters):
            if pyopencl_platform == platform_adapter._pyopencl_platform:  # noqa: SLF001
                return cls(api_adapter, pyopencl_platform, platform_idx)

        # Sanity check, should not be reachable as long as `pyopencl` is consistent.
        raise RuntimeError(
            f"{pyopencl_platform} was not found among OpenCL platforms"
        )  # pragma: no cover

    def __init__(
        self, api_adapter: APIAdapter, pyopencl_platform: pyopencl.Platform, platform_idx: int
    ):
        self._api_adapter = api_adapter
        self._pyopencl_platform = pyopencl_platform
        self._platform_idx = platform_idx

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, OclPlatformAdapter)
            and self._pyopencl_platform == other._pyopencl_platform
        )

    def __hash__(self) -> int:
        return hash((type(self), self._pyopencl_platform))

    @property
    def api_adapter(self) -> APIAdapter:
        return self._api_adapter

    @property
    def platform_idx(self) -> int:
        return self._platform_idx

    @property
    def name(self) -> str:
        return self._pyopencl_platform.name

    @property
    def vendor(self) -> str:
        return self._pyopencl_platform.vendor

    @property
    def version(self) -> str:
        return self._pyopencl_platform.version

    @property
    def device_count(self) -> int:
        return len(self._pyopencl_platform.get_devices())

    def get_device_adapters(self) -> tuple[OclDeviceAdapter, ...]:
        return tuple(
            OclDeviceAdapter(self, device, device_idx)
            for device_idx, device in enumerate(self._pyopencl_platform.get_devices())
        )


class OclDeviceAdapter(DeviceAdapter):
    @classmethod
    def from_pyopencl_device(cls, pyopencl_device: pyopencl.Device) -> OclDeviceAdapter:
        """Creates this object from a PyOpenCL ``Device`` object."""
        platform_adapter = OclPlatformAdapter.from_pyopencl_platform(pyopencl_device.platform)
        for device_idx, device_adapter in enumerate(platform_adapter.get_device_adapters()):
            if pyopencl_device == device_adapter._pyopencl_device:  # noqa: SLF001
                return cls(platform_adapter, pyopencl_device, device_idx)

        # Sanity check, should not be reachable as long as `pyopencl` is consistent.
        raise RuntimeError(
            f"{pyopencl_device} was not found among OpenCL devices"
        )  # pragma: no cover

    def __init__(
        self, platform_adapter: PlatformAdapter, pyopencl_device: pyopencl.Device, device_idx: int
    ):
        self._platform_adapter = platform_adapter
        self._device_idx = device_idx
        self._pyopencl_device = pyopencl_device
        self._params: OclDeviceParameters | None = None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, OclDeviceAdapter) and self._pyopencl_device == other._pyopencl_device
        )

    def __hash__(self) -> int:
        return hash((type(self), self._pyopencl_device))

    @property
    def platform_adapter(self) -> PlatformAdapter:
        return self._platform_adapter

    @property
    def device_idx(self) -> int:
        return self._device_idx

    @property
    def name(self) -> str:
        return self._pyopencl_device.name

    @property
    def params(self) -> DeviceParameters:
        if self._params is None:
            self._params = OclDeviceParameters(self._pyopencl_device)
        return self._params


class OclDeviceParameters(DeviceParameters):
    def __init__(self, pyopencl_device: pyopencl.Device):
        self._type = (
            DeviceType.CPU if pyopencl_device.type == pyopencl.device_type.CPU else DeviceType.GPU
        )

        if (
            pyopencl_device.platform.name == "Apple"
            and pyopencl_device.type == pyopencl.device_type.CPU
        ):
            self._max_local_sizes: tuple[int, ...]

            # Apple is being funny again.
            # As of MacOS 10.15.0, it reports the maximum local size as 1024 (for the device),
            # when even for the simplest kernel it turns out to be 128.
            # Moreover, if local_barrier() is used in the kernel, it becomes 1.
            self._max_total_local_size = 1
            self._max_local_sizes = (1, 1, 1)
        else:
            self._max_total_local_size = pyopencl_device.max_work_group_size
            self._max_local_sizes = tuple(pyopencl_device.max_work_item_sizes)

        # The limit of `2**address_bits` is for global size
        max_size = 2**pyopencl_device.address_bits // self._max_total_local_size
        self._max_num_groups = (max_size, max_size, max_size)

        if pyopencl_device.type == pyopencl.device_type.CPU:
            # For CPU both values do not make much sense
            self._local_mem_banks = 1
            self._warp_size = 1
        elif "cl_nv_device_attribute_query" in pyopencl_device.extensions:
            # If NV extensions are available, use them to query info
            cc = getattr(pyopencl_device, "compute_capability_major_nv", 1)
            self._local_mem_banks = 16 if cc < 2 else 32  # noqa: PLR2004
            self._warp_size = pyopencl_device.warp_size_nv
        elif pyopencl_device.vendor == "NVIDIA":
            # nVidia device, but no extensions.
            # Must be APPLE OpenCL implementation.
            # Assuming CC>=3
            self._local_mem_banks = 32
            self._warp_size = 32
        else:
            # AMD card.
            # Do not know how to query this info, so settle for most probable values.

            self._local_mem_banks = 32

            # An alternative is to query CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
            # for some arbitrary kernel.
            self._warp_size = 32

        self._local_mem_size = pyopencl_device.local_mem_size
        self._compute_units = pyopencl_device.max_compute_units

    @property
    def max_total_local_size(self) -> int:
        return self._max_total_local_size

    @property
    def max_local_sizes(self) -> tuple[int, ...]:
        return self._max_local_sizes

    @property
    def warp_size(self) -> int:
        return self._warp_size

    @property
    def max_num_groups(self) -> tuple[int, ...]:
        return self._max_num_groups

    @property
    def type(self) -> DeviceType:
        return self._type

    @property
    def local_mem_size(self) -> int:
        return self._local_mem_size

    @property
    def local_mem_banks(self) -> int:
        return self._local_mem_banks

    @property
    def compute_units(self) -> int:
        return self._compute_units

    def align_words(self, word_size: int) -> int:
        # TODO: is it possible to query it from the API?
        return {4: 16, 8: 16, 16: 8}[word_size]


class OclContextAdapter(ContextAdapter):
    @classmethod
    def from_pyopencl_devices(
        cls, pyopencl_devices: Sequence[pyopencl.Device]
    ) -> OclContextAdapter:
        """Creates a context based on one or several (distinct) PyOpenCL ``Device`` objects."""
        pyopencl_devices = normalize_object_sequence(pyopencl_devices, pyopencl.Device)
        return cls(pyopencl.Context(pyopencl_devices), pyopencl_devices=pyopencl_devices)

    @classmethod
    def from_pyopencl_context(cls, pyopencl_context: pyopencl.Context) -> OclContextAdapter:
        """Creates a context based on a (possibly multi-device) PyOpenCL ``Context`` object."""
        return cls(pyopencl_context)

    @classmethod
    def from_device_adapters(cls, device_adapters: Sequence[OclDeviceAdapter]) -> OclContextAdapter:
        """
        Creates a context based on one or several (distinct)
        :py:class:`OclDeviceAdapter` objects.
        """
        return cls.from_pyopencl_devices(
            [device_adapter._pyopencl_device for device_adapter in device_adapters]  # noqa: SLF001
        )

    def __init__(
        self,
        pyopencl_context: pyopencl.Context,
        pyopencl_devices: Sequence[pyopencl.Device] | None = None,
    ):
        if pyopencl_devices is None:
            pyopencl_devices = pyopencl_context.devices

        # Not checking here that all the `pyopencl_devices` are actually present
        # in `pyopencl_context`, since the constructor should only be called
        # from the trusted classmethods, which already ensure that.

        self._pyopencl_context = pyopencl_context

        device_adapters = [
            OclDeviceAdapter.from_pyopencl_device(device) for device in pyopencl_devices
        ]

        self._device_order = [device_adapter.device_idx for device_adapter in device_adapters]

        self._pyopencl_devices = dict(zip(self._device_order, pyopencl_devices, strict=True))

        self._device_adapters = {
            device_adapter.device_idx: device_adapter for device_adapter in device_adapters
        }

        self.platform_adapter = device_adapters[0].platform_adapter

        # On an Apple platform, in a multi-device context with nVidia cards it is necessary to have
        # any created OpenCL buffers allocated on the host (with ALLOC_HOST_PTR flag).
        # If it is not the case, using a subregion of such a buffer leads to a crash.
        is_multi_device = len(pyopencl_devices) > 1
        is_apple_platform = pyopencl_devices[0].platform.name == "Apple"
        has_geforce_device = any("GeForce" in device.name for device in pyopencl_devices)
        self._buffers_host_allocation = is_multi_device and is_apple_platform and has_geforce_device

    @property
    def device_adapters(self) -> Mapping[int, DeviceAdapter]:
        return self._device_adapters

    @property
    def device_order(self) -> list[int]:
        return self._device_order

    @staticmethod
    def render_prelude(*, fast_math: bool = False) -> str:
        return _PRELUDE.render(fast_math=fast_math, dtypes=dtypes)

    def compile_single_device(
        self,
        device_adapter: DeviceAdapter,
        prelude: str,
        src: str,
        *,
        keep: bool = False,
        fast_math: bool = False,
        compiler_options: Iterable[str] = [],
        constant_arrays: Mapping[str, ArrayMetadata] = {},
    ) -> OclProgramAdapter:
        # Will be checked in the upper levels.
        assert isinstance(device_adapter, OclDeviceAdapter)  # noqa: S101

        # Sanity check: should have been caught in compile()
        assert not constant_arrays  # noqa: S101

        src = prelude + src

        if keep:
            temp_dir = Path(mkdtemp())
            temp_file_path = temp_dir / "kernel.cl"

            with temp_file_path.open("w") as f:
                # `str()` is for convenience of mocking;
                # during a normal operation `src` is already a string
                f.write(str(src))

            print("*** compiler output in", temp_dir)  # noqa: T201

        else:
            temp_dir = None

        options = list(compiler_options) + (
            ["-cl-mad-enable", "-cl-fast-relaxed-math"] if fast_math else []
        )

        pyopencl_program = pyopencl.Program(self._pyopencl_context, src)

        try:
            pyopencl_program.build(
                devices=[self._pyopencl_devices[device_adapter.device_idx]],
                options=options,
                cache_dir=str(temp_dir) if temp_dir else None,
            )
        except pyopencl.RuntimeError as exc:
            raise AdapterCompilationError(exc, src) from exc

        return OclProgramAdapter(self, device_adapter, pyopencl_program, src)

    def allocate(self, device_adapter: DeviceAdapter, size: int) -> OclBufferAdapter:
        # Will be checked in the upper levels.
        assert isinstance(device_adapter, OclDeviceAdapter)  # noqa: S101

        flags: int = pyopencl.mem_flags.READ_WRITE
        if self._buffers_host_allocation:
            flags |= pyopencl.mem_flags.ALLOC_HOST_PTR

        pyopencl_buffer = pyopencl.Buffer(self._pyopencl_context, flags, size=size)

        return OclBufferAdapter(self, device_adapter, pyopencl_buffer)

    def make_queue_adapter(self, device_adapter: DeviceAdapter) -> OclQueueAdapter:
        # Will be checked in the upper levels.
        assert isinstance(device_adapter, OclDeviceAdapter)  # noqa: S101
        queue = pyopencl.CommandQueue(
            self._pyopencl_context,
            device=self._device_adapters[device_adapter.device_idx]._pyopencl_device,  # noqa: SLF001
        )

        return OclQueueAdapter(self, device_adapter, queue)

    def deactivate(self) -> None:
        # No need to do anything for OpenCL
        pass


class OclBufferAdapter(BufferAdapter):
    def __init__(
        self,
        context_adapter: OclContextAdapter,
        device_adapter: OclDeviceAdapter,
        pyopencl_buffer: pyopencl.Buffer,
    ):
        self._context_adapter = context_adapter
        self._pyopencl_buffer = pyopencl_buffer
        self._device_adapter = device_adapter

    @property
    def kernel_arg(self) -> pyopencl.Buffer:
        return self._pyopencl_buffer

    @property
    def size(self) -> int:
        return self._pyopencl_buffer.size

    @property
    def offset(self) -> int:
        return self._pyopencl_buffer.offset

    def set(
        self,
        queue_adapter: QueueAdapter,
        source: NDArray[Any] | BufferAdapter,
        *,
        sync: bool = False,
    ) -> None:
        # Will be checked in the upper levels.
        assert isinstance(queue_adapter, OclQueueAdapter)  # noqa: S101

        pyopencl_queue = queue_adapter._pyopencl_queue  # noqa: SLF001

        if isinstance(source, BufferAdapter):
            # Will be checked in the upper levels.
            assert isinstance(source, OclBufferAdapter)  # noqa: S101

            pyopencl.enqueue_copy(
                pyopencl_queue,
                self._pyopencl_buffer,
                source._pyopencl_buffer,  # noqa: SLF001
            )
        else:
            pyopencl.enqueue_copy(
                pyopencl_queue,
                self._pyopencl_buffer,
                source,
                is_blocking=sync,
            )

    def get(
        self,
        queue_adapter: QueueAdapter,
        host_array: NDArray[Any],
        *,
        async_: bool = False,
    ) -> None:
        # Will be checked in the upper levels.
        assert isinstance(queue_adapter, OclQueueAdapter)  # noqa: S101
        pyopencl.enqueue_copy(
            queue_adapter._pyopencl_queue,  # noqa: SLF001
            host_array,
            self._pyopencl_buffer,
            is_blocking=not async_,
        )

    def get_sub_region(self, origin: int, size: int) -> OclBufferAdapter:
        return OclBufferAdapter(
            self._context_adapter,
            self._device_adapter,
            self._pyopencl_buffer.get_sub_region(origin, size),
        )


class OclQueueAdapter(QueueAdapter):
    def __init__(
        self,
        context_adapter: OclContextAdapter,
        device_adapter: OclDeviceAdapter,
        pyopencl_queue: pyopencl.CommandQueue,
    ):
        self._context_adapter = context_adapter
        self._device_adapter = device_adapter
        self._pyopencl_queue = pyopencl_queue

    def synchronize(self) -> None:
        self._pyopencl_queue.finish()


class OclProgramAdapter(ProgramAdapter):
    def __init__(
        self,
        context_adapter: OclContextAdapter,
        device_adapter: OclDeviceAdapter,
        pyopencl_program: pyopencl.Program,
        source: str,
    ):
        self._context_adapter = context_adapter
        self._device_adapter = device_adapter
        self._pyopencl_program = pyopencl_program
        self._source = source

    @property
    def source(self) -> str:
        return self._source

    # Note that `pyopencl.Program` does not cache kernels because in PyOpenCL they are mutable.
    # We don't use any of their mutable capabilities, so it is safe for us to cache them.
    @functools.cache  # noqa: B019
    def __getattr__(self, kernel_name: str) -> OclKernelAdapter:
        pyopencl_kernel = getattr(self._pyopencl_program, kernel_name)
        return OclKernelAdapter(self, self._device_adapter, pyopencl_kernel)

    def set_constant_buffer(
        self,
        _queue_adapter: QueueAdapter,
        _name: str,
        _arr: BufferAdapter | NDArray[Any],
    ) -> None:
        raise RuntimeError("OpenCL does not allow setting constant arrays externally")


class OclKernelAdapter(KernelAdapter):
    def __init__(
        self,
        program_adapter: OclProgramAdapter,
        device_adapter: OclDeviceAdapter,
        pyopencl_kernel: pyopencl.Kernel,
    ):
        self._program_adapter = program_adapter
        self._device_adapter = device_adapter
        self._pyopencl_kernel = pyopencl_kernel

    @property
    def program_adapter(self) -> OclProgramAdapter:
        return self._program_adapter

    def prepare(
        self, global_size: Sequence[int], local_size: Sequence[int] | None = None
    ) -> OclPreparedKernelAdapter:
        return OclPreparedKernelAdapter(self, global_size, local_size)

    @property
    def max_total_local_size(self) -> int:
        return self._pyopencl_kernel.get_work_group_info(
            pyopencl.kernel_work_group_info.WORK_GROUP_SIZE,
            self._device_adapter._pyopencl_device,  # noqa: SLF001
        )


class OclPreparedKernelAdapter(PreparedKernelAdapter):
    def __init__(
        self,
        kernel_adapter: OclKernelAdapter,
        global_size: Sequence[int],
        local_size: Sequence[int] | None = None,
    ):
        self._kernel_adapter = kernel_adapter
        self._local_size = tuple(local_size) if local_size is not None else None
        self._global_size = tuple(global_size)
        self._device_adapter = kernel_adapter._device_adapter  # noqa: SLF001
        self._pyopencl_kernel = kernel_adapter._pyopencl_kernel  # noqa: SLF001

    def __call__(
        self,
        queue_adapter: QueueAdapter,
        *args: BufferAdapter | numpy.generic,
        cu_dynamic_local_mem: int = 0,
    ) -> pyopencl.Event:
        if cu_dynamic_local_mem != 0:
            raise ValueError(
                "`cu_dynamic_local_mem` must be zero for OpenCL kernels; "
                "dynamic local memory allocation is not supported"
            )

        # We have to keep the signature more general because of the base class,
        # but the upper levels will ensure this is the case.
        assert isinstance(queue_adapter, OclQueueAdapter)  # noqa: S101

        # Sanity check. If it happened, there's something wrong in the abstraction layer logic
        # (Program and PreparedKernel).
        assert self._device_adapter == queue_adapter._device_adapter  # noqa: S101, SLF001

        backend_args: list[pyopencl.Buffer | numpy.generic] = []
        for arg in args:
            if isinstance(arg, BufferAdapter):
                karg = arg.kernel_arg
                # Need this to pass it to PyOpenCL. It may be an actual Buffer, or a virtual buffer
                assert isinstance(karg, pyopencl.Buffer)  # noqa: S101
                backend_args.append(karg)
            else:
                backend_args.append(arg)

        return self._pyopencl_kernel(
            queue_adapter._pyopencl_queue,  # noqa: SLF001
            self._global_size,
            self._local_size,
            *backend_args,
        )

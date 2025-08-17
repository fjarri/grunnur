from __future__ import annotations

from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import numpy

# Skipping coverage count - can't test properly
try:  # pragma: no cover
    import pycuda.compiler as pycuda_compiler
    import pycuda.driver as pycuda_driver
    import pycuda.tools as pycuda_tools
except ImportError:  # pragma: no cover
    # these variables are used for a PyCUDA mock during tests
    pycuda_driver = None  # type: ignore[assignment]
    pycuda_compiler = None  # type: ignore[assignment]
    pycuda_tools = None  # type: ignore[assignment]

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
from ._utils import get_launch_size, normalize_object_sequence, prod

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import NDArray

    from ._array_metadata import ArrayMetadata


# Another way would be to place it in the try block and only set `_avaialable`
# if CUDA was initialized correctly, but it is better to distinguish import errors
# (PyCUDA not installed, which is quite common) and initialization errors
# (some problem with the package).
#
# Skipping coverage count - can't test properly
if pycuda_driver is not None:  # pragma: no cover
    pycuda_driver.init()


_API_ID = APIID("cuda")
_TEMPLATE = Template.from_associated_file(__file__)
_PRELUDE = _TEMPLATE.get_def("prelude")
_CONSTANT_ARRAYS_DEF = _TEMPLATE.get_def("constant_arrays_def")


class CuAPIAdapterFactory(APIAdapterFactory):
    @property
    def api_id(self) -> APIID:
        return _API_ID

    @property
    def available(self) -> bool:
        return pycuda_driver is not None

    def make_api_adapter(self) -> CuAPIAdapter:
        if not self.available:
            raise ImportError(
                "CUDA API is not operational. Check if PyCUDA is installed correctly."
            )

        return CuAPIAdapter()


class CuAPIAdapter(APIAdapter):
    @property
    def id(self) -> APIID:
        return _API_ID

    @property
    def platform_count(self) -> int:
        return 1

    def get_platform_adapters(self) -> tuple[CuPlatformAdapter, ...]:
        return (CuPlatformAdapter(self),)

    def isa_backend_device(self, obj: Any) -> bool:
        return isinstance(obj, pycuda_driver.Device)

    def isa_backend_platform(self, _obj: Any) -> bool:
        return False  # CUDA backend doesn't have platforms

    def isa_backend_context(self, obj: Any) -> bool:
        return isinstance(obj, pycuda_driver.Context)

    def make_device_adapter(self, pycuda_device: pycuda_driver.Device) -> CuDeviceAdapter:
        return CuDeviceAdapter.from_pycuda_device(pycuda_device)

    def make_platform_adapter(self, _pycuda_platform: Any) -> CuPlatformAdapter:  # pragma: no cover
        # Not going to be called since `isa_backend_platform()` always returns `False`
        raise RuntimeError("CUDA does not have the concept of a platform")

    def make_context_adapter_from_backend_contexts(
        self, backend_contexts: Sequence[Any], *, take_ownership: bool
    ) -> CuContextAdapter:
        return CuContextAdapter.from_pycuda_contexts(
            backend_contexts, take_ownership=take_ownership
        )

    def make_context_adapter_from_device_adapters(
        self, device_adapters: Sequence[DeviceAdapter]
    ) -> CuContextAdapter:
        cu_device_adapters = normalize_object_sequence(device_adapters, CuDeviceAdapter)
        return CuContextAdapter.from_device_adapters(cu_device_adapters)


class CuPlatformAdapter(PlatformAdapter):
    def __init__(self, api_adapter: CuAPIAdapter):
        self._api_adapter = api_adapter

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CuPlatformAdapter) and self._api_adapter == other._api_adapter

    def __hash__(self) -> int:
        return hash((type(self), self._api_adapter))

    @property
    def api_adapter(self) -> CuAPIAdapter:
        return self._api_adapter

    @property
    def platform_idx(self) -> int:
        return 0

    @property
    def name(self) -> str:
        return "nVidia CUDA"

    @property
    def vendor(self) -> str:
        return "nVidia"

    @property
    def version(self) -> str:
        return "CUDA " + ".".join(str(x) for x in pycuda_driver.get_version())

    @property
    def device_count(self) -> int:
        return pycuda_driver.Device.count()

    def get_device_adapters(self) -> tuple[CuDeviceAdapter, ...]:
        return tuple(
            CuDeviceAdapter(self, pycuda_driver.Device(device_idx), device_idx)
            for device_idx in range(self.device_count)
        )


class CuDeviceAdapter(DeviceAdapter):
    @classmethod
    def from_pycuda_device(cls, pycuda_device: pycuda_driver.Device) -> CuDeviceAdapter:
        """Creates this object from a PyCuda ``Device`` object."""
        platform_adapter = CuPlatformAdapter(CuAPIAdapter())
        for device_idx, device_adapter in enumerate(platform_adapter.get_device_adapters()):
            if pycuda_device == device_adapter.pycuda_device:
                return cls(platform_adapter, pycuda_device, device_idx)

        # Sanity check, should not be reachable as long as `pycuda` is consistent.
        raise RuntimeError(f"{pycuda_device} was not found among CUDA devices")  # pragma: no cover

    def __init__(
        self,
        platform_adapter: CuPlatformAdapter,
        pycuda_device: pycuda_driver.Device,
        device_idx: int,
    ):
        self._platform_adapter = platform_adapter
        self._device_idx = device_idx
        self.pycuda_device = pycuda_device
        self._params: CuDeviceParameters | None = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CuDeviceAdapter) and self.pycuda_device == other.pycuda_device

    def __hash__(self) -> int:
        return hash((type(self), self.pycuda_device))

    @property
    def platform_adapter(self) -> CuPlatformAdapter:
        return self._platform_adapter

    @property
    def device_idx(self) -> int:
        return self._device_idx

    @property
    def name(self) -> str:
        return self.pycuda_device.name()

    @property
    def params(self) -> CuDeviceParameters:
        if self._params is None:
            self._params = CuDeviceParameters(self.pycuda_device)
        return self._params


class CuDeviceParameters(DeviceParameters):
    def __init__(self, pycuda_device: pycuda_driver.Device):
        self._type = DeviceType.GPU

        self._max_total_local_size = pycuda_device.max_threads_per_block
        self._max_local_sizes = (
            pycuda_device.max_block_dim_x,
            pycuda_device.max_block_dim_y,
            pycuda_device.max_block_dim_z,
        )

        self._max_num_groups = (
            pycuda_device.max_grid_dim_x,
            pycuda_device.max_grid_dim_y,
            pycuda_device.max_grid_dim_z,
        )

        # there is no corresponding constant in the API at the moment
        self._local_mem_banks = 16 if pycuda_device.compute_capability()[0] < 2 else 32  # noqa: PLR2004

        self._warp_size = pycuda_device.warp_size
        self._local_mem_size = pycuda_device.max_shared_memory_per_block
        self._compute_units = pycuda_device.multiprocessor_count
        self._devdata = pycuda_tools.DeviceData(pycuda_device)

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
        return self._devdata.align_words(word_size)


class _ContextStack:
    """A helper class that keeps the CUDA context stack state."""

    def __init__(self, pycuda_contexts: Sequence[pycuda_driver.Context], *, take_ownership: bool):
        self._active_context: int | None = None
        self._owns_contexts = take_ownership
        self._pycuda_contexts: dict[int, pycuda_driver.Context] = {}

        self.device_order: list[int] = []
        self.device_adapters: dict[int, CuDeviceAdapter] = {}

        # A shortcut for the case when we're given a single external context
        # and told not to take ownership.
        dont_push = len(pycuda_contexts) == 1 and not take_ownership

        for context in pycuda_contexts:
            if not dont_push:
                context.push()
            device = context.get_device()
            if not dont_push:
                context.pop()

            device_adapter = CuDeviceAdapter.from_pycuda_device(device)
            self.device_adapters[device_adapter.device_idx] = device_adapter
            self._pycuda_contexts[device_adapter.device_idx] = context
            self.device_order.append(device_adapter.device_idx)

    def deactivate(self) -> None:
        if self._active_context is not None and self._owns_contexts:
            self._pycuda_contexts[self._active_context].pop()
            self._active_context = None

    def activate(self, device_idx: int) -> None:
        if self._active_context != device_idx and self._owns_contexts:
            self.deactivate()
            self._pycuda_contexts[device_idx].push()
            self._active_context = device_idx

    def __del__(self) -> None:
        self.deactivate()


class CuContextAdapter(ContextAdapter):
    """Wraps CUDA contexts for several devices."""

    @classmethod
    def from_pycuda_devices(
        cls, pycuda_devices: Sequence[pycuda_driver.Device]
    ) -> CuContextAdapter:
        """Creates a context based on one or several (distinct) PyCuda ``Device`` objects."""
        pycuda_devices = normalize_object_sequence(pycuda_devices, pycuda_driver.Device)
        contexts = []
        for device in pycuda_devices:
            context = device.make_context()
            pycuda_driver.Context.pop()
            contexts.append(context)
        return cls(contexts, take_ownership=True)

    @classmethod
    def from_pycuda_contexts(
        cls, pycuda_contexts: Sequence[pycuda_driver.Context], *, take_ownership: bool
    ) -> CuContextAdapter:
        """
        Creates a context based on one or several (distinct) PyCuda ``Context`` objects.
        If ``take_ownership`` is ``False``, and there is only one context given,
        it should be pushed to the top of the context stack.
        Otherwise, none of the PyCuda contexts should be pushed to the context stack.
        """
        if len(pycuda_contexts) > 1 and not take_ownership:
            raise ValueError(
                "When dealing with multiple CUDA contexts, Grunnur must be the one managing them"
            )
        pycuda_contexts = normalize_object_sequence(pycuda_contexts, pycuda_driver.Context)
        return cls(pycuda_contexts, take_ownership=take_ownership)

    @classmethod
    def from_device_adapters(cls, device_adapters: Sequence[CuDeviceAdapter]) -> CuContextAdapter:
        """
        Creates a context based on one or several (distinct)
        :py:class:`CuDeviceAdapter` objects.
        """
        return cls.from_pycuda_devices(
            [device_adapter.pycuda_device for device_adapter in device_adapters]
        )

    def __init__(self, pycuda_contexts: Sequence[pycuda_driver.Context], *, take_ownership: bool):
        self._context_stack = _ContextStack(pycuda_contexts, take_ownership=take_ownership)
        self._device_adapters = self._context_stack.device_adapters
        self._device_order = self._context_stack.device_order

        # Eager activation to help some tests.
        self._context_stack.activate(next(iter(self._device_order)))

    @property
    def device_adapters(self) -> Mapping[int, CuDeviceAdapter]:
        return self._device_adapters

    @property
    def device_order(self) -> list[int]:
        return self._device_order

    def activate_device(self, device_idx: int) -> None:
        """
        Activates the device ``device_idx``.
        Pops a previous context from the stack, if there was one pushed before,
        and pushes the corresponding context to the stack.
        """
        self._context_stack.activate(device_idx)

    def deactivate(self) -> None:
        self._context_stack.deactivate()

    @staticmethod
    def render_prelude(*, fast_math: bool = False) -> str:
        return _PRELUDE.render(fast_math=fast_math)

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
    ) -> CuProgramAdapter:
        # Will be checked in the higher levels
        assert isinstance(device_adapter, CuDeviceAdapter)  # noqa: S101

        normalized_constant_arrays = {}
        for name, metadata in constant_arrays.items():
            normalized_constant_arrays[name] = (metadata.buffer_size, metadata.dtype)

        constant_arrays_src = _CONSTANT_ARRAYS_DEF.render(
            dtypes=dtypes, constant_arrays=normalized_constant_arrays
        )

        options = list(compiler_options) + (["-use_fast_math"] if fast_math else [])
        full_src = prelude + constant_arrays_src + src
        self.activate_device(device_adapter.device_idx)

        # For some reason, `keep=True` does not work without an explicit `cache_dir`.
        # A new temporary dir is still created, and everything is placed there,
        # and `cache_dir` receives a copy of `kernel.cubin`.
        cache_dir = mkdtemp() if keep else None

        try:
            module = pycuda_compiler.SourceModule(
                full_src, no_extern_c=True, options=options, keep=keep, cache_dir=cache_dir
            )
        except pycuda_driver.CompileError as exc:
            raise AdapterCompilationError(exc, full_src) from exc

        return CuProgramAdapter(self, device_adapter.device_idx, module, full_src)

    def allocate(self, device_adapter: DeviceAdapter, size: int) -> CuBufferAdapter:
        # Will be checked in the higher levels
        assert isinstance(device_adapter, CuDeviceAdapter)  # noqa: S101
        return CuBufferAdapter(self, device_adapter.device_idx, size)

    def make_queue_adapter(self, device_adapter: DeviceAdapter) -> CuQueueAdapter:
        # Will be checked in the higher levels
        assert isinstance(device_adapter, CuDeviceAdapter)  # noqa: S101
        self.activate_device(device_adapter.device_idx)
        stream = pycuda_driver.Stream()
        return CuQueueAdapter(self, device_adapter.device_idx, stream)


class CuBufferAdapter(BufferAdapter):
    def __init__(
        self,
        context_adapter: CuContextAdapter,
        device_idx: int,
        size: int,
        *,
        offset: int = 0,
        managed: bool = False,  # noqa: ARG002
        base_buffer: CuBufferAdapter | None = None,
    ):
        if base_buffer is None:
            context_adapter.activate_device(device_idx)
            base_allocation = pycuda_driver.mem_alloc(size)
        else:
            base_allocation = base_buffer._base_allocation  # noqa: SLF001

        self._size = size
        self._offset = offset
        self._context_adapter = context_adapter
        self._device_idx = device_idx

        self._base_buffer = base_buffer
        self._base_allocation: pycuda_driver.DeviceAllocation = base_allocation
        self._ptr = numpy.uintp(self._base_allocation) + numpy.uintp(offset)

    @property
    def kernel_arg(self) -> numpy.uintp:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    @property
    def offset(self) -> int:
        return self._offset

    def get_sub_region(self, origin: int, size: int) -> CuBufferAdapter:
        # Sanity check, should be already checked in `Buffer`
        assert origin + size <= self._size  # noqa: S101
        base_buffer = self if self._base_buffer is None else self._base_buffer
        return CuBufferAdapter(
            self._context_adapter,
            self._device_idx,
            size,
            offset=self._offset + origin,
            base_buffer=base_buffer,
        )

    def set(
        self,
        queue_adapter: QueueAdapter,
        source: NDArray[Any] | BufferAdapter,
        *,
        sync: bool = False,
    ) -> None:
        # Will be checked in the upper levels.
        assert isinstance(queue_adapter, CuQueueAdapter)  # noqa: S101
        device_idx = queue_adapter._device_idx  # noqa: SLF001
        self._context_adapter.activate_device(device_idx)

        # PyCUDA needs pointers to be passed as `numpy.number` to kernels,
        # but `memcpy` functions require Python `int`s.
        ptr = int(self._ptr) if isinstance(self._ptr, numpy.number) else self._ptr

        if isinstance(source, numpy.ndarray):
            if sync:
                pycuda_driver.memcpy_htod(ptr, source)
            else:
                pycuda_driver.memcpy_htod_async(ptr, source, stream=queue_adapter._pycuda_stream)  # noqa: SLF001
        else:
            # Will be checked in the upper levels.
            assert isinstance(source, CuBufferAdapter)  # noqa: S101
            buf_ptr = int(source._ptr) if isinstance(source._ptr, numpy.number) else source._ptr  # noqa: SLF001
            if sync:
                pycuda_driver.memcpy_dtod(ptr, buf_ptr, source.size)
            else:
                pycuda_driver.memcpy_dtod_async(
                    ptr,
                    buf_ptr,
                    source.size,
                    stream=queue_adapter._pycuda_stream,  # noqa: SLF001
                )

    def get(
        self,
        queue_adapter: QueueAdapter,
        host_array: NDArray[Any],
        *,
        async_: bool = False,
    ) -> None:
        # Will be checked in the upper levels.
        assert isinstance(queue_adapter, CuQueueAdapter)  # noqa: S101
        device_idx = queue_adapter._device_idx  # noqa: SLF001
        self._context_adapter.activate_device(device_idx)

        # PyCUDA needs pointers to be passed as `numpy.number` to kernels,
        # but `memcpy` functions require Python `int`s.
        ptr = int(self._ptr) if isinstance(self._ptr, numpy.number) else self._ptr

        if async_:
            pycuda_driver.memcpy_dtoh_async(host_array, ptr, stream=queue_adapter._pycuda_stream)  # noqa: SLF001
        else:
            pycuda_driver.memcpy_dtoh(host_array, ptr)


class CuQueueAdapter(QueueAdapter):
    def __init__(
        self,
        context_adapter: CuContextAdapter,
        device_idx: int,
        pycuda_stream: pycuda_driver.Stream,
    ):
        self._context_adapter = context_adapter
        self._device_idx = device_idx
        self._pycuda_stream = pycuda_stream

    def synchronize(self) -> None:
        self._context_adapter.activate_device(self._device_idx)
        self._pycuda_stream.synchronize()


class CuProgramAdapter(ProgramAdapter):
    def __init__(
        self,
        context_adapter: CuContextAdapter,
        device_idx: int,
        pycuda_program: pycuda_compiler.SourceModule,
        source: str,
    ):
        self._context_adapter = context_adapter
        self._device_idx = device_idx
        self._pycuda_program = pycuda_program
        self._source = source

    @property
    def source(self) -> str:
        return self._source

    def __getattr__(self, kernel_name: str) -> CuKernelAdapter:
        pycuda_kernel = self._pycuda_program.get_function(kernel_name)
        return CuKernelAdapter(self, self._device_idx, pycuda_kernel)

    def set_constant_buffer(
        self,
        queue_adapter: QueueAdapter,
        name: str,
        arr: BufferAdapter | NDArray[Any],
    ) -> None:
        """Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context."""
        # Will be checked in the upper levels
        assert isinstance(queue_adapter, CuQueueAdapter)  # noqa: S101

        self._context_adapter.activate_device(self._device_idx)
        symbol, size = self._pycuda_program.get_global(name)

        pycuda_stream = queue_adapter._pycuda_stream  # noqa: SLF001

        if isinstance(arr, BufferAdapter):
            # Will be checked in the upper levels
            assert isinstance(arr, CuBufferAdapter)  # noqa: S101
            transfer_size = arr.size
        elif isinstance(arr, numpy.ndarray):
            transfer_size = prod(arr.shape) * arr.dtype.itemsize
        else:  # pragma: no cover
            # Shouldn't reach this path because the type is already checked by the caller.
            # Nevertheless leaving it here as a sanity check.
            raise TypeError(f"Unsupported array type: {type(arr)}")

        if transfer_size != size:
            raise ValueError(
                f"Incorrect size of the constant buffer; expected {size} bytes, got {transfer_size}"
            )

        if isinstance(arr, CuBufferAdapter):
            pycuda_driver.memcpy_dtod_async(symbol, int(arr._ptr), arr.size, stream=pycuda_stream)  # noqa: SLF001
        else:
            # This serves two purposes:
            # 1. Gives us a pagelocked array, as PyCUDA requires
            # 2. Makes the array contiguous
            # Constant array are usually quite small, so it won't affect the performance.
            buf = pycuda_driver.pagelocked_empty(arr.shape, arr.dtype)
            numpy.copyto(buf, arr)
            pycuda_driver.memcpy_htod_async(symbol, buf, stream=pycuda_stream)


class CuKernelAdapter(KernelAdapter):
    def __init__(
        self,
        program_adapter: CuProgramAdapter,
        device_idx: int,
        pycuda_function: pycuda_compiler.Function,
    ):
        self._context_adapter = program_adapter._context_adapter  # noqa: SLF001
        self._program_adapter = program_adapter
        self._device_idx = device_idx
        self._pycuda_function = pycuda_function

    @property
    def program_adapter(self) -> CuProgramAdapter:
        return self._program_adapter

    @property
    def max_total_local_size(self) -> int:
        return self._pycuda_function.get_attribute(
            pycuda_driver.function_attribute.MAX_THREADS_PER_BLOCK
        )

    def prepare(
        self, global_size: Sequence[int], local_size: Sequence[int] | None = None
    ) -> CuPreparedKernelAdapter:
        return CuPreparedKernelAdapter(self, global_size, local_size)


class CuPreparedKernelAdapter(PreparedKernelAdapter):
    def __init__(
        self,
        kernel_adapter: CuKernelAdapter,
        global_size: Sequence[int],
        local_size: Sequence[int] | None = None,
    ):
        context_adapter = kernel_adapter._context_adapter  # noqa: SLF001
        device_idx = kernel_adapter._device_idx  # noqa: SLF001

        device_adapter = context_adapter._device_adapters[device_idx]  # noqa: SLF001

        grid, block = get_launch_size(
            device_adapter.params.max_local_sizes,
            device_adapter.params.max_total_local_size,
            global_size,
            local_size,
        )

        # append missing dimensions, otherwise PyCUDA will complain
        max_dims = len(device_adapter.params.max_local_sizes)
        self._block = block + (1,) * (max_dims - len(block))
        self._grid = grid + (1,) * (max_dims - len(grid))

        self._device_idx = device_idx
        self._context_adapter = context_adapter
        self._pycuda_function = kernel_adapter._pycuda_function  # noqa: SLF001

    def __call__(
        self,
        queue_adapter: QueueAdapter,
        *args: BufferAdapter | numpy.generic,
        cu_dynamic_local_mem: int = 0,
    ) -> None:
        # Will be checked in the upper levels
        assert isinstance(queue_adapter, CuQueueAdapter)  # noqa: S101

        # Sanity check. If it happened, there's something wrong in the abstraction layer logic
        # (Program and PreparedKernel).
        assert self._device_idx == queue_adapter._device_idx  # noqa: SLF001, S101

        backend_args = []
        for arg in args:
            if isinstance(arg, BufferAdapter):
                karg = arg.kernel_arg
                backend_args.append(karg)
            else:
                backend_args.append(arg)

        self._context_adapter.activate_device(self._device_idx)
        return self._pycuda_function(
            *backend_args,
            grid=self._grid,
            block=self._block,
            stream=queue_adapter._pycuda_stream,  # noqa: SLF001
            shared=cu_dynamic_local_mem,
        )

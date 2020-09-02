from __future__ import annotations

from tempfile import mkdtemp
from typing import Iterable, Union, Tuple, Sequence

import numpy

# Skipping coverage count - can't test properly
try: # pragma: no cover
    import pycuda.driver as pycuda_driver
    import pycuda.compiler as pycuda_compiler
except ImportError: # pragma: no cover
    # these variables are used for a PyCUDA mock during tests
    pycuda_driver = None
    pycuda_compiler = None

from .utils import wrap_in_tuple, prod, normalize_object_sequence, get_launch_size
from .template import Template
from . import dtypes
from .adapter_base import (
    APIID, DeviceType, APIAdapterFactory, APIAdapter, PlatformAdapter, DeviceAdapter,
    DeviceParameters, ContextAdapter, BufferAdapter, QueueAdapter, ProgramAdapter, KernelAdapter,
    AdapterCompilationError, PreparedKernelAdapter)


# Another way would be to place it in the try block and only set `_avaialable`
# if CUDA was initialized correctly, but it is better to distinguish import errors
# (PyCUDA not installed, which is quite common) and initialization errors
# (some problem with the package).
#
# Skipping coverage count - can't test properly
if pycuda_driver is not None: # pragma: no cover
    pycuda_driver.init()


_API_ID = APIID('cuda')
_TEMPLATE = Template.from_associated_file(__file__)
_PRELUDE = _TEMPLATE.get_def('prelude')
_CONSTANT_ARRAYS_DEF = _TEMPLATE.get_def('constant_arrays_def')


class CuAPIAdapterFactory(APIAdapterFactory):

    @property
    def api_id(self):
        return _API_ID

    @property
    def available(self):
        return pycuda_driver is not None

    def make_api_adapter(self):
        if not self.available:
            raise ImportError(
                "CUDA API is not operational. Check if PyCUDA is installed correctly.")

        return CuAPIAdapter()


class CuAPIAdapter(APIAdapter):

    @property
    def id(self):
        return _API_ID

    @property
    def platform_count(self):
        return 1

    def get_platform_adapters(self):
        return [CuPlatformAdapter(self)]

    def isa_backend_device(self, obj):
        return isinstance(obj, pycuda_driver.Device)

    def isa_backend_platform(self, _):
        return False # CUDA backend doesn't have platforms

    def isa_backend_context(self, obj):
        return isinstance(obj, pycuda_driver.Context)

    def make_device_adapter(self, pycuda_device):
        return CuDeviceAdapter.from_pycuda_device(pycuda_device)

    def make_platform_adapter(self, pycuda_platform): # pragma: no cover
        # Not going to be called since `isa_backend_platform()` always returns `False`
        raise Exception("CUDA does not have the concept of a platform")

    def make_context_adapter_from_backend_contexts(self, backend_contexts, take_ownership):
        return CuContextAdapter.from_pycuda_contexts(backend_contexts, take_ownership)

    def make_context_adapter_from_device_adapters(self, device_adapters):
        return CuContextAdapter.from_device_adapters(device_adapters)


class CuPlatformAdapter(PlatformAdapter):

    def __init__(self, api_adapter):
        self._api_adapter = api_adapter

    def __eq__(self, other):
        return type(self) == type(other) and self._api_adapter == other._api_adapter

    def __hash__(self):
        return hash((type(self), self._api_adapter))

    @property
    def api_adapter(self):
        return self._api_adapter

    @property
    def platform_idx(self):
        return 0

    @property
    def name(self):
        return "nVidia CUDA"

    @property
    def vendor(self):
        return "nVidia"

    @property
    def version(self):
        return "CUDA " + ".".join(str(x) for x in pycuda_driver.get_version())

    @property
    def device_count(self):
        return pycuda_driver.Device.count()

    def get_device_adapters(self):
        return [
            CuDeviceAdapter(self, pycuda_driver.Device(device_idx), device_idx)
            for device_idx in range(self.device_count)]


class CuDeviceAdapter(DeviceAdapter):

    @classmethod
    def from_pycuda_device(cls, pycuda_device: pycuda_driver.Device) -> CuDeviceAdapter:
        """
        Creates this object from a PyCuda ``Device`` object.
        """
        platform_adapter = CuPlatformAdapter(CuAPIAdapter())
        for device_idx, device_adapter in enumerate(platform_adapter.get_device_adapters()):
            if pycuda_device == device_adapter.pycuda_device:
                return cls(platform_adapter, pycuda_device, device_idx)

        # Sanity check, should not be reachable as long as `pycuda` is consistent.
        raise RuntimeError(f"{pycuda_device} was not found among CUDA devices") # pragma: no cover

    def __init__(self, platform_adapter, pycuda_device, device_idx):
        self._platform_adapter = platform_adapter
        self._device_idx = device_idx
        self.pycuda_device = pycuda_device
        self._params = None

    def __eq__(self, other):
        return type(self) == type(other) and self.pycuda_device == other.pycuda_device

    def __hash__(self):
        return hash((type(self), self.pycuda_device))

    @property
    def platform_adapter(self):
        return self._platform_adapter

    @property
    def device_idx(self):
        return self._device_idx

    @property
    def name(self):
        return self.pycuda_device.name()

    @property
    def params(self):
        if self._params is None:
            self._params = CuDeviceParameters(self.pycuda_device)
        return self._params


class CuDeviceParameters(DeviceParameters):

    def __init__(self, pycuda_device):

        self._type = DeviceType.GPU

        self._max_total_local_size = pycuda_device.max_threads_per_block
        self._max_local_sizes = (
            pycuda_device.max_block_dim_x,
            pycuda_device.max_block_dim_y,
            pycuda_device.max_block_dim_z)

        self._max_num_groups = (
            pycuda_device.max_grid_dim_x,
            pycuda_device.max_grid_dim_y,
            pycuda_device.max_grid_dim_z)

        # there is no corresponding constant in the API at the moment
        self._local_mem_banks = 16 if pycuda_device.compute_capability()[0] < 2 else 32

        self._warp_size = pycuda_device.warp_size
        self._local_mem_size = pycuda_device.max_shared_memory_per_block
        self._compute_units = pycuda_device.multiprocessor_count

    @property
    def max_total_local_size(self):
        return self._max_total_local_size

    @property
    def max_local_sizes(self):
        return self._max_local_sizes

    @property
    def warp_size(self):
        return self._warp_size

    @property
    def max_num_groups(self):
        return self._max_num_groups

    @property
    def type(self):
        return self._type

    @property
    def local_mem_size(self):
        return self._local_mem_size

    @property
    def local_mem_banks(self):
        return self._local_mem_banks

    @property
    def compute_units(self):
        return self._compute_units


class _ContextStack:
    """
    A helper class that keeps the CUDA context stack state.
    """

    def __init__(self, pycuda_contexts, take_ownership):
        self._pycuda_contexts = pycuda_contexts
        self._active_context = None
        self._owns_contexts = take_ownership

    def deactivate(self):
        if self._active_context is not None and self._owns_contexts:
            self._pycuda_contexts[self._active_context].pop()
            self._active_context = None

    def activate(self, device_idx):
        if self._active_context != device_idx and self._owns_contexts:
            self.deactivate()
            self._pycuda_contexts[device_idx].push()
            self._active_context = device_idx

    def __del__(self):
        self.deactivate()


class CuContextAdapter(ContextAdapter):
    """
    Wraps CUDA contexts for several devices.
    """

    @classmethod
    def from_pycuda_devices(
            cls, pycuda_devices: Iterable[pycuda_driver.Device]) -> CuContextAdapter:
        """
        Creates a context based on one or several (distinct) PyCuda ``Device`` objects.
        """
        pycuda_devices = normalize_object_sequence(pycuda_devices, pycuda_driver.Device)
        contexts = []
        for device in pycuda_devices:
            context = device.make_context()
            pycuda_driver.Context.pop()
            contexts.append(context)
        return cls(contexts, take_ownership=True)

    @classmethod
    def from_pycuda_contexts(
            cls, pycuda_contexts: Sequence[pycuda_driver.Context],
            take_ownership: bool) -> CuContextAdapter:
        """
        Creates a context based on one or several (distinct) PyCuda ``Context`` objects.
        None of the PyCuda contexts should be pushed to the context stack.
        """
        if len(pycuda_contexts) > 1 and not take_ownership:
            raise ValueError(
                "When dealing with multiple CUDA contexts, Grunnur must be the one managing them")
        pycuda_contexts = normalize_object_sequence(pycuda_contexts, pycuda_driver.Context)
        return cls(pycuda_contexts, take_ownership)

    @classmethod
    def from_device_adapters(cls, device_adapters: Sequence[CuDeviceAdapter]) -> CuContextAdapter:
        """
        Creates a context based on one or several (distinct) :py:class:`CuDeviceAdapter` objects.
        """
        device_adapters = normalize_object_sequence(device_adapters, CuDeviceAdapter)
        return cls.from_pycuda_devices(
            [device_adapter.pycuda_device for device_adapter in device_adapters])

    def __init__(self, pycuda_contexts: Iterable[pycuda_driver.Context], take_ownership):

        self._context_stack = _ContextStack(pycuda_contexts, take_ownership)

        self._pycuda_devices = []
        for context_num, context in enumerate(pycuda_contexts):
            self.activate_device(context_num)
            self._pycuda_devices.append(context.get_device())

        self._device_adapters = tuple(
            CuDeviceAdapter.from_pycuda_device(device) for device in self._pycuda_devices)
        self._device_idxs = list(range(len(self._device_adapters)))

        self.activate_device(0)

    @property
    def device_adapters(self) -> Tuple[CuDeviceAdapter, ...]:
        return self._device_adapters

    def activate_device(self, device_idx: int):
        """
        Activates the device ``device_idx``.
        Pops a previous context from the stack, if there was one pushed before,
        and pushes the corresponding context to the stack.
        """
        self._context_stack.activate(device_idx)

    def deactivate(self):
        self._context_stack.deactivate()

    @staticmethod
    def render_prelude(fast_math=False, constant_arrays=None):
        return _PRELUDE.render(
            fast_math=fast_math,
            dtypes=dtypes,
            constant_arrays=constant_arrays)

    def compile_single_device(
            self, device_idx, prelude, src, keep=False, fast_math=False, compiler_options=[],
            constant_arrays={}):

        constant_arrays = normalize_constant_arrays(constant_arrays)
        constant_arrays_src = _CONSTANT_ARRAYS_DEF.render(
            dtypes=dtypes,
            constant_arrays=constant_arrays)

        options = compiler_options + (['-use_fast_math'] if fast_math else [])
        full_src = prelude + constant_arrays_src + src
        self.activate_device(device_idx)

        # For some reason, `keep=True` does not work without an explicit `cache_dir`.
        # A new temporary dir is still created, and everything is placed there,
        # and `cache_dir` receives a copy of `kernel.cubin`.
        if keep:
            cache_dir = mkdtemp()
        else:
            cache_dir = None

        try:
            module = pycuda_compiler.SourceModule(
                full_src, no_extern_c=True, options=options, keep=keep, cache_dir=cache_dir)
        except pycuda_driver.CompileError as e:
            raise AdapterCompilationError(e, full_src)

        return CuProgramAdapter(self, device_idx, module, full_src)

    def allocate(self, size):
        managed = len(self._device_adapters) > 1
        self.activate_device(0)
        return CuBufferAdapter(self, size=size, managed=managed)

    def make_queue_adapter(self, device_idxs):
        device_adapters = {
            device_idx: self._device_adapters[device_idx] for device_idx in device_idxs}

        streams = {}
        for device_idx in self._device_idxs:
            self.activate_device(device_idx)
            stream = pycuda_driver.Stream()
            streams[device_idx] = stream

        return CuQueueAdapter(self, device_adapters, streams)


class CuBufferAdapter(BufferAdapter):

    def __init__(self, context_adapter, size, offset=0, managed=False, ptr=None, base_buffer=None):

        if ptr is None:
            assert offset == 0
            if managed:
                arr = pycuda_driver.managed_empty(
                    shape=size, dtype=numpy.uint8, mem_flags=pycuda_driver.mem_attach_flags.GLOBAL)
                ptr = arr.base
            else:
                ptr = pycuda_driver.mem_alloc(size)

        self._size = size
        self._offset = offset
        self._context_adapter = context_adapter

        self._base_buffer = base_buffer
        self._ptr = ptr
        self.kernel_arg = self._ptr

    @property
    def size(self):
        return self._size

    @property
    def offset(self):
        return self._offset

    def get_sub_region(self, origin, size):
        assert origin + size <= self._size
        if self._base_buffer is None:
            base_buffer = self
        else:
            base_buffer = self._base_buffer
        new_ptr = numpy.uintp(self._ptr) + numpy.uintp(origin)
        return CuBufferAdapter(
            self._context_adapter, size,
            offset=self._offset + origin, ptr=new_ptr, base_buffer=base_buffer)

    def set(self, queue_adapter, device_idx, buf, no_async=False):
        queue_adapter._synchronize_other_streams(device_idx)

        self._context_adapter.activate_device(device_idx)

        # PyCUDA needs pointers to be passed as `numpy.number` to kernels,
        # but `memcpy` functions require Python `int`s.
        ptr = int(self._ptr) if isinstance(self._ptr, numpy.number) else self._ptr

        if isinstance(buf, numpy.ndarray):
            if no_async:
                pycuda_driver.memcpy_htod(ptr, buf)
            else:
                pycuda_driver.memcpy_htod_async(
                    ptr, buf, stream=queue_adapter._pycuda_streams[device_idx])
        else:
            buf_ptr = int(buf._ptr) if isinstance(buf._ptr, numpy.number) else buf._ptr
            if no_async:
                pycuda_driver.memcpy_dtod(ptr, buf_ptr, buf.size)
            else:
                pycuda_driver.memcpy_dtod_async(
                    ptr, buf_ptr, buf.size, stream=queue_adapter._pycuda_streams[device_idx])

    def get(self, queue_adapter, device_idx, host_array, async_=False):
        queue_adapter._synchronize_other_streams(device_idx)

        self._context_adapter.activate_device(device_idx)

        # PyCUDA needs pointers to be passed as `numpy.number` to kernels,
        # but `memcpy` functions require Python `int`s.
        ptr = int(self._ptr) if isinstance(self._ptr, numpy.number) else self._ptr

        if async_:
            pycuda_driver.memcpy_dtoh_async(
                host_array, ptr, stream=queue_adapter._pycuda_streams[device_idx])
        else:
            pycuda_driver.memcpy_dtoh(host_array, ptr)

    def migrate(self, queue_adapter, device_idx):
        pass


def normalize_constant_arrays(constant_arrays):
    normalized = {}
    for name, metadata in constant_arrays.items():
        if isinstance(metadata, (list, tuple)):
            shape, dtype = metadata
            shape = wrap_in_tuple(shape)
            dtype = dtypes.normalize_type(dtype)
            length = prod(shape)
        elif hasattr(metadata, 'shape') and hasattr(metadata, 'dtype'):
            dtype = metadata.dtype
            length = prod(metadata.shape)
        else:
            raise TypeError(f"Unknown constant array metadata type: {type(metadata)}")

        normalized[name] = (length, dtype)

    return normalized


class CuQueueAdapter(QueueAdapter):

    def __init__(self, context_adapter: CuContextAdapter, device_adapters, pycuda_streams):
        self._context_adapter = context_adapter
        self._device_adapters = device_adapters
        self._pycuda_streams = pycuda_streams

    @property
    def device_adapters(self):
        return self._device_adapters

    def synchronize(self):
        for device_idx, pycuda_stream in self._pycuda_streams.items():
            self._context_adapter.activate_device(device_idx)
            pycuda_stream.synchronize()

    def _synchronize_other_streams(self, skip_device_idx):
        for device_idx, pycuda_stream in self._pycuda_streams.items():
            if device_idx != skip_device_idx:
                self._context_adapter.activate_device(device_idx)
                pycuda_stream.synchronize()


class CuProgramAdapter(ProgramAdapter):

    def __init__(self, context_adapter, device_idx, pycuda_program, source):
        self._context_adapter = context_adapter
        self._device_idx = device_idx
        self._pycuda_program = pycuda_program
        self.source = source

    def __getattr__(self, kernel_name):
        pycuda_kernel = self._pycuda_program.get_function(kernel_name)
        return CuKernelAdapter(self, self._device_idx, pycuda_kernel)

    def set_constant_buffer(
            self, queue_adapter: CuQueueAdapter,
            name: str, arr: Union[CuBufferAdapter, numpy.ndarray]):
        """
        Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
        """
        self._context_adapter.activate_device(self._device_idx)
        symbol, size = self._pycuda_program.get_global(name)

        pycuda_stream = queue_adapter._pycuda_streams[self._device_idx]

        if isinstance(arr, CuBufferAdapter):
            transfer_size = arr.size
        elif isinstance(arr, numpy.ndarray):
            transfer_size = prod(arr.shape) * arr.dtype.itemsize
        else: # pragma: no cover
            # Shouldn't reach this path because the type is already checked by the caller.
            # Nevertheless leaving it here as a sanity check.
            raise TypeError(f"Unsupported array type: {type(arr)}")

        if transfer_size != size:
            raise ValueError(
                f"Incorrect size of the constant buffer; "
                f"expected {size} bytes, got {transfer_size}")

        if isinstance(arr, CuBufferAdapter):
            pycuda_driver.memcpy_dtod_async(symbol, arr.kernel_arg, arr.size, stream=pycuda_stream)
        else:
            # This serves two purposes:
            # 1. Gives us a pagelocked array, as PyCUDA requires
            # 2. Makes the array contiguous
            # Constant array are usually quite small, so it won't affect the performance.
            buf = pycuda_driver.pagelocked_empty(arr.shape, arr.dtype)
            numpy.copyto(buf, arr)
            pycuda_driver.memcpy_htod_async(symbol, buf, stream=pycuda_stream)


class CuKernelAdapter(KernelAdapter):

    def __init__(self, program_adapter: CuProgramAdapter, device_idx: int, pycuda_function):
        self._program_adapter = program_adapter
        self._device_idx = device_idx
        self._pycuda_function = pycuda_function

    @property
    def max_total_local_size(self) -> int:
        return self._pycuda_function.get_attribute(
            pycuda_driver.function_attribute.MAX_THREADS_PER_BLOCK)

    def prepare(
            self, queue_adapter: CuQueueAdapter, global_size: Tuple[int, ...],
            local_size: Tuple[int, ...]) -> CuPreparedKernelAdapter:
        return CuPreparedKernelAdapter(self, queue_adapter, global_size, local_size)


class CuPreparedKernelAdapter(PreparedKernelAdapter):

    def __init__(
            self, kernel_adapter: CuKernelAdapter,
            queue_adapter: CuQueueAdapter,
            global_size: Tuple[int, ...],
            local_size: Tuple[int, ...]):

        device_adapter = queue_adapter.device_adapters[kernel_adapter._device_idx]

        grid, block = get_launch_size(
            device_adapter.params.max_local_sizes,
            device_adapter.params.max_total_local_size,
            global_size, local_size)

        # append missing dimensions, otherwise PyCUDA will complain
        max_dims = len(device_adapter.params.max_local_sizes)
        self._block = block + (1,) * (max_dims - len(block))
        self._grid = grid + (1,) * (max_dims - len(grid))

        self._context_adapter = queue_adapter._context_adapter
        self._device_idx = kernel_adapter._device_idx
        self._pycuda_stream = queue_adapter._pycuda_streams[self._device_idx]
        self._pycuda_function = kernel_adapter._pycuda_function

    def __call__(self, *args, local_mem: int=0):
        self._context_adapter.activate_device(self._device_idx)
        return self._pycuda_function(
            *args, grid=self._grid, block=self._block, stream=self._pycuda_stream, shared=local_mem)

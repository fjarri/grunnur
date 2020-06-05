from __future__ import annotations

from typing import Iterable, Union, Optional, Tuple, List, Sequence, cast

import numpy

try:
    import pycuda.driver as pycuda_drv
except ImportError:
    pycuda_drv = None # this variable is used for a PyCUDA mock during tests

from .utils import all_same, all_different, wrap_in_tuple, prod, normalize_object_sequence
from .template import Template
from . import dtypes
from .adapter_base import (
    APIID, DeviceType, APIAdapterFactory, APIAdapter, PlatformAdapter, DeviceAdapter,
    DeviceParameters, ContextAdapter, BufferAdapter, QueueAdapter, ProgramAdapter, KernelAdapter)


# Another way would be to place it in the try block and only set `_avaialable`
# if CUDA was initialized correctly, but it is better to distinguish import errors
# (PyCUDA not installed, which is quite common) and initialization errors
# (some problem with the package).
if pycuda_drv is not None:
    pycuda_drv.init()
    import pycuda.compiler


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
        return pycuda_drv is not None

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
    def num_platforms(self):
        return 1

    # TODO: have instead get_platform(platform_num)?
    def get_platform_adapters(self):
        return [CuPlatformAdapter(self)]

    def isa_backend_device(self, obj):
        return isinstance(obj, pycuda_drv.Device)

    def isa_backend_context(self, obj):
        return isinstance(obj, pycuda_drv.Context)

    def make_context_from_backend_devices(self, backend_devices):
        return CuContextAdapter.from_pycuda_devices(backend_devices)

    def make_context_from_backend_contexts(self, backend_contexts):
        return CuContextAdapter.from_pycuda_contexts(backend_contexts)


class CuPlatformAdapter(PlatformAdapter):

    def __init__(self, api_adapter):
        self._api_adapter = api_adapter

    @property
    def api_adapter(self):
        return self._api_adapter

    @property
    def platform_num(self):
        return 0

    @property
    def name(self):
        return "nVidia CUDA"

    @property
    def vendor(self):
        return "nVidia"

    @property
    def version(self):
        return ".".join(str(x) for x in pycuda_drv.get_version())

    @property
    def num_devices(self):
        return pycuda_drv.Device.count()

    def get_device_adapters(self):
        return [
            CuDeviceAdapter(self, pycuda_drv.Device(device_num), device_num)
            for device_num in range(self.num_devices)]

    def make_context(self, device_adapters):
        return CuContextAdapter.from_device_adapters(device_adapters)


class CuDeviceAdapter(DeviceAdapter):

    @classmethod
    def from_pycuda_device(cls, pycuda_device: pycuda_drv.Device) -> CuDeviceAdapter:
        """
        Creates this object from a PyCuda ``Device`` object.
        """
        platform_adapter = CuPlatformAdapter(CuAPIAdapter())
        for device_num, device_adapter in enumerate(platform_adapter.get_device_adapters()):
            if pycuda_device == device_adapter.pycuda_device:
                return cls(platform_adapter, pycuda_device, device_num)

        raise Exception(f"{pycuda_device} was not found among CUDA devices")

    def __init__(self, platform_adapter, pycuda_device, device_num):
        self._platform_adapter = platform_adapter
        self._device_num = device_num
        self.pycuda_device = pycuda_device
        self._params = None

    @property
    def platform_adapter(self):
        return self._platform_adapter

    @property
    def device_num(self):
        return self._device_num

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

    def __init__(self, pycuda_contexts, owns_contexts=False):
        self._pycuda_contexts = pycuda_contexts
        self._active_context = None
        self._owns_contexts = owns_contexts

    def deactivate(self):
        if self._active_context is not None:
            self._active_context = None

            # This can happen in tests when the PyCuda module is mocked.
            if pycuda_drv is not None:
                pycuda_drv.Context.pop()
        # TODO: raise an exception on deactivate() of an already inactive stack?

    def activate(self, device_num):
        if self._active_context != device_num:
            self.deactivate()
            self._pycuda_contexts[device_num].push()
            self._active_context = device_num

    def __del__(self):
        self.deactivate()


class CuContextAdapter(ContextAdapter):
    """
    Wraps CUDA contexts for several devices.
    """

    @classmethod
    def from_any_base(cls, objs) -> CuContextAdapter:
        """
        Create a context based on any object supported by other ``from_*`` methods.
        """
        objs = wrap_in_tuple(objs)
        if isinstance(objs[0], pycuda_drv.Device):
            return cls.from_pycuda_devices(objs)
        elif isinstance(objs[0], pycuda_drv.Context):
            return cls.from_pycuda_contexts(objs)
        elif isinstance(objs[0], CuDeviceAdapter):
            return cls.from_device_adapters(objs)
        else:
            raise TypeError(f"Do not know how to create a context out of {type(objs[0])}")

    @classmethod
    def from_pycuda_devices(cls, pycuda_devices: Iterable[pycuda_drv.Device]) -> CuContextAdapter:
        """
        Creates a context based on one or several (distinct) PyCuda ``Device`` objects.
        """
        pycuda_devices = normalize_object_sequence(pycuda_devices, pycuda_drv.Device)
        contexts = []
        for device in pycuda_devices:
            context = device.make_context()
            pycuda_drv.Context.pop()
            contexts.append(context)
        return cls(contexts, owns_contexts=True)

    @classmethod
    def from_pycuda_contexts(cls, pycuda_contexts: Iterable[pycuda_drv.Context]) -> CuContextAdapter:
        """
        Creates a context based on one or several (distinct) PyCuda ``Context`` objects.
        None of the PyCuda contexts should be pushed to the context stack.
        """
        pycuda_contexts = normalize_object_sequence(pycuda_contexts, pycuda_drv.Context)
        return cls(pycuda_contexts, owns_contexts=False)

    @classmethod
    def from_device_adapters(cls, device_adapters: Iterable[CuDeviceAdapter]) -> CuContextAdapter:
        """
        Creates a context based on one or several (distinct) :py:class:`CuDeviceAdapter` objects.
        """
        device_adapters = normalize_object_sequence(device_adapters, CuDeviceAdapter)
        return cls.from_pycuda_devices(
            [device_adapter.pycuda_device for device_adapter in device_adapters])

    def __init__(self, pycuda_contexts: Iterable[pycuda_drv.Context], owns_contexts=False):

        self._context_stack = _ContextStack(pycuda_contexts, owns_contexts=owns_contexts)

        self._pycuda_devices = []
        for context_num, context in enumerate(pycuda_contexts):
            self.activate_device(context_num)
            self._pycuda_devices.append(context.get_device())

        self._device_adapters = tuple(
            CuDeviceAdapter.from_pycuda_device(device) for device in self._pycuda_devices)
        self._device_nums = list(range(len(self._device_adapters)))

        # TODO: do we activate here if owns_context=False? What are the general behavior
        # rules for owns_context=False?
        self.activate_device(0)

    @property
    def device_adapters(self) -> Tuple[CuDeviceAdapter, ...]:
        return self._device_adapters

    def activate_device(self, device_num: int):
        """
        Activates the device ``device_num``.
        Pops a previous context from the stack, if there was one pushed before,
        and pushes the corresponding context to the stack.
        """
        self._context_stack.activate(device_num)

    def deactivate(self):
        """
        Pops a context from the stack, if there was one pushed before.
        """
        self._context_stack.deactivate()

    def render_prelude(self, fast_math=False, constant_arrays=None):
        return _PRELUDE.render(
            fast_math=fast_math,
            dtypes=dtypes,
            constant_arrays=constant_arrays)

    @property
    def compile_error_class(self):
        return pycuda_drv.CompileError

    def compile_single_device(
            self, device_num, prelude, src, keep=False, fast_math=False, compiler_options=[],
            constant_arrays=None):

        if constant_arrays is not None:
            constant_arrays = normalize_constant_arrays(constant_arrays)
            constant_arrays_src = _CONSTANT_ARRAYS_DEF.render(
                dtypes=dtypes,
                constant_arrays=constant_arrays)
        else:
            constant_arrays_src = ""

        options = compiler_options + (['-use_fast_math'] if fast_math else [])
        full_src = prelude + constant_arrays_src + src
        self.activate_device(device_num)
        module = pycuda.compiler.SourceModule(
            full_src, no_extern_c=True, options=options, keep=keep)
        return CuProgram(self, device_num, module)

    def allocate(self, size):
        managed = len(self._device_adapters) > 1
        self.activate_device(0)
        return CuBufferAdapter(self, size=size, managed=managed)

    def make_queue_adapter(self, device_nums):
        device_adapters = {
            device_num: self._device_adapters[device_num] for device_num in device_nums}

        streams = {}
        for device_num in self._device_nums:
            self.activate_device(device_num)
            stream = pycuda_drv.Stream()
            streams[device_num] = stream

        return CuQueueAdapter(self, device_adapters, streams)


class CuBufferAdapter(BufferAdapter):

    def __init__(self, context_adapter, size, offset=0, managed=False, ptr=None, base_buffer=None):

        if ptr is None:
            assert offset == 0
            if managed:
                arr = pycuda_drv.managed_empty(
                    shape=size, dtype=numpy.uint8, mem_flags=pycuda_drv.mem_attach_flags.GLOBAL)
                ptr = arr.base
            else:
                ptr = pycuda_drv.mem_alloc(size)

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

    @property
    def context_adapter(self):
        return self._context_adapter

    def get_sub_region(self, origin, size):
        assert origin + size <= self._size
        if self._base_buffer is None:
            base_buffer = self
        else:
            base_buffer = self._base_buffer
        new_ptr = numpy.uintp(self._ptr) + numpy.uintp(origin)
        return CuBuffer(
            self._context_adapter, size,
            offset=self._offset + origin, ptr=new_ptr, base_buffer=base_buffer)

    def set(self, queue_adapter, device_num, host_array, async_=False, dont_sync_other_devices=False):
        # TODO: is there a way to keep the whole thing async, but still wait until
        # all current tasks on other devices finish, like with events in OpenCL?

        if not dont_sync_other_devices:
            queue_adapter._synchronize_other_streams(device_num)

        self._context_adapter.activate_device(device_num)

        if async_:
            pycuda_drv.memcpy_htod_async(
                self._ptr, host_array, stream=queue_adapter._pycuda_streams[device_num])
        else:
            pycuda_drv.memcpy_htod(self._ptr, host_array)

    def get(self, queue_adapter, device_num, host_array, async_=False, dont_sync_other_devices=False):
        if not dont_sync_other_devices:
            queue_adapter._synchronize_other_streams(device_num)

        self._context_adapter.activate_device(device_num)

        if async_:
            pycuda_drv.memcpy_dtoh_async(
                host_array, self._ptr, stream=queue_adapter._pycuda_streams[device_num])
        else:
            pycuda_drv.memcpy_dtoh(host_array, self._ptr)

    def migrate(self, queue_adapter, device_num):
        pass

    def __del__(self):
        if self._base_buffer is None:
            self._ptr.free()


def normalize_constant_arrays(constant_arrays):
    normalized = {}
    for name, metadata in constant_arrays.items():
        if isinstance(metadata, (list, tuple)):
            shape, dtype = metadata
            shape = wrap_in_tuple(shape)
            dtype = normalize_type(dtype)
            length = prod(shape)
        elif isinstance(metadata, numpy.ndarray):
            dtype = metadata.dtype
            length = metadata.size
        elif isinstance(metadata, CuArray):
            dtype = metadata.dtype
            length = metadata.buffer_size
        else:
            raise TypeError(f"Unknown constant array metadata type: {type(metadata)}")

        normalized[name] = (length, dtype)

    return normalized


class CuQueueAdapter(QueueAdapter):

    def __init__(self, context_adapter: CuContextAdapter, device_adapters, pycuda_streams):
        self.context_adapter = context_adapter
        self.device_adapters = device_adapters
        self._pycuda_streams = pycuda_streams

    def synchronize(self):
        for device_num, pycuda_streams in self._pycuda_streams.items():
            self.context_adapter.activate_device(device_num)
            self._pycuda_streams[device_num].synchronize()

    def _synchronize_other_streams(self, skip_device_num):
        for device_num, pycuda_streams in self._pycuda_streams.items():
            if device_num != skip_device_num:
                self.context_adapter.activate_device(device_num)
                self._pycuda_streams[device_num].synchronize()


class CuProgram(ProgramAdapter):

    def __init__(self, context_adapter, device_num, pycuda_program):
        self.context_adapter = context_adapter
        self._device_num = device_num
        self._pycuda_program = pycuda_program

    def __getattr__(self, kernel_name):
        pycuda_kernel = self._pycuda_program.get_function(kernel_name)
        return CuKernel(self, self._device_num, pycuda_kernel)

    def set_constant_buffer(
            self, name: str, arr: Union[CuBufferAdapter, numpy.ndarray], queue: Optional[CuQueue]=None):
        """
        Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
        """
        self.context_adapter.activate_device(self._device_num)
        symbol, size = self._pycuda_program.get_global(name)

        if queue is not None:
            pycuda_stream = queue._pycuda_streams[self._device_num]

        if isinstance(arr, CuBufferAdapter):
            if size != arr.size:
                raise ValueError(
                    f"Incorrect size of the constant buffer; "
                    f"expected {size} bytes, got {arr.size}")

            if queue is None:
                pycuda_drv.memcpy_dtod(symbol, arr.kernel_arg, arr.size)
            else:
                pycuda_drv.memcpy_dtod_async(symbol, arr.kernel_arg, arr.size, stream=pycuda_stream)
        elif isinstance(arr, numpy.ndarray):
            if queue is None:
                pycuda_drv.memcpy_htod(symbol, arr)
            else:
                pycuda_drv.memcpy_htod_async(symbol, arr, stream=pycuda_stream)
        else:
            raise TypeError(f"Uunsupported array type: {type(arr)}")


def max_factor(x: int, y: int) -> int:
    """
    Find the maximum `d` such that `x % d == 0` and `d <= y`.
    """
    if x <= y:
        return x

    # TODO: speed up

    for d in range(y, 0, -1):
        if x % d == 0:
            return d

    return 1


def find_local_size(
        global_size: Sequence[int],
        max_local_sizes: Sequence[int],
        max_total_local_size: int) -> Tuple[int, ...]:
    """
    Mimics the OpenCL local size finding algorithm.
    Returns the tuple of the same length as ``global_size``, with every element
    being a factor of the corresponding element of ``global_size``.
    Neither of the elements of ``local_size`` are greater then the corresponding element
    of ``max_local_sizes``, and their product is not greater than ``max_total_local_size``.
    """
    if max_total_local_size == 1:
        return (1,) * len(global_size)

    local_size = []
    for gs, mls in zip(global_size, max_local_sizes):
        d = max_factor(gs, min(mls, max_total_local_size))
        max_total_local_size //= d
        local_size.append(d)

    return tuple(local_size)


def get_launch_size(
        max_local_sizes: Sequence[int],
        max_total_local_size: int,
        global_size: Sequence[int],
        local_size: Optional[Sequence[int]]=None) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Constructs the grid and block tuples to launch a CUDA kernel
    based on the provided global and local sizes.
    """

    if len(global_size) > len(max_local_sizes):
        raise ValueError("Global size has too many dimensions")

    if local_size is not None:
        local_size = wrap_in_tuple(local_size)
        if len(local_size) != len(global_size):
            raise ValueError("Global/local work sizes have differing dimensions")
    else:
        local_size = find_local_size(global_size, max_local_sizes, max_total_local_size)

    grid_list = []
    for gs, ls in zip(global_size, local_size):
        if gs % ls != 0:
            raise ValueError("Global sizes must be multiples of corresponding local sizes")
        grid_list.append(gs // ls)

    # append missing dimensions, otherwise PyCUDA will complain
    block = local_size + (1,) * (3 - len(grid_list))
    grid: Tuple[int, ...] = tuple(grid_list) + (1,) * (3 - len(grid_list))

    return grid, block


class CuKernel(KernelAdapter):

    def __init__(self, program_adapter, device_num, pycuda_function):
        self.program_adapter = program_adapter
        self._device_num = device_num
        self._pycuda_function = pycuda_function

    @property
    def max_total_local_size(self):
        return self._pycuda_function.get_attribute(
            pycuda_drv.function_attribute.MAX_THREADS_PER_BLOCK)

    def __call__(self, queue_adapter, global_size, local_size, *args, local_mem=0):

        device_adapter = queue_adapter.device_adapters[self._device_num]

        grid, block = get_launch_size(
            device_adapter.params.max_local_sizes,
            device_adapter.params.max_total_local_size,
            global_size, local_size)

        self.program_adapter.context_adapter.activate_device(self._device_num)
        self._pycuda_function(
            *args, grid=grid, block=block, stream=queue_adapter._pycuda_streams[self._device_num],
            shared=local_mem)
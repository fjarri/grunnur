# Avoids errors from using PyCUDA types as annotations when PyCUDA is not present
from __future__ import annotations

from typing import Iterable, Union, Optional

import numpy

try:
    import pycuda.driver as pycuda_drv
except ImportError:
    pycuda_drv = None # this variable is used for a PyCUDA mock during tests

from .base_classes import (
    APIFactory, API, Platform, APIID, DeviceID, PlatformID,
    Device, DeviceType, DeviceParameters, Context, Queue, Program, Kernel,
    SingleDeviceProgram, SingleDeviceKernel,
    normalize_base_objects, process_arg, Array, Buffer
    )
from .utils import all_same, all_different, wrap_in_tuple, prod, factors
from .template import Template
from . import dtypes


_TEMPLATE = Template.from_associated_file(__file__)
_PRELUDE = _TEMPLATE.get_def('prelude')
_CONSTANT_ARRAYS_DEF = _TEMPLATE.get_def('constant_arrays_def')


# Another way would be to place it in the try block and only set `_avaialable`
# if CUDA was initialized correctly, but it is better to distinguish import errors
# (PyCUDA not installed, which is quite common) and initialization errors
# (some problem with the package).
if pycuda_drv is not None:
    pycuda_drv.init()
    import pycuda.compiler


class CuAPI(API):

    def get_platforms(self):
        return [CuPlatform(PlatformID(self.id, 0))]

    @property
    def _context_class(self):
        return CuContext


class CuAPIFactory(APIFactory):

    @property
    def available(self):
        return pycuda_drv is not None

    def make_api(self):
        if not self.available:
            raise ImportError(
                "CUDA API is not operational. Check if PyCUDA is installed correctly.")

        return _CUDA_API


CUDA_API_ID = APIID('cuda')
_CUDA_API = CuAPI(CUDA_API_ID)
CUDA_API_FACTORY = CuAPIFactory(CUDA_API_ID)


def make_cuda_api():
    return CUDA_API_FACTORY.make_api()


class CuPlatform(Platform):

    def __init__(self, platform_id: PlatformID):
        super().__init__(_CUDA_API, platform_id)

    @property
    def name(self):
        return "nVidia CUDA"

    @property
    def vendor(self):
        return "nVidia"

    @property
    def version(self):
        return ".".join(str(x) for x in pycuda_drv.get_version())

    def get_devices(self):
        return [
            CuDevice(self, DeviceID(self.id, i), pycuda_drv.Device(i))
            for i in range(pycuda_drv.Device.count())]


class CuDevice(Device):

    def __init__(self, platform: CuPlatform, device_id: DeviceID, pycuda_device: pycuda_drv.Device):
        super().__init__(platform, device_id)
        self._pycuda_device = pycuda_device
        self._params = None

    @property
    def name(self):
        return self._pycuda_device.name()

    @property
    def params(self):
        if self._params is None:
            self._params = CuDeviceParameters(self._pycuda_device)
        return self._params

    @classmethod
    def from_pycuda_device(cls, pycuda_device: pycuda_drv.Device) -> CuDevice:
        """
        Creates this object from a PyCuda ``Device`` object.
        """
        platform = _CUDA_API.get_platforms()[0]
        for device_num, device in enumerate(platform.get_devices()):
            if pycuda_device == device._pycuda_device:
                return cls(platform, device.id, pycuda_device)

        raise Exception(f"{pycuda_device} was not found among CUDA devices")


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


class CuBuffer(Buffer):

    def __init__(self, context, size, offset=0, managed=False, ptr=None, base_buffer=None):

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
        self._context = context

        self._base_buffer = base_buffer
        self._ptr = ptr

        # Allows this object to be passed as an argument to PyCuda kernels
        self.gpudata = ptr

    @property
    def size(self):
        return self._size

    @property
    def offset(self):
        return self._offset

    @property
    def context(self):
        return self._context

    def get_sub_region(self, origin, size):
        assert origin + size <= self._size
        if self._base_buffer is None:
            base_buffer = self
        else:
            base_buffer = self._base_buffer
        new_ptr = numpy.uintp(self._ptr) + numpy.uintp(origin)
        return CuBuffer(
            self._context, size, offset=self._offset + origin, ptr=new_ptr, base_buffer=base_buffer)

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


class CuContext(Context):
    """
    Wraps CUDA contexts for several devices.
    """

    @classmethod
    def from_any_base(cls, objs) -> CuContext:
        """
        Create a context based on any object supported by other ``from_*`` methods.
        """
        objs = wrap_in_tuple(objs)
        if isinstance(objs[0], pycuda_drv.Device):
            return cls.from_pycuda_devices(objs)
        elif isinstance(objs[0], pycuda_drv.Context):
            return cls.from_pycuda_contexts(objs)
        elif isinstance(objs[0], CuDevice):
            return cls.from_devices(objs)
        else:
            raise TypeError(f"Do not know how to create a context out of {type(objs[0])}")

    @classmethod
    def from_pycuda_devices(cls, pycuda_devices: Iterable[pycuda_drv.Device]) -> CuContext:
        """
        Creates a context based on one or several (distinct) PyCuda ``Device`` objects.
        """
        pycuda_devices = normalize_base_objects(pycuda_devices, pycuda_drv.Device)
        contexts = []
        for device in pycuda_devices:
            context = device.make_context()
            pycuda_drv.Context.pop()
            contexts.append(context)
        return cls(contexts, owns_contexts=True)

    @classmethod
    def from_pycuda_contexts(cls, pycuda_contexts: Iterable[pycuda_drv.Context]) -> CuContext:
        """
        Creates a context based on one or several (distinct) PyCuda ``Context`` objects.
        None of the PyCuda contexts should be pushed to the context stack.
        """
        pycuda_contexts = normalize_base_objects(pycuda_contexts, pycuda_drv.Context)
        return cls(pycuda_contexts, owns_contexts=False)

    @classmethod
    def from_devices(cls, devices: Iterable[CuDevice]) -> CuContext:
        """
        Creates a context based on one or several (distinct) :py:class:`CuDevice` objects.
        """
        devices = normalize_base_objects(devices, CuDevice)
        return cls.from_pycuda_devices([device._pycuda_device for device in devices])

    def __init__(self, pycuda_contexts: Iterable[pycuda_drv.Context], owns_contexts=False):

        self._context_stack = _ContextStack(pycuda_contexts, owns_contexts=owns_contexts)

        self._pycuda_devices = []
        for context_num, context in enumerate(pycuda_contexts):
            self.activate_device(context_num)
            self._pycuda_devices.append(context.get_device())

        self.devices = [CuDevice.from_pycuda_device(device) for device in self._pycuda_devices]

        # TODO: do we activate here if owns_context=False? What are the general behavior
        # rules for owns_context=False?
        self.activate_device(0)

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

    @property
    def api(self):
        return _CUDA_API

    def _render_prelude(self, fast_math=False, constant_arrays=None):
        return _PRELUDE.render(
            fast_math=fast_math,
            dtypes=dtypes,
            constant_arrays=constant_arrays)

    @property
    def _compile_error_class(self):
        return pycuda_drv.CompilerError

    def _compile_single_device(
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
        return CuSingleDeviceProgram(self, device_num, module)

    @property
    def _program_class(self):
        return CuProgram

    def allocate(self, size):
        managed = len(self.devices) > 1
        self.activate_device(0)
        return CuBuffer(self, size=size, managed=managed)

    @property
    def _array_class(self):
        return CuArray

    @property
    def _queue_class(self):
        return CuQueue


class CuQueue(Queue):

    def __init__(self, context, device_nums=None):
        if device_nums is None:
            device_nums = tuple(range(len(context.devices)))
        else:
            device_nums = tuple(sorted(device_nums))

        streams = []
        for device_num in device_nums:
            context.activate_device(device_num)
            stream = pycuda_drv.Stream()
            streams.append(stream)

        self._context = context
        self._pycuda_streams = streams
        self._device_nums = device_nums
        self.devices = tuple(context.devices[device_num] for device_num in device_nums)

    @property
    def context(self):
        return self._context

    @property
    def device_nums(self):
        return self._device_nums

    def synchronize(self):
        for device_num, context_device_num in enumerate(self.device_nums):
            self.context.activate_device(context_device_num)
            self._pycuda_streams[device_num].synchronize()


class CuArray(Array):

    def __init__(self, queue, *args, **kwds):

        if 'allocator' not in kwds or kwds['allocator'] is None:
            kwds['allocator'] = queue.context.allocate

        super().__init__(*args, **kwds)

        self.context = queue.context
        self._queue = queue

    def _new_like_me(self, *args, device_num=None, **kwds):
        if device_num is None:
            return CuArray(self._queue, *args, **kwds)
        else:
            return CuSingleDeviceArray(self._queue, device_num, *args, **kwds)

    def _synchronize_other_streams(self):
        for device_num, context_device_num in enumerate(self._queue.device_nums[1:]):
            self.context.activate_device(context_device_num)
            self._queue._pycuda_streams[device_num].synchronize()

    def set(self, array, async_=False):
        self._synchronize_other_streams()
        array_device = self._queue.device_nums[0]
        self.context.activate_device(array_device)

        if async_:
            pycuda_drv.memcpy_htod_async(
                self.data._ptr, array,
                stream=self._queue._pycuda_streams[array_device])
        else:
            pycuda_drv.memcpy_htod(self.data._ptr, array)

    def get(self, dest=None, async_=False):
        if dest is None:
            dest = numpy.empty(self.shape, self.dtype)

        self._synchronize_other_streams()
        array_device = self._queue.device_nums[0]
        self.context.activate_device(array_device)

        if async_:
            pycuda_drv.memcpy_dtoh_async(
                dest, self.data._ptr,
                stream=self._queue._pycuda_streams[array_device])
        else:
            pycuda_drv.memcpy_dtoh(dest, self.data._ptr)

        return dest


class CuSingleDeviceArray(CuArray):

    def __init__(self, context, device_num, *args, **kwds):
        super().__init__(context, *args, **kwds)
        self._device_num = device_num


class CuSingleDeviceProgram(SingleDeviceProgram):

    def __init__(self, context, device_num, pycuda_module):
        self.context = context
        self._device_num = device_num
        self._pycuda_module = pycuda_module

    def __getattr__(self, kernel_name):
        func = self._pycuda_module.get_function(kernel_name)
        return CuSingleDeviceKernel(self, self._device_num, func)

    def set_constant_array(
            self, name: str, arr: Union[CuArray, numpy.ndarray], queue: Optional[Queue]=None):
        """
        Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
        """
        if queue is not None:
            if queue.context is not self.context:
                raise ValueError(
                    "The provided queue must belong to the same context as this program uses")
            if self._device_num not in queue.device_nums:
                raise ValueError(
                    f"The provided queue must include the device this program uses ({self._device_num})")

        self.context.activate_device(self._device_num)
        symbol, size = self._pycuda_module.get_global(name)

        if queue is not None:
            pycuda_stream = queue._pycuda_streams[self._device_num]

        if isinstance(arr, CuArray):
            # TODO: check that array is contiguous, offset is 0 and there's no padding in the end
            if size != arr.buffer_size:
                raise ValueError(
                    f"Incorrect size of the constant array; "
                    f"expected {size} bytes, got {arr.buffer_size}")

            if queue is None:
                pycuda_drv.memcpy_dtod(symbol, arr.data._ptr, arr.buffer_size)
            else:
                pycuda_drv.memcpy_dtod_async(
                    symbol, arr.data._ptr, arr.buffer_size, stream=pycuda_stream)
        elif isinstance(arr, numpy.ndarray):
            if queue is None:
                pycuda_drv.memcpy_htod(symbol, arr)
            else:
                pycuda_drv.memcpy_htod_async(symbol, arr, stream=pycuda_stream)
        else:
            raise ValueError(f"Uunsupported array type: {type(arr)}")


class CuProgram(Program):

    @property
    def _kernel_class(self):
        return CuKernel

    def set_constant_array(
            self, name: str, arr: Union[CuArray, numpy.ndarray], queue: Optional[Queue]=None):
        """
        Uploads a constant array ``arr`` corresponding to the symbol ``name`` to the context.
        """
        for sd_program in self._sd_programs.values():
            sd_program.set_constant_array(name, arr, queue=queue)


def find_local_size(global_size, max_work_item_sizes, max_total_local_size):
    """
    Mimics the OpenCL local size finding algorithm.
    Returns the tuple of the same length as ``global_size``, with every element
    being a factor of the corresponding element of ``global_size``.
    Neither of the elements of ``local_size`` are greater then the corresponding element
    of ``max_work_item_sizes``, and their product is not greater than ``max_total_local_size``.
    """
    if len(global_size) == 0:
        return tuple()

    if max_total_local_size == 1:
        return (1,) * len(global_size)

    gs_factors = factors(global_size[0], limit=min(max_work_item_sizes[0], max_total_local_size))
    local_size_1d, _ = gs_factors[-1]

    # TODO:
    #local_size_1d = max_factor(global_size[0], min(max_work_item_sizes[0], max_total_local_size))
    # this will be much faster

    remainder = find_local_size(
        global_size[1:], max_work_item_sizes[1:], max_total_local_size // local_size_1d)

    return (local_size_1d,) + remainder


def get_launch_size(max_local_sizes, max_total_local_size, global_size, local_size=None):
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

    grid = []
    for gs, ls in zip(global_size, local_size):
        if gs % ls != 0:
            raise ValueError("Global sizes must be multiples of corresponding local sizes")
        grid.append(gs // ls)

    # append missing dimensions, otherwise PyCUDA will complain
    block = local_size + (1,) * (3 - len(grid))
    grid = tuple(grid) + (1,) * (3 - len(grid))

    return grid, block


class CuKernel(Kernel):
    pass


class CuSingleDeviceKernel(SingleDeviceKernel):

    def __init__(self, program, device_num, pycuda_function):
        self.program = program
        self._device_num = device_num
        self._pycuda_function = pycuda_function

    @property
    def max_total_local_size(self):
        return self._pycuda_function.get_attribute(
            pycuda_drv.function_attribute.MAX_THREADS_PER_BLOCK)

    def __call__(self, queue, global_size, local_size, *args, local_mem=0):

        if local_size is not None:
            local_size = wrap_in_tuple(local_size)
        global_size = wrap_in_tuple(global_size)

        args = process_arg(args)

        assert self._device_num in queue.device_nums

        device = queue.devices[self._device_num]

        grid, block = get_launch_size(
            device.params.max_local_sizes, device.params.max_total_local_size,
            global_size, local_size)

        self.program.context.activate_device(self._device_num)
        self._pycuda_function(
            *args, grid=grid, block=block, stream=queue._pycuda_streams[self._device_num],
            shared=local_mem)

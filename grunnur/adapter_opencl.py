from __future__ import annotations

import os
from tempfile import mkdtemp
from typing import Iterable, Tuple

# Skipping coverage count - can't test properly
try: # pragma: no cover
    import pyopencl
except ImportError: # pragma: no cover
    pyopencl = None

from .utils import normalize_object_sequence
from .template import Template
from . import dtypes
from .adapter_base import (
    DeviceType, APIID, APIAdapterFactory, APIAdapter, PlatformAdapter, DeviceAdapter,
    DeviceParameters, ContextAdapter, BufferAdapter, QueueAdapter, ProgramAdapter, KernelAdapter,
    AdapterCompilationError, PreparedKernelAdapter)


_API_ID = APIID('opencl')
_TEMPLATE = Template.from_associated_file(__file__)
_PRELUDE = _TEMPLATE.get_def('prelude')


class OclAPIAdapterFactory(APIAdapterFactory):

    @property
    def api_id(self):
        return _API_ID

    @property
    def available(self):
        return pyopencl is not None

    def make_api_adapter(self):
        if not self.available:
            raise ImportError(
                "OpenCL API is not operational. Check if PyOpenCL is installed correctly.")

        return OclAPIAdapter()


class OclAPIAdapter(APIAdapter):

    @property
    def id(self):
        return _API_ID

    @property
    def platform_count(self):
        return len(pyopencl.get_platforms())

    def get_platform_adapters(self):
        return [
            OclPlatformAdapter(self, platform, platform_idx)
            for platform_idx, platform in enumerate(pyopencl.get_platforms())]

    def isa_backend_device(self, obj):
        return isinstance(obj, pyopencl.Device)

    def isa_backend_platform(self, obj):
        return isinstance(obj, pyopencl.Platform)

    def isa_backend_context(self, obj):
        return isinstance(obj, pyopencl.Context)

    def make_device_adapter(self, pyopencl_device):
        return OclDeviceAdapter.from_pyopencl_device(pyopencl_device)

    def make_platform_adapter(self, pyopencl_platform):
        return OclPlatformAdapter.from_pyopencl_platform(pyopencl_platform)

    def make_context_adapter_from_device_adapters(self, device_adapters):
        return OclContextAdapter.from_device_adapters(device_adapters)

    def make_context_adapter_from_backend_contexts(self, pyopencl_contexts, take_ownership):
        if len(pyopencl_contexts) > 1:
            raise ValueError("Cannot make one OpenCL context out of several contexts")
        return OclContextAdapter.from_pyopencl_context(pyopencl_contexts[0])


class OclPlatformAdapter(PlatformAdapter):

    @classmethod
    def from_pyopencl_platform(cls, pyopencl_platform: pyopencl.Platform):
        """
        Creates this object from a PyOpenCL ``Platform`` object.
        """
        api_adapter = OclAPIAdapter()
        platform_adapters = api_adapter.get_platform_adapters()
        for platform_idx, platform_adapter in enumerate(platform_adapters):
            if pyopencl_platform == platform_adapter.pyopencl_platform:
                return cls(api_adapter, pyopencl_platform, platform_idx)

        # Sanity check, should not be reachable as long as `pyopencl` is consistent.
        raise RuntimeError(
            f"{pyopencl_platform} was not found among OpenCL platforms") # pragma: no cover

    def __init__(self, api_adapter, pyopencl_platform, platform_idx):
        self._api_adapter = api_adapter
        self.pyopencl_platform = pyopencl_platform
        self._platform_idx = platform_idx

    def __eq__(self, other):
        return type(self) == type(other) and self.pyopencl_platform == other.pyopencl_platform

    def __hash__(self):
        return hash((type(self), self.pyopencl_platform))

    @property
    def api_adapter(self):
        return self._api_adapter

    @property
    def platform_idx(self):
        return self._platform_idx

    @property
    def name(self):
        return self.pyopencl_platform.name

    @property
    def vendor(self):
        return self.pyopencl_platform.vendor

    @property
    def version(self):
        return self.pyopencl_platform.version

    @property
    def device_count(self):
        return len(self.pyopencl_platform.get_devices())

    def get_device_adapters(self):
        return [
            OclDeviceAdapter(self, device, device_idx)
            for device_idx, device in enumerate(self.pyopencl_platform.get_devices())]


class OclDeviceAdapter(DeviceAdapter):

    @classmethod
    def from_pyopencl_device(cls, pyopencl_device: pyopencl.Device) -> OclDeviceAdapter:
        """
        Creates this object from a PyOpenCL ``Device`` object.
        """
        platform_adapter = OclPlatformAdapter.from_pyopencl_platform(pyopencl_device.platform)
        for device_idx, device_adapter in enumerate(platform_adapter.get_device_adapters()):
            if pyopencl_device == device_adapter.pyopencl_device:
                return cls(platform_adapter, pyopencl_device, device_idx)

        # Sanity check, should not be reachable as long as `pyopencl` is consistent.
        raise RuntimeError(
            f"{pyopencl_device} was not found among OpenCL devices") # pragma: no cover

    def __init__(self, platform_adapter, pyopencl_device, device_idx):
        self._platform_adapter = platform_adapter
        self._device_idx = device_idx
        self.pyopencl_device = pyopencl_device
        self._params = None

    def __eq__(self, other):
        return type(self) == type(other) and self.pyopencl_device == other.pyopencl_device

    def __hash__(self):
        return hash((type(self), self.pyopencl_device))

    @property
    def platform_adapter(self):
        return self._platform_adapter

    @property
    def device_idx(self):
        return self._device_idx

    @property
    def name(self):
        return self.pyopencl_device.name

    @property
    def params(self):
        if self._params is None:
            self._params = OclDeviceParameters(self.pyopencl_device)
        return self._params


class OclDeviceParameters(DeviceParameters):

    def __init__(self, pyopencl_device):
        self._type = (
            DeviceType.CPU if pyopencl_device.type == pyopencl.device_type.CPU else DeviceType.GPU)

        if (pyopencl_device.platform.name == 'Apple' and
                pyopencl_device.type == pyopencl.device_type.CPU):
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
            self._local_mem_banks = 16 if pyopencl_device.compute_capability_major_nv < 2 else 32
            self._warp_size = pyopencl_device.warp_size_nv
        elif pyopencl_device.vendor == 'NVIDIA':
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


class OclContextAdapter(ContextAdapter):

    @classmethod
    def from_pyopencl_devices(
            cls, pyopencl_devices: Iterable[pyopencl.Device]) -> OclContextAdapter:
        """
        Creates a context based on one or several (distinct) PyOpenCL ``Device`` objects.
        """
        pyopencl_devices = normalize_object_sequence(pyopencl_devices, pyopencl.Device)
        return cls(pyopencl.Context(pyopencl_devices), pyopencl_devices=pyopencl_devices)

    @classmethod
    def from_pyopencl_context(cls, pyopencl_context: pyopencl.Context) -> OclContextAdapter:
        """
        Creates a context based on a (possibly multi-device) PyOpenCL ``Context`` object.
        """
        return cls(pyopencl_context)

    @classmethod
    def from_device_adapters(cls, device_adapters: Iterable[OclDeviceAdapter]) -> OclContextAdapter:
        """
        Creates a context based on one or several (distinct) :py:class:`OclDeviceAdapter` objects.
        """
        device_adapters = normalize_object_sequence(device_adapters, OclDeviceAdapter)
        return cls.from_pyopencl_devices(
            [device_adapter.pyopencl_device for device_adapter in device_adapters])

    def __init__(self, pyopencl_context: pyopencl.Context, pyopencl_devices=None):
        if pyopencl_devices is None:
            pyopencl_devices = pyopencl_context.devices

        # Not checking here that all the `pyopencl_devices` are actually present
        # in `pyopencl_context`, since the constructor should only be called
        # from the trusted classmethods, which already ensure that.

        self._pyopencl_context = pyopencl_context
        self._pyopencl_devices = pyopencl_devices

        self._device_adapters = tuple(
            OclDeviceAdapter.from_pyopencl_device(device) for device in self._pyopencl_devices)
        self.platform_adapter = self.device_adapters[0].platform_adapter

        # On an Apple platform, in a multi-device context with nVidia cards it is necessary to have
        # any created OpenCL buffers allocated on the host (with ALLOC_HOST_PTR flag).
        # If it is not the case, using a subregion of such a buffer leads to a crash.
        is_multi_device = len(self._pyopencl_devices) > 1
        is_apple_platform = self._pyopencl_devices[0].platform.name == 'Apple'
        has_geforce_device = any('GeForce' in device.name for device in self._pyopencl_devices)
        self._buffers_host_allocation = is_multi_device and is_apple_platform and has_geforce_device

    @property
    def device_adapters(self) -> Tuple[OclDeviceAdapter, ...]:
        return self._device_adapters

    def deactivate(self):
        pass

    @staticmethod
    def render_prelude(fast_math=False):
        return _PRELUDE.render(
            fast_math=fast_math,
            dtypes=dtypes)

    def compile_single_device(
            self, device_idx, prelude, src,
            keep=False, fast_math=False, compiler_options=[], constant_arrays={}):

        # Sanity check: should have been caught in compile()
        assert not constant_arrays

        if keep:
            temp_dir = mkdtemp()
            temp_file_path = os.path.join(temp_dir, 'kernel.cl')

            with open(temp_file_path, 'w') as f:
                # `str()` is for convenience of mocking;
                # during a normal operation `src` is already a string
                f.write(str(src))

            print("*** compiler output in", temp_dir)

        else:
            temp_dir = None

        options = (
            compiler_options +
            (["-cl-mad-enable", "-cl-fast-relaxed-math"] if fast_math else []))
        full_src = prelude + src

        pyopencl_program = pyopencl.Program(self._pyopencl_context, full_src)

        try:
            pyopencl_program.build(
                devices=[self._pyopencl_devices[device_idx]], options=options, cache_dir=temp_dir)
        except pyopencl.RuntimeError as e:
            raise AdapterCompilationError(e, full_src)

        return OclProgramAdapter(self, device_idx, pyopencl_program, full_src)

    def make_queue_adapter(self, device_idxs):
        device_adapters = {
            device_idx: self._device_adapters[device_idx] for device_idx in device_idxs}
        pyopencl_queues = {
            device_idx: pyopencl.CommandQueue(
                self._pyopencl_context, device=self._device_adapters[device_idx].pyopencl_device)
            for device_idx in device_idxs}
        return OclQueueAdapter(self, device_adapters, pyopencl_queues)

    def allocate(self, size):
        flags = pyopencl.mem_flags.READ_WRITE
        if self._buffers_host_allocation:
            flags |= pyopencl.mem_flags.ALLOC_HOST_PTR

        pyopencl_buffer = pyopencl.Buffer(self._pyopencl_context, flags, size=size)
        return OclBufferAdapter(self, pyopencl_buffer)


class OclBufferAdapter(BufferAdapter):

    def __init__(self, context_adapter, pyopencl_buffer):
        self._context_adapter = context_adapter
        self.pyopencl_buffer = pyopencl_buffer
        self.kernel_arg = pyopencl_buffer

    @property
    def size(self) -> int:
        return self.pyopencl_buffer.size

    @property
    def offset(self) -> int:
        return self.pyopencl_buffer.offset

    def set(self, queue_adapter, device_idx, buf, no_async=False):
        buf_data = buf.pyopencl_buffer if isinstance(buf, OclBufferAdapter) else buf
        kwds = dict(wait_for=queue_adapter._other_device_events(device_idx))
        # This keyword is only supported for transfers involving hosts in PyOpenCL
        if not isinstance(buf, OclBufferAdapter):
            kwds['is_blocking'] = no_async
        pyopencl.enqueue_copy(
            queue_adapter._pyopencl_queues[device_idx], self.pyopencl_buffer, buf_data, **kwds)

    def get(self, queue_adapter, device_idx, host_array, async_=False):
        wait_for = queue_adapter._other_device_events(device_idx)
        pyopencl.enqueue_copy(
            queue_adapter._pyopencl_queues[device_idx], host_array, self.pyopencl_buffer,
            wait_for=wait_for, is_blocking=not async_)

    def get_sub_region(self, origin, size):
        return OclBufferAdapter(
            self._context_adapter, self.pyopencl_buffer.get_sub_region(origin, size))

    def migrate(self, queue_adapter, device_idx):
        pyopencl.enqueue_migrate_mem_objects(
            queue_adapter._pyopencl_queues[device_idx], [self.pyopencl_buffer])


class OclQueueAdapter(QueueAdapter):

    def __init__(self, context_adapter, device_adapters, pyopencl_queues):
        self._context_adapter = context_adapter
        self._device_adapters = device_adapters
        self._pyopencl_queues = pyopencl_queues

    @property
    def device_adapters(self):
        return self._device_adapters

    def _other_device_events(self, skip_device_idx):
        return [
            pyopencl.enqueue_marker(self._pyopencl_queues[device_idx])
            for device_idx in self._pyopencl_queues
            if device_idx != skip_device_idx]

    def synchronize(self):
        for pyopencl_queue in self._pyopencl_queues.values():
            pyopencl_queue.finish()


class OclProgramAdapter(ProgramAdapter):

    def __init__(self, context_adapter, device_idx, pyopencl_program, source):
        self._context_adapter = context_adapter
        self._device_idx = device_idx
        self._pyopencl_program = pyopencl_program
        self.source = source

    def __getattr__(self, kernel_name):
        pyopencl_kernel = getattr(self._pyopencl_program, kernel_name)
        return OclKernelAdapter(self, self._device_idx, pyopencl_kernel)


class OclKernelAdapter(KernelAdapter):

    def __init__(self, program_adapter: OclProgramAdapter, device_idx: int, pyopencl_kernel):
        self._program_adapter = program_adapter
        self._device_idx = device_idx
        self._pyopencl_kernel = pyopencl_kernel

    def prepare(
            self, queue_adapter: OclQueueAdapter, global_size: Tuple[int, ...],
            local_size: Tuple[int, ...]) -> OclPreparedKernelAdapter:
        return OclPreparedKernelAdapter(self, queue_adapter, global_size, local_size)

    @property
    def max_total_local_size(self) -> int:
        return self._pyopencl_kernel.get_work_group_info(
            pyopencl.kernel_work_group_info.WORK_GROUP_SIZE,
            self._program_adapter._context_adapter.device_adapters[self._device_idx].pyopencl_device)


class OclPreparedKernelAdapter(PreparedKernelAdapter):

    def __init__(
            self, kernel_adapter: OclKernelAdapter,
            queue_adapter: OclQueueAdapter, global_size: Tuple[int, ...],
            local_size: Tuple[int, ...]):
        self._kernel_adapter = kernel_adapter
        self._local_size = local_size
        self._global_size = global_size
        self._pyopencl_kernel = kernel_adapter._pyopencl_kernel
        self._pyopencl_queue = queue_adapter._pyopencl_queues[kernel_adapter._device_idx]

    def __call__(self, *args):
        return self._pyopencl_kernel(
            self._pyopencl_queue, self._global_size, self._local_size, *args)

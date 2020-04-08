from __future__ import annotations

from typing import Iterable, List, Tuple, Optional, Sequence

import numpy

try:
    import pyopencl
    import pyopencl.array
except ImportError:
    pyopencl = None

from .utils import all_same, all_different, wrap_in_tuple, normalize_base_objects
from .template import Template
from . import dtypes
from .adapter_base import (
    DeviceType, APIID, APIAdapterFactory, APIAdapter, PlatformAdapter, DeviceAdapter,
    DeviceParameters, ContextAdapter, BufferAdapter, QueueAdapter, ProgramAdapter, KernelAdapter)


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
                "OpenCL API is not operational. Check if PyCUDA is installed correctly.")

        return OclAPIAdapter()


class OclAPIAdapter(APIAdapter):

    @property
    def id(self):
        return _API_ID

    @property
    def num_platforms(self):
        return len(pyopencl.get_platforms())

    def get_platform_adapters(self):
        return [
            OclPlatformAdapter(self, platform, platform_num)
            for platform_num, platform in enumerate(pyopencl.get_platforms())]

    def isa_backend_device(self, obj):
        return isinstance(obj, pyopencl.Device)

    def isa_backend_context(self, obj):
        return isinstance(obj, pyopencl.Context)

    def make_context_from_backend_devices(self, backend_devices):
        raise NotImplementedError

    def make_context_from_backend_contexts(self, backend_contexts):
        raise NotImplementedError


class OclPlatformAdapter(PlatformAdapter):

    @classmethod
    def from_pyopencl_platform(cls, pyopencl_platform: pyopencl.Platform):
        """
        Creates this object from a PyOpenCL ``Platform`` object.
        """
        api_adapter = OclAPIAdapter()
        platform_adapters = api_adapter.get_platform_adapters()
        for platform_num, platform_adapter in enumerate(platform_adapters):
            if pyopencl_platform == platform_adapter.pyopencl_platform:
                return cls(api_adapter, pyopencl_platform, platform_num)

        raise Exception(f"{pyopencl_platform} was not found among OpenCL platforms")

    def __init__(self, api_adapter, pyopencl_platform, platform_num):
        self._api_adapter = api_adapter
        self.pyopencl_platform = pyopencl_platform
        self._platform_num = platform_num

    @property
    def api_adapter(self):
        return self._api_adapter

    @property
    def platform_num(self):
        return self._platform_num

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
    def num_devices(self):
        return len(self.pyopencl_platform.get_devices())

    def get_device_adapters(self):
        return [
            OclDeviceAdapter(self, device, device_num)
            for device_num, device in enumerate(self.pyopencl_platform.get_devices())]

    def make_context(self, device_adapters):
        return OclContextAdapter.from_device_adapters(device_adapters)


class OclDeviceAdapter(DeviceAdapter):

    @classmethod
    def from_pyopencl_device(cls, pyopencl_device: pyopencl.Device) -> OclDeviceAdapter:
        """
        Creates this object from a PyOpenCL ``Device`` object.
        """
        platform_adapter = OclPlatformAdapter.from_pyopencl_platform(pyopencl_device.platform)
        for device_num, device_adapter in enumerate(platform_adapter.get_device_adapters()):
            if pyopencl_device == device_adapter.pyopencl_device:
                return cls(platform_adapter, pyopencl_device, device_num)

        raise Exception(f"{pyopencl_device} was not found among OpenCL devices")

    def __init__(self, platform_adapter, pyopencl_device, device_num):
        self._platform_adapter = platform_adapter
        self._device_num = device_num
        self.pyopencl_device = pyopencl_device
        self._params = None

    @property
    def platform_adapter(self):
        return self._platform_adapter

    @property
    def device_num(self):
        return self._device_num

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
        # TODO: support other device types
        self._type = DeviceType.CPU if pyopencl_device.type == pyopencl.device_type.CPU else DeviceType.GPU

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

        max_size = 2**pyopencl_device.address_bits
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
    def from_any_base(cls, objs) -> OclContextAdapter:
        """
        Create a context based on any object supported by other ``from_*`` methods.
        """
        objs = wrap_in_tuple(objs)
        if isinstance(objs[0], pyopencl.Device):
            return cls.from_pyopencl_devices(objs)
        elif isinstance(objs[0], pyopencl.Context):
            return cls.from_pyopencl_context(objs)
        elif isinstance(objs[0], OclDeviceAdapter):
            return cls.from_devices(objs)
        else:
            raise TypeError(f"Do not know how to create a context out of {type(objs[0])}")

    @classmethod
    def from_pyopencl_devices(cls, pyopencl_devices: Iterable[pyopencl.Device]) -> OclContextAdapter:
        """
        Creates a context based on one or several (distinct) PyOpenCL ``Device`` objects.
        """
        pyopencl_devices = normalize_base_objects(pyopencl_devices, pyopencl.Device)
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
        device_adapters = normalize_base_objects(device_adapters, OclDeviceAdapter)
        if not all_same(device_adapter.platform_adapter for device_adapter in device_adapters):
            raise ValueError("All devices must belong to the same platform")
        return cls.from_pyopencl_devices(
            [device_adapter.pyopencl_device for device_adapter in device_adapters])

    def __init__(self, pyopencl_context: pyopencl.Context, pyopencl_devices=None):
        if pyopencl_devices is None:
            pyopencl_devices = pyopencl_context.devices
        else:
            if not set(pyopencl_context.devices) == set(pyopencl_devices):
                raise ValueError(
                    "`pyopencl_devices` argument must contain the same devices "
                    "as the provided context")

        self._pyopencl_context = pyopencl_context
        self._pyopencl_devices = pyopencl_devices

        self._device_adapters = tuple(
            OclDeviceAdapter.from_pyopencl_device(device) for device in self._pyopencl_devices)
        self.platform_adapter = self.device_adapters[0].platform_adapter

        # On an Apple platform, in a multi-device context with nVidia cards it is necessary to have
        # any created OpenCL buffers allocated on the host (with ALLOC_HOST_PTR flag).
        # If it is not the case, using a subregion of such a buffer leads to a crash.
        self._buffers_host_allocation = (
            len(self._pyopencl_devices) > 1 and
            self._pyopencl_devices[0].platform.name == 'Apple' and
            any('GeForce' in device.name for device in self._pyopencl_devices))

    @property
    def device_adapters(self) -> Tuple[OclDeviceAdapter, ...]:
        return self._device_adapters

    @property
    def compile_error_class(self):
        return pyopencl.RuntimeError

    def render_prelude(self, fast_math=False):
        return _PRELUDE.render(
            fast_math=fast_math,
            dtypes=dtypes)

    def compile_single_device(
            self, device_num, prelude, src,
            keep=False, fast_math=False, compiler_options=[], constant_arrays={}):

        # Sanity check: should have been caught in compile()
        assert not constant_arrays

        if keep:
            temp_dir = mkdtemp()
            temp_file_path = os.path.join(temp_dir, 'kernel.cl')

            with open(temp_file_path, 'w') as f:
                f.write(src)

            print("*** compiler output in", temp_dir)

        else:
            temp_dir = None

        options = compiler_options + (["-cl-mad-enable", "-cl-fast-relaxed-math"] if fast_math else [])

        pyopencl_program = pyopencl.Program(self._pyopencl_context, prelude + src)
        pyopencl_program.build(
            devices=[self._pyopencl_devices[device_num]], options=options, cache_dir=temp_dir)
        return OclProgram(self, device_num, pyopencl_program)

    def make_queue_adapter(self, device_nums):
        device_adapters = {
            device_num: self._device_adapters[device_num] for device_num in device_nums}
        pyopencl_queues = {
            device_num: pyopencl.CommandQueue(
                self._pyopencl_context, device=self._device_adapters[device_num].pyopencl_device)
            for device_num in device_nums}
        return OclQueueAdapter(self, device_adapters, pyopencl_queues)

    def allocate(self, size):
        flags = pyopencl.mem_flags.READ_WRITE
        if self._buffers_host_allocation:
            flags |= pyopencl.mem_flags.ALLOC_HOST_PTR

        pyopencl_buffer = pyopencl.Buffer(self._pyopencl_context, flags, size=size)
        return OclBufferAdapter(self, pyopencl_buffer)


class OclBufferAdapter(BufferAdapter):

    def __init__(self, context_adapter, pyopencl_buffer):
        self.context_adapter = context_adapter
        self.pyopencl_buffer = pyopencl_buffer
        self.kernel_arg = pyopencl_buffer

    @property
    def size(self) -> int:
        return self.pyopencl_buffer.size

    @property
    def offset(self) -> int:
        return self.pyopencl_buffer.offset

    def set(self, queue_adapter, device_num, host_array, async_=False, dont_sync_other_devices=False):
        if dont_sync_other_devices:
            wait_for = []
        else:
            wait_for = queue_adapter._other_device_events(device_num)
        pyopencl.enqueue_copy(
            queue_adapter._pyopencl_queues[device_num], self.pyopencl_buffer, host_array,
            wait_for=wait_for, is_blocking=not async_)

    def get(self, queue_adapter, device_num, host_array, async_=False, dont_sync_other_devices=False):
        if dont_sync_other_devices:
            wait_for = []
        else:
            wait_for = queue_adapter._other_device_events(device_num)
        pyopencl.enqueue_copy(
            queue_adapter._pyopencl_queues[device_num], host_array, self.pyopencl_buffer,
            wait_for=wait_for, is_blocking=not async_)

    def get_sub_region(self, origin, size):
        return OclBufferAdapter(self.context_adapter, self.pyopencl_buffer.get_sub_region(origin, size))

    def migrate(self, queue_adapter, device_num):
        pyopencl.enqueue_migrate_mem_objects(
            queue_adapter._pyopencl_queues[device_num], [self.pyopencl_buffer])


class OclQueueAdapter(QueueAdapter):

    def __init__(self, context_adapter, device_adapters, pyopencl_queues):
        self.context_adapter = context_adapter
        self.device_adapters = device_adapters
        self._pyopencl_queues = pyopencl_queues

    def _other_device_events(self, skip_device_num):
        return [
            pyopencl.enqueue_marker(self._pyopencl_queues[device_num])
            for device_num in self._pyopencl_queues
            if device_num != skip_device_num]

    def synchronize(self, device_num=None):
        if device_num is None:
            device_nums = self._pyopencl_queues.keys()
        else:
            device_nums = [device_num]

        for device_num in device_nums:
            self._pyopencl_queues[device_num].finish()


class OclProgram(ProgramAdapter):

    def __init__(self, context_adapter, device_num, pyopencl_program):
        self.context_adapter = context_adapter
        self._device_num = device_num
        self._pyopencl_program = pyopencl_program

    def __getattr__(self, kernel_name):
        pyopencl_kernel = getattr(self._pyopencl_program, kernel_name)
        return OclKernel(self, self._device_num, pyopencl_kernel)


class OclKernel(KernelAdapter):

    def __init__(self, program_adapter, device_num, pyopencl_kernel):
        self.program_adapter = program_adapter
        self._device_num = device_num
        self._pyopencl_kernel = pyopencl_kernel

    @property
    def max_total_local_size(self):
        return self._pyopencl_kernel.get_work_group_info(
            pyopencl.kernel_work_group_info.WORK_GROUP_SIZE,
            self.program_adapter.context_adapter.device_adapters[self._device_num].pyopencl_device)

    def __call__(self, queue_adapter, global_size, local_size, *args):
        return self._pyopencl_kernel(
            queue_adapter._pyopencl_queues[self._device_num], global_size, local_size, *args)

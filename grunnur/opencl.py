# Avoids errors from using PyOpenCL types as annotations when PyOpenCL is not present
from __future__ import annotations

from typing import Iterable, List, Tuple, Optional, Sequence

import numpy

try:
    import pyopencl
    import pyopencl.array
except ImportError:
    pyopencl = None

from .base_classes import (
    APIFactory, APIID, PlatformID, DeviceID, API, Platform,
    Device, DeviceType, DeviceParameters, Context, Queue, Program, Kernel,
    SingleDeviceProgram, SingleDeviceKernel, Buffer,
    normalize_base_objects, process_arg, Array)
from .utils import all_same, all_different, wrap_in_tuple
from .template import Template
from . import dtypes


_TEMPLATE = Template.from_associated_file(__file__)
_PRELUDE = _TEMPLATE.get_def('prelude')


class OclAPI(API):

    def get_platforms(self):
        return [
            OclPlatform(PlatformID(self.id, i), platform)
            for i, platform in enumerate(pyopencl.get_platforms())]

    @property
    def _context_class(self):
        return OclContext


class OclAPIFactory(APIFactory):

    @property
    def available(self):
        return pyopencl is not None

    def make_api(self):
        if not self.available:
            raise ImportError(
                "OpenCL API is not operational. Check if PyOpenCL is installed correctly.")

        return _OPENCL_API


OPENCL_API_ID = APIID('opencl')
_OPENCL_API = OclAPI(OPENCL_API_ID)
OPENCL_API_FACTORY = OclAPIFactory(OPENCL_API_ID)


def make_opencl_api():
    return OPENCL_API_FACTORY.make_api()


class OclPlatform(Platform):

    def __init__(self, platform_id: PlatformID, pyopencl_platform: pyopencl.Platform):
        super().__init__(_OPENCL_API, platform_id)
        self.pyopencl_platform = pyopencl_platform

    @property
    def name(self):
        return self.pyopencl_platform.name

    @property
    def vendor(self):
        return self.pyopencl_platform.vendor

    @property
    def version(self):
        return self.pyopencl_platform.version

    def get_devices(self):
        return [
            OclDevice(self, DeviceID(self.id, i), device)
            for i, device in enumerate(self.pyopencl_platform.get_devices())]

    @classmethod
    def from_pyopencl_platform(cls, pyopencl_platform: pyopencl.Platform):
        """
        Creates this object from a PyOpenCL ``Platform`` object.
        """
        platforms = _OPENCL_API.get_platforms()
        for platform in platforms:
            if pyopencl_platform == platform.pyopencl_platform:
                return cls(platform.id, pyopencl_platform)

        raise Exception(f"{pyopencl_platform} was not found among OpenCL platforms")


class OclDevice(Device):

    def __init__(self, platform: OclPlatform, device_id: DeviceID, pyopencl_device: pyopencl.Device):
        super().__init__(platform, device_id)
        self.pyopencl_device = pyopencl_device
        self._params = None

    @property
    def name(self):
        return self.pyopencl_device.name

    @property
    def params(self):
        if self._params is None:
            self._params = OclDeviceParameters(self.pyopencl_device)
        return self._params

    @classmethod
    def from_pyopencl_device(cls, pyopencl_device: pyopencl.Device) -> OclDevice:
        """
        Creates this object from a PyOpenCL ``Device`` object.
        """
        platform = OclPlatform.from_pyopencl_platform(pyopencl_device.platform)
        for device_num, device in enumerate(platform.get_devices()):
            if pyopencl_device == device.pyopencl_device:
                return cls(platform, device.id, pyopencl_device)

        raise Exception(f"{pyopencl_device} was not found among OpenCL devices")


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


class OclContext(Context):

    @classmethod
    def from_any_base(cls, objs) -> OclContext:
        """
        Create a context based on any object supported by other ``from_*`` methods.
        """
        objs = wrap_in_tuple(objs)
        if isinstance(objs[0], pyopencl.Device):
            return cls.from_pyopencl_devices(objs)
        elif isinstance(objs[0], pyopencl.Context):
            return cls.from_pyopencl_context(objs)
        elif isinstance(objs[0], OclDevice):
            return cls.from_devices(objs)
        else:
            raise TypeError(f"Do not know how to create a context out of {type(objs[0])}")

    @classmethod
    def from_pyopencl_devices(cls, pyopencl_devices: Iterable[pyopencl.Device]) -> OclContext:
        """
        Creates a context based on one or several (distinct) PyOpenCL ``Device`` objects.
        """
        pyopencl_devices = normalize_base_objects(pyopencl_devices, pyopencl.Device)
        return cls(pyopencl.Context(pyopencl_devices), pyopencl_devices=pyopencl_devices)

    @classmethod
    def from_pyopencl_context(cls, pyopencl_context: pyopencl.Context) -> OclContext:
        """
        Creates a context based on a (possibly multi-device) PyOpenCL ``Context`` object.
        """
        return cls(pyopencl_context)

    @classmethod
    def from_devices(cls, devices: Iterable[OclDevice]) -> OclContext:
        """
        Creates a context based on one or several (distinct) :py:class:`OclDevice` objects.
        """
        devices = normalize_base_objects(devices, OclDevice)
        if not all_same(device.platform for device in devices):
            raise ValueError("All devices must belong to the same platform")
        return cls.from_pyopencl_devices([device.pyopencl_device for device in devices])

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

        self._devices = tuple(
            OclDevice.from_pyopencl_device(device) for device in self._pyopencl_devices)
        self.platform = self.devices[0].platform

        # On an Apple platform, in a multi-device context with nVidia cards it is necessary to have
        # any created OpenCL buffers allocated on the host (with ALLOC_HOST_PTR flag).
        # If it is not the case, using a subregion of such a buffer leads to a crash.
        self._buffers_host_allocation = (
            len(self._pyopencl_devices) > 1 and
            self._pyopencl_devices[0].platform.name == 'Apple' and
            any('GeForce' in device.name for device in self._pyopencl_devices))

    @property
    def devices(self) -> Tuple[OclDevice, ...]:
        return self._devices

    @property
    def api(self):
        return _OPENCL_API

    @property
    def _compile_error_class(self):
        return pyopencl.RuntimeError

    def _render_prelude(self, fast_math=False):
        return _PRELUDE.render(
            fast_math=fast_math,
            dtypes=dtypes)

    def _compile_single_device(
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
        return OclSingleDeviceProgram(self, device_num, pyopencl_program)

    @property
    def _program_class(self):
        return OclProgram

    def compile(self, *args, constant_arrays=None, **kwds):
        if constant_arrays is not None:
            # Check here to fail earlier and reduce the traceback size.
            raise ValueError("OpenCL does not support compile-time constant arrays")
        return super().compile(*args, **kwds)

    def allocate(self, size):
        return OclBuffer.allocate(self, size)

    @property
    def _array_class(self):
        return OclArray

    @property
    def _queue_class(self):
        return OclQueue


class OclBuffer(Buffer):

    @classmethod
    def allocate(cls, context, size):
        flags = pyopencl.mem_flags.READ_WRITE
        if context._buffers_host_allocation:
            flags |= pyopencl.mem_flags.ALLOC_HOST_PTR

        pyopencl_buffer = pyopencl.Buffer(context._pyopencl_context, flags, size=size)

        return cls(context, pyopencl_buffer)

    def __init__(self, context, pyopencl_buffer):

        self._context = context
        self._size = pyopencl_buffer.size
        self._offset = pyopencl_buffer.offset
        self._pyopencl_buffer = pyopencl_buffer

    @property
    def backend_buffer(self):
        return self._pyopencl_buffer

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
        return OclBuffer(self._context, self._pyopencl_buffer.get_sub_region(origin, size))


class OclQueue(Queue):

    @classmethod
    def from_pyopencl_commandqueues(
            cls, pyopencl_queues: Iterable[pyopencl.CommandQueue]) -> OclQueue:
        """
        Create this object, and an associated :py:class:`OclContext`
        from a (possibly multi-device) PyOpenCL ``CommandQueue``.
        """
        pyopencl_queues = normalize_base_objects(pyopencl_queues, pyopencl.CommandQueue)

        pyopencl_contexts = [queue.context for queue in pyopencl_queues]
        if not all_same(pyopencl_contexts):
            raise ValueError("All CommandQueue objects must belong to the same context")

        pyopencl_devices = [queue.device for queue in pyopencl_queues]
        if not all_different(pyopencl_devices):
            raise ValueError("All CommandQueue objects must run on different devices")

        context = OclContext.from_pyopencl_context(pyopencl_contexts[0])
        devices = [
            OclDevice.from_pyopencl_device(pyopencl_device)
            for pyopencl_device in pyopencl_devices]

        device_nums = [context.devices.index(device) for device in devices]

        return cls(context, device_nums=device_nums)

    def __init__(self, context: OclContext, device_nums: Optional[Sequence[int]]=None):
        super().__init__(context, device_nums=device_nums)

        self._pyopencl_queues = [
            pyopencl.CommandQueue(
                context._pyopencl_context,
                device=context._pyopencl_devices[device_num])
            for device_num in self._device_nums]

    def synchronize(self):
        for queue in self._pyopencl_queues:
            queue.finish()


class OclArray(Array):

    def __init__(self, queue, *args, **kwds):
        super().__init__(queue, *args, **kwds)
        self._queue = queue
        self._pyopencl_queues = queue._pyopencl_queues
        self._default_queue = self._pyopencl_queues[0]

    def _new_like_me(self, *args, device_num=None, **kwds):
        if device_num is None:
            return OclArray(self._queue, *args, **kwds)
        else:
            return OclSingleDeviceArray(self._queue, device_num, *args, **kwds)

    def _events(self):
        return [pyopencl.enqueue_marker(queue) for queue in self._pyopencl_queues[1:]]

    def set(self, array, async_=False):
        pyopencl.enqueue_copy(
            self._default_queue, self.data.backend_buffer,
            array, wait_for=self._events(), is_blocking=not async_)

    def get(self, dest=None, async_=False):
        if dest is None:
            dest = numpy.empty(self.shape, self.dtype)
        pyopencl.enqueue_copy(
            self._default_queue, dest, self.data.backend_buffer,
            wait_for=self._events(), is_blocking=not async_)
        return dest


class OclSingleDeviceArray(OclArray):

    def __init__(self, queue, device_num, *args, **kwds):
        super().__init__(queue, *args, **kwds)
        self._device_num = device_num
        self._default_queue = queue._pyopencl_queues[device_num]
        pyopencl.enqueue_migrate_mem_objects(self._default_queue, [self.data.backend_buffer])

    def _events(self):
        return []


class OclProgram(Program):

    @property
    def _kernel_class(self):
        return OclKernel


class OclSingleDeviceProgram(SingleDeviceProgram):

    def __init__(self, context, device_num, pyopencl_program):
        self.context = context
        self._device_num = device_num
        self._pyopencl_program = pyopencl_program

    def __getattr__(self, kernel_name):
        pyopencl_kernel = getattr(self._pyopencl_program, kernel_name)
        return OclSingleDeviceKernel(self, self._device_num, pyopencl_kernel)


class OclKernel(Kernel):
    pass


class OclSingleDeviceKernel(SingleDeviceKernel):

    def __init__(self, sd_program, device_num, pyopencl_kernel):
        self.program = sd_program
        self._device_num = device_num
        self._pyopencl_kernel = pyopencl_kernel

    @property
    def max_total_local_size(self):
        return self._pyopencl_kernel.get_work_group_info(
            pyopencl.kernel_work_group_info.WORK_GROUP_SIZE,
            self.program.context.devices[self._device_num].pyopencl_device)

    def __call__(self, queue, global_size, local_size, *args):

        if local_size is not None:
            local_size = wrap_in_tuple(local_size)
        global_size = wrap_in_tuple(global_size)

        args = process_arg(args)

        assert self._device_num in queue.device_nums

        return self._pyopencl_kernel(
            queue._pyopencl_queues[self._device_num], global_size, local_size, *args)

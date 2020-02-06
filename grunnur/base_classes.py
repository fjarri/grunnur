from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from math import log10
from typing import List, Type, Optional, Tuple, Iterable, Mapping, Union, Callable
import re

import numpy

from .array_metadata import ArrayMetadata
from .modules import render_with_modules
from .utils import all_same, all_different, wrap_in_tuple


class APIID:
    """
    An ID of an :py:class:`API` object.

    .. py:attribute:: shortcut

        This API's shortcut.

    .. py:attribute:: short_name

        This API's short name.
    """

    def __init__(self, shortcut: str):
        self.shortcut = shortcut
        self.short_name = f"id({self.shortcut})"

    def __eq__(self, other):
        return self.shortcut == other.shortcut

    def __hash__(self):
        return hash((type(self), self.shortcut))


class PlatformID:
    """
    An ID of a :py:class:`Platform` object.

    .. py:attribute:: api_id

        An :py:class:`APIID` of the parent :py:class:`API` object.

    .. py:attribute:: shortcut

        This Platform ID's shortcut.

    .. py:attribute:: short_name

        This Platform ID's short name.
    """

    def __init__(self, api_id: APIID, platform_num: int):
        self.api_id = api_id
        self._platform_num = platform_num
        self.shortcut = f"{api_id.shortcut},{platform_num}"
        self.short_name = f"id({self.shortcut})"

    def __eq__(self, other):
        return self.api_id == other.api_id and self._platform_num == other._platform_num

    def __hash__(self):
        return hash((type(self), self.api_id, self._platform_num))


class DeviceID:
    """
    An ID of a :py:class:`Device` object.

    .. py:attribute:: platform_id

        An :py:class:`PlatformID` of the parent :py:class:`Platform` object.

    .. py:attribute:: shortcut

        This Device ID's shortcut.

    .. py:attribute:: short_name

        This Device ID's short name.
    """

    def __init__(self, platform_id: PlatformID, device_num: int):
        self.platform_id = platform_id
        self._device_num = device_num
        self.shortcut = f"{platform_id.shortcut},{device_num}"
        self.short_name = f"id({self.shortcut})"

    def __eq__(self, other):
        return self.platform_id == other.platform_id and self._device_num == other._device_num

    def __hash__(self):
        return hash((type(self), self.platform_id, self._device_num))


class APIFactory(ABC):
    """
    A helper class that allows handling cases when an API's backend is unavailable
    or temporarily replaced by a mock object.
    """

    def __init__(self, api_id):
        self.api_id = api_id
        self.short_name = f"api_factory({api_id.shortcut})"

    @abstractmethod
    def make_api(self):
        pass

    @property
    @abstractmethod
    def available(self):
        pass


class API(ABC):
    """
    A generalized GPGPU API.

    .. py:attribute:: id

        This API's ID, an :py:class:`APIID` object.

    .. py:attribute:: short_name

        This API's short name.
    """

    def __init__(self, api_id: APIID):
        self.id = api_id
        self.short_name = f"api({api_id.shortcut})"

    @abstractmethod
    def get_platforms(self) -> List[Platform]:
        """
        Returns a list of platforms available for this API.
        """
        pass

    @property
    @abstractmethod
    def _context_class(self) -> Type[Context]:
        """
        Returns the context class specialized for this API.
        """
        pass

    @property
    def shortcut(self) -> str:
        """
        Returns a shortcut for this API (to use in :py:func:`~grunnur.find_apis`).
        """
        return self.id.shortcut

    def find_platforms(
            self,
            include_masks: Optional[Iterable[str]]=None,
            exclude_masks: Optional[Iterable[str]]=None) -> List[Platform]:
        """
        Returns a list of all platforms with names satisfying the given criteria.

        :param include_masks: a list of strings (treated as regexes),
            one of which must match with the platform name.
        :param exclude_masks: a list of strings (treated as regexes),
            neither of which must match with the platform name.
        """
        return [
            platform for platform in self.get_platforms()
            if _name_matches_masks(
                platform.name, include_masks=include_masks, exclude_masks=exclude_masks)
            ]

    def find_devices(
            self,
            quantity: Optional[int]=1,
            platform_include_masks: Optional[Iterable[str]]=None,
            platform_exclude_masks: Optional[Iterable[str]]=None,
            device_include_masks: Optional[Iterable[str]]=None,
            device_exclude_masks: Optional[Iterable[str]]=None,
            unique_devices_only: bool=False,
            include_pure_parallel_devices: bool=False) -> Tuple[Platform, List[Device]]:
        """
        Returns all tuples (platform, list of devices) where the platform name and device names
        satisfy the given criteria, and there are at least ``quantity`` devices in the list.

        :param quantity: the number of devices to find. If ``None``,
            find all matching devices belonging to a single platform.
        :param platform_include_masks: passed to :py:meth:`find_platforms`.
        :param platform_exclude_masks: passed to :py:meth:`find_platforms`.
        :param device_include_masks: passed to :py:meth:`Platform.find_devices`.
        :param device_exclude_masks: passed to :py:meth:`Platform.find_devices`.
        :param unique_devices_only: passed to :py:meth:`Platform.find_devices`.
        :param include_pure_parallel_devices: passed to :py:meth:`Platform.find_devices`.
        """

        results = []

        suitable_platforms = self.find_platforms(
            include_masks=platform_include_masks, exclude_masks=platform_exclude_masks)

        for platform in suitable_platforms:

            suitable_devices = platform.find_devices(
                include_masks=device_include_masks, exclude_masks=device_exclude_masks,
                unique_devices_only=unique_devices_only,
                include_pure_parallel_devices=include_pure_parallel_devices)

            if ((quantity is None and len(suitable_devices) > 0) or
                    (quantity is not None and len(suitable_devices) >= quantity)):
                results.append((platform, suitable_devices))

        return results

    def select_devices(
            self, interactive: bool=False, quantity: Optional[int]=1,
            **device_filters) -> List[Device]:
        """
        Using the results from :py:meth:`find_devices`, either lets the user
        select the devices (from the ones matching the criteria) interactively,
        or takes the first matching list of ``quantity`` devices.

        :param interactive: if ``True``, shows a dialog to select the devices.
            If ``False``, selects the first matching ones.
        :param quantity: passed to :py:meth:`find_devices`.
        :param device_filters: passed to :py:meth:`find_devices`.
        """
        suitable_pds = self.find_devices(quantity, **device_filters)

        if len(suitable_pds) == 0:
            quantity_val = "any" if quantity is None else quantity
            raise ValueError(
                f"Could not find {quantity_val} devices on a single platform "
                "matching the given criteria")

        if interactive:
            return _select_devices_interactive(suitable_pds, quantity=quantity)
        else:
            _, devices = suitable_pds[0]
            return devices if quantity is None else devices[:quantity]

    def create_some_context(
            self, interactive: bool=False, quantity: Optional[int]=1, **device_filters) -> Context:
        """
        Finds devices matching the given criteria and creates a
        :py:class:`Context` object out of them.

        :param interactive: passed to :py:meth:`select_devices`.
        :param quantity: passed to :py:meth:`select_devices`.
        :param device_filters: passed to :py:meth:`select_devices`.
        """
        devices = self.select_devices(interactive=interactive, quantity=quantity, **device_filters)
        return self.create_context(devices)

    def create_context(self, context_base) -> Context:
        """
        Creates a :py:class:`Context` object based on ``context_base``, which can be
        a :py:class:`Device`, a list of :py:class:`Device` objects
        (from this API and belonging to the same platform), or some additional API-specific types.
        See classmethods of :py:class:`~grunnur.cuda.CuContext` and
        :py:class:`~grunnur.opencl.OclContext` for details on the latter.

        :param context_base: an object to base the context on.
        """
        return self._context_class.from_any_base(context_base)


class Platform(ABC):
    """
    A generalized GPGPU platform.

    .. py:attribute:: api

        The :py:class:`API` object this platform belongs to.

    .. py:attribute:: id

        This platform's ID, a :py:class:`PlatformID` object.

    .. py:attribute:: short_name

        This platform's short name.
    """

    def __init__(self, api: API, platform_id: PlatformID):
        self.api = api
        self.id = platform_id
        self.short_name = f"platform({platform_id.shortcut})"

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Platform name.
        """
        pass

    @property
    @abstractmethod
    def vendor(self) -> str:
        """
        Platform vendor.
        """
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """
        Platform version.
        """
        pass

    @abstractmethod
    def get_devices(self) -> List[Device]:
        """
        Returns a list of all devices available for this platform.
        """
        pass

    def find_devices(
            self,
            include_masks: Optional[Iterable[str]]=None,
            exclude_masks: Optional[Iterable[str]]=None,
            unique_devices_only: bool=False,
            include_pure_parallel_devices: bool=False):
        """
        Returns a list of all devices satisfying the given criteria.

        :param include_masks: a list of strings (treated as regexes),
            one of which must match with the device name.
        :param exclude_masks: a list of strings (treated as regexes),
            neither of which must match with the device name.
        :param unique_devices_only: if ``True``, only return devices with unique names.
        :param include_pure_parallel_devices: if ``True``, include devices with
            :py:meth:`~Device.max_total_local_size` equal to 1.
        """

        seen_devices = set()
        devices = []

        for device in self.get_devices():
            if not _name_matches_masks(
                    device.name, include_masks=include_masks, exclude_masks=exclude_masks):
                continue

            if unique_devices_only and device.name in seen_devices:
                continue

            if not include_pure_parallel_devices and device.params.max_total_local_size == 1:
                continue

            seen_devices.add(device.name)
            devices.append(device)

        return devices

    def __str__(self):
        return self.name + " " + self.version


class Device(ABC):
    """
    A generalized GPGPU device.

    .. py:attribute:: platform

        The :py:class:`Platform` object this device belongs to.

    .. py:attribute:: id

        This device's ID, a :py:class:`DeviceID` object.

    .. py:attribute:: short_name

        This device's short name.
    """

    def __init__(self, platform: Platform, device_id: DeviceID):
        self.platform = platform
        self.id = device_id
        self.short_name = f"device({device_id.shortcut})"

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Device name
        """
        pass

    @property
    @abstractmethod
    def params(self) -> DeviceParameters:
        """
        This device's parameters.
        """
        pass

    def __hash__(self):
        return hash((type(self), self.id))


class DeviceType(Enum):
    """
    An enum representing a device's type.
    """

    CPU = 1
    "CPU type"

    GPU = 2
    "GPU type"


class DeviceParameters(ABC):
    """
    An object containing device's specifications.
    """

    @property
    @abstractmethod
    def type(self) -> DeviceType:
        """
        Device type.
        """
        pass

    @property
    @abstractmethod
    def max_total_local_size(self) -> int:
        """
        The maximum total number of threads in one block (CUDA),
        or work items in one work group (OpenCL).
        """
        pass

    @property
    @abstractmethod
    def max_local_sizes(self) -> Tuple[int]:
        """
        The maximum number of threads in one block (CUDA),
        or work items in one work group (OpenCL) for each of the available dimensions.
        """
        pass

    @property
    @abstractmethod
    def warp_size(self) -> int:
        """
        The number of threads (CUDA)/work items (OpenCL) that are executed synchronously
        (within one multiprocessor/compute unit).
        """
        pass

    @property
    @abstractmethod
    def max_num_groups(self) -> Tuple[int]:
        """
        The maximum number of blocks (CUDA)/work groups (OpenCL)
        for each of the available dimensions.
        """
        pass

    @property
    @abstractmethod
    def local_mem_size(self) -> int:
        """
        The size of shared (CUDA)/local (OpenCL) memory (in bytes).
        """
        pass

    @property
    @abstractmethod
    def local_mem_banks(self) -> int:
        """
        The number of independent channels for shared (CUDA)/local (OpenCL) memory,
        which can be used from one warp without request serialization.
        """
        pass

    @property
    @abstractmethod
    def compute_units(self) -> int:
        """
        The number of multiprocessors (CUDA)/compute units (OpenCL) for the device.
        """
        pass


class Context(ABC):
    """
    GPGPU context.
    """

    @property
    @abstractmethod
    def api(self) -> API:
        """
        The :py:class:`API` object this context belongs to.
        """
        pass

    @property
    @abstractmethod
    def _compile_error_class(self) -> Type[Exception]:
        """
        The exception class thrown by backend's compilation function.
        """
        pass

    @abstractmethod
    def _render_prelude(self, fast_math: bool=False) -> str:
        """
        Renders the prelude allowing one to write kernels compiling
        both in CUDA and OpenCL.

        :param fast_math: whether the compilation with fast math is requested.
        """
        # TODO: it doesn't really need the context object, move to API and make a class method?
        pass

    @abstractmethod
    def _compile_single_device(
            self,
            device_num: int,
            prelude: str,
            src: str,
            keep: bool=False,
            fast_math: bool=False,
            compiler_options: Iterable[str]=[],
            constant_arrays: Mapping[str, Tuple[int, numpy.dtype]]={}) -> SingleDeviceProgram:
        """
        Compiles the given source with the given prelude on a single device.

        :param device_num: the number of the device to compile on.
        :param prelude: the source of the prelude.
        :param src: the source of the kernels to be compiled.
        :param keep: see :py:meth:`compile`.
        :param fast_math: see :py:meth:`compile`.
        :param compiler_options: see :py:meth:`compile`.
        :param constant_arrays: (**CUDA only**) see :py:meth:`compile`.
        """
        pass

    def _render_and_compile_single_device(
            self,
            device_num: int,
            template_src: str,
            no_prelude: bool=False,
            fast_math: bool=False,
            render_args: Iterable=[],
            render_globals: Mapping={},
            **kwds) -> SingleDeviceProgram:
        """
        Renders and compiles the given template on a single device.

        :param device_num: the number of the device to compile on.
        :param template_src: see :py:meth:`compile`.
        :param no_prelude: see :py:meth:`compile`.
        :param fast_math: see :py:meth:`compile`.
        :param render_args: see :py:meth:`compile`.
        :param render_globals: see :py:meth:`compile`.
        :param kwds: additional parameters for compilation, see :py:func:`compile`.
        """

        src = render_with_modules(
            template_src, render_args=render_args, render_globals=render_globals)

        if no_prelude:
            prelude = ""
        else:
            prelude = self._render_prelude(fast_math=fast_math)

        try:
            return self._compile_single_device(
                device_num, prelude, src, fast_math=fast_math, **kwds)
        except self._compile_error_class:
            print(f"Failed to compile on device {device_num} ({self.devices[device_num]})")

            lines = src.split("\n")
            max_num_len = int(log10(len(lines))) + 1
            for i, l in enumerate(lines):
                print(str(i+1).rjust(max_num_len) + ": " + l)

            raise

    @property
    @abstractmethod
    def _program_class(self) -> Type[Program]:
        pass

    def compile(
            self,
            template_src: str,
            device_nums: Optional[Iterable[int]]=None,
            no_prelude: bool=False,
            fast_math: bool=False,
            render_args: Iterable=[],
            render_globals: Mapping={},
            compiler_options: Iterable[str]=[],
            keep: bool=False,
            constant_arrays: Mapping[str, Tuple[int, numpy.dtype]]={}) -> Program:
        """
        Compiles the given source on multiple devices.

        :param template_src: a string with the source code, or a Mako template source to render.
        :param device_nums: a list of device numbers to compile on.
        :param no_prelude: do not add prelude to the rendered source.
        :param fast_math: compile using fast (but less accurate) math functions.
        :param render_args: a list of positional args to pass to the template.
        :param render_globals: a dictionary of globals to pass to the template.
        :param compiler_options: a list of options to pass to the backend compiler.
        :param keep: keep the intermediate files in a temporary directory.
        :param constant_arrays: (**CUDA only**) a dictionary ``name: (size, dtype)``
            of global constant arrays to be declared in the program.
        """
        if device_nums is None:
            device_nums = range(len(self.devices))
        else:
            device_nums = sorted(device_nums)

        sd_programs = {}
        for device_num in device_nums:
            sd_program = self._render_and_compile_single_device(
                device_num, template_src,
                no_prelude=no_prelude,
                fast_math=fast_math,
                render_args=render_args,
                render_globals=render_globals,
                compiler_options=compiler_options,
                keep=keep,
                constant_arrays=constant_arrays)
            sd_programs[device_num] = sd_program

        return self._program_class(self, sd_programs)

    @abstractmethod
    def allocate(self, size: int) -> Buffer:
        """
        Allocates a memory buffer of size ``size`` bytes
        that can be shared between the devices in this context.
        """
        pass

    @property
    @abstractmethod
    def _array_class(self) -> Type[Array]:
        pass

    def empty_array(
            self,
            queue: Queue,
            shape: Union[int, Iterable[int]],
            dtype: numpy.dtype,
            strides: Optional[Iterable[int]]=None,
            first_element_offset: int=0,
            buffer_size: int=None,
            data: Buffer=None,
            allocator: Callable[[int], Buffer]=None) -> Array:
        """
        Returns an uninitialized :py:class:`Array` object connected to the given ``queue``.
        """
        metadata = ArrayMetadata(
            shape, dtype,
            strides=strides, first_element_offset=first_element_offset, buffer_size=buffer_size)
        return self._array_class(queue, metadata, data=data, allocator=allocator)

    def upload(self, queue: Queue, array: numpy.ndarray) -> Array:
        """
        Create a new :py:class:`Array` in the context
        and fill it with the data from a host array ``array``.
        """
        array_dev = self.empty_array(queue, array.shape, array.dtype)
        array_dev.set(array)
        return array_dev

    @property
    @abstractmethod
    def _queue_class(self) -> Type[Queue]:
        pass

    def make_queue(self, device_nums: Optional[Iterable[int]]=None) -> Queue:
        """
        Returns a new :py:class:`Queue` object running on ``device_nums``.
        If it is ``None``, all the context's devices are used.
        """
        return self._queue_class(self, device_nums=device_nums)


class Queue(ABC):
    """
    A queue on multiple devices.
    """

    @property
    @abstractmethod
    def context(self) -> Context:
        """
        The :py:class:`Context` object this queue belongs to.
        """
        pass

    @property
    @abstractmethod
    def device_nums(self) -> Tuple[int]:
        """
        Device numbers (in the context, sorted) this queue executes on.
        """
        pass

    @abstractmethod
    def synchronize(self):
        """
        Blocks until sub-queues on all devices are empty.
        """
        pass


class Program(ABC):
    """
    A program compiled for multiple devices.
    """

    def __init__(self, context: Context, sd_programs: Iterable[SingleDeviceProgram]):
        self.context = context
        self._sd_programs = sd_programs

    @property
    @abstractmethod
    def _kernel_class(self) -> Type[Kernel]:
        pass

    def __getattr__(self, kernel_name: str) -> Kernel:
        """
        Returns a :py:class:`Kernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        sd_kernels = {
            device_num: getattr(sd_program, kernel_name)
            for device_num, sd_program in self._sd_programs.items()}
        return self._kernel_class(self, sd_kernels)


class SingleDeviceProgram(ABC):
    """
    A program compiled for a single device.
    """

    @abstractmethod
    def __getattr__(self, kernel_name: str) -> SingleDeviceKernel:
        """
        Returns a :py:class:`SingleDeviceKernel` object for a function (CUDA)/kernel (OpenCL)
        with the name ``kernel_name``.
        """
        pass


class Buffer(ABC):
    """
    A memory buffer on the device.
    """

    @property
    @abstractmethod
    def context(self) -> Context:
        """
        The :py:class:`Context` object this buffer object was created in.
        """
        return self._context

    @property
    @abstractmethod
    def size(self) -> int:
        """
        This buffer's size (in bytes).
        """
        return self._size

    @property
    @abstractmethod
    def offset(self) -> int:
        """
        This buffer's offset from the start of the physical memory allocation
        (will be non-zero for buffers created using :py:meth:`get_sub_region`).
        """
        return self._offset

    @abstractmethod
    def get_sub_region(self, origin: int, size: int) -> Buffer:
        """
        Returns a buffer sub-region starting at ``origin`` and of length ``size`` (in bytes).
        """
        pass


class Array:
    """
    Array on the device.

    .. py:attribute:: shape: Tuple[int]

        Array shape.

    .. py:attribute:: dtype: numpy.dtype

        Array item data type.

    .. py:attribute:: strides: Tuple[int]

        Array strides.

    .. py:attribute:: first_element_offset: int

        Offset of the first element of the array.

    .. py:attribute:: buffer_size: int

        The total memory taken by the array in the buffer.
    """

    def __init__(self, array_metadata, data=None, allocator=None):

        self._metadata = array_metadata

        self.shape = self._metadata.shape
        self.dtype = self._metadata.dtype
        self.strides = self._metadata.strides
        self.first_element_offset = self._metadata.first_element_offset
        self.buffer_size = self._metadata.buffer_size

        if data is None:
            data = allocator(self.buffer_size)
        self.data = data

    def single_device_view(self, device_num: int) -> SingleDeviceFactory:
        """
        Returns a subscriptable object that produces sub-arrays based on the device ``device_num``.
        """
        return SingleDeviceFactory(self, device_num)

    def _view(self, slices, subregion=False, device_num=None):
        new_metadata = self._metadata[slices]

        if subregion:
            origin, size, new_metadata = new_metadata.minimal_subregion()
            data = self.data.get_sub_region(origin, size)
        else:
            data = self.data

        return self._new_like_me(new_metadata, device_num=device_num, data=data)

    @abstractmethod
    def set(self, array: numpy.ndarray, async_: bool=False):
        """
        Sets the data in this array from a CPU array.
        If ``async_`` is ``True``, this call blocks.
        """
        pass

    @abstractmethod
    def get(self, dest: Optional[numpy.ndarray]=None, async_: bool=False) -> numpy.ndarray:
        """
        Gets the data from this array to a CPU array.
        If ``dest`` is ``None``, the target array is created.
        If ``async_`` is ``True``, this call blocks.
        Returns the created CPU array, or ``dest`` if it was provided.
        """
        pass


class SingleDeviceFactory:
    """
    A subscriptable object that produces sub-arrays based on a single device.
    """

    def __init__(self, array, device_num):
        self._array = array
        self._device_num = device_num

    def __getitem__(self, slices) -> Array:
        """
        Return a view of the parent array bound to the device this factory was created for
        (see :py:meth:`Array.single_device_view`).
        """
        return self._array._view(slices, subregion=True, device_num=self._device_num)


def process_arg(arg):
    if isinstance(arg, Array):
        return arg.data
    elif isinstance(arg, (list, tuple)):
        return [process_arg(arg_part) for arg_part in arg]
    else:
        return arg


class Kernel(ABC):
    """
    A kernel compiled for multiple devices.
    """

    def __init__(self, program: Program, sd_kernels: Iterable[SingleDeviceKernel]):
        self._program = program
        self._device_nums = set(sd_kernels)
        self._sd_kernels = sd_kernels

    def __call__(
            self,
            queue: Queue,
            global_size: Union[int, Iterable[int]],
            local_size: Union[int, Iterable[int], None],
            *args,
            device_nums: Optional[Iterable[int]]=None,
            **kwds):
        """
        Enqueues the kernel on the chosen devices.

        :param queue: the multi-device queue to use.
        :param global_size: the total number of threads (CUDA)/work items (OpenCL) in each dimension
            (column-major). Note that there may be a maximum size in each dimension as well
            as the maximum number of dimensions. See :py:class:`DeviceParameters` for details.
        :param local_size: the number of threads in a block (CUDA)/work items in a
            work group (OpenCL) in each dimension (column-major).
            If ``None``, it will be chosen automatically.
        :param args: kernel arguments. Can be: :py:class:`Array` objects,
            :py:class:`Buffer` objects, ``numpy`` scalars.
        :param device_nums: the devices to enqueue the kernel on
            (*in the context, not in the queue*). Must be a subset of the devices of the ``queue``.
            If ``None``, all the ``queue``'s devices are used.
            Note that the used devices must be among the ones the parent :py:class:`Program`
            was compiled for.
        :param kwds: backend-specific keyword parameters.
        """

        if local_size is not None:
            local_size = wrap_in_tuple(local_size)
        global_size = wrap_in_tuple(global_size)

        args = process_arg(args)

        if device_nums is None:
            device_nums = queue.device_nums
        else:
            device_nums = list(device_nums)

        # TODO: speed this up. Probably shouldn't create sets on every kernel call.
        if not set(device_nums).issubset(self._device_nums):
            missing_dev_nums = [
                device_num for device_num in device_nums if device_num not in self._device_nums]
            raise ValueError(
                f"This kernel's program was not compiled for devices {missing_dev_nums.join(', ')}")

        ret_vals = []
        for i, device_num in enumerate(device_nums):
            kernel_args = [arg[i] if isinstance(arg, (list, tuple)) else arg for arg in args]
            ret_val = self._sd_kernels[device_num](queue, global_size, local_size, *kernel_args, **kwds)
            ret_vals.append(ret_val)

        return ret_vals


class SingleDeviceKernel(ABC):
    """
    A kernel compiled for a single device.
    """
    @property
    @abstractmethod
    def max_total_local_size(self) -> int:
        """
        The maximum possible number of threads in a block (CUDA)/work items in a work group (OpenCL)
        for this kernel.
        """
        pass

    @abstractmethod
    def __call__(
            self,
            queue: Queue,
            global_size: Union[int, Iterable[int]],
            local_size: Union[int, Iterable[int], None],
            *args):
        """
        Enqueues the kernel on a single device.

        :param queue: see :py:meth:`Kernel.__call__`.
        :param global_size: see :py:meth:`Kernel.__call__`.
        :param local_size: see :py:meth:`Kernel.__call__`.
        :param args: see :py:meth:`Kernel.__call__`.
        """
        pass


def normalize_base_objects(objs, expected_cls):

    objs = wrap_in_tuple(objs)

    elems = list(objs) # in case it is a generic iterable
    if len(elems) == 0:
        raise ValueError("The iterable of base objects for the context cannot be empty")

    if not all_different(elems):
        raise ValueError("All base objects must be different")

    types = [type(elem) for elem in elems]
    if not all(issubclass(tp, expected_cls) for tp in types):
        raise ValueError(f"The iterable must contain only subclasses of {expected_cls}, got {types}")

    return objs


def _name_matches_masks(name, include_masks=None, exclude_masks=None):

    if include_masks is None:
        include_masks = []
    if exclude_masks is None:
        exclude_masks = []

    if len(include_masks) > 0:
        for include_mask in include_masks:
            if re.search(include_mask, name):
                break
        else:
            return False

    if len(exclude_masks) > 0:
        for exclude_mask in exclude_masks:
            if re.search(exclude_mask, name):
                return False

    return True


def _select_devices_interactive(suitable_pds, quantity=1):

    if len(suitable_pds) == 1:
        platform, devices = suitable_pds[0]
        print(f"Platform: {platform.name}")
    else:
        print("Platforms:")
        for pnum, pd in enumerate(suitable_pds):
            platform, _ = pd
            print(f"[{pnum}]: {platform.name}")

        default_pnum = 0
        print(f"Choose the platform [{default_pnum}]: ", end='')
        selected_pnum = input()
        if selected_pnum == '':
            selected_pnum = default_pnum
        else:
            selected_pnum = int(selected_pnum)

        platform, devices = suitable_pds[selected_pnum]

    if quantity is None or (quantity is not None and len(devices) == quantity):
        selected_devices = devices
        print(f"Devices: {[device.name for device in selected_devices]}")
    else:
        print("Devices:")
        default_dnums = list(range(quantity))
        for dnum, device in enumerate(devices):
            print(f"[{dnum}]: {device.name}")
        default_dnums_str = ', '.join(str(dnum) for dnum in default_dnums)
        print(f"Choose the device(s), comma-separated [{default_dnums_str}]: ", end='')
        selected_dnums = input()
        if selected_dnums == '':
            selected_dnums = default_dnums
        else:
            selected_dnums = [int(dnum) for dnum in selected_dnums.split(',')]
            if len(selected_dnums) != quantity:
                raise ValueError(f"Exactly {quantity} devices must be selected")

        selected_devices = [devices[dnum] for dnum in selected_dnums]

    return selected_devices

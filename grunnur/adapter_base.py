from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Type, Mapping, Iterable

import numpy


class DeviceType(Enum):
    """
    An enum representing a device's type.
    """

    CPU = 1
    "CPU type"

    GPU = 2
    "GPU type"


class APIID:
    """
    An ID of an :py:class:`API` object.

    .. py:attribute:: shortcut

        This API's shortcut.

    .. py:attribute:: short_name

        This API's short name.
    """

    def __init__(self, shortcut):
        self.shortcut = shortcut
        self.short_name = f"id({self.shortcut})"

    def __eq__(self, other):
        return self.shortcut == other.shortcut

    def __hash__(self):
        return hash((type(self), self.shortcut))

    def __str__(self):
        return self.short_name


class APIAdapterFactory(ABC):
    """
    A helper class that allows handling cases when an API's backend is unavailable
    or temporarily replaced by a mock object.
    """
    @property
    @abstractmethod
    def api_id(self):
        pass

    @property
    @abstractmethod
    def available(self):
        pass

    @abstractmethod
    def make_api_adapter(self):
        pass


class APIAdapter(ABC):

    @property
    @abstractmethod
    def id(self):
        pass

    @property
    @abstractmethod
    def num_platforms(self):
        pass

    # TODO: have instead get_platform(platform_num)?
    @abstractmethod
    def get_platform_adapters(self):
        pass

    @abstractmethod
    def isa_backend_device(self, obj):
        pass

    @abstractmethod
    def isa_backend_context(self, obj):
        pass

    @abstractmethod
    def make_context_from_backend_devices(self, backend_devices):
        pass

    @abstractmethod
    def make_context_from_backend_contexts(self, backend_contexts):
        pass


class PlatformAdapter(ABC):

    @property
    @abstractmethod
    def api_adapter(self):
        pass

    @property
    @abstractmethod
    def platform_num(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def vendor(self):
        pass

    @property
    @abstractmethod
    def version(self):
        pass

    @property
    @abstractmethod
    def num_devices(self):
        pass

    @abstractmethod
    def get_device_adapters(self):
        pass

    @abstractmethod
    def make_context(self, device_adapters):
        pass


class DeviceAdapter(ABC):

    @property
    @abstractmethod
    def platform_adapter(self):
        pass

    @property
    @abstractmethod
    def device_num(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass


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


class ContextAdapter(ABC):

    @classmethod
    @abstractmethod
    def from_device_adapters(cls, device_adapters: Iterable[DeviceAdapter]) -> ContextAdapter:
        """
        Creates a context based on one or several (distinct) :py:class:`OclDeviceAdapter` objects.
        """
        pass

    @property
    @abstractmethod
    def device_adapters(self) -> Tuple[DeviceAdapter, ...]:
        pass

    @abstractmethod
    def make_queue_adapter(self, device_nums):
        pass

    @abstractmethod
    def allocate(self, size):
        pass

    @property
    @abstractmethod
    def compile_error_class(self) -> Type[Exception]:
        """
        The exception class thrown by backend's compilation function.
        """
        pass

    @abstractmethod
    def render_prelude(self, fast_math: bool=False) -> str:
        """
        Renders the prelude allowing one to write kernels compiling
        both in CUDA and OpenCL.

        :param fast_math: whether the compilation with fast math is requested.
        """
        # TODO: it doesn't really need the context object, move to API and make a class method?
        pass

    @property
    @abstractmethod
    def compile_error_class(self):
        pass

    @abstractmethod
    def compile_single_device(
            self,
            device_num: int,
            prelude: str,
            src: str,
            keep: bool=False,
            fast_math: bool=False,
            compiler_options: Iterable[str]=[],
            constant_arrays: Mapping[str, Tuple[int, numpy.dtype]]={}):
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


class BufferAdapter(ABC):
    """
    A memory buffer on the device.
    """

    #@property
    #@abstractmethod
    #def context_adapter(self):
    #    """
    #    The :py:class:`Context` object this buffer object was created in.
    #    """
    #    pass

    @property
    @abstractmethod
    def size(self) -> int:
        """
        This buffer's size (in bytes).
        """
        pass

    @property
    @abstractmethod
    def offset(self) -> int:
        """
        This buffer's offset from the start of the physical memory allocation
        (will be non-zero for buffers created using :py:meth:`get_sub_region`).
        """
        pass

    @abstractmethod
    def get_sub_region(self, origin: int, size: int):
        """
        Returns a buffer sub-region starting at ``origin`` and of length ``size`` (in bytes).
        """
        pass

    @abstractmethod
    def set(self, queue_adapter, device_num, host_array, async_=False, dont_sync_other_devices=False):
        pass

    @abstractmethod
    def get(self, queue_adapter, device_num, host_array, async_=False, dont_sync_other_devices=False):
        pass

    @abstractmethod
    def migrate(self, queue_adapter, device_num):
        pass


class QueueAdapter(ABC):

    @abstractmethod
    def synchronize(self):
        pass


class ProgramAdapter(ABC):

    @abstractmethod
    def __getattr__(self, kernel_name):
        pass


class KernelAdapter(ABC):

    @property
    @abstractmethod
    def max_total_local_size(self):
        pass

    @abstractmethod
    def __call__(self, queue_adapter, global_size, local_size, *args, local_mem=0):
        pass

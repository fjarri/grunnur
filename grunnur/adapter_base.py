from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Mapping, Iterable, Sequence, List, Mapping, TypeVar, Union, Protocol, Optional, runtime_checkable

import numpy


@runtime_checkable
class ArrayMetadataLike(Protocol):
    """
    A protocol for an object providing array metadata.
    `numpy.ndarray` or :py:class:`Array` follow this protocol.
    """

    shape: Tuple[int, ...]
    """Array shape."""

    dtype: numpy.dtype
    """The type of an array element."""


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
    An ID of an :py:class:`~grunnur.API` object.
    """

    shortcut: str
    """This API's shortcut."""

    def __init__(self, shortcut):
        self.shortcut = shortcut

    def __eq__(self, other):
        return self.shortcut == other.shortcut

    def __hash__(self):
        return hash((type(self), self.shortcut))

    def __str__(self):
        return f"id({self.shortcut})"


class APIAdapterFactory(ABC):
    """
    A helper class that allows handling cases when an API's backend is unavailable
    or temporarily replaced by a mock object.
    """
    @property
    @abstractmethod
    def api_id(self) -> "APIID":
        pass

    @property
    @abstractmethod
    def available(self) -> bool:
        pass

    @abstractmethod
    def make_api_adapter(self) -> "APIAdapter":
        pass


class APIAdapter(ABC):

    @property
    @abstractmethod
    def id(self) -> APIID:
        pass

    @property
    @abstractmethod
    def platform_count(self) -> int:
        pass

    @abstractmethod
    def get_platform_adapters(self) -> Sequence["PlatformAdapter"]:
        pass

    @abstractmethod
    def isa_backend_device(self, obj) -> bool:
        pass

    @abstractmethod
    def isa_backend_context(self, obj) -> bool:
        pass

    @abstractmethod
    def make_context_adapter_from_device_adapters(self, device_adapters: Sequence["DeviceAdapter"]) -> "ContextAdapter":
        pass

    @abstractmethod
    def make_context_adapter_from_backend_contexts(self, backend_contexts, take_ownership: bool) -> "ContextAdapter":
        pass

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash((type(self), self.id))


class PlatformAdapter(ABC):

    @property
    @abstractmethod
    def api_adapter(self) -> APIAdapter:
        pass

    @property
    @abstractmethod
    def platform_idx(self) -> int:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def vendor(self) -> str:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    @abstractmethod
    def device_count(self) -> int:
        pass

    @abstractmethod
    def get_device_adapters(self) -> List["DeviceAdapter"]:
        pass


class DeviceAdapter(ABC):

    @property
    @abstractmethod
    def platform_adapter(self) -> PlatformAdapter:
        pass

    @property
    @abstractmethod
    def device_idx(self) -> int:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def params(self) -> "DeviceParameters":
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
    def max_local_sizes(self) -> Tuple[int, ...]:
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
    def max_num_groups(self) -> Tuple[int, ...]:
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
    def from_device_adapters(cls, device_adapters) -> 'ContextAdapter':
        """
        Creates a context based on one or several (distinct) :py:class:`OclDeviceAdapter` objects.
        """
        pass

    @property
    @abstractmethod
    def device_adapters(self) -> Mapping[int, DeviceAdapter]:
        pass

    @property
    @abstractmethod
    def device_order(self) -> List[int]:
        pass

    @abstractmethod
    def make_queue_adapter(self, device_idxs: Sequence[int]):
        pass

    @abstractmethod
    def allocate(self, size: int, device_idx: int):
        pass

    @staticmethod
    @abstractmethod
    def render_prelude(fast_math: bool=False) -> str:
        """
        Renders the prelude allowing one to write kernels compiling
        both in CUDA and OpenCL.

        :param fast_math: whether the compilation with fast math is requested.
        """
        pass

    @abstractmethod
    def compile_single_device(
            self,
            device_idx: int,
            prelude: str,
            src: str,
            keep: bool=False,
            fast_math: bool=False,
            compiler_options: Optional[Iterable[str]]=None,
            constant_arrays: Optional[Mapping[str, ArrayMetadataLike]]=None) -> "ProgramAdapter":
        """
        Compiles the given source with the given prelude on a single device.

        :param device_idx: the number of the device to compile on.
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
    def set(self, queue_adapter, host_array, no_async=False):
        pass

    @abstractmethod
    def get(self, queue_adapter, host_array, async_=False):
        pass


class QueueAdapter(ABC):

    @abstractmethod
    def synchronize(self):
        pass


class AdapterCompilationError(RuntimeError):

    def __init__(self, backend_exception, source):
        super().__init__(str(backend_exception))
        self.backend_exception = backend_exception
        self.source = source


class ProgramAdapter(ABC):

    @abstractmethod
    def __getattr__(self, kernel_name: str) -> 'KernelAdapter':
        pass

    @abstractmethod
    def set_constant_buffer(
            self, queue_adapter: QueueAdapter,
            name: str, arr: Union[BufferAdapter, numpy.ndarray]):
        pass

    @property
    @abstractmethod
    def source(self) -> str:
        pass


class KernelAdapter(ABC):

    @property
    @abstractmethod
    def program_adapter(self) -> ProgramAdapter:
        pass

    @property
    @abstractmethod
    def max_total_local_size(self) -> int:
        pass

    @abstractmethod
    def prepare(self, global_size: Sequence[int], local_size: Optional[Sequence[int]] = None):
        pass


class PreparedKernelAdapter(ABC):

    @abstractmethod
    def __call__(self, queue_adapter: QueueAdapter, *args, local_mem: int=0):
        pass

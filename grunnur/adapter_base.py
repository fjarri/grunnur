from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    import numpy

    from .array_metadata import ArrayMetadataLike


class DeviceType(Enum):
    """An enum representing a device's type."""

    CPU = 1
    "CPU type"

    GPU = 2
    "GPU type"


class APIID:
    """An ID of an :py:class:`~grunnur.API` object."""

    shortcut: str
    """This API's shortcut."""

    def __init__(self, shortcut: str):
        self.shortcut = shortcut

    def __eq__(self, other: object) -> bool:
        return isinstance(other, APIID) and self.shortcut == other.shortcut

    def __hash__(self) -> int:
        return hash((type(self), self.shortcut))

    def __str__(self) -> str:
        return f"id({self.shortcut})"


class APIAdapterFactory(ABC):
    """
    A helper class that allows handling cases when an API's backend is unavailable
    or temporarily replaced by a mock object.
    """

    @property
    @abstractmethod
    def api_id(self) -> APIID:
        pass

    @property
    @abstractmethod
    def available(self) -> bool:
        pass

    @abstractmethod
    def make_api_adapter(self) -> APIAdapter:
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
    def get_platform_adapters(self) -> tuple[PlatformAdapter, ...]:
        pass

    @abstractmethod
    def isa_backend_device(self, obj: Any) -> bool:
        pass

    @abstractmethod
    def isa_backend_platform(self, obj: Any) -> bool:
        pass

    @abstractmethod
    def isa_backend_context(self, obj: Any) -> bool:
        pass

    @abstractmethod
    def make_device_adapter(self, backend_device: Any) -> DeviceAdapter:
        pass

    @abstractmethod
    def make_platform_adapter(self, backend_platform: Any) -> PlatformAdapter:
        pass

    @abstractmethod
    def make_context_adapter_from_device_adapters(
        self, device_adapters: Sequence[DeviceAdapter]
    ) -> ContextAdapter:
        pass

    @abstractmethod
    def make_context_adapter_from_backend_contexts(
        self, backend_contexts: Sequence[Any], *, take_ownership: bool
    ) -> ContextAdapter:
        pass

    def __eq__(self, other: object) -> bool:
        return isinstance(other, APIAdapter) and self.id == other.id

    def __hash__(self) -> int:
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
    def get_device_adapters(self) -> tuple[DeviceAdapter, ...]:
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
    def params(self) -> DeviceParameters:
        pass


class DeviceParameters(ABC):
    """An object containing device's specifications."""

    @property
    @abstractmethod
    def type(self) -> DeviceType:
        """Device type."""

    @property
    @abstractmethod
    def max_total_local_size(self) -> int:
        """
        The maximum total number of threads in one block (CUDA),
        or work items in one work group (OpenCL).
        """

    @property
    @abstractmethod
    def max_local_sizes(self) -> tuple[int, ...]:
        """
        The maximum number of threads in one block (CUDA),
        or work items in one work group (OpenCL) for each of the available dimensions.
        """

    @property
    @abstractmethod
    def warp_size(self) -> int:
        """
        The number of threads (CUDA)/work items (OpenCL) that are executed synchronously
        (within one multiprocessor/compute unit).
        """

    @property
    @abstractmethod
    def max_num_groups(self) -> tuple[int, ...]:
        """
        The maximum number of blocks (CUDA)/work groups (OpenCL)
        for each of the available dimensions.
        """

    @property
    @abstractmethod
    def local_mem_size(self) -> int:
        """The size of shared (CUDA)/local (OpenCL) memory (in bytes)."""

    @property
    @abstractmethod
    def local_mem_banks(self) -> int:
        """
        The number of independent channels for shared (CUDA)/local (OpenCL) memory,
        which can be used from one warp without request serialization.
        """

    @property
    @abstractmethod
    def compute_units(self) -> int:
        """The number of multiprocessors (CUDA)/compute units (OpenCL) for the device."""

    @abstractmethod
    def align_words(self, word_size: int) -> int:
        """
        The distance (in ``word_size`` units) between global memory base addresses
        that allow accesses of word-size ``word_size`` bytes to get coalesced.
        """


class ContextAdapter(ABC):
    @property
    @abstractmethod
    def device_adapters(self) -> Mapping[int, DeviceAdapter]:
        pass

    @property
    @abstractmethod
    def device_order(self) -> list[int]:
        pass

    @abstractmethod
    def make_queue_adapter(self, device_adapter: DeviceAdapter) -> QueueAdapter:
        pass

    @abstractmethod
    def allocate(self, device_adapter: DeviceAdapter, size: int) -> BufferAdapter:
        pass

    @staticmethod
    @abstractmethod
    def render_prelude(*, fast_math: bool = False) -> str:
        """
        Renders the prelude allowing one to write kernels compiling
        both in CUDA and OpenCL.

        :param fast_math: whether the compilation with fast math is requested.
        """

    @abstractmethod
    def compile_single_device(
        self,
        device_adapter: DeviceAdapter,
        prelude: str,
        src: str,
        *,
        keep: bool = False,
        fast_math: bool = False,
        compiler_options: Sequence[str] | None = None,
        constant_arrays: Mapping[str, ArrayMetadataLike] | None = None,
    ) -> ProgramAdapter:
        """
        Compiles the given source with the given prelude on a single device.

        :param device_idx: the number of the device to compile on.
        :param prelude: the source of the prelude to prepend to the main source.
        :param src: the source of the kernels to be compiled.
        :param keep: see :py:meth:`compile`.
        :param fast_math: see :py:meth:`compile`.
        :param compiler_options: see :py:meth:`compile`.
        :param constant_arrays: (**CUDA only**) see :py:meth:`compile`.
        """

    @abstractmethod
    def deactivate(self) -> None:
        """
        For CUDA API: deactivates this context, popping all the CUDA context objects from the stack.
        Other APIs: no effect.
        """


class BufferAdapter(ABC):
    """A memory buffer on the device."""

    @property
    @abstractmethod
    def kernel_arg(self) -> Any:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """The buffer's size (in bytes)."""

    @property
    @abstractmethod
    def offset(self) -> int:
        """
        The buffer's offset from the start of the physical memory allocation
        (will be non-zero for buffers created using :py:meth:`get_sub_region`).
        """

    @abstractmethod
    def get_sub_region(self, origin: int, size: int) -> BufferAdapter:
        """Returns a buffer sub-region starting at ``origin`` and of length ``size`` (in bytes)."""

    @abstractmethod
    def set(
        self,
        queue_adapter: QueueAdapter,
        source: numpy.ndarray[Any, numpy.dtype[Any]] | BufferAdapter,
        *,
        sync: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def get(
        self,
        queue_adapter: QueueAdapter,
        host_array: numpy.ndarray[Any, numpy.dtype[Any]],
        *,
        async_: bool = False,
    ) -> None:
        pass


class QueueAdapter(ABC):
    @abstractmethod
    def synchronize(self) -> None:
        pass


class AdapterCompilationError(RuntimeError):
    def __init__(self, backend_exception: Exception, source: str):
        super().__init__(str(backend_exception))
        self.backend_exception = backend_exception
        self.source = source


class ProgramAdapter(ABC):
    @abstractmethod
    def __getattr__(self, kernel_name: str) -> KernelAdapter:
        pass

    @abstractmethod
    def set_constant_buffer(
        self,
        queue_adapter: QueueAdapter,
        name: str,
        arr: BufferAdapter | numpy.ndarray[Any, numpy.dtype[Any]],
    ) -> None:
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
    def prepare(
        self, global_size: Sequence[int], local_size: Sequence[int] | None = None
    ) -> PreparedKernelAdapter:
        pass


class PreparedKernelAdapter(ABC):
    @abstractmethod
    def __call__(
        self,
        queue_adapter: QueueAdapter,
        *args: BufferAdapter | numpy.generic,
        cu_dynamic_local_mem: int = 0,
    ) -> Any:
        pass

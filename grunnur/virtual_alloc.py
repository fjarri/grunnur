from __future__ import annotations

from abc import abstractmethod
from collections import namedtuple, Counter
from collections.abc import Iterable
from typing import Set, Dict, Optional, cast
import weakref
from weakref import ReferenceType

from .sorted_list import SortedList
from .adapter_base import BufferAdapter
from .buffer import Buffer
from .array import Array
from .queue import Queue


def extract_dependencies(dependencies) -> Set[int]:
    """
    Recursively extracts allocation identifiers from an iterable or an ``Array`` object.
    """
    if isinstance(dependencies, VirtualBufferAdapter):
        return {dependencies._id}
    if isinstance(dependencies, Buffer):
        return extract_dependencies(dependencies._buffer_adapter)
    if isinstance(dependencies, Array):
        return extract_dependencies(dependencies.data)
    if isinstance(dependencies, Iterable):
        results = set()
        for dep in dependencies:
            results.update(extract_dependencies(dep))
        return results
    if hasattr(dependencies, '__virtual_allocations__'):
        # a hook for exposing nested virtual allocations in arbitrary classes
        return extract_dependencies(dependencies.__virtual_allocations__)

    return set()


class VirtualAllocator:
    """
    A helper callable object to use as an allocator
    for :py:class:`~grunnur.Array` creation.
    Encapsulates the dependencies (as identifiers, doesn't hold references for actual objects).
    """

    def __init__(self, manager: VirtualManager, dependencies: Set[int]):
        self.manager = manager
        self.dependencies = dependencies

    def __call__(self, size: int) -> Buffer:
        return self.manager._allocate_virtual(size, self.dependencies)


class VirtualBufferAdapter(BufferAdapter):
    """
    A virtual buffer object.
    """

    def __init__(self, manager: VirtualManager, size: int, id_: int):
        self._manager = manager
        self._id = id_
        self._size = size
        self._real_buffer_adapter: Optional[BufferAdapter] = None

    @property
    def kernel_arg(self):
        return self._real_buffer_adapter.kernel_arg

    def get_sub_region(self, origin, size):
        # FIXME: how to handle it?
        raise NotImplementedError("Virtual buffers do not support subregions")

    def get(self, *args, **kwds):
        return self._real_buffer_adapter.get(*args, **kwds)

    def set(self, *args, **kwds):
        return self._real_buffer_adapter.set(*args, **kwds)

    @property
    def offset(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return self._size

    def migrate(self, *args, **kwds):
        self._real_buffer_adapter.migrate(*args, **kwds)

    def _set_real_buffer_adapter(self, buf: Buffer):
        self._real_buffer_adapter = buf._buffer_adapter


class VirtualManager:
    """
    Base class for a manager of virtual allocations.

    :param queue: an instance of :py:class:`~grunnur.Queue`.
    :param pack_on_alloc: whether to repack allocations when a new allocation is requested.
    :param pack_on_free: whether to repack allocations when an allocation is freed.

    .. note::

        The allocators returned by this object must be used for arrays attached to the same queue.
    """

    def __init__(self, queue: Queue, pack_on_alloc: bool=False, pack_on_free: bool=False):
        self.queue = queue
        self._id_counter = 0
        self._virtual_buffers: Dict[int, ReferenceType[VirtualBufferAdapter]] = {}
        self._pack_on_alloc = pack_on_alloc
        self._pack_on_free = pack_on_free

    def allocator(self, dependencies=None) -> VirtualAllocator:
        """
        Create a callable to use for :py:class:`~grunnur.Array` creation.

        :param dependencies: can be a :py:class:`~grunnur.Array` instance
            (the ones containing persistent allocations will be ignored),
            an iterable with valid values,
            or an object with the attribute ``__virtual_allocations__`` which is a valid value
            (the last two will be processed recursively).
        """
        dependencies = extract_dependencies(dependencies)
        return VirtualAllocator(self, dependencies)

    def _allocate_virtual(self, size: int, dependencies: Set[int]) -> Buffer:

        new_id = self._id_counter
        self._id_counter += 1

        if not dependencies.issubset(self._virtual_buffers):
            missing_deps = dependencies.difference(self._virtual_buffers)
            missing_deps_str = ", ".join(str(dep) for dep in missing_deps)
            raise ValueError(
                f"Some of the declared dependencies (with IDs {missing_deps}) do not exist")

        self._allocate_specific(new_id, size, dependencies, self._pack_on_alloc)
        vbuf = VirtualBufferAdapter(self, size, new_id)
        buf = Buffer(self.queue.context, vbuf)
        self._virtual_buffers[new_id] = weakref.ref(vbuf, lambda _: self._free(new_id))

        if self._pack_on_alloc:
            self._update_all()
        else:
            self._update_buffer(new_id)

        return buf

    def _update_buffer(self, id_: int):
        vbuf = self._virtual_buffers[id_]()
        assert vbuf is not None
        buf = self._get_real_buffer(id_)
        vbuf._set_real_buffer_adapter(buf)

    def _update_all(self):
        for id_ in self._virtual_buffers:
            self._update_buffer(id_)

    def _free(self, id_: int):
        del self._virtual_buffers[id_]
        self._free_specific(id_, self._pack_on_free)
        if self._pack_on_free:
            self._update_all()

    def pack(self):
        """
        Packs the real allocations possibly reducing total memory usage.
        This process can be slow and may synchronize the base queue.
        """
        self._pack_specific()
        self._update_all()

    def statistics(self) -> VirtualAllocationStatistics:
        """
        Returns allocation statistics.
        """
        return VirtualAllocationStatistics(
            self._real_buffers(),
            # cast() to override inference here - we know that vb() will not return `None`
            [cast(VirtualBufferAdapter, vb()) for vb in self._virtual_buffers.values()])

    @abstractmethod
    def _allocate_specific(self, new_id: int, size:int, dependencies: Set[int], pack: bool):
        pass

    @abstractmethod
    def _get_real_buffer(self, id_: int) -> Buffer:
        pass

    @abstractmethod
    def _real_buffers(self) -> Iterable[Buffer]:
        pass

    @abstractmethod
    def _free_specific(self, id_: int, _pack: bool):
        pass

    @abstractmethod
    def _pack_specific(self):
        pass


class VirtualAllocationStatistics:
    """
    Virtual allocation details.
    """

    real_size_total: int
    """The total size of physical allocations (in bytes)."""

    real_num: int
    """The number of physical allocations."""

    real_sizes: Dict[int, int]
    """A dictionary ``size: count`` with the counts for physical allocations of each size."""

    virtual_size_total: int
    """The total size of virtual allocations (in bytes)."""

    virtual_num: int
    """The number of virtual allocations."""

    virtual_sizes: Dict[int, int]
    """A dictionary ``size: count`` with the counts for virtual allocations of each size."""

    def __init__(
            self, real_buffers: Iterable[Buffer], virtual_buffers: Iterable[VirtualBufferAdapter]):

        real_sizes = [rb.size for rb in real_buffers]
        virtual_sizes = [vb.size for vb in virtual_buffers]

        self.real_size_total: int = sum(real_sizes)
        self.real_num: int = len(real_sizes)
        self.real_sizes: Dict[int, int] = dict(Counter(real_sizes))

        self.virtual_size_total: int = sum(virtual_sizes)
        self.virtual_num: int = len(virtual_sizes)
        self.virtual_sizes: Dict[int, int] = dict(Counter(virtual_sizes))

    def __eq__(self, other):
        return (
            self.real_size_total == other.real_size_total
            and self.real_num == other.real_num
            and self.real_sizes == other.real_sizes
            and self.virtual_size_total == other.virtual_size_total
            and self.virtual_num == other.virtual_num
            and self.virtual_sizes == other.virtual_sizes
            )

    def __str__(self):
        real_buffers = ", ".join(
            f"{num}x{size}b" for size, num in sorted(self.real_sizes.items()))
        virtual_buffers = ", ".join(
            f"{num}x{size}b" for size, num in sorted(self.virtual_sizes.items()))
        return (
            f"VirtualAllocationStatistics("
            f"real: {self.real_num} allocs, "
            f"total size {self.real_size_total}b, "
            f"buffers: {real_buffers}; "
            f"virtual: {self.virtual_num} allocs, "
            f"total size {self.virtual_size_total}b, "
            f"buffers: {virtual_buffers})"
            )


class TrivialManager(VirtualManager):
    """
    Trivial manager --- allocates a separate buffer for each allocation request.
    """

    def __init__(self, *args, **kwds):
        VirtualManager.__init__(self, *args, **kwds)
        self._rbuffers = {}

    def _allocate_specific(self, new_id, size, _dependencies, _pack):
        buf = Buffer.allocate(self.queue.context, size)
        self._rbuffers[new_id] = buf

    def _get_real_buffer(self, id_):
        return self._rbuffers[id_]

    def _free_specific(self, id_, _pack):
        del self._rbuffers[id_]

    def _pack_specific(self):
        pass

    def _real_buffers(self):
        return self._rbuffers.values()


class ZeroOffsetManager(VirtualManager):
    """
    Tries to assign several allocation requests to a single real allocation,
    if dependencies allow that.
    All virtual allocations start from the beginning of real allocations.
    """

    VirtualAllocation = namedtuple('VirtualAllocation', ['size', 'dependencies'])
    RealAllocation = namedtuple('RealAllocation', ['buffer', 'virtual_ids'])
    RealSize = namedtuple('RealSize', ['size', 'real_id'])
    VirtualMapping = namedtuple('VirtualMapping', ['real_id', 'sub_region'])

    def __init__(self, *args, **kwds):
        VirtualManager.__init__(self, *args, **kwds)

        self._virtual_allocations = {} # id -> VirtualAllocation
        self._real_sizes = SortedList(key=lambda x: x.size) # RealSize objects, sorted by size
        self._virtual_to_real = {} # id -> VirtualMapping
        self._real_allocations = {} # real_id -> RealAllocation
        self._real_id_counter = 0

    def _allocate_specific(self, new_id, size, dependencies, pack):

        # Dependencies should be bidirectional.
        # So if some new allocation says it depends on earlier ones,
        # we need to update their dependency lists.
        for dep in dependencies:
            self._virtual_allocations[dep].dependencies.add(new_id)

        # Save virtual allocation parameters
        self._virtual_allocations[new_id] = self.VirtualAllocation(size, dependencies)

        if pack:
            # If pack is requested, we can just do full re-pack right away.
            self._pack_specific()
        else:
            # If not, find a real allocation using the greedy algorithm.
            self._fast_add(new_id, size, dependencies)

    def _fast_add(self, new_id, size, dependencies):
        """
        Greedy algorithm to find a real allocation for a given virtual allocation.
        """

        # Find the smallest real allocation which can hold the requested virtual allocation.
        try:
            idx_start = self._real_sizes.argfind_ge(size)
        except ValueError:
            idx_start = len(self._real_sizes)

        # Check all real allocations with suitable sizes, starting from the smallest one.
        # Use the first real allocation which does not contain ``new_id``'s dependencies.
        for idx in range(idx_start, len(self._real_sizes)):
            real_id = self._real_sizes[idx].real_id
            buf = self._real_allocations[real_id].buffer
            virtual_ids = self._real_allocations[real_id].virtual_ids
            if virtual_ids.isdisjoint(dependencies):
                virtual_ids.add(new_id)
                break
        else:
            # If no suitable real allocation is found, create a new one.
            buf = Buffer.allocate(self.queue.context, size)
            real_id = self._real_id_counter
            self._real_id_counter += 1

            self._real_allocations[real_id] = self.RealAllocation(buf, set([new_id]))
            self._real_sizes.insert(self.RealSize(size, real_id))

        # TODO: Here it would be more appropriate to use buffer.get_sub_region(0, size),
        # but OpenCL does not allow several overlapping subregions to be used in a single kernel
        # for both read and write, which ruins the whole idea.
        # So we are passing full buffers and hope that the Array class takes care of sizes.
        self._virtual_to_real[new_id] = self.VirtualMapping(
            real_id, self._real_allocations[real_id].buffer)

    def _get_real_buffer(self, id_: int) -> Buffer:
        return self._virtual_to_real[id_].sub_region

    def _free_specific(self, id_: int, pack: bool=False):
        # Remove the allocation from the dependency lists of its dependencies
        dependencies = self._virtual_allocations[id_].dependencies
        for dep in dependencies:
            self._virtual_allocations[dep].dependencies.remove(id_)

        vtr = self._virtual_to_real[id_]

        # Clear virtual allocation data
        del self._virtual_allocations[id_]
        del self._virtual_to_real[id_]

        if pack:
            self._pack_specific()
        else:
            # Fast and non-optimal free.
            # Remove the virtual allocation from the real allocation,
            # and delete the real allocation if its no longer used by other virtual allocations.
            ra = self._real_allocations[vtr.real_id]
            ra.virtual_ids.remove(id_)
            if len(ra.virtual_ids) == 0:
                del self._real_allocations[vtr.real_id]
                self._real_sizes.remove(self.RealSize(ra.buffer.size, vtr.real_id))

    def _pack_specific(self):
        """
        Full memory re-pack.
        In theory, should find the optimal (with the minimal real allocation size) distribution
        of virtual allocations.
        """

        # Need to synchronize, because we are going to change allocation addresses,
        # and we do not want to free the memory some kernel is reading from.
        self.queue.synchronize()

        # Clear all real allocation data.
        self._real_sizes.clear()
        self._real_allocations = {}
        self._real_id_counter = 0

        va = self._virtual_allocations

        # Sort all virtual allocations by size
        virtual_sizes = sorted(
            [(va[id_].size, id_) for id_ in va],
            key=lambda x: x[0])

        # Application of greedy algorithm for virtual allocations starting from the largest one
        # should give the optimal distribution.
        for size, id_ in reversed(virtual_sizes):
            self._fast_add(id_, size, self._virtual_allocations[id_].dependencies)

    def _real_buffers(self):
        return [ra.buffer for ra in self._real_allocations.values()]

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple, Sequence, Union, Iterable, Dict, TypeVar

import numpy

from .array_metadata import ArrayMetadata
from .context import Context
from .device import Device
from .queue import Queue, MultiQueue
from .buffer import Buffer
from .utils import min_blocks


class Array:
    """
    Array on a single device.
    """

    context: 'Context'
    """Context this array is allocated on."""

    shape: Tuple[int, ...]
    """Array shape."""

    dtype: numpy.dtype
    """Array item data type."""

    strides: Tuple[int, ...]
    """Array strides."""

    @classmethod
    def from_host(cls, queue: 'Queue', host_arr: numpy.ndarray) -> 'Array':
        """
        Creates an array object from a host array.

        :param queue: the queue to use for the transfer.
        :param host_arr: the source array.
        """
        array = cls.empty(queue.context, host_arr.shape, host_arr.dtype, device_idx=queue.device_idx)
        array.set(queue, host_arr)
        return array

    @classmethod
    def empty(
            cls, context: 'Context', shape: Sequence[int],
            dtype: numpy.dtype, allocator: Callable[[int, int], 'Buffer']=None,
            device_idx: Optional[int]=None) -> 'Array':
        """
        Creates an empty array.

        :param shape: array shape.
        :param dtype: array data type.
        :param allocator: an optional callable taking two integer arguments
            (buffer size in bytes, and the device to allocate it on)
            and returning a :py:class:`Buffer` object.
        :param device_idx: the index of the device on which to allocate the array.
        """
        if device_idx is None:
            if len(context.devices) == 1:
                device_idx = 0
            else:
                raise ValueError("device_idx must be specified in a multi-device context")

        metadata = ArrayMetadata(shape, dtype)
        size = metadata.buffer_size

        if allocator is None:
            data = Buffer.allocate(context, size, device_idx=device_idx)
        else:
            data = allocator(size, device_idx)

        return cls(context, metadata, data)

    def __init__(self, context: Context, array_metadata: ArrayMetadata, data: Buffer):

        self._metadata = array_metadata

        self.context = context
        self.shape = self._metadata.shape
        self.dtype = self._metadata.dtype
        self.strides = self._metadata.strides
        self.first_element_offset = self._metadata.first_element_offset
        self.buffer_size = self._metadata.buffer_size

        if data.size < self.buffer_size:
            raise ValueError(
                "Provided data buffer is not big enough to hold the array "
                "(minimum required {self.buffer_size})")

        self.data = data

    def _view(self, slices):
        new_metadata = self._metadata[slices]

        origin, size, new_metadata = new_metadata.minimal_subregion()
        data = self.data.get_sub_region(origin, size)
        return Array(self.context, new_metadata, data)

    def set(self, queue: 'Queue', array: Union[numpy.ndarray, 'Array'], no_async: bool=False):
        """
        Copies the contents of the host array to the array.

        :param queue: the queue to use for the transfer.
        :param array: the source array.
        :param no_async: if `True`, the transfer blocks until completion.
        """
        if isinstance(array, numpy.ndarray):
            array_data = array
        elif isinstance(array, Array):
            if not array._metadata.contiguous:
                raise ValueError("Setting from a non-contiguous device array is not supported")
            array_data = array.data
        else:
            raise TypeError(f"Cannot set from an object of type {type(array)}")

        if self.shape != array.shape:
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {array.shape}")
        if self.dtype != array.dtype:
            raise ValueError(f"Dtype mismatch: expected {self.dtype}, got {array.dtype}")

        self.data.set(queue, array_data, no_async=no_async)

    def get(self, queue: 'Queue', dest: Optional[numpy.ndarray]=None, async_: bool=False) -> numpy.ndarray:
        """
        Copies the contents of the array to the host array and returns it.

        :param queue: the queue to use for the transfer.
        :param dest: the destination array. If ``None``, the target array is created.
        :param async_: if `True`, the transfer is performed asynchronously.
        """
        if dest is None:
            dest = numpy.empty(self.shape, self.dtype)
        self.data.get(queue, dest, async_=async_)
        return dest

    def __getitem__(self, slices) -> 'Array':
        """
        Returns a view of this array.
        """
        return self._view(slices)


class BaseSplay(ABC):
    """
    Base class for splay strategies for :py:class:`~grunnur.MultiArray`.
    """

    ArrayLike = TypeVar("ArrayLike")
    """
    The type of an array-like object (the one having a ``shape``
    and supporting views via ``__getitem__()``)
    """

    @abstractmethod
    def __call__(self, arr: 'ArrayLike', devices: Dict[int, 'grunnur.Device']) -> Dict[int, 'ArrayLike']:
        """
        Creates a dictionary of views of an array-like object for each of the given devices.

        :param arr: an array-like object.
        :param devices: a dictionary of device indices matched to device objects.
        """
        pass


class EqualSplay(BaseSplay):
    """
    Splays the given array equally between the devices using the outermost dimension.
    The outermost dimension should be larger or equal to the number of devices.
    """

    def __call__(self, arr, devices):
        parts = len(devices)
        outer_dim = arr.shape[0]
        assert parts <= outer_dim
        chunk_size = min_blocks(outer_dim, parts)
        rem = outer_dim % parts

        subarrays = {}
        ptr = 0
        for part, device_idx in enumerate(sorted(devices)):
            l = chunk_size if rem == 0 or part < rem else chunk_size - 1
            subarrays[device_idx] = arr[ptr:ptr+l]
            ptr += l

        return subarrays


class CloneSplay(BaseSplay):
    """
    Copies the given array to each device.
    """

    def __call__(self, arr, devices):
        return {device_idx: arr for device_idx in devices}


class MultiArray:
    """
    An array on multiple devices.
    """

    context: 'Context'
    """Context this array is allocated on."""

    shape: Tuple[int, ...]
    """Array shape."""

    dtype: numpy.dtype
    """Array item data type."""

    shapes: Dict[int, Tuple[int, ...]]
    """Sub-array shapes matched to device indices."""

    EqualSplay = EqualSplay
    CloneSplay = CloneSplay

    @classmethod
    def from_host(
            cls, mqueue: 'MultiQueue', host_arr: numpy.ndarray,
            splay: Optional['grunnur.array.BaseSplay']=None
            ) -> 'MultiArray':
        """
        Creates an array object from a host array.

        :param mqueue: the queue to use for the transfer.
        :param host_arr: the source array.
        :param splay: the splay strategy (if ``None``, an :py:class:`EqualSplay` object is used).
        """

        if splay is None:
            splay = EqualSplay()

        host_subarrays = splay(host_arr, mqueue.devices)

        subarrays = {
            device_idx: Array.from_host(mqueue.queues[device_idx], host_subarrays[device_idx])
            for device_idx in mqueue.devices
        }

        return cls(mqueue.context, host_arr.shape, host_arr.dtype, subarrays, splay)

    @classmethod
    def empty(
            cls, context: 'Context', shape: Sequence[int],
            dtype: numpy.dtype, allocator: Callable[[int, int], 'Buffer']=None,
            device_idxs: Optional[Iterable[int]]=None,
            splay: Optional['grunnur.array.BaseSplay']=None,
            ) -> 'MultiArray':
        """
        Creates an empty array.

        :param shape: array shape.
        :param dtype: array data type.
        :param allocator: an optional callable taking two integer arguments
            (buffer size in bytes, and the device to allocate it on)
            and returning a :py:class:`Buffer` object.
        :param device_idx: the index of the device on which to allocate the array.
        :param splay: the splay strategy (if ``None``, an :py:class:`EqualSplay` object is used).
        """

        if splay is None:
            splay = EqualSplay()

        if device_idxs is None:
            device_idxs = list(range(len(context.devices)))

        devices = {device_idx: context.devices[device_idx] for device_idx in device_idxs}

        metadata = ArrayMetadata(shape, dtype)
        submetadatas = splay(metadata, devices)

        subarrays = {
            device_idx: Array.empty(
                context, submetadata.shape, submetadata.dtype, allocator=allocator, device_idx=device_idx)
            for device_idx, submetadata in submetadatas.items()
            }

        return cls(context, shape, dtype, subarrays, splay)

    def __init__(self, context, shape, dtype, subarrays, splay):
        self.context = context
        self.shape = shape
        self.dtype = dtype
        self.subarrays = subarrays
        self.shapes = {device_idx: subarray.shape for device_idx, subarray in self.subarrays.items()}

        self._devices = {device_idx: self.context.devices[device_idx] for device_idx in self.subarrays}
        self._splay = splay

    def get(self, mqueue: 'MultiQueue', dest: Optional[numpy.ndarray]=None, async_: bool=False) -> numpy.ndarray:
        """
        Copies the contents of the array to the host array and returns it.

        :param mqueue: the queue to use for the transfer.
        :param dest: the destination array. If ``None``, the target array is created.
        :param async_: if `True`, the transfer is performed asynchronously.
        """

        if dest is None:
            dest = numpy.empty(self.shape, self.dtype)

        dest_subarrays = self._splay(dest, self._devices)

        for device_idx, subarray in self.subarrays.items():
            subarray.get(mqueue.queues[device_idx], dest_subarrays[device_idx], async_=async_)

        return dest

    def set(self, mqueue: 'MultiQueue', array: Union[numpy.ndarray, 'MultiArray'], no_async: bool=False):
        """
        Copies the contents of the host array to the array.

        :param mqueue: the queue to use for the transfer.
        :param array: the source array.
        :param no_async: if `True`, the transfer blocks until completion.
        """

        if isinstance(array, numpy.ndarray):
            subarrays = self._splay(array, self._devices)
        else:
            subarrays = array.subarrays

        if self.subarrays.keys() != subarrays.keys():
            raise ValueError("Mismatched device sets in the source and the destination")

        for device_idx in self.subarrays:
            self.subarrays[device_idx].set(
                mqueue.queues[device_idx], subarrays[device_idx], no_async=no_async)

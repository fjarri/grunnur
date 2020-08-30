from typing import Optional, Callable, Tuple, Sequence, Union

import numpy

from .array_metadata import ArrayMetadata
from .queue import Queue
from .buffer import Buffer


class Array:
    """
    Array on the device.
    """

    shape: Tuple[int, ...]
    """Array shape."""

    dtype: numpy.dtype
    """Array item data type."""

    strides: Tuple[int, ...]
    """Array strides."""

    @classmethod
    def from_host(cls, queue: Queue, host_arr: numpy.ndarray) -> 'Array':
        """
        Creates an array object from a host array.

        :param queue: the queue to use for the transfer.
        :param host_arr: the source array.
        """
        metadata = ArrayMetadata.from_arraylike(host_arr)
        array = cls(queue, metadata)
        array.set(host_arr)
        return array

    @classmethod
    def empty(
            cls, queue: Queue, shape: Sequence[int],
            dtype: numpy.dtype, allocator: Callable[[int], Buffer]=None) -> 'Array':
        """
        Creates an empty array.

        :param queue: the queue to use for the transfer.
        :param shape: array shape.
        :param dtype: array data type.
        :param allocator: an optional callable returning a :py:class:`Buffer` object.
        """
        metadata = ArrayMetadata(shape, dtype)
        return cls(queue, metadata, allocator=allocator)

    def __init__(
            self, queue: Queue, array_metadata: ArrayMetadata,
            data: Optional[Buffer]=None, allocator: Callable[[int], Buffer]=None):

        if allocator is None:
            allocator = lambda size: Buffer.allocate(queue.context, size)

        self._queue = queue
        self._metadata = array_metadata

        self.shape = self._metadata.shape
        self.dtype = self._metadata.dtype
        self.strides = self._metadata.strides
        self.first_element_offset = self._metadata.first_element_offset
        self.buffer_size = self._metadata.buffer_size

        if data is None:
            data = allocator(self.buffer_size)
            data.bind(queue.device_idxs[0])
        else:
            if data.size < self.buffer_size:
                raise ValueError(
                    "Provided data buffer is not big enough to hold the array "
                    "(minimum required {self.buffer_size})")

        self.data = data

    def single_device_view(self, device_idx: int) -> 'SingleDeviceFactory':
        """
        Returns a subscriptable object that produces sub-arrays based on the device ``device_idx``.
        """
        if device_idx not in self._queue.devices:
            device_nums = ', '.join(str(i) for i in self._queue.devices)
            raise ValueError(
                f"The device number must be one of those present in the queue ({device_nums})")
        return SingleDeviceFactory(self, device_idx)

    def _view(self, slices, device_idx=None):
        new_metadata = self._metadata[slices]

        origin, size, new_metadata = new_metadata.minimal_subregion()
        data = self.data.get_sub_region(origin, size)
        if device_idx is not None:
            data.bind(device_idx)
            data.migrate(self._queue)

        return Array(self._queue, new_metadata, data=data)

    def set(self, array: Union[numpy.ndarray, 'Array'], no_async: bool=False):
        """
        Copy the contents of the host array to the array.

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

        self.data.set(self._queue, array_data, no_async=no_async)

    def get(self, dest: Optional[numpy.ndarray]=None, async_: bool=False) -> numpy.ndarray:
        """
        Copy the contents of the array to the host array and return it.

        :param dest: the destination array. If ``None``, the target array is created.
        :param async_: if `True`, the transfer is performed asynchronously.
        """
        if dest is None:
            dest = numpy.empty(self.shape, self.dtype)
        self.data.get(self._queue, dest, async_=async_)
        return dest

    def __getitem__(self, slices) -> 'Array':
        """
        Return a view of this array.
        """
        return self._view(slices)


class SingleDeviceFactory:
    """
    A subscriptable object that produces sub-arrays based on a single device.
    """

    def __init__(self, array, device_idx):
        self._array = array
        self._device_idx = device_idx

    def __getitem__(self, slices) -> Array:
        """
        Return a view of the parent array bound to the device this factory was created for
        (see :py:meth:`~grunnur.Array.single_device_view`).
        """
        return self._array._view(slices, device_idx=self._device_idx)

from typing import Optional, Callable

import numpy

from .array_metadata import ArrayMetadata
from .queue import Queue
from .buffer import Buffer


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

    @classmethod
    def from_host(cls, queue, host_arr):
        metadata = ArrayMetadata.from_arraylike(host_arr)
        array = cls(queue, metadata)
        array.set(host_arr)
        return array

    @classmethod
    def empty(cls, queue, shape, dtype, allocator=None):
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

    def single_device_view(self, device_idx: int):
        """
        Returns a subscriptable object that produces sub-arrays based on the device ``device_idx``.
        """
        if device_idx not in self._queue.devices:
            device_nums = ', '.join(str(i) for i in self._queue.devices)
            raise ValueError(
                f"The device number must be one of those present in the queue ({device_nums})")
        return SingleDeviceFactory(self, device_idx)

    def _view(self, slices, device_idx):
        new_metadata = self._metadata[slices]

        origin, size, new_metadata = new_metadata.minimal_subregion()
        data = self.data.get_sub_region(origin, size)
        data.bind(device_idx)
        data.migrate(self._queue)

        return Array(self._queue, new_metadata, data=data)

    def set(self, array: numpy.ndarray, no_async: bool=False):
        """
        Sets the data in this array from a CPU array.
        If ``async_`` is ``True``, this call blocks.
        """
        self.data.set(self._queue, array, no_async=no_async)

    def get(self, dest: Optional[numpy.ndarray]=None, async_: bool=False) -> numpy.ndarray:
        """
        Gets the data from this array to a CPU array.
        If ``dest`` is ``None``, the target array is created.
        If ``async_`` is ``True``, this call blocks.
        Returns the created CPU array, or ``dest`` if it was provided.
        """
        # TODO: check if the array is contiguous
        if dest is None:
            dest = numpy.empty(self.shape, self.dtype)
        self.data.get(self._queue, dest, async_=async_)
        return dest


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
        (see :py:meth:`Array.single_device_view`).
        """
        return self._array._view(slices, self._device_idx)

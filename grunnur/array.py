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
        buf = Buffer.allocate(queue.context, metadata.buffer_size)
        array = cls(queue, metadata, buf)
        array.set(host_arr)
        return array

    @classmethod
    def empty(cls, queue, shape, dtype, allocator=None):
        metadata = ArrayMetadata(shape, dtype)
        return cls(queue, metadata, allocator=allocator)

    def __init__(
            self, queue: Queue, array_metadata: ArrayMetadata,
            data: Optional[Buffer]=None, allocator: Callable[[int], Buffer]=None,
            device_num=None):

        if allocator is None:
            allocator = lambda size: Buffer.allocate(queue.context, size)

        self._queue = queue
        self._metadata = array_metadata

        self.device_num = device_num
        self._queue_device_num = device_num if device_num is not None else 0

        self.shape = self._metadata.shape
        self.dtype = self._metadata.dtype
        self.strides = self._metadata.strides
        self.first_element_offset = self._metadata.first_element_offset
        self.buffer_size = self._metadata.buffer_size

        if data is None:
            data = allocator(self.buffer_size)

        self.data = data

        if device_num is not None:
            self.data.migrate(queue, device_num)

    def single_device_view(self, device_num: int):
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

        return Array(self._queue, new_metadata, device_num=device_num, data=data)

    def set(self, array: numpy.ndarray, async_: bool=False):
        """
        Sets the data in this array from a CPU array.
        If ``async_`` is ``True``, this call blocks.
        """
        self.data.set(self._queue, self._queue_device_num, array)

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
        self.data.get(self._queue, self._queue_device_num, dest, async_=async_)
        return dest


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

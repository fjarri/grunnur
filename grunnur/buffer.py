from typing import Union, Optional

import numpy

from .adapter_base import BufferAdapter
from .context import Context
from .queue import Queue


class Buffer:
    """
    A memory buffer on device.
    """

    context: Context
    """Context this buffer is allocated on."""

    device_idx: int
    """The index of the device this buffer is allocated on."""

    @classmethod
    def allocate(cls, context: Context, size: int, device_idx: Optional[int]=None) -> 'Buffer':
        """
        Allocate a buffer of ``size`` bytes.

        :param context: the context to use.
        :param size: the buffer's size in bytes.
        :param device_idx: the device to allocate on (can be omitted in a single-device context).
        """
        if device_idx is None:
            if len(context.devices) == 1:
                device_idx = 0
            else:
                raise ValueError("device_idx must be specified in a multi-device context")

        buffer_adapter = context._context_adapter.allocate(size, device_idx)
        return cls(context, device_idx, buffer_adapter)

    def __init__(self, context: Context, device_idx: int, buffer_adapter: BufferAdapter):
        self.context = context
        self._buffer_adapter = buffer_adapter
        self.device_idx = device_idx

    @property
    def kernel_arg(self):
        # Has to be a property since buffer_adapter can be externally updated
        # (e.g. if it's a virtual buffer)
        return self._buffer_adapter.kernel_arg

    @property
    def offset(self) -> int:
        """
        Offset of this buffer (in bytes) from the beginning
        of the physical allocation it resides in.
        """
        return self._buffer_adapter.offset

    @property
    def size(self) -> int:
        """
        This buffer's size (in bytes).
        """
        return self._buffer_adapter.size

    def set(self, queue: Queue, buf: Union[numpy.ndarray, 'Buffer'], no_async: bool=False):
        """
        Copy the contents of the host array or another buffer to this buffer.

        :param queue: the queue to use for the transfer.
        :param buf: the source - ``numpy`` array or a :py:class:`Buffer` object.
        :param no_async: if `True`, the transfer blocks until completion.
        """
        if queue.device_idx != self.device_idx:
            raise ValueError(
                f"Mismatched devices: queue on device {queue.device_idx}, "
                f"buffer on device {self.device_idx}")

        if isinstance(buf, numpy.ndarray):
            buf_adapter = numpy.ascontiguousarray(buf)
        elif isinstance(buf, Buffer):
            buf_adapter = buf._buffer_adapter
        else:
            raise TypeError(f"Cannot set from an object of type {type(buf)}")

        self._buffer_adapter.set(queue._queue_adapter, buf_adapter, no_async=no_async)

    def get(self, queue: Queue, host_array: numpy.ndarray, async_: bool=False):
        """
        Copy the contents of the buffer to the host array.

        :param queue: the queue to use for the transfer.
        :param host_array: the destination array.
        :param async_: if `True`, the transfer is performed asynchronously.
        """
        if queue.device_idx != self.device_idx:
            raise ValueError(
                f"Mismatched devices: queue on device {queue.device_idx}, "
                f"buffer on device {self.device_idx}")

        self._buffer_adapter.get(queue._queue_adapter, host_array, async_=async_)

    def get_sub_region(self, origin: int, size: int) -> 'Buffer':
        """
        Return a buffer object describing a subregion of this buffer.

        :param origin: the offset of the subregion.
        :param size: the size of the subregion.
        """
        return Buffer(self.context, self.device_idx, self._buffer_adapter.get_sub_region(origin, size))

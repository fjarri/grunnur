from __future__ import annotations

from typing import Any, Union, Optional

import numpy

from .adapter_base import BufferAdapter
from .context import BoundDevice
from .queue import Queue


class Buffer:
    """
    A memory buffer on device.
    """

    device: BoundDevice
    """Device on which this buffer is allocated."""

    @classmethod
    def allocate(cls, device: BoundDevice, size: int) -> "Buffer":
        """
        Allocate a buffer of ``size`` bytes.

        :param device: the device on which this buffer will be allocated.
        :param size: the buffer's size in bytes.
        """
        buffer_adapter = device.context._context_adapter.allocate(device._device_adapter, size)
        return cls(device, buffer_adapter)

    def __init__(self, device: BoundDevice, buffer_adapter: BufferAdapter):
        self.device = device
        self._buffer_adapter = buffer_adapter

    @property
    def kernel_arg(self) -> Any:
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

    def set(
        self,
        queue: Queue,
        buf: Union["numpy.ndarray[Any, numpy.dtype[Any]]", "Buffer"],
        no_async: bool = False,
    ) -> None:
        """
        Copy the contents of the host array or another buffer to this buffer.

        :param queue: the queue to use for the transfer.
        :param buf: the source - ``numpy`` array or a :py:class:`Buffer` object.
        :param no_async: if `True`, the transfer blocks until completion.
        """
        if queue.device != self.device:
            raise ValueError(
                f"Mismatched devices: queue on device {queue.device}, "
                f"buffer on device {self.device}"
            )

        buf_adapter: Union["numpy.ndarray[Any, numpy.dtype[Any]]", BufferAdapter]
        if isinstance(buf, numpy.ndarray):
            buf_adapter = numpy.ascontiguousarray(buf)
        elif isinstance(buf, Buffer):
            buf_adapter = buf._buffer_adapter
        else:
            raise TypeError(f"Cannot set from an object of type {type(buf)}")

        self._buffer_adapter.set(queue._queue_adapter, buf_adapter, no_async=no_async)

    def get(
        self, queue: Queue, host_array: "numpy.ndarray[Any, numpy.dtype[Any]]", async_: bool = False
    ) -> None:
        """
        Copy the contents of the buffer to the host array.

        :param queue: the queue to use for the transfer.
        :param host_array: the destination array.
        :param async_: if `True`, the transfer is performed asynchronously.
        """
        if queue.device != self.device:
            raise ValueError(
                f"Mismatched devices: queue on device {queue.device}, "
                f"buffer on device {self.device}"
            )

        self._buffer_adapter.get(queue._queue_adapter, host_array, async_=async_)

    def get_sub_region(self, origin: int, size: int) -> "Buffer":
        """
        Return a buffer object describing a subregion of this buffer.

        :param origin: the offset of the subregion.
        :param size: the size of the subregion.
        """
        return Buffer(self.device, self._buffer_adapter.get_sub_region(origin, size))

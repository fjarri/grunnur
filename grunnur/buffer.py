from typing import Union

import numpy

from .adapter_base import BufferAdapter
from .context import Context
from .queue import Queue


class Buffer:
    """
    A memory buffer on device.
    """

    @classmethod
    def allocate(cls, context: Context, size: int) -> 'Buffer':
        """
        Allocate a buffer of ``size`` bytes.

        :param context: the context to use.
        :param size: the buffer's size in bytes.
        """
        buffer_adapter = context._context_adapter.allocate(size)
        return cls(context, buffer_adapter)

    def __init__(self, context: Context, buffer_adapter: BufferAdapter):
        self.context = context
        self._buffer_adapter = buffer_adapter
        self._device_idx = None if len(context.devices) > 1 else 0

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
        if self._device_idx is None:
            raise RuntimeError("This buffer has not been bound to any device yet")

        if isinstance(buf, numpy.ndarray):
            buf_adapter = buf
        elif isinstance(buf, Buffer):
            buf_adapter = buf._buffer_adapter
        else:
            raise TypeError(f"Cannot set from an object of type {type(buf)}")

        self._buffer_adapter.set(
            queue._queue_adapter, self._device_idx, buf_adapter, no_async=no_async)
        self.migrate(queue)

    def get(self, queue: Queue, host_array: numpy.ndarray, async_: bool=False):
        """
        Copy the contents of the buffer to the host array.

        :param queue: the queue to use for the transfer.
        :param host_array: the destination array.
        :param async_: if `True`, the transfer is performed asynchronously.
        """
        if self._device_idx is None:
            raise RuntimeError("This buffer has not been bound to any device yet")
        self._buffer_adapter.get(queue._queue_adapter, self._device_idx, host_array, async_=async_)

    def get_sub_region(self, origin: int, size: int) -> 'Buffer':
        """
        Return a buffer object describing a subregion of this buffer.

        :param origin: the offset of the subregion.
        :param size: the size of the subregion.
        """
        return Buffer(self.context, self._buffer_adapter.get_sub_region(origin, size))

    def bind(self, device_idx: int):
        if device_idx >= len(self.context.devices):
            max_idx = len(self.context.devices) - 1
            raise ValueError(
                f"Device index {device_idx} out of available range for this context (0-{max_idx})")

        if self._device_idx is None:
            self._device_idx = device_idx
        elif self._device_idx != device_idx:
            raise ValueError(f"The buffer is already bound to device {self._device_idx}")

    def migrate(self, queue: Queue):
        # Normally, a buffer will migrate automatically to the device,
        # but on some platforms the lack of explicit migration might lead
        # to performance degradation.
        # (e.g. `examples/multi_device_comparison.py` on a multi-Tesla AWS instance).
        if self._device_idx is None:
            raise RuntimeError("This buffer has not been bound to any device yet")

        # Automatic migration works well enough for one-device contexts
        if len(self.context.devices) > 1:
            self._buffer_adapter.migrate(queue._queue_adapter, self._device_idx)

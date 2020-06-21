class Buffer:

    @classmethod
    def allocate(cls, context, size):
        buffer_adapter = context._context_adapter.allocate(size)
        return cls(context, buffer_adapter)

    def __init__(self, context, buffer_adapter):
        self.context = context
        self._buffer_adapter = buffer_adapter
        self._device_idx = None if len(context.devices) > 1 else 0

    @property
    def kernel_arg(self):
        # Has to be a property since buffer_adapter can be externally updated
        # (e.g. if it's a virtual buffer)
        return self._buffer_adapter.kernel_arg

    @property
    def offset(self):
        return self._buffer_adapter.offset

    @property
    def size(self):
        return self._buffer_adapter.size

    def set(self, queue, host_array, async_=False):
        if self._device_idx is None:
            raise RuntimeError("This buffer has not been bound to any device yet")
        self._buffer_adapter.set(queue._queue_adapter, self._device_idx, host_array, async_=async_)
        self.migrate(queue)

    def get(self, queue, host_array, async_=False):
        if self._device_idx is None:
            raise RuntimeError("This buffer has not been bound to any device yet")
        self._buffer_adapter.get(queue._queue_adapter, self._device_idx, host_array, async_=async_)

    def get_sub_region(self, origin, size):
        return Buffer(self.context, self._buffer_adapter.get_sub_region(origin, size))

    def bind(self, device_idx):
        if device_idx >= len(self.context.devices):
            max_idx = len(self.context.devices) - 1
            raise ValueError(f"Device index {device_idx} out of available range for this context (0-{max_idx})")

        if self._device_idx is None:
            self._device_idx = device_idx
        elif self._device_idx != device_idx:
            raise ValueError(f"The buffer is already bound to device {self._device_idx}")

    def migrate(self, queue):
        # Normally, a buffer will migrate automatically to the device,
        # but on some platforms the lack of explicit migration might lead to performance degradation.
        # (e.g. `examples/multi_device_comparison.py` on a multi-Tesla AWS instance).
        if self._device_idx is None:
            raise RuntimeError("This buffer has not been bound to any device yet")

        # Automatic migration works well enough for one-device contexts
        if len(self.context.devices) > 1:
            self._buffer_adapter.migrate(queue._queue_adapter, self._device_idx)

from .utils import wrap_in_tuple, normalize_base_objects
from .api import API
from .device import Device


class Context:
    """
    GPGPU context.
    """

    @classmethod
    def from_devices(cls, devices):
        devices = wrap_in_tuple(devices)
        platform = devices[0].platform

        device_adapters = [device._device_adapter for device in devices]
        context_adapter = platform._platform_adapter.make_context(device_adapters)

        return cls(context_adapter)

    @classmethod
    def from_backend_devices(cls, backend_devices):
        #devices = normalize_base_objects(devices)
        for api in API.all():
            if api._api_adapter.isa_backend_device(backend_devices[0]):
                context_adapter = api._api_adapter.make_context_from_devices(backend_devices)
                return cls(context_adapter)
        raise TypeError(
            f"{type(backend_devices[0])} objects were not recognized as devices by any API")

    @classmethod
    def from_backend_contexts(cls, backend_contexts):
        #contexts = normalize_base_objects(contexts)
        for api in API.all():
            if api._api_adapter.isa_backend_context(backend_contexts[0]):
                context_adapter = api._api_adapter.make_context_from_contexts(backend_contexts)
                return cls(context_adapter)
        raise TypeError(
            f"{type(backend_contexts[0])} objects were not recognized as contexts by any API")

    def __init__(self, context_adapter):
        self._context_adapter = context_adapter
        self.devices = [Device.from_backend_device(device) for device in context_adapter.devices]
        self.platform = self.devices[0].platform
        self.api = self.platform.api

'''
# FIXME: leftover from the refactoring
    def create_some_context(
            self, interactive: bool=False, quantity: Optional[int]=1, **device_filters) -> Context:
        """
        Finds devices matching the given criteria and creates a
        :py:class:`Context` object out of them.

        :param interactive: passed to :py:meth:`select_devices`.
        :param quantity: passed to :py:meth:`select_devices`.
        :param device_filters: passed to :py:meth:`select_devices`.
        """
        devices = self.select_devices(interactive=interactive, quantity=quantity, **device_filters)
        return self.create_context(devices)

    def create_context(self, context_base) -> Context:
        """
        Creates a :py:class:`Context` object based on ``context_base``, which can be
        a :py:class:`Device`, a list of :py:class:`Device` objects
        (from this API and belonging to the same platform), or some additional API-specific types.
        See classmethods of :py:class:`~grunnur.cuda.CuContext` and
        :py:class:`~grunnur.opencl.OclContext` for details on the latter.

        :param context_base: an object to base the context on.
        """
        return self._context_class.from_any_base(context_base)
'''

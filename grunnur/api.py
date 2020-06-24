from __future__ import annotations

from typing import Optional, Iterable, List

from .adapter_base import APIID
from .adapter_cuda import CuAPIAdapterFactory
from .adapter_opencl import OclAPIAdapterFactory


CUDA_API_ID = CuAPIAdapterFactory().api_id


OPENCL_API_ID = OclAPIAdapterFactory().api_id


_ALL_API_ADAPTER_FACTORIES = {
    factory.api_id: factory for factory in [
        CuAPIAdapterFactory(),
        OclAPIAdapterFactory(),
        ]
    }


def all_api_ids() -> List[APIID]:
    return list(_ALL_API_ADAPTER_FACTORIES.keys())


class API:
    """
    A generalized GPGPU API.

    .. py:attribute:: id

        This API's ID, an :py:class:`APIID` object.

    .. py:attribute:: short_name

        This API's short name.

    .. py:attribute:: shortcut

        A shortcut for this API (to use in :py:func:`~grunnur.find_apis`).
    """

    @classmethod
    def all(cls) -> List[API]:
        """
        Returns a list of :py:class:`~grunnur.base_classes.API` objects
        for which backends are available.
        """
        # TODO: rename to `all_available()`?
        return [
            cls.from_api_id(api_id)
            for api_id, api_factory in _ALL_API_ADAPTER_FACTORIES.items()
            if api_factory.available]

    @classmethod
    def all_by_shortcut(cls, shortcut: Optional[str]=None) -> List[API]:
        """
        If ``shortcut`` is a string, returns a list of one :py:class:`~grunnur.base_classes.API` object
        whose :py:attr:`~grunnur.base_classes.API.id` attribute has its
        :py:attr:`~grunnur.base_classes.APIID.shortcut` attribute equal to it
        (or raises an error if it was not found, or its backend is not available).

        If ``shortcut`` is ``None``, returns a list of all available
        py:class:`~grunnur.base_classes.API` objects.

        :param shortcut: an API shortcut to match.
        """
        if shortcut is None:
            apis = cls.all()
        else:
            for api_id, api_factory in _ALL_API_ADAPTER_FACTORIES.items():
                if shortcut == api_id.shortcut:
                    if not api_factory.available:
                        raise ValueError(str(shortcut) + " API is not available")
                    apis = [cls.from_api_id(api_id)]
                    break
            else:
                raise ValueError("Invalid API shortcut: " + str(shortcut))

        return apis

    @classmethod
    def from_api_id(cls, api_id):
        api_adapter = _ALL_API_ADAPTER_FACTORIES[api_id].make_api_adapter()
        return cls(api_adapter)

    def __init__(self, api_adapter):
        self._api_adapter = api_adapter
        self.id = api_adapter.id
        self.shortcut = self.id.shortcut
        self.short_name = f"api({self.shortcut})"

    def __eq__(self, other):
        return isinstance(other, API) and self._api_adapter == other._api_adapter

    def __hash__(self):
        return hash((type(self), self._api_adapter))

    def __getitem__(self, idx):
        from .platform import Platform # avoiding circular imports
        return Platform.all(self)[idx]

from __future__ import annotations

from typing import Any, Optional, List, TYPE_CHECKING

from .adapter_base import APIID, APIAdapter
from .adapter_cuda import CuAPIAdapterFactory
from .adapter_opencl import OclAPIAdapterFactory

if TYPE_CHECKING:  # pragma: no cover
    from .platform import Platform


def cuda_api_id() -> APIID:
    """
    Returns the identifier of CUDA API.
    """
    return CuAPIAdapterFactory().api_id


def opencl_api_id() -> APIID:
    """
    Returns the identifier of OpenCL API.
    """
    return OclAPIAdapterFactory().api_id


_ALL_API_ADAPTER_FACTORIES = {
    factory.api_id: factory
    for factory in [
        CuAPIAdapterFactory(),
        OclAPIAdapterFactory(),
    ]
}


def all_api_ids() -> List[APIID]:
    """
    Returns a list of identifiers for all APIs available.
    """
    return list(_ALL_API_ADAPTER_FACTORIES.keys())


class API:
    """
    A generalized GPGPU API.
    """

    id: APIID
    """This API's ID."""

    shortcut: str
    """
    A shortcut for this API (to use in :py:meth:`all_by_shortcut`,
    usually coming from some kind of a CLI).
    Equal to ``id.shortcut``.
    """

    @classmethod
    def all_available(cls) -> List["API"]:
        """
        Returns a list of :py:class:`~grunnur.API` objects
        for which backends are available.
        """
        return [
            cls.from_api_id(api_id)
            for api_id, api_factory in _ALL_API_ADAPTER_FACTORIES.items()
            if api_factory.available
        ]

    @classmethod
    def all_by_shortcut(cls, shortcut: Optional[str] = None) -> List["API"]:
        """
        If ``shortcut`` is a string, returns a list of one :py:class:`~grunnur.API` object
        whose :py:attr:`~API.id` attribute has its
        :py:attr:`~grunnur.adapter_base.APIID.shortcut` attribute equal to it
        (or raises an error if it was not found, or its backend is not available).

        If ``shortcut`` is ``None``, returns a list of all available :py:class:`~grunnur.API` objects.

        :param shortcut: an API shortcut to match.
        """
        if shortcut is None:
            apis = cls.all_available()
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
    def from_api_id(cls, api_id: APIID) -> "API":
        """
        Creates an :py:class:`~grunnur.API` object out of an identifier.

        :param api_id: API identifier.
        """
        api_adapter = _ALL_API_ADAPTER_FACTORIES[api_id].make_api_adapter()
        return cls(api_adapter)

    def __init__(self, api_adapter: APIAdapter):
        self._api_adapter = api_adapter
        self.id = api_adapter.id
        self.shortcut = self.id.shortcut

    @property
    def platforms(self) -> List["Platform"]:
        """
        A list of this API's :py:class:`Platform` objects.
        """
        from .platform import Platform  # avoiding circular imports

        return Platform.all(self)

    def __eq__(self, other: Any) -> bool:
        return (
            type(self) == type(other)
            and isinstance(other, API)
            and self._api_adapter == other._api_adapter
        )

    def __hash__(self) -> int:
        return hash((type(self), self._api_adapter))

    def __str__(self) -> str:
        return f"api({self.shortcut})"

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._adapter_cuda import CuAPIAdapterFactory
from ._adapter_opencl import OclAPIAdapterFactory

if TYPE_CHECKING:  # pragma: no cover
    from ._adapter_base import APIID, APIAdapter
    from ._platform import Platform


def cuda_api_id() -> APIID:
    """Returns the identifier of CUDA API."""
    return CuAPIAdapterFactory().api_id


def opencl_api_id() -> APIID:
    """Returns the identifier of OpenCL API."""
    return OclAPIAdapterFactory().api_id


_ALL_API_ADAPTER_FACTORIES = {
    factory.api_id: factory
    for factory in [
        CuAPIAdapterFactory(),
        OclAPIAdapterFactory(),
    ]
}


def all_api_ids() -> list[APIID]:
    """
    Returns a list of identifiers for all APIs
    (not necessarily available in the current system).
    """
    return list(_ALL_API_ADAPTER_FACTORIES.keys())


class API:
    """A generalized GPGPU API."""

    id: APIID
    """This API's ID."""

    shortcut: str
    """
    A shortcut for this API (to use in :py:meth:`all_by_shortcut`,
    usually coming from some kind of a CLI).
    Equal to ``id.shortcut``.
    """

    @classmethod
    def all_available(cls) -> list[API]:
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
    def all_by_shortcut(cls, shortcut: str | None = None) -> list[API]:
        """
        If ``shortcut`` is a string, returns a list of one :py:class:`~grunnur.API` object
        whose :py:attr:`~API.id` attribute has its
        :py:attr:`~grunnur._adapter_base.APIID.shortcut` attribute equal to it
        (or raises an error if it was not found, or its backend is not available).

        If ``shortcut`` is ``None``, returns a list of all available
        :py:class:`~grunnur.API` objects.

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
    def from_api_id(cls, api_id: APIID) -> API:
        """
        Creates an :py:class:`~grunnur.API` object out of an identifier.

        :param api_id: API identifier.
        """
        api_adapter = _ALL_API_ADAPTER_FACTORIES[api_id].make_api_adapter()
        return cls(api_adapter)

    @staticmethod
    def cuda() -> API:
        """Returns a CUDA API object, if CUDA backend (that is, ``pycuda`` package) is available."""
        return API.from_api_id(cuda_api_id())

    @staticmethod
    def opencl() -> API:
        """
        Returns an OpenCL API object, if OpenCL backend
        (that is, ``pyopencl`` package) is available.
        """
        return API.from_api_id(opencl_api_id())

    @staticmethod
    def any() -> API:
        """
        Returns an API object for some available backend.

        Raises ``RuntimeError`` if no backends are available.
        """
        apis = API.all_available()
        if len(apis) == 0:
            raise RuntimeError("No APIs are available. Please install either PyCUDA or PyOpenCL")
        return apis[0]

    def __init__(self, api_adapter: APIAdapter):
        self._api_adapter = api_adapter
        self.id = api_adapter.id
        self.shortcut = self.id.shortcut

    @property
    def platforms(self) -> list[Platform]:
        """A list of this API's :py:class:`Platform` objects."""
        from ._platform import Platform  # noqa: PLC0415

        return Platform.all(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, API) and self._api_adapter == other._api_adapter

    def __hash__(self) -> int:
        return hash((type(self), self._api_adapter))

    def __str__(self) -> str:
        return f"api({self.shortcut})"

from __future__ import annotations

from typing import Optional, List, Sequence

from .adapter_base import PlatformAdapter
from .api import API
from .utils import string_matches_masks


class Platform:
    """
    A generalized GPGPU platform.
    """

    api: API
    """The :py:class:`~grunnur.API` object this platform belongs to."""

    name: str
    """The platform's name."""

    vendor: str
    """The platform's vendor."""

    version: str
    """The platform's version."""

    @classmethod
    def all(cls, api: API) -> List[Platform]:
        """
        Returns a list of platforms available for the given API.

        :param api: the API to search in.
        """
        return [
            Platform.from_index(api, platform_idx)
            for platform_idx in range(api._api_adapter.platform_count)]

    @classmethod
    def all_by_masks(
            cls,
            api: API,
            include_masks: Optional[Sequence[str]]=None,
            exclude_masks: Optional[Sequence[str]]=None) -> List[Platform]:
        """
        Returns a list of all platforms with names satisfying the given criteria.

        :param api: the API to search in.
        :param include_masks: a list of strings (treated as regexes),
            one of which must match with the platform name.
        :param exclude_masks: a list of strings (treated as regexes),
            neither of which must match with the platform name.
        """
        return [
            platform for platform in cls.all(api)
            if string_matches_masks(
                platform.name, include_masks=include_masks, exclude_masks=exclude_masks)
            ]

    @classmethod
    def from_backend_platform(cls, obj) -> Platform:
        """
        Wraps a backend platform object into a Grunnur platform object.
        """
        for api in API.all_available():
            if api._api_adapter.isa_backend_platform(obj):
                platform_adapter = api._api_adapter.make_platform_adapter(obj)
                return cls(platform_adapter)

        raise TypeError(f"{obj} was not recognized as a platform object by any available API")

    @classmethod
    def from_index(cls, api: API, platform_idx: int) -> Platform:
        """
        Creates a platform based on its index in the list returned by the API.

        :param api: the API to search in.
        :param platform_idx: the target platform's index.
        """
        platform_adapter = api._api_adapter.get_platform_adapters()[platform_idx]
        return cls(platform_adapter)

    def __init__(self, platform_adapter: PlatformAdapter):
        self.api = API(platform_adapter.api_adapter)
        self._platform_adapter = platform_adapter
        self.name = self._platform_adapter.name

        self.shortcut = f"{self.api.shortcut},{platform_adapter.platform_idx}"

        self.vendor = platform_adapter.vendor
        self.version = platform_adapter.version

    @property
    def devices(self):
        """
        A list of this device's :py:class:`Device` objects.
        """
        from .device import Device # avoiding circular imports
        return Device.all(self)

    def __eq__(self, other):
        return isinstance(other, Platform) and self._platform_adapter == other._platform_adapter

    def __hash__(self):
        return hash((type(self), self._platform_adapter))

    def __str__(self):
        return f"platform({self.shortcut})"

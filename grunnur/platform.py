from __future__ import annotations

from .api import API
from .utils import string_matches_masks


class Platform:
    """
    A generalized GPGPU platform.

    .. py:attribute:: api

        The :py:class:`API` object this platform belongs to.

    .. py:attribute:: id

        This platform's ID, a :py:class:`PlatformID` object.

    .. py:attribute:: short_name

        This platform's short name.
    """

    @classmethod
    def all(cls, api):
        """
        Returns a list of platforms available for this API.
        """
        return [
            Platform.from_index(api, platform_idx)
            for platform_idx in range(api._api_adapter.platform_count)]

    @classmethod
    def all_by_masks(
            cls,
            api,
            include_masks: Optional[Iterable[str]]=None,
            exclude_masks: Optional[Iterable[str]]=None) -> List[Platform]:
        """
        Returns a list of all platforms with names satisfying the given criteria.

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
    def from_backend_platform(cls, obj):
        for api in API.all():
            if api._api_adapter.isa_backend_platform(obj):
                platform_adapter = api._api_adapter.make_platform_adapter(obj)
                return cls(platform_adapter)

        raise TypeError(f"{obj} was not recognized as a platform object by any available API")

    @classmethod
    def from_index(cls, api, platform_idx):
        platform_adapter = api._api_adapter.get_platform_adapters()[platform_idx]
        return cls(platform_adapter)

    def __init__(self, platform_adapter):
        self.api = API(platform_adapter.api_adapter)
        self._platform_adapter = platform_adapter
        self.name = self._platform_adapter.name

        self.shortcut = f"{self.api.shortcut},{platform_adapter.platform_idx}"
        self.short_name = f"platform({self.shortcut})"

        self.vendor = platform_adapter.vendor
        self.version = platform_adapter.version

    def __eq__(self, other):
        return isinstance(other, Platform) and self._platform_adapter == other._platform_adapter

    def __hash__(self):
        return hash((type(self), self._platform_adapter))

    def __getitem__(self, idx):
        from .device import Device # avoiding circular imports
        return Device.all(self)[idx]

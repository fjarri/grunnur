from __future__ import annotations

from .api import API
from .utils import name_matches_masks


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
            Platform.from_number(api, platform_num)
            for platform_num in range(api._backend_api.num_platforms)]

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
            if name_matches_masks(
                platform.name, include_masks=include_masks, exclude_masks=exclude_masks)
            ]

    @classmethod
    def from_backend_platform(cls, backend_platform):
        api = API(backend_platform.api)
        return cls(api, backend_platform)

    @classmethod
    def from_backend_object(cls, obj):
        for api in API.all():
            if api.is_backend_platform(obj):
                backend_platform = api.make_platform(obj)
                return cls(api, backend_platform)

        raise ValueError(f"{obj} was not recognized as a platform object by any available API")

    @classmethod
    def from_number(cls, api, platform_num):
        backend_platform = api._backend_api.get_platforms()[platform_num]
        return cls(api, backend_platform)

    def __init__(self, api, backend_platform):
        self.api = api
        self._backend_platform = backend_platform
        self.name = self._backend_platform.name

        self.shortcut = f"{api.shortcut},{backend_platform.platform_num}"
        self.short_name = f"platform({self.shortcut})"

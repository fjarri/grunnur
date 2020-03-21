"""
This module contains functions for API discovery.
"""

from typing import List, Optional

from .base_classes import API
from .cuda import CUDA_API_FACTORY
from .opencl import OPENCL_API_FACTORY


def all_api_factories():
    return [CUDA_API_FACTORY, OPENCL_API_FACTORY]


def available_apis() -> List[API]:
    """
    Returns a list of :py:class:`~grunnur.base_classes.API` objects
    for which backends are available.
    """
    return [api_factory.make_api() for api_factory in all_api_factories() if api_factory.available]


def find_apis(shortcut: Optional[str]=None) -> List[API]:
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
        apis = available_apis()
    else:
        for api_factory in all_api_factories():
            if shortcut == api_factory.api_id.shortcut:
                if not api_factory.available:
                    raise ValueError(str(shortcut) + " API is not available")
                apis = [api_factory.make_api()]
                break
        else:
            raise ValueError("Invalid API shortcut: " + str(shortcut))

    return apis

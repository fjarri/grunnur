"""
This module contains :py:class:`~grunnur.Module` factories
which are used to compensate for the lack of complex number operations in OpenCL,
and the lack of C++ synthax which would allow one to write them.
"""

from typing import Optional
from warnings import warn

import numpy

from .template import Template
from . import dtypes
from .modules import Module


TEMPLATE = Template.from_associated_file(__file__)


def check_information_loss(out_dtype: numpy.dtype, expected_dtype: numpy.dtype):
    if dtypes.is_complex(expected_dtype) and not dtypes.is_complex(out_dtype):
        warn(
            "Imaginary part ignored during the downcast from " +
            str(expected_dtype) + " to " + str(out_dtype),
            numpy.ComplexWarning)


def derive_out_dtype(*in_dtypes: numpy.dtype, out_dtype: Optional[numpy.dtype]=None) -> numpy.dtype:
    expected_dtype = dtypes.result_type(*in_dtypes)
    if out_dtype is None:
        out_dtype = expected_dtype
    else:
        check_information_loss(out_dtype, expected_dtype)
    return out_dtype


def cast(in_dtype: numpy.dtype, out_dtype: numpy.dtype) -> Module:
    """
    Returns a :py:class:`~grunnur.Module` with a function of one argument
    that casts values of ``in_dtype`` to ``out_dtype``.

    :param in_dtype:
    :param out_dtype:
    """
    return Module(
        TEMPLATE.get_def('cast'),
        render_globals=dict(dtypes=dtypes, out_dtype=out_dtype, in_dtype=in_dtype))


def add(*in_dtypes: numpy.dtype, out_dtype: Optional[numpy.dtype]=None) -> Module:
    """
    Returns a :py:class:`~grunnur.Module`  with a function of
    ``len(in_dtypes)`` arguments that adds values of types ``in_dtypes``.
    If ``out_dtype`` is given, it will be set as a return type for this function.

    This is necessary since on some platforms complex numbers are based on 2-vectors,
    and therefore the ``+`` operator for a complex and a real number
    works in an unexpected way (returning ``(a.x + b, a.y + b)`` instead of ``(a.x + b, a.y)``).

    :param in_dtypes:
    :param out_dtype:
    """
    out_dtype = derive_out_dtype(*in_dtypes, out_dtype=out_dtype)
    return Module(
        TEMPLATE.get_def('add_or_mul'),
        render_globals=dict(dtypes=dtypes, op='add', out_dtype=out_dtype, in_dtypes=in_dtypes))


def mul(*in_dtypes: numpy.dtype, out_dtype: Optional[numpy.dtype]=None) -> Module:
    """
    Returns a :py:class:`~grunnur.Module`  with a function of
    ``len(in_dtypes)`` arguments that multiplies values of types ``in_dtypes``.
    If ``out_dtype`` is given, it will be set as a return type for this function.

    :param in_dtypes:
    :param out_dtype:
    """
    out_dtype = derive_out_dtype(*in_dtypes, out_dtype=out_dtype)
    return Module(
        TEMPLATE.get_def('add_or_mul'),
        render_globals=dict(dtypes=dtypes, op='mul', out_dtype=out_dtype, in_dtypes=in_dtypes))


def div(
        dividend_dtype: numpy.dtype, divisor_dtype: numpy.dtype,
        out_dtype: Optional[numpy.dtype]=None) -> Module:
    """
    Returns a :py:class:`~grunnur.Module` with a function of two arguments
    that divides a value of type ``dividend_dtype`` by a value of type ``divisor_dtype``.
    If ``out_dtype`` is given, it will be set as a return type for this function.

    :param dividend_dtype:
    :param divisor_dtype:
    :param out_dtype:
    """
    out_dtype = derive_out_dtype(dividend_dtype, divisor_dtype, out_dtype=out_dtype)
    return Module(
        TEMPLATE.get_def('div'),
        render_globals=dict(
            dtypes=dtypes, out_dtype=out_dtype,
            dividend_dtype=dividend_dtype, divisor_dtype=divisor_dtype))


def conj(dtype: numpy.dtype) -> Module:
    """
    Returns a :py:class:`~grunnur.Module` with a function of one argument
    that conjugates the value of type ``dtype``
    (if it is not a complex data type, the value will not be modified).

    :param dtype:
    """
    return Module(
        TEMPLATE.get_def('conj'),
        render_globals=dict(dtypes=dtypes, dtype=dtype))


def polar_unit(dtype: numpy.dtype) -> Module:
    """
    Returns a :py:class:`~grunnur.Module` with a function of one argument
    that returns a complex number ``exp(i * theta) == (cos(theta), sin(theta))``
    for a value ``theta`` of type ``dtype`` (must be a real data type).

    :param dtype:
    """
    if not dtypes.is_real(dtype):
        raise ValueError("polar_unit() can only be applied to real dtypes")

    return Module(
        TEMPLATE.get_def('polar_unit'),
        render_globals=dict(dtypes=dtypes, dtype=dtype))


def norm(dtype: numpy.dtype) -> Module:
    """
    Returns a :py:class:`~grunnur.Module` with a function of one argument
    that returns the 2-norm of the value of type ``dtype``
    (product by the complex conjugate if the value is complex, square otherwise).

    :param dtype:
    """
    return Module(
        TEMPLATE.get_def('norm'),
        render_globals=dict(dtypes=dtypes, dtype=dtype))


def exp(dtype: numpy.dtype) -> Module:
    """
    Returns a :py:class:`~grunnur.Module` with a function of one argument
    that exponentiates the value of type ``dtype``
    (must be a real or a complex data type).

    :param dtype:
    """
    # Supporting this will require an explicit output type specification.
    if dtypes.is_integer(dtype):
        raise ValueError(f"exp() of {dtype} is not supported")

    if dtypes.is_real(dtype):
        polar_unit_ = None
    else:
        polar_unit_ = polar_unit(dtypes.real_for(dtype))
    return Module(
        TEMPLATE.get_def('exp'),
        render_globals=dict(dtypes=dtypes, dtype=dtype, polar_unit_=polar_unit_))


def pow(
        base_dtype: numpy.dtype, exponent_dtype: Optional[numpy.dtype]=None,
        out_dtype: Optional[numpy.dtype]=None) -> Module:
    """
    Returns a :py:class:`~grunnur.Module` with a function of two arguments
    that raises the first argument of type ``base_dtype``
    to the power of the second argument of type ``exponent_dtype``
    (an integer or real data type).

    If ``exponent_dtype`` or ``out_dtype`` are not given, they default to ``base_dtype``.
    If ``base_dtype`` is not the same as ``out_dtype``,
    the input is cast to ``out_dtype`` *before* exponentiation.
    If ``exponent_dtype`` is real, but both ``base_dtype`` and ``out_dtype`` are integer,
    a ``ValueError`` is raised.

    :param base_dtype:
    :param exponent_dtype:
    :param out_dtype:
    """
    if exponent_dtype is None:
        exponent_dtype = base_dtype

    if out_dtype is None:
        out_dtype = base_dtype

    if dtypes.is_complex(exponent_dtype):
        raise ValueError("pow() with a complex exponent is not supported")

    if dtypes.is_real(exponent_dtype):
        if dtypes.is_complex(out_dtype):
            exponent_dtype = dtypes.real_for(out_dtype)
        elif dtypes.is_real(out_dtype):
            exponent_dtype = out_dtype
        else:
            raise ValueError("pow(integer, float): integer is not supported")

    kwds = dict(
        dtypes=dtypes,
        base_dtype=base_dtype, exponent_dtype=exponent_dtype, out_dtype=out_dtype,
        div_=None, mul_=None, cast_=None, polar_=None)
    if out_dtype != base_dtype:
        kwds['cast_'] = cast(base_dtype, out_dtype)
    if dtypes.is_integer(exponent_dtype) and not dtypes.is_real(out_dtype):
        kwds['mul_'] = mul(out_dtype, out_dtype)
        kwds['div_'] = div(out_dtype, out_dtype)
    if dtypes.is_complex(out_dtype):
        kwds['polar_'] = polar(dtypes.real_for(out_dtype))

    return Module(TEMPLATE.get_def('pow'), render_globals=kwds)


def polar(dtype: numpy.dtype) -> Module:
    """
    Returns a :py:class:`~grunnur.Module` with a function of two arguments
    that returns the complex-valued ``rho * exp(i * theta)``
    for values ``rho, theta`` of type ``dtype`` (must be a real data type).

    :param dtype:
    """
    if not dtypes.is_real(dtype):
        raise ValueError("polar() of " + str(dtype) + " is not supported")

    return Module(
        TEMPLATE.get_def('polar'),
        render_globals=dict(dtypes=dtypes, dtype=dtype, polar_unit_=polar_unit(dtype)))

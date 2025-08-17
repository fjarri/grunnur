import itertools
import re
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any
from warnings import catch_warnings, filterwarnings

import numpy
import pytest
from numpy.typing import DTypeLike, NDArray

from grunnur import Array, Context, Module, Program, Queue, RenderError, dtypes, functions
from grunnur._modules import render_with_modules
from grunnur._utils import prod
from utils import get_test_array


def get_func_kernel(
    func_module: Module, out_dtype: DTypeLike, in_dtypes: Iterable[DTypeLike]
) -> str:
    src = """
    <%
        argnames = ["a" + str(i + 1) for i in range(len(in_dtypes))]
        in_ctypes = list(map(dtypes.ctype, in_dtypes))
        out_ctype = dtypes.ctype(out_dtype)
    %>
    KERNEL void test(
        GLOBAL_MEM ${out_ctype} *dest
        %for arg, ctype in zip(argnames, in_ctypes, strict=True):
        , GLOBAL_MEM ${ctype} *${arg}
        %endfor
        )
    {
        const SIZE_T i = get_global_id(0);
        %for arg, ctype in zip(argnames, in_ctypes, strict=True):
        ${ctype} ${arg}_load = ${arg}[i];
        %endfor

        dest[i] = ${func}(${", ".join([arg + "_load" for arg in argnames])});
    }
    """

    return render_with_modules(
        src,
        render_globals=dict(
            dtypes=dtypes, in_dtypes=in_dtypes, out_dtype=out_dtype, func=func_module
        ),
    )


def generate_dtypes(
    out_code: str, in_codes: Iterable[str]
) -> tuple[numpy.dtype[Any], list[numpy.dtype[Any]]]:
    def test_dtype(idx: str) -> numpy.dtype[Any]:
        return numpy.dtype(dict(i=numpy.int32, f=numpy.float32, c=numpy.complex64)[idx])

    in_dtypes = list(map(test_dtype, in_codes))
    out_dtype = dtypes.result_type(*in_dtypes) if out_code == "auto" else test_dtype(out_code)

    if not any(map(dtypes.is_double, in_dtypes)):
        # numpy thinks that int32 * float32 == float64,
        # but we still need to run this test on older videocards
        if dtypes.is_complex(out_dtype):
            out_dtype = numpy.dtype(numpy.complex64)
        elif dtypes.is_real(out_dtype):
            out_dtype = numpy.dtype(numpy.float32)

    return out_dtype, in_dtypes


def check_func(
    context: Context,
    func_module: Module,
    reference_func: Callable[..., NDArray[Any]],
    out_dtype: DTypeLike,
    in_dtypes: Iterable[DTypeLike],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    is_mocked: bool = False,
) -> None:
    test_size = 256

    full_src = get_func_kernel(func_module, out_dtype, in_dtypes)

    # Can't test anything else if we don't have a real context
    if is_mocked:
        return

    program = Program([context.device], full_src)
    test = program.kernel.test

    queue = Queue(context.device)

    arrays = [get_test_array((test_size,), dt, no_zeros=True, high=8) for dt in in_dtypes]
    arrays_dev = [Array.from_host(queue, array) for array in arrays]
    dest_dev = Array.empty(context.device, [test_size], out_dtype)

    test(queue, [test_size], None, dest_dev, *arrays_dev)

    assert numpy.allclose(
        dest_dev.get(queue), reference_func(*arrays).astype(out_dtype), atol=atol, rtol=rtol
    )


# exp()


_test_exp_parameters = pytest.mark.parametrize(("out_code", "in_codes"), [("f", "f"), ("c", "c")])


def _test_exp(context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool) -> None:
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context, functions.exp(in_dtypes[0]), numpy.exp, out_dtype, in_dtypes, is_mocked=is_mocked
    )


@_test_exp_parameters
def test_exp(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_exp(context, out_code, in_codes, is_mocked=False)


@_test_exp_parameters
def test_exp_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_exp(mock_context, out_code, in_codes, is_mocked=True)


def test_exp_of_integer() -> None:
    with pytest.raises(ValueError, match=re.escape("exp() of int32 is not supported")):
        functions.exp(numpy.int32)


# pow()


_test_pow_parameters = pytest.mark.parametrize(
    ("out_code", "in_codes"), [("f", "fi"), ("c", "ci"), ("f", "ff"), ("c", "cf"), ("i", "ii")]
)


def _test_pow(context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool) -> None:
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    func = functions.pow(in_dtypes[0], exponent_dtype=in_dtypes[1], out_dtype=out_dtype)
    check_func(context, func, numpy.power, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_pow_parameters
def test_pow(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_pow(context, out_code, in_codes, is_mocked=False)


@_test_pow_parameters
def test_pow_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_pow(mock_context, out_code, in_codes, is_mocked=True)


def _test_pow_defaults(context: Context, *, is_mocked: bool) -> None:
    func = functions.pow(numpy.float32)  # check that exponent and output default to the base dtype
    check_func(
        context,
        func,
        numpy.power,
        numpy.float32,
        [numpy.float32, numpy.float32],
        is_mocked=is_mocked,
    )


def test_pow_defaults(context: Context) -> None:
    _test_pow_defaults(context, is_mocked=False)


def test_pow_defaults_mocked(mock_context: Context) -> None:
    _test_pow_defaults(mock_context, is_mocked=True)


def _test_pow_cast_output(context: Context, *, is_mocked: bool) -> None:
    func = functions.pow(numpy.int32, exponent_dtype=numpy.int32, out_dtype=numpy.int64)
    check_func(
        context, func, numpy.power, numpy.int64, [numpy.int32, numpy.int32], is_mocked=is_mocked
    )


def test_pow_cast_output(context: Context) -> None:
    _test_pow_cast_output(context, is_mocked=False)


def test_pow_cast_output_mocked(mock_context: Context) -> None:
    _test_pow_cast_output(mock_context, is_mocked=True)


def test_pow_complex_exponent() -> None:
    with pytest.raises(
        ValueError, match=re.escape("pow() with a complex exponent is not supported")
    ):
        functions.pow(numpy.float32, exponent_dtype=numpy.complex64)


def test_pow_int_to_float() -> None:
    with pytest.raises(
        ValueError, match=re.escape("pow(integer, float): integer is not supported")
    ):
        functions.pow(numpy.int32, exponent_dtype=numpy.float32, out_dtype=numpy.int32)


@pytest.mark.parametrize(("out_code", "in_codes"), [("c", "cf"), ("f", "ff")])
def test_pow_zero_base(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    """Specific tests for 0^0 and 0^x."""
    test_size = 256

    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    func_module = functions.pow(in_dtypes[0], exponent_dtype=in_dtypes[1], out_dtype=out_dtype)
    full_src = get_func_kernel(func_module, out_dtype, in_dtypes)
    program = Program([context.device], full_src)
    test = program.kernel.test

    queue = Queue(context.device)
    bases = Array.from_host(queue, numpy.zeros(test_size, in_dtypes[0]))

    # zero exponents
    exponents = Array.from_host(queue, numpy.zeros(test_size, in_dtypes[1]))
    dest_dev = Array.empty(context.device, [test_size], out_dtype)
    test(queue, [test_size], None, dest_dev, bases, exponents)
    assert numpy.allclose(dest_dev.get(queue), numpy.ones(test_size, in_dtypes[0]))

    # non-zero exponents
    exponents = Array.from_host(queue, numpy.ones(test_size, in_dtypes[1]))
    dest_dev = Array.empty(context.device, [test_size], out_dtype)
    test(queue, [test_size], None, dest_dev, bases, exponents)
    assert numpy.allclose(dest_dev.get(queue), numpy.zeros(test_size, in_dtypes[0]))


# polar_unit()


_test_polar_unit_parameters = pytest.mark.parametrize(("out_code", "in_codes"), [("c", "f")])


def _test_polar_unit(
    context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool
) -> None:
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context,
        functions.polar_unit(in_dtypes[0]),
        lambda theta: numpy.exp(1j * theta),
        out_dtype,
        in_dtypes,
        is_mocked=is_mocked,
    )


@_test_polar_unit_parameters
def test_polar_unit(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_polar_unit(context, out_code, in_codes, is_mocked=False)


@_test_polar_unit_parameters
def test_polar_unit_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_polar_unit(mock_context, out_code, in_codes, is_mocked=True)


def test_polar_unit_of_complex() -> None:
    with pytest.raises(
        ValueError, match=re.escape("polar_unit() can only be applied to real dtypes")
    ):
        functions.polar_unit(numpy.complex64)


# polar()


_test_polar_parameters = pytest.mark.parametrize(("out_code", "in_codes"), [("c", "ff")])


def _test_polar(
    context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool
) -> None:
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context,
        functions.polar(in_dtypes[0]),
        lambda rho, theta: rho * numpy.exp(1j * theta),
        out_dtype,
        in_dtypes,
        is_mocked=is_mocked,
    )


@_test_polar_parameters
def test_polar(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_polar(context, out_code, in_codes, is_mocked=False)


@_test_polar_parameters
def test_polar_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_polar(mock_context, out_code, in_codes, is_mocked=True)


def test_polar_of_complex() -> None:
    with pytest.raises(ValueError, match=re.escape("polar() of complex64 is not supported")):
        functions.polar(numpy.complex64)


# norm()


_test_norm_parameters = pytest.mark.parametrize(
    ("out_code", "in_codes"), [("f", "c"), ("f", "f"), ("i", "i")]
)


def _test_norm(
    context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool
) -> None:
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context,
        functions.norm(in_dtypes[0]),
        lambda x: numpy.abs(x) ** 2,
        out_dtype,
        in_dtypes,
        is_mocked=is_mocked,
    )


@_test_norm_parameters
def test_norm(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_norm(context, out_code, in_codes, is_mocked=False)


@_test_norm_parameters
def test_norm_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_norm(mock_context, out_code, in_codes, is_mocked=True)


# conj()


_test_conj_parameters = pytest.mark.parametrize(("out_code", "in_codes"), [("c", "c"), ("f", "f")])


def _test_conj(
    context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool
) -> None:
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context, functions.conj(in_dtypes[0]), numpy.conj, out_dtype, in_dtypes, is_mocked=is_mocked
    )


@_test_conj_parameters
def test_conj(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_conj(context, out_code, in_codes, is_mocked=False)


@_test_conj_parameters
def test_conj_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_conj(mock_context, out_code, in_codes, is_mocked=True)


# cast()


_test_cast_parameters = pytest.mark.parametrize(
    ("out_code", "in_codes"), [("c", "f"), ("f", "f"), ("c", "c")]
)


def _test_cast(
    context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool
) -> None:
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context,
        functions.cast(in_dtypes[0], out_dtype),
        partial(numpy.asarray, dtype=out_dtype),
        out_dtype,
        in_dtypes,
        is_mocked=is_mocked,
    )


@_test_cast_parameters
def test_cast(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_cast(context, out_code, in_codes, is_mocked=False)


@_test_cast_parameters
def test_cast_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_cast(mock_context, out_code, in_codes, is_mocked=True)


def test_cast_complex_to_real(context: Context) -> None:
    out_dtype = numpy.float32
    in_dtypes = [numpy.complex64]
    message = re.escape("cast from complex64 to float32 is not supported")
    with pytest.raises(ValueError, match=message):
        check_func(
            context,
            functions.cast(in_dtypes[0], out_dtype),
            partial(numpy.asarray, dtype=out_dtype),
            out_dtype,
            in_dtypes,
        )


# div()


_test_div_parameters = pytest.mark.parametrize(
    ("out_code", "in_codes"), [("f", "ff"), ("c", "cc"), ("c", "cf"), ("c", "fc"), ("f", "if")]
)


def _test_div(context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool) -> None:
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    dividend, divisor = in_dtypes
    check_func(
        context,
        functions.div(dividend, divisor, out_dtype=out_dtype),
        lambda x, y: numpy.asarray(x / y, out_dtype),
        out_dtype,
        in_dtypes,
        is_mocked=is_mocked,
    )


@_test_div_parameters
def test_div(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_div(context, out_code, in_codes, is_mocked=False)


@_test_div_parameters
def test_div_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_div(mock_context, out_code, in_codes, is_mocked=True)


# add()


def make_reference_add(out_dtype: numpy.dtype[Any]) -> Callable[..., NDArray[Any]]:
    def reference_add(*args: NDArray[Any]) -> NDArray[Any]:
        res = sum(args, start=numpy.zeros_like(args[0]))
        if not dtypes.is_complex(out_dtype) and dtypes.is_complex(res.dtype):
            res = res.real
        return res.astype(out_dtype)

    return reference_add


_test_add_parameters = pytest.mark.parametrize("in_codes", ["ff", "cc", "cf", "fc", "ifccfi"])


def _test_add(context: Context, in_codes: Iterable[str], *, is_mocked: bool) -> None:
    """Checks multi-argument add() with a variety of data types."""
    out_dtype, in_dtypes = generate_dtypes("auto", in_codes)
    reference_add = make_reference_add(out_dtype)
    add = functions.add(*in_dtypes, out_dtype=out_dtype)
    check_func(context, add, reference_add, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_add_parameters
def test_add(context: Context, in_codes: Iterable[str]) -> None:
    _test_add(context, in_codes, is_mocked=False)


@_test_add_parameters
def test_add_mocked(mock_context: Context, in_codes: Iterable[str]) -> None:
    _test_add(mock_context, in_codes, is_mocked=True)


_test_add_cast_parameters = pytest.mark.parametrize(
    ("out_code", "in_codes"), [("c", "ff"), ("c", "cc"), ("f", "ff"), ("f", "cc")]
)


def _test_add_cast(
    context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool
) -> None:
    """Check that add() casts the result correctly."""
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    reference_add = make_reference_add(out_dtype)

    # Temporarily catching imaginary part truncation warnings
    with catch_warnings():
        filterwarnings("ignore", "", numpy.exceptions.ComplexWarning)
        add = functions.add(*in_dtypes, out_dtype=out_dtype)

    check_func(context, add, reference_add, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_add_cast_parameters
def test_add_cast(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_add_cast(context, out_code, in_codes, is_mocked=False)


@_test_add_cast_parameters
def test_add_cast_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_add_cast(mock_context, out_code, in_codes, is_mocked=True)


# mul()


def make_reference_mul(out_dtype: numpy.dtype[Any]) -> Callable[..., NDArray[Any]]:
    def reference_mul(*args: NDArray[Any]) -> NDArray[Any]:
        res = prod(args)
        if not dtypes.is_complex(out_dtype) and dtypes.is_complex(res.dtype):
            res = res.real
        return res.astype(out_dtype)

    return reference_mul


_test_mul_parameters = pytest.mark.parametrize("in_codes", ["ff", "cc", "cf", "fc", "ifccfi"])


def _test_mul(context: Context, in_codes: Iterable[str], *, is_mocked: bool) -> None:
    """Checks multi-argument mul() with a variety of data types."""
    out_dtype, in_dtypes = generate_dtypes("auto", in_codes)
    reference_mul = make_reference_mul(out_dtype)
    mul = functions.mul(*in_dtypes, out_dtype=out_dtype)
    check_func(context, mul, reference_mul, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_mul_parameters
def test_mul(context: Context, in_codes: Iterable[str]) -> None:
    _test_mul(context, in_codes, is_mocked=False)


@_test_mul_parameters
def test_mul_mocked(mock_context: Context, in_codes: Iterable[str]) -> None:
    _test_mul(mock_context, in_codes, is_mocked=True)


_test_mul_cast_parameters = pytest.mark.parametrize(
    ("out_code", "in_codes"), [("c", "ff"), ("c", "cc"), ("f", "ff"), ("f", "cc")]
)


def _test_mul_cast(
    context: Context, out_code: str, in_codes: Iterable[str], *, is_mocked: bool
) -> None:
    """Check that mul() casts the result correctly."""
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    reference_mul = make_reference_mul(out_dtype)

    # Temporarily catching imaginary part truncation warnings
    with catch_warnings():
        filterwarnings("ignore", "", numpy.exceptions.ComplexWarning)
        mul = functions.mul(*in_dtypes, out_dtype=out_dtype)

    # Relax tolerance a little - in single precision the difference may sometimes go to 1e-5
    check_func(context, mul, reference_mul, out_dtype, in_dtypes, is_mocked=is_mocked, rtol=1e-4)


@_test_mul_cast_parameters
def test_mul_cast(context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_mul_cast(context, out_code, in_codes, is_mocked=False)


@_test_mul_cast_parameters
def test_mul_cast_mocked(mock_context: Context, out_code: str, in_codes: Iterable[str]) -> None:
    _test_mul_cast(mock_context, out_code, in_codes, is_mocked=True)

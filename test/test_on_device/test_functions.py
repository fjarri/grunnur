import numpy
import itertools
from warnings import catch_warnings, filterwarnings

import pytest

from grunnur import Program, Queue, Array
from grunnur.modules import render_with_modules
import grunnur.dtypes as dtypes
import grunnur.functions as functions
from grunnur.template import RenderError
from grunnur.utils import prod

from ..utils import get_test_array


def get_func_kernel(func_module, out_dtype, in_dtypes):
    src = """
    <%
        argnames = ["a" + str(i + 1) for i in range(len(in_dtypes))]
        in_ctypes = list(map(dtypes.ctype, in_dtypes))
        out_ctype = dtypes.ctype(out_dtype)
    %>
    KERNEL void test(
        GLOBAL_MEM ${out_ctype} *dest
        %for arg, ctype in zip(argnames, in_ctypes):
        , GLOBAL_MEM ${ctype} *${arg}
        %endfor
        )
    {
        const SIZE_T i = get_global_id(0);
        %for arg, ctype in zip(argnames, in_ctypes):
        ${ctype} ${arg}_load = ${arg}[i];
        %endfor

        dest[i] = ${func}(${", ".join([arg + "_load" for arg in argnames])});
    }
    """

    full_src = render_with_modules(
        src,
        render_globals=dict(dtypes=dtypes, in_dtypes=in_dtypes, out_dtype=out_dtype, func=func_module))

    return full_src


def generate_dtypes(out_code, in_codes):
    test_dtype = lambda idx: dict(i=numpy.int32, f=numpy.float32, c=numpy.complex64)[idx]
    in_dtypes = list(map(test_dtype, in_codes))
    out_dtype = dtypes.result_type(*in_dtypes) if out_code == 'auto' else test_dtype(out_code)

    if not any(map(dtypes.is_double, in_dtypes)):
        # numpy thinks that int32 * float32 == float64,
        # but we still need to run this test on older videocards
        if dtypes.is_complex(out_dtype):
            out_dtype = numpy.complex64
        elif dtypes.is_real(out_dtype):
            out_dtype = numpy.float32

    return out_dtype, in_dtypes


def check_func(context, func_module, reference_func, out_dtype, in_dtypes, atol=1e-8, rtol=1e-5, is_mocked=False):
    N = 256

    full_src = get_func_kernel(func_module, out_dtype, in_dtypes)

    # Can't test anything else if we don't have a real context
    if is_mocked:
        return

    program = Program(context, full_src)
    test = program.kernel.test

    queue = Queue.on_all_devices(context)

    arrays = [get_test_array(N, dt, no_zeros=True, high=8) for dt in in_dtypes]
    arrays_dev = [Array.from_host(queue, array) for array in arrays]
    dest_dev = Array.empty(queue, N, out_dtype)

    test(queue, N, None, dest_dev, *arrays_dev)

    assert numpy.allclose(
        dest_dev.get(),
        reference_func(*arrays).astype(out_dtype), atol=atol, rtol=rtol)


# exp()


_test_exp_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('f', 'f'), ('c', 'c')])


def _test_exp(context, out_code, in_codes, is_mocked):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(context, functions.exp(in_dtypes[0]), numpy.exp, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_exp_parameters
def test_exp(context, out_code, in_codes):
    _test_exp(context, out_code, in_codes, is_mocked=False)


@_test_exp_parameters
def test_exp_mocked(mock_context, out_code, in_codes):
    _test_exp(mock_context, out_code, in_codes, is_mocked=True)


def test_exp_of_integer():
    with pytest.raises(ValueError):
        functions.exp(numpy.int32)


# pow()


_test_pow_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('f', 'fi'), ('c', 'ci'), ('f', 'ff'), ('c', 'cf'), ('i', 'ii')])


def _test_pow(context, out_code, in_codes, is_mocked):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    func = functions.pow(in_dtypes[0], exponent_dtype=in_dtypes[1], out_dtype=out_dtype)
    check_func(context, func, numpy.power, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_pow_parameters
def test_pow(context, out_code, in_codes):
    _test_pow(context, out_code, in_codes, is_mocked=False)


@_test_pow_parameters
def test_pow_mocked(mock_context, out_code, in_codes):
    _test_pow(mock_context, out_code, in_codes, is_mocked=True)


def _test_pow_defaults(context, is_mocked):
    func = functions.pow(numpy.float32) # check that exponent and output default to the base dtype
    check_func(context, func, numpy.power, numpy.float32, [numpy.float32, numpy.float32], is_mocked=is_mocked)


def test_pow_defaults(context):
    _test_pow_defaults(context, is_mocked=False)


def test_pow_defaults_mocked(mock_context):
    _test_pow_defaults(mock_context, is_mocked=True)


def _test_pow_cast_output(context, is_mocked):
    func = functions.pow(numpy.int32, exponent_dtype=numpy.int32, out_dtype=numpy.int64)
    check_func(context, func, numpy.power, numpy.int64, [numpy.int32, numpy.int32], is_mocked=is_mocked)


def test_pow_cast_output(context):
    _test_pow_cast_output(context, is_mocked=False)


def test_pow_cast_output_mocked(mock_context):
    _test_pow_cast_output(mock_context, is_mocked=True)


def test_pow_complex_exponent():
    with pytest.raises(ValueError):
        functions.pow(numpy.float32, exponent_dtype=numpy.complex64)


def test_pow_int_to_float():
    with pytest.raises(ValueError):
        functions.pow(numpy.int32, exponent_dtype=numpy.float32, out_dtype=numpy.int32)


@pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'cf'), ('f', 'ff')])
def test_pow_zero_base(context, out_code, in_codes):
    """
    Specific tests for 0^0 and 0^x.
    """
    N = 256

    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    func_module = functions.pow(in_dtypes[0], exponent_dtype=in_dtypes[1], out_dtype=out_dtype)
    full_src = get_func_kernel(func_module, out_dtype, in_dtypes)
    program = Program(context, full_src)
    test = program.kernel.test

    queue = Queue.on_all_devices(context)
    bases = Array.from_host(queue, numpy.zeros(N, in_dtypes[0]))

    # zero exponents
    exponents = Array.from_host(queue, numpy.zeros(N, in_dtypes[1]))
    dest_dev = Array.empty(queue, N, out_dtype)
    test(queue, N, None, dest_dev, bases, exponents)
    assert numpy.allclose(dest_dev.get(), numpy.ones(N, in_dtypes[0]))

    # non-zero exponents
    exponents = Array.from_host(queue, numpy.ones(N, in_dtypes[1]))
    dest_dev = Array.empty(queue, N, out_dtype)
    test(queue, N, None, dest_dev, bases, exponents)
    assert numpy.allclose(dest_dev.get(), numpy.zeros(N, in_dtypes[0]))


# polar_unit()


_test_polar_unit_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'f')])


def _test_polar_unit(context, out_code, in_codes, is_mocked):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context, functions.polar_unit(in_dtypes[0]),
        lambda theta: numpy.exp(1j * theta), out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_polar_unit_parameters
def test_polar_unit(context, out_code, in_codes):
    _test_polar_unit(context, out_code, in_codes, is_mocked=False)


@_test_polar_unit_parameters
def test_polar_unit_mocked(mock_context, out_code, in_codes):
    _test_polar_unit(mock_context, out_code, in_codes, is_mocked=True)


def test_polar_unit_of_complex():
    with pytest.raises(ValueError):
        functions.polar_unit(numpy.complex64)


# polar()


_test_polar_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'ff')])


def _test_polar(context, out_code, in_codes, is_mocked):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context, functions.polar(in_dtypes[0]),
        lambda rho, theta: rho * numpy.exp(1j * theta), out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_polar_parameters
def test_polar(context, out_code, in_codes):
    _test_polar(context, out_code, in_codes, is_mocked=False)


@_test_polar_parameters
def test_polar_mocked(mock_context, out_code, in_codes):
    _test_polar(mock_context, out_code, in_codes, is_mocked=True)


def test_polar_of_complex():
    with pytest.raises(ValueError):
        functions.polar(numpy.complex64)


# norm()


_test_norm_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('f', 'c'), ('f', 'f'), ('i', 'i')])


def _test_norm(context, out_code, in_codes, is_mocked):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context, functions.norm(in_dtypes[0]),
        lambda x: numpy.abs(x) ** 2, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_norm_parameters
def test_norm(context, out_code, in_codes):
    _test_norm(context, out_code, in_codes, is_mocked=False)


@_test_norm_parameters
def test_norm_mocked(mock_context, out_code, in_codes):
    _test_norm(mock_context, out_code, in_codes, is_mocked=True)


# conj()


_test_conj_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'c'), ('f', 'f')])


def _test_conj(context, out_code, in_codes, is_mocked):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context, functions.conj(in_dtypes[0]),
        numpy.conj, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_conj_parameters
def test_conj(context, out_code, in_codes):
    _test_conj(context, out_code, in_codes, is_mocked=False)


@_test_conj_parameters
def test_conj_mocked(mock_context, out_code, in_codes):
    _test_conj(mock_context, out_code, in_codes, is_mocked=True)


# cast()


_test_cast_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'f'), ('f', 'f'), ('c', 'c')])


def _test_cast(context, out_code, in_codes, is_mocked):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context, functions.cast(in_dtypes[0], out_dtype),
        dtypes.cast(out_dtype), out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_cast_parameters
def test_cast(context, out_code, in_codes):
    _test_cast(context, out_code, in_codes, is_mocked=False)


@_test_cast_parameters
def test_cast_mocked(mock_context, out_code, in_codes):
    _test_cast(mock_context, out_code, in_codes, is_mocked=True)


def test_cast_complex_to_real(context):
    out_dtype = numpy.float32
    in_dtypes = [numpy.complex64]
    with pytest.raises(RenderError, match="ValueError"):
        check_func(
            context, functions.cast(in_dtypes[0], out_dtype),
            dtypes.cast(out_dtype), out_dtype, in_dtypes)


# div()


_test_div_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('f', 'ff'), ('c', 'cc'), ('c', 'cf'), ('c', 'fc'), ('f', 'if')])


def _test_div(context, out_code, in_codes, is_mocked):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        context, functions.div(*in_dtypes, out_dtype=out_dtype),
        lambda x, y: dtypes.cast(out_dtype)(x / y), out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_div_parameters
def test_div(context, out_code, in_codes):
    _test_div(context, out_code, in_codes, is_mocked=False)


@_test_div_parameters
def test_div_mocked(mock_context, out_code, in_codes):
    _test_div(mock_context, out_code, in_codes, is_mocked=True)


# add()


def make_reference_add(out_dtype):

    def reference_add(*args):
        res = sum(args)
        if not dtypes.is_complex(out_dtype) and dtypes.is_complex(res.dtype):
            res = res.real
        return res.astype(out_dtype)

    return reference_add


_test_add_parameters = pytest.mark.parametrize('in_codes', ["ff", "cc", "cf", "fc", "ifccfi"])


def _test_add(context, in_codes, is_mocked):
    """
    Checks multi-argument add() with a variety of data types.
    """
    out_dtype, in_dtypes = generate_dtypes('auto', in_codes)
    reference_add = make_reference_add(out_dtype)
    add = functions.add(*in_dtypes, out_dtype=out_dtype)
    check_func(context, add, reference_add, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_add_parameters
def test_add(context, in_codes):
    _test_add(context, in_codes, is_mocked=False)


@_test_add_parameters
def test_add_mocked(mock_context, in_codes):
    _test_add(mock_context, in_codes, is_mocked=True)


_test_add_cast_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'ff'), ('c', 'cc'), ('f', 'ff'), ('f', 'cc')])


def _test_add_cast(context, out_code, in_codes, is_mocked):
    """
    Check that add() casts the result correctly.
    """
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    reference_add = make_reference_add(out_dtype)

    # Temporarily catching imaginary part truncation warnings
    with catch_warnings():
        filterwarnings("ignore", "", numpy.ComplexWarning)
        add = functions.add(*in_dtypes, out_dtype=out_dtype)

    check_func(context, add, reference_add, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_add_cast_parameters
def test_add_cast(context, out_code, in_codes):
    _test_add_cast(context, out_code, in_codes, is_mocked=False)


@_test_add_cast_parameters
def test_add_cast_mocked(mock_context, out_code, in_codes):
    _test_add_cast(mock_context, out_code, in_codes, is_mocked=True)


# mul()


def make_reference_mul(out_dtype):

    def reference_mul(*args):
        res = prod(args)
        if not dtypes.is_complex(out_dtype) and dtypes.is_complex(res.dtype):
            res = res.real
        return res.astype(out_dtype)

    return reference_mul


_test_mul_parameters = pytest.mark.parametrize('in_codes', ["ff", "cc", "cf", "fc", "ifccfi"])


def _test_mul(context, in_codes, is_mocked):
    """
    Checks multi-argument mul() with a variety of data types.
    """
    out_dtype, in_dtypes = generate_dtypes('auto', in_codes)
    reference_mul = make_reference_mul(out_dtype)
    mul = functions.mul(*in_dtypes, out_dtype=out_dtype)
    check_func(context, mul, reference_mul, out_dtype, in_dtypes, is_mocked=is_mocked)


@_test_mul_parameters
def test_mul(context, in_codes):
    _test_mul(context, in_codes, is_mocked=False)


@_test_mul_parameters
def test_mul_mocked(mock_context, in_codes):
    _test_mul(mock_context, in_codes, is_mocked=True)


_test_mul_cast_parameters = pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'ff'), ('c', 'cc'), ('f', 'ff'), ('f', 'cc')])


def _test_mul_cast(context, out_code, in_codes, is_mocked):
    """
    Check that mul() casts the result correctly.
    """
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    reference_mul = make_reference_mul(out_dtype)

    # Temporarily catching imaginary part truncation warnings
    with catch_warnings():
        filterwarnings("ignore", "", numpy.ComplexWarning)
        mul = functions.mul(*in_dtypes, out_dtype=out_dtype)

    # Relax tolerance a little - in single precision the difference may sometimes go to 1e-5
    check_func(context, mul, reference_mul, out_dtype, in_dtypes, is_mocked=is_mocked, rtol=1e-4)


@_test_mul_cast_parameters
def test_mul_cast(context, out_code, in_codes):
    _test_mul_cast(context, out_code, in_codes, is_mocked=False)


@_test_mul_cast_parameters
def test_mul_cast_mocked(mock_context, out_code, in_codes):
    _test_mul_cast(mock_context, out_code, in_codes, is_mocked=True)

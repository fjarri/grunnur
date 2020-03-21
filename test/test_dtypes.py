import numpy
import pytest

from grunnur import dtypes
from grunnur.modules import render_with_modules


def test_normalize_type():
    dtype = dtypes.normalize_type(numpy.int32)
    assert dtype == numpy.int32
    assert type(dtype) == numpy.dtype


def test_ctype_builtin():
    assert dtypes.ctype(numpy.int32) == 'int'


def test_is_complex():
    assert dtypes.is_complex(numpy.complex64)
    assert dtypes.is_complex(numpy.complex128)
    assert not dtypes.is_complex(numpy.float64)


def test_is_double():
    assert dtypes.is_double(numpy.float64)
    assert dtypes.is_double(numpy.complex128)
    assert not dtypes.is_double(numpy.complex64)


def test_is_integer():
    assert dtypes.is_integer(numpy.int32)
    assert not dtypes.is_integer(numpy.float32)


def test_is_real():
    assert dtypes.is_real(numpy.float32)
    assert not dtypes.is_real(numpy.complex64)
    assert not dtypes.is_real(numpy.int32)


def test_promote_type():
    assert dtypes._promote_type(numpy.int8) == numpy.int32
    assert dtypes._promote_type(numpy.uint8) == numpy.uint32
    assert dtypes._promote_type(numpy.float16) == numpy.float32
    assert dtypes._promote_type(numpy.int32) == numpy.int32


def test_result_type():
    assert dtypes.result_type(numpy.int32, numpy.float32) == numpy.float64


def test_min_scalar_type():
    assert dtypes.min_scalar_type(1) == numpy.uint32
    assert dtypes.min_scalar_type(-1) == numpy.int32
    assert dtypes.min_scalar_type(1.) == numpy.float32

    assert dtypes.min_scalar_type(2**31-1, force_signed=True) == numpy.int32
    # 2**31 will not fit into int32 type
    assert dtypes.min_scalar_type(2**31, force_signed=True) == numpy.int64


def test_detect_type():
    assert dtypes.detect_type(numpy.int8(-1)) == numpy.int32
    assert dtypes.detect_type(numpy.int64(-1)) == numpy.int64
    assert dtypes.detect_type(-1) == numpy.int32
    assert dtypes.detect_type(-1.) == numpy.float32


def test_complex_for():
    assert dtypes.complex_for(numpy.float32) == numpy.complex64
    assert dtypes.complex_for(numpy.float64) == numpy.complex128
    with pytest.raises(ValueError):
        assert dtypes.complex_for(numpy.complex64)
    with pytest.raises(ValueError):
        assert dtypes.complex_for(numpy.int32)


def test_real_for():
    assert dtypes.real_for(numpy.complex64) == numpy.float32
    assert dtypes.real_for(numpy.complex128) == numpy.float64
    with pytest.raises(ValueError):
        assert dtypes.real_for(numpy.float32)
    with pytest.raises(ValueError):
        assert dtypes.real_for(numpy.int32)


def test_complex_ctr():
    assert dtypes.complex_ctr(numpy.complex64) == "COMPLEX_CTR(float2)"


def test_cast():
    cast = dtypes.cast(numpy.uint64)
    for val in [cast(1), cast(numpy.int32(1)), cast(numpy.uint64(1))]:
        assert val.dtype == numpy.uint64 and val == 1


def test_c_constant():
    # scalar values
    assert dtypes.c_constant(1) == "1"
    assert dtypes.c_constant(numpy.uint64(1)) == "1UL"
    assert dtypes.c_constant(numpy.int64(-1)) == "-1L"
    assert dtypes.c_constant(numpy.float64(1.)) == "1.0"
    assert dtypes.c_constant(numpy.float32(1.)) == "1.0f"
    assert dtypes.c_constant(numpy.complex64(1 + 2j)) == "COMPLEX_CTR(float2)(1.0f, 2.0f)"
    assert dtypes.c_constant(numpy.complex128(1 + 2j)) == "COMPLEX_CTR(double2)(1.0, 2.0)"

    # array
    assert dtypes.c_constant(numpy.array([1, 2, 3], numpy.float32)) == "{1.0f, 2.0f, 3.0f}"

    # struct type
    dtype = numpy.dtype([('val1', numpy.int32), ('val2', numpy.float32)])
    val = numpy.empty((), dtype)
    val['val1'] = 1
    val['val2'] = 2
    assert dtypes.c_constant(val) == "{1, 2.0f}"

    # custom dtype
    assert dtypes.c_constant(1, numpy.float32) == "1.0f"


def test__align_simple():
    dtype = numpy.dtype('int32')
    res = dtypes._align(dtype)
    ref = dtypes.WrappedType(dtype, dtype.itemsize)
    assert res == ref


def test__align_array():
    dtype = numpy.dtype('int32')
    dtype_arr = numpy.dtype((dtype, 3))
    res = dtypes._align(dtype_arr)
    ref = dtypes.WrappedType(dtype_arr, dtype.itemsize)
    assert res == ref


def test__align_non_aligned_struct():
    dtype = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32]))
    res = dtypes._align(dtype)

    dtype_aligned = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 2, 4],
        itemsize=8,
        aligned=True))

    wt_x = dtypes.WrappedType(numpy.dtype('int8'), 1)
    wt_y = dtypes.WrappedType(numpy.dtype('int16'), 2)
    wt_z = dtypes.WrappedType(numpy.dtype('int32'), 4)
    ref = dtypes.WrappedType(
        dtype_aligned, 4, explicit_alignment=None, wrapped_fields=dict(x=wt_x, y=wt_y, z=wt_z),
        field_alignments=dict(x=None, y=None, z=None))
    assert res == ref


def test__align_aligned_struct():
    dtype_aligned = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 2, 4],
        itemsize=8,
        aligned=True))

    res = dtypes._align(dtype_aligned)

    wt_x = dtypes.WrappedType(numpy.dtype('int8'), 1)
    wt_y = dtypes.WrappedType(numpy.dtype('int16'), 2)
    wt_z = dtypes.WrappedType(numpy.dtype('int32'), 4)
    ref = dtypes.WrappedType(
        dtype_aligned, 4, explicit_alignment=None, wrapped_fields=dict(x=wt_x, y=wt_y, z=wt_z),
        field_alignments=dict(x=None, y=None, z=None))
    assert res == ref


def test__align_aligned_struct_custom_itemsize():
    dtype_aligned = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 2, 4],
        itemsize=16,
        aligned=True))

    res = dtypes._align(dtype_aligned)

    wt_x = dtypes.WrappedType(numpy.dtype('int8'), 1)
    wt_y = dtypes.WrappedType(numpy.dtype('int16'), 2)
    wt_z = dtypes.WrappedType(numpy.dtype('int32'), 4)
    ref = dtypes.WrappedType(
        dtype_aligned, 16, explicit_alignment=16, wrapped_fields=dict(x=wt_x, y=wt_y, z=wt_z),
        field_alignments=dict(x=None, y=None, z=None))
    assert res == ref


def test__align_custom_field_offsets():
    dtype = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 4, 16],
        itemsize=32))

    dtype_aligned = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 4, 16],
        itemsize=32,
        aligned=True))

    res = dtypes._align(dtype_aligned)

    wt_x = dtypes.WrappedType(numpy.dtype('int8'), 1)
    wt_y = dtypes.WrappedType(numpy.dtype('int16'), 2)
    wt_z = dtypes.WrappedType(numpy.dtype('int32'), 4)
    ref = dtypes.WrappedType(
        dtype_aligned, 16, explicit_alignment=None, wrapped_fields=dict(x=wt_x, y=wt_y, z=wt_z),
        field_alignments=dict(x=None, y=4, z=16))
    assert res == ref


def test__align_aligned_struct_invalid_itemsize():
    dtype_aligned = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 2, 4],
        itemsize=20, # not a power of 2, an error should be raised
        aligned=True))

    with pytest.raises(ValueError):
        dtypes._align(dtype_aligned)


def test_align_nested():
    dtype_nested = numpy.dtype(dict(
        names=['val1', 'pad'],
        formats=[numpy.int8, numpy.int8]))

    dtype = numpy.dtype(dict(
        names=['pad', 'struct_arr', 'regular_arr'],
        formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))]))

    dtype_ref = numpy.dtype(dict(
        names=['pad','struct_arr','regular_arr'],
        formats=[numpy.int32, (dtype_nested, (2,)), (numpy.int16, (3,))],
        offsets=[0,4,8],
        itemsize=16))

    dtype_aligned = dtypes.align(dtype)

    assert dtype_aligned.isalignedstruct
    assert dtype_aligned == dtype_ref


def test_align_preserve_nested_aligned():

    dtype_int3 = numpy.dtype(dict(names=['x'], formats=[(numpy.int32, 3)], itemsize=16, aligned=True))

    dtype = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int32, dtype_int3, numpy.int32]))

    dtype_ref = numpy.dtype(dict(
        names=['x','y','z'],
        formats=[numpy.int32, dtype_int3, numpy.int32],
        offsets=[0,16,32],
        itemsize=48,
        aligned=True))

    dtype_aligned = dtypes.align(dtype)

    assert dtype_aligned.isalignedstruct
    assert dtype_aligned == dtype_ref


def test_lcm():
    assert dtypes._lcm(10) == 10
    assert dtypes._lcm(15, 20) == 60
    assert dtypes._lcm(16, 32, 24) == 96


def test_find_minimum_alignment():
    # simple case: base alignment is enough because 12 is the next multiple of 4 after 9
    assert dtypes._find_minimum_alignment(12, 4, 9) == 4
    # the next multiple of 4 is 12, but we want offset 16 - this means we need to set
    # the alignment equal to 8, because 16 is the next multiple of 8 after 9.
    assert dtypes._find_minimum_alignment(16, 4, 9) == 8

    # incorrect offset (not a multiple of the base alignment)
    with pytest.raises(ValueError):
        dtypes._find_minimum_alignment(13, 4, 9)

    # offset too large and not a power of 2 - cannot achieve that with alignment only,
    # will need explicit padding
    with pytest.raises(ValueError):
        dtypes._find_minimum_alignment(24, 4, 9)


def test_wrapped_type_repr():
    dtype_aligned = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 4, 16],
        itemsize=32,
        aligned=True))
    wt_x = dtypes.WrappedType(numpy.dtype('int8'), 1)
    wt_y = dtypes.WrappedType(numpy.dtype('int16'), 2)
    wt_z = dtypes.WrappedType(numpy.dtype('int32'), 4)
    wt = dtypes.WrappedType(
        dtype_aligned, 16, explicit_alignment=None, wrapped_fields=dict(x=wt_x, y=wt_y, z=wt_z),
        field_alignments=dict(x=None, y=4, z=16))

    assert eval(
        repr(wt),
        dict(
            numpy=numpy, WrappedType=dtypes.WrappedType,
            int8=numpy.int8, int16=numpy.int16, int32=numpy.int32)) == wt


def test_ctype_struct():
    dtype = dtypes.align(numpy.dtype([('val1', numpy.int32), ('val2', numpy.float32)]))
    ctype = dtypes.ctype(dtype)
    src = render_with_modules("${ctype}", render_globals=dict(ctype=ctype)).strip()

    assert src == (
        'typedef struct _mod__module_0__ {\n'
        '    int  val1;\n'
        '    float  val2;\n'
        '}  _mod__module_0_;\n\n\n'
        '_mod__module_0_')


def test_ctype_struct_nested():

    dtype_nested = numpy.dtype(dict(
        names=['val1', 'pad'],
        formats=[numpy.int8, numpy.int8]))

    dtype = numpy.dtype(dict(
        names=['pad', 'struct_arr', 'regular_arr'],
        formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))]))

    dtype = dtypes.align(dtype)
    ctype = dtypes.ctype(dtype)
    src = render_with_modules("${ctype}", render_globals=dict(ctype=ctype)).strip()

    assert src == (
        'typedef struct _mod__module_1__ {\n'
        '    char  val1;\n'
        '    char  pad;\n'
        '}  _mod__module_1_;\n\n\n'
        'typedef struct _mod__module_0__ {\n'
        '    int  pad;\n'
        '    _mod__module_1_  struct_arr[2];\n'
        '    short  regular_arr[3];\n'
        '}  _mod__module_0_;\n\n\n'
        '_mod__module_0_')


def test_ctype_to_ctype_struct():
    # Checks that ctype() on an unknown type calls ctype_struct()
    dtype = dtypes.align(numpy.dtype([('val1', numpy.int32), ('val2', numpy.float32)]))
    ctype = dtypes.ctype(dtype)
    src = render_with_modules("${ctype}", render_globals=dict(ctype=ctype)).strip()

    assert src == (
        'typedef struct _mod__module_0__ {\n'
        '    int  val1;\n'
        '    float  val2;\n'
        '}  _mod__module_0_;\n\n\n'
        '_mod__module_0_')


def test_ctype_struct():

    dtype = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 4, 16],
        itemsize=64,
        aligned=True))
    ctype = dtypes.ctype_struct(dtype)
    src = render_with_modules("${ctype}", render_globals=dict(ctype=ctype)).strip()
    assert src == (
        'typedef struct _mod__module_0__ {\n'
        '    char  x;\n'
        '    short ALIGN(4) y;\n'
        '    int ALIGN(16) z;\n'
        '} ALIGN(64) _mod__module_0_;\n\n\n'
        '_mod__module_0_')


def test_ctype_struct_ignore_alignment():

    dtype = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 4, 16],
        itemsize=64,
        aligned=True))
    ctype = dtypes.ctype_struct(dtype, ignore_alignment=True)
    src = render_with_modules("${ctype}", render_globals=dict(ctype=ctype)).strip()
    assert src == (
        'typedef struct _mod__module_0__ {\n'
        '    char  x;\n'
        '    short  y;\n'
        '    int  z;\n'
        '}  _mod__module_0_;\n\n\n'
        '_mod__module_0_')


def test_ctype_struct_checks_alignment():
    dtype = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32]))
    with pytest.raises(ValueError):
        dtypes.ctype_struct(dtype)


def test_ctype_struct_for_non_struct():
    dtype = numpy.dtype((numpy.int32, 3))
    with pytest.raises(ValueError):
        dtypes.ctype_struct(dtype)

    # ctype_struct() is not applicable for simple types
    with pytest.raises(ValueError):
        dtypes.ctype_struct(numpy.int32)


def test_flatten_dtype():
    dtype_nested = numpy.dtype(dict(
        names=['val1', 'pad'],
        formats=[numpy.int8, numpy.int8]))

    dtype = numpy.dtype(dict(
        names=['pad', 'struct_arr', 'regular_arr'],
        formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))]))

    res = dtypes.flatten_dtype(dtype)
    ref = [
        (['pad'], numpy.dtype('int32')),
        (['struct_arr', 0, 'val1'], numpy.dtype('int8')),
        (['struct_arr', 0, 'pad'], numpy.dtype('int8')),
        (['struct_arr', 1, 'val1'], numpy.dtype('int8')),
        (['struct_arr', 1, 'pad'], numpy.dtype('int8')),
        (['regular_arr', 0], numpy.dtype('int16')),
        (['regular_arr', 1], numpy.dtype('int16')),
        (['regular_arr', 2], numpy.dtype('int16'))]

    assert dtypes.flatten_dtype(dtype) == ref


def test_c_path():
    assert dtypes.c_path(['struct_arr', 0, 'val1']) == 'struct_arr[0].val1'


def test_extract_field():
    dtype_nested = numpy.dtype(dict(
        names=['val1', 'pad'],
        formats=[numpy.int8, numpy.int8]))

    dtype = numpy.dtype(dict(
        names=['pad', 'struct_arr', 'regular_arr'],
        formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))]))

    a = numpy.empty(16, dtype)
    a['struct_arr']['val1'][:,1] = numpy.arange(16)
    assert (dtypes.extract_field(a, ['struct_arr', 1, 'val1']) == numpy.arange(16)).all()

    b = numpy.empty(16, dtype_nested)
    b['val1'] = numpy.arange(16)
    assert (dtypes.extract_field(b, ['val1']) == numpy.arange(16)).all()

import re
from collections.abc import Mapping

import numpy
import pytest

from grunnur import dtypes
from grunnur._modules import render_with_modules
from grunnur.dtypes import FieldInfo, _WrappedType


def test_ctype_builtin() -> None:
    assert dtypes.ctype(numpy.int32) == "int"

    with pytest.raises(ValueError, match="object is not a built-in data type"):
        dtypes.ctype(numpy.object_)


def test_is_complex() -> None:
    assert dtypes.is_complex(numpy.complex64)
    assert dtypes.is_complex(numpy.complex128)
    assert not dtypes.is_complex(numpy.float64)


def test_is_double() -> None:
    assert dtypes.is_double(numpy.float64)
    assert dtypes.is_double(numpy.complex128)
    assert not dtypes.is_double(numpy.complex64)


def test_is_integer() -> None:
    assert dtypes.is_integer(numpy.int32)
    assert not dtypes.is_integer(numpy.float32)


def test_is_real() -> None:
    assert dtypes.is_real(numpy.float32)
    assert not dtypes.is_real(numpy.complex64)
    assert not dtypes.is_real(numpy.int32)


def test_promote_type() -> None:
    assert dtypes._promote_type(numpy.dtype("int8")) == numpy.int32
    assert dtypes._promote_type(numpy.dtype("uint8")) == numpy.uint32
    assert dtypes._promote_type(numpy.dtype("float16")) == numpy.float32
    assert dtypes._promote_type(numpy.dtype("csingle")) == numpy.complex64
    assert dtypes._promote_type(numpy.dtype("int32")) == numpy.int32
    assert dtypes._promote_type(numpy.dtype("int64")) == numpy.int64


def test_result_type() -> None:
    assert dtypes.result_type(numpy.int32, numpy.float32) == numpy.float64


def test_min_scalar_type() -> None:
    assert dtypes.min_scalar_type(1) == numpy.uint32
    assert dtypes.min_scalar_type(-1) == numpy.int32
    assert dtypes.min_scalar_type(1.0) == numpy.float32
    assert dtypes.min_scalar_type(1 + 2j) == numpy.complex64


def test_complex_for() -> None:
    assert dtypes.complex_for(numpy.float32) == numpy.complex64
    assert dtypes.complex_for(numpy.float64) == numpy.complex128
    with pytest.raises(ValueError, match="complex64 does not have a corresponding complex type"):
        assert dtypes.complex_for(numpy.complex64)
    with pytest.raises(ValueError, match="int32 does not have a corresponding complex type"):
        assert dtypes.complex_for(numpy.int32)


def test_real_for() -> None:
    assert dtypes.real_for(numpy.complex64) == numpy.float32
    assert dtypes.real_for(numpy.complex128) == numpy.float64
    with pytest.raises(ValueError, match="float32 does not have a corresponding real type"):
        assert dtypes.real_for(numpy.float32)
    with pytest.raises(ValueError, match="int32 does not have a corresponding real type"):
        assert dtypes.real_for(numpy.int32)


def test_complex_ctr() -> None:
    assert dtypes.complex_ctr(numpy.complex64) == "COMPLEX_CTR(float2)"


def test_c_constant() -> None:
    # scalar values
    assert dtypes.c_constant(1) == "1"
    assert dtypes.c_constant(numpy.uint64(1)) == "1UL"
    assert dtypes.c_constant(numpy.int64(-1)) == "-1L"
    assert dtypes.c_constant(numpy.float64(1.0)) == "1.0"
    assert dtypes.c_constant(numpy.float32(1.0)) == "1.0f"
    assert dtypes.c_constant(numpy.complex64(1 + 2j)) == "COMPLEX_CTR(float2)(1.0f, 2.0f)"
    assert dtypes.c_constant(numpy.complex128(1 + 2j)) == "COMPLEX_CTR(double2)(1.0, 2.0)"

    # array
    assert dtypes.c_constant(numpy.array([1, 2, 3], numpy.float32)) == "{1.0f, 2.0f, 3.0f}"

    # struct type
    dtype = numpy.dtype([("val1", numpy.int32), ("val2", numpy.float32)])
    val = numpy.empty((), dtype)
    val["val1"] = 1
    val["val2"] = 2
    assert dtypes.c_constant(val) == "{1, 2.0f}"

    # custom dtype
    assert dtypes.c_constant(1, numpy.float32) == "1.0f"

    message = r"Cannot render a value of type <class 'numpy.str_'> as a C constant"
    with pytest.raises(TypeError, match=message):
        dtypes.c_constant(numpy.array(["a", "b"]))


def test_align_builtin() -> None:
    dtype = numpy.dtype("int32")
    assert dtypes.align(dtype) == dtype


def test_wrap_builtin() -> None:
    dtype = numpy.dtype("int32")
    ref = _WrappedType(
        dtype=dtype,
        alignment=4,
        wrapped_fields={},
        explicit_alignment=None,
        explicit_field_alignments={},
    )
    assert _WrappedType.wrap(dtype) == ref


def test_align_array() -> None:
    dtype = numpy.dtype("int32")
    dtype_arr = numpy.dtype((dtype, 3))
    assert dtypes.align(dtype_arr) == dtype_arr


def test_wrap_array() -> None:
    dtype = numpy.dtype("int32")
    dtype_arr = numpy.dtype((dtype, 3))
    ref = _WrappedType(
        dtype=dtype_arr,
        alignment=4,
        wrapped_fields={},
        explicit_alignment=None,
        explicit_field_alignments={},
    )
    assert _WrappedType.wrap(dtype_arr) == ref


def test_align_sets_aligned_attribute() -> None:
    dtype = numpy.dtype(dict(names=["x", "y", "z"], formats=[numpy.int8, numpy.int16, numpy.int32]))
    res = dtypes.align(dtype)

    assert res.isalignedstruct
    assert isinstance(res.fields, Mapping)
    assert dict(res.fields) == dict(
        x=(numpy.dtype(numpy.int8), 0),
        y=(numpy.dtype(numpy.int16), 2),
        z=(numpy.dtype(numpy.int32), 4),
    )
    assert res.itemsize == 8
    assert res.alignment == 4


def test_wrap_aligned_struct() -> None:
    dtype = numpy.dtype(
        dict(
            names=["x", "y", "z"],
            formats=[numpy.int8, numpy.int16, numpy.int32],
            offsets=[0, 2, 4],
            itemsize=8,
            aligned=True,
        )
    )

    wt_x = _WrappedType.wrap(numpy.dtype("int8"))
    wt_y = _WrappedType.wrap(numpy.dtype("int16"))
    wt_z = _WrappedType.wrap(numpy.dtype("int32"))
    ref = _WrappedType(
        dtype=dtype,
        alignment=4,
        explicit_alignment=None,
        wrapped_fields=dict(x=wt_x, y=wt_y, z=wt_z),
        explicit_field_alignments=dict(x=None, y=None, z=None),
    )
    assert _WrappedType.wrap(dtype) == ref


def test_align_ignores_offsets_and_itemsize() -> None:
    dtype = numpy.dtype(
        dict(
            names=["x", "y", "z"],
            formats=[numpy.int8, numpy.int16, numpy.int32],
            offsets=[0, 4, 16],
            itemsize=32,
            aligned=True,
        )
    )

    # Aligning will ignore all existing offets and itemsizes,
    # so the itemsize will be reset to 8 and alignment to 4.
    res = dtypes.align(dtype)

    assert res.isalignedstruct
    assert isinstance(res.fields, Mapping)
    assert dict(res.fields) == dict(
        x=(numpy.dtype(numpy.int8), 0),
        y=(numpy.dtype(numpy.int16), 2),
        z=(numpy.dtype(numpy.int32), 4),
    )
    assert res.itemsize == 8
    assert res.alignment == 4


def test_wrap_custom_offsets_and_itemsize() -> None:
    dtype = numpy.dtype(
        dict(
            names=["x", "y", "z"],
            formats=[numpy.int8, numpy.int16, numpy.int32],
            offsets=[0, 4, 16],
            itemsize=32,
            aligned=True,
        )
    )

    wt_x = _WrappedType.wrap(numpy.dtype("int8"))
    wt_y = _WrappedType.wrap(numpy.dtype("int16"))
    wt_z = _WrappedType.wrap(numpy.dtype("int32"))

    # The offsets & itemsize are possible to achieve with alignments,
    # so this won't fail.
    res = _WrappedType.wrap(dtype)
    ref = _WrappedType(
        dtype=dtype,
        alignment=16,
        explicit_alignment=None,
        wrapped_fields=dict(x=wt_x, y=wt_y, z=wt_z),
        explicit_field_alignments=dict(x=None, y=4, z=16),
    )
    assert res == ref


def test_wrap_invalid_itemsize() -> None:
    dtype = numpy.dtype(
        dict(
            names=["x", "y", "z"],
            formats=[numpy.int8, numpy.int16, numpy.int32],
            offsets=[0, 2, 4],
            # This itemsize is not the implicit one based on the field sizes and alignments,
            # and is not a power of 2, so it cannot be caused by an explicit alignment.
            # An error should be raised.
            itemsize=20,
            aligned=True,
        )
    )

    with pytest.raises(
        ValueError, match="An itemsize that requires an explicit alignment must be a power of 2"
    ):
        _WrappedType.wrap(dtype)


def test_align_nested() -> None:
    dtype_nested = numpy.dtype(dict(names=["val1", "pad"], formats=[numpy.int8, numpy.int8]))

    dtype = numpy.dtype(
        dict(
            names=["pad", "struct_arr", "regular_arr"],
            formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))],
        )
    )

    dtype_ref = numpy.dtype(
        dict(
            names=["pad", "struct_arr", "regular_arr"],
            formats=[numpy.int32, (dtype_nested, (2,)), (numpy.int16, (3,))],
            offsets=[0, 4, 8],
            itemsize=16,
            aligned=True,
        )
    )

    dtype_aligned = dtypes.align(dtype)

    assert dtype_aligned == dtype_ref


def test_align_nested_ignores_itemsize() -> None:
    dtype_int3 = numpy.dtype(
        dict(names=["x"], formats=[(numpy.int32, 3)], itemsize=16, aligned=True)
    )
    dtype = numpy.dtype(
        dict(names=["x", "y", "z"], formats=[numpy.int32, dtype_int3, numpy.int32], aligned=True)
    )

    # Alignment will ignore the existing itemsize and pack the components tighter.
    dtype_int3_ref = numpy.dtype(
        dict(names=["x"], formats=[(numpy.int32, 3)], itemsize=12, aligned=True)
    )
    dtype_ref = numpy.dtype(
        dict(
            names=["x", "y", "z"],
            formats=[numpy.int32, dtype_int3_ref, numpy.int32],
            offsets=[0, 4, 16],
            itemsize=20,
            aligned=True,
        )
    )

    dtype_aligned = dtypes.align(dtype)

    assert dtype_aligned.isalignedstruct
    assert dtype_aligned == dtype_ref


def test_lcm() -> None:
    assert dtypes._lcm(10) == 10
    assert dtypes._lcm(15, 20) == 60
    assert dtypes._lcm(16, 32, 24) == 96


def test_find_minimum_alignment() -> None:
    # simple case: base alignment is enough because 12 is the next multiple of 4 after 9
    assert dtypes._find_minimum_alignment(12, 4, 9) == 4
    # the next multiple of 4 is 12, but we want offset 16 - this means we need to set
    # the alignment equal to 8, because 16 is the next multiple of 8 after 9.
    assert dtypes._find_minimum_alignment(16, 4, 9) == 8

    # incorrect offset (not a multiple of the base alignment)
    message = re.escape("Field offset (13) must be a multiple of the base alignment (4)")
    with pytest.raises(ValueError, match=message):
        dtypes._find_minimum_alignment(13, 4, 9)

    # offset too large and not a power of 2 - cannot achieve that with alignment only,
    # will need explicit padding
    message = (
        "Could not find a suitable alignment for the field at offset 24; "
        "consider adding explicit padding"
    )
    with pytest.raises(ValueError, match=message):
        dtypes._find_minimum_alignment(24, 4, 9)


def test_ctype_struct() -> None:
    dtype = dtypes.align(numpy.dtype([("val1", numpy.int32), ("val2", numpy.float32)]))
    ctype = dtypes.ctype(dtype)
    src = render_with_modules("${ctype}", render_globals=dict(ctype=ctype)).strip()

    assert src == (
        "typedef struct _mod__module_0__ {\n"
        "    int  val1;\n"
        "    float  val2;\n"
        "}  _mod__module_0_;\n\n\n"
        "_mod__module_0_"
    )


def test_ctype_struct_nested() -> None:
    dtype_nested = numpy.dtype(dict(names=["val1", "pad"], formats=[numpy.int8, numpy.int8]))

    dtype = numpy.dtype(
        dict(
            names=["pad", "struct_arr", "regular_arr"],
            formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))],
        )
    )

    dtype = dtypes.align(dtype)
    ctype = dtypes.ctype(dtype)
    src = render_with_modules("${ctype}", render_globals=dict(ctype=ctype)).strip()

    assert src == (
        "typedef struct _mod__module_1__ {\n"
        "    char  val1;\n"
        "    char  pad;\n"
        "}  _mod__module_1_;\n\n\n"
        "typedef struct _mod__module_0__ {\n"
        "    int  pad;\n"
        "    _mod__module_1_  struct_arr[2];\n"
        "    short  regular_arr[3];\n"
        "}  _mod__module_0_;\n\n\n"
        "_mod__module_0_"
    )


def test_ctype_struct_aligned() -> None:
    dtype = numpy.dtype(
        dict(
            names=["x", "y", "z"],
            formats=[numpy.int8, numpy.int16, numpy.int32],
            offsets=[0, 4, 16],
            itemsize=64,
            aligned=True,
        )
    )
    ctype = dtypes.ctype(dtype)
    src = render_with_modules("${ctype}", render_globals=dict(ctype=ctype)).strip()
    assert src == (
        "typedef struct _mod__module_0__ {\n"
        "    char  x;\n"
        "    short ALIGN(4) y;\n"
        "    int ALIGN(16) z;\n"
        "} ALIGN(64) _mod__module_0_;\n\n\n"
        "_mod__module_0_"
    )


def test_ctype_checks_alignment() -> None:
    dtype = numpy.dtype(dict(names=["x", "y", "z"], formats=[numpy.int8, numpy.int16, numpy.int32]))
    message = re.escape(
        "Failed to find alignment for field `y`: "
        "Field offset (1) must be a multiple of the base alignment (2)."
    )
    with pytest.raises(ValueError, match=message):
        dtypes.ctype(dtype)


def test_ctype_for_array() -> None:
    dtype = numpy.dtype((numpy.int32, 3))
    with pytest.raises(ValueError, match="The data type cannot be an array"):
        dtypes.ctype(dtype)


def test_flatten_dtype() -> None:
    dtype_nested = numpy.dtype(dict(names=["val1", "pad"], formats=[numpy.int8, numpy.int8]))

    dtype = numpy.dtype(
        dict(
            names=["pad", "struct_arr", "regular_arr"],
            formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))],
        )
    )

    res = dtypes.flatten_dtype(dtype)
    ref = [
        FieldInfo(path=["pad"], dtype=numpy.dtype("int32"), offset=0),
        FieldInfo(path=["struct_arr", 0, "val1"], dtype=numpy.dtype("int8"), offset=4),
        FieldInfo(path=["struct_arr", 0, "pad"], dtype=numpy.dtype("int8"), offset=5),
        FieldInfo(path=["struct_arr", 1, "val1"], dtype=numpy.dtype("int8"), offset=6),
        FieldInfo(path=["struct_arr", 1, "pad"], dtype=numpy.dtype("int8"), offset=7),
        FieldInfo(path=["regular_arr", 0], dtype=numpy.dtype("int16"), offset=8),
        FieldInfo(path=["regular_arr", 1], dtype=numpy.dtype("int16"), offset=10),
        FieldInfo(path=["regular_arr", 2], dtype=numpy.dtype("int16"), offset=12),
    ]

    assert res == ref


def test_c_path() -> None:
    field_info = FieldInfo(path=["struct_arr", 0, "val1"], dtype=numpy.dtype(numpy.int8), offset=0)
    assert field_info.c_path == ".struct_arr[0].val1"


def test_c_path_primitive_type() -> None:
    flat_dtype = dtypes.flatten_dtype(numpy.int32)
    assert flat_dtype[0].c_path == ""


def test_extract_field() -> None:
    dtype_nested = numpy.dtype(dict(names=["val1", "pad"], formats=[numpy.int8, numpy.int8]))

    dtype = numpy.dtype(
        dict(
            names=["pad", "struct_arr", "regular_arr"],
            formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))],
        )
    )

    a = numpy.empty(16, dtype)
    a["struct_arr"]["val1"][:, 1] = numpy.arange(16)
    assert (dtypes.extract_field(a, ["struct_arr", 1, "val1"]) == numpy.arange(16)).all()

    b = numpy.empty(16, dtype_nested)
    b["val1"] = numpy.arange(16)
    assert (dtypes.extract_field(b, ["val1"]) == numpy.arange(16)).all()

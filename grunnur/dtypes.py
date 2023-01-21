from __future__ import annotations

import itertools
from math import gcd
import platform
from typing import (
    NamedTuple,
    Callable,
    Any,
    Sequence,
    Optional,
    Tuple,
    Dict,
    Iterable,
    Union,
    List,
    Type,
    TypeVar,
    Mapping,
    overload,
    cast,
)

import numpy
from numpy.typing import DTypeLike

from .utils import bounding_power_of_2, log2, min_blocks
from .modules import Module


_DTYPE_TO_BUILTIN_CTYPE = {
    numpy.dtype("bool"): "bool",
    numpy.dtype("int8"): "char",
    numpy.dtype("uint8"): "unsigned char",
    numpy.dtype("int16"): "short",
    numpy.dtype("uint16"): "unsigned short",
    numpy.dtype("int32"): "int",
    numpy.dtype("uint32"): "unsigned int",
    numpy.dtype("float32"): "float",
    numpy.dtype("float64"): "double",
    numpy.dtype("complex64"): "float2",
    numpy.dtype("complex128"): "double2",
    numpy.dtype("int64"): "long long" if platform.system() == "Windows" else "long",
    numpy.dtype("uint64"): "unsigned "
    + ("long long" if platform.system() == "Windows" else "long"),
}


def _ctype_builtin(dtype: DTypeLike) -> str:
    dtype = _normalize_type(dtype)
    if dtype in _DTYPE_TO_BUILTIN_CTYPE:
        return _DTYPE_TO_BUILTIN_CTYPE[dtype]
    raise ValueError(f"{dtype} is not a built-in data type")


def ctype(dtype: DTypeLike) -> Union[str, Module]:
    """
    Returns an object that can be passed as a global to :py:meth:`~grunnur.Program`
    and used to render a C equivalent of the given ``numpy`` dtype.
    If there is a built-in C equivalent, the object is just a string with the type name;
    otherwise it is a :py:class:`~grunnur.Module` object containing
    the corresponding ``struct`` declaration.

    .. note::

        If ``dtype`` is a struct type, it needs to be aligned
        (see :py:func:`ctype_struct` and :py:func:`align`).
    """
    dtype = _normalize_type(dtype)
    try:
        return _ctype_builtin(dtype)
    except ValueError:
        return ctype_struct(dtype)


def _normalize_type(dtype: DTypeLike) -> "numpy.dtype[Any]":
    """
    Numpy's dtype shortcuts (e.g. ``numpy.int32``) are ``type`` objects
    and have slightly different properties from actual ``numpy.dtype`` objects.
    This function converts the former to ``numpy.dtype`` and keeps the latter unchanged.
    """
    return numpy.dtype(dtype)


def is_complex(dtype: DTypeLike) -> bool:
    """
    Returns ``True`` if ``dtype`` is complex.
    """
    dtype = _normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.complexfloating)


def is_double(dtype: DTypeLike) -> bool:
    """
    Returns ``True`` if ``dtype`` is double precision floating point.
    """
    dtype = _normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.float_) or numpy.issubdtype(dtype, numpy.complex_)


def is_integer(dtype: DTypeLike) -> bool:
    """
    Returns ``True`` if ``dtype`` is an integer.
    """
    dtype = _normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.integer)


def is_real(dtype: DTypeLike) -> bool:
    """
    Returns ``True`` if ``dtype`` is a real number (but not complex).
    """
    dtype = _normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.floating)


def _promote_type(dtype: "numpy.dtype[Any]") -> "numpy.dtype[Any]":
    # not all numpy datatypes are supported by GPU, so we may need to promote
    if issubclass(dtype.type, numpy.signedinteger) and dtype.itemsize < 4:
        return numpy.dtype("int32")
    elif issubclass(dtype.type, numpy.unsignedinteger) and dtype.itemsize < 4:
        return numpy.dtype("uint32")
    elif issubclass(dtype.type, numpy.floating) and dtype.itemsize < 4:
        return numpy.dtype("float32")
    # Not checking complex dtypes since there is nothing smaller than complex32, which is supported
    return dtype


def result_type(*dtypes: DTypeLike) -> "numpy.dtype[Any]":
    """
    Wrapper for :py:func:`numpy.result_type`
    which takes into account types supported by GPUs.
    """
    return _promote_type(numpy.result_type(*dtypes))


def min_scalar_type(
    val: Union[int, float, complex, "numpy.number[Any]"], force_signed: bool = False
) -> "numpy.dtype[Any]":
    """
    Wrapper for :py:func:`numpy.min_scalar_type`
    which takes into account types supported by GPUs.
    """
    if isinstance(val, numpy.number):
        dtype = val.dtype
    else:
        dtype = numpy.min_scalar_type(val)
    return _promote_type(dtype)


def complex_for(dtype: DTypeLike) -> "numpy.dtype[Any]":
    """
    Returns complex dtype corresponding to given floating point ``dtype``.
    """
    dtype = _normalize_type(dtype)
    if dtype == numpy.float32:
        return numpy.dtype("complex64")
    if dtype == numpy.float64:
        return numpy.dtype("complex128")
    raise ValueError(f"{dtype} does not have a corresponding complex type")


def real_for(dtype: DTypeLike) -> "numpy.dtype[Any]":
    """
    Returns floating point dtype corresponding to given complex ``dtype``.
    """
    dtype = _normalize_type(dtype)
    if dtype == numpy.complex64:
        return numpy.dtype("float32")
    if dtype == numpy.complex128:
        return numpy.dtype("float64")
    raise ValueError(f"{dtype} does not have a corresponding real type")


def complex_ctr(dtype: DTypeLike) -> str:
    """
    Returns name of the constructor for the given ``dtype``.
    """
    return "COMPLEX_CTR(" + _ctype_builtin(dtype) + ")"


def _c_constant_arr(val: Any, shape: Sequence[int]) -> str:
    if len(shape) == 0:
        return c_constant(val)
    return "{" + ", ".join(_c_constant_arr(val[i], shape[1:]) for i in range(shape[0])) + "}"


def c_constant(
    val: Union[int, float, complex, numpy.generic, "numpy.ndarray[Any, numpy.dtype[Any]]"],
    dtype: Optional[DTypeLike] = None,
) -> str:
    """
    Returns a C-style numerical constant.
    If ``val`` has a struct dtype, the generated constant will have the form ``{ ... }``
    and can be used as an initializer for a variable.
    """
    if dtype is not None:
        dtype = _promote_type(_normalize_type(dtype))
    elif isinstance(val, (int, float, complex)):
        dtype = min_scalar_type(val)
    else:
        dtype = _promote_type(val.dtype)

    numpy_val: Union[numpy.generic, "numpy.ndarray[Any, numpy.dtype[Any]]"]
    if isinstance(val, numpy.ndarray):
        numpy_val = numpy.cast[dtype](val)
    else:
        numpy_val = numpy.cast[dtype](val).flat[0]

    if len(numpy_val.shape) > 0:
        return _c_constant_arr(numpy_val, numpy_val.shape)

    scalar_val: numpy.generic
    if isinstance(numpy_val, numpy.ndarray):
        scalar_val = numpy_val.flat[0]
    else:
        scalar_val = numpy_val

    if isinstance(scalar_val, numpy.void) and scalar_val.dtype.names is not None:
        return (
            "{" + ", ".join([c_constant(scalar_val[name]) for name in scalar_val.dtype.names]) + "}"
        )

    if isinstance(scalar_val, numpy.complexfloating):
        return (
            f"COMPLEX_CTR({_ctype_builtin(dtype)})"
            + f"({c_constant(scalar_val.real)}, {c_constant(scalar_val.imag)})"
        )

    if isinstance(scalar_val, numpy.integer):
        if dtype.itemsize > 4:
            postfix = "L" if numpy.issubdtype(scalar_val.dtype, numpy.signedinteger) else "UL"
        else:
            postfix = ""
        return str(scalar_val) + postfix

    if isinstance(scalar_val, numpy.floating):
        return repr(float(scalar_val)) + ("f" if scalar_val.dtype.itemsize <= 4 else "")

    raise TypeError(f"Cannot render a value of type {type(val)} as a C constant")


def _struct_alignment(alignments: Iterable[int]) -> int:
    """
    Returns the minimum alignment for a structure given alignments for its fields.
    According to the C standard, it the lowest common multiple of the alignments
    of all of the members of the struct rounded up to the nearest power of two.
    """
    return bounding_power_of_2(_lcm(*alignments))


def _find_minimum_alignment(offset: int, base_alignment: int, prev_end: int) -> int:
    """
    Returns the minimum alignment that must be set for a field with
    ``base_alignment`` (the one inherent to the type),
    so that the compiler positioned it at ``offset`` given that the previous field
    ends at the position ``prev_end``.
    """
    # Essentially, we need to find the minimum k such that:
    # 1) offset = m * base_alignment * 2**k, where m > 0 and k >= 0;
    #    (by definition of alignment)
    # 2) offset - prev_offset < base_alignment * 2**k
    #    (otherwise the compiler can just as well take m' = m - 1).
    if offset % base_alignment != 0:
        raise ValueError(
            f"Field offset ({offset}) must be a multiple of the base alignment ({base_alignment})."
        )

    alignment = base_alignment
    while offset % alignment == 0:
        if offset - prev_end < alignment:
            return alignment

        alignment *= 2

    raise ValueError(
        f"Could not find a suitable alignment for the field at offset {offset}; "
        "consider adding explicit padding."
    )


class WrappedType(NamedTuple):
    """
    Contains an accompanying information for an aligned dtype.
    """

    dtype: "numpy.dtype[Any]"

    # This type's alignment
    alignment: int

    # An in integer if the type's alignment requires
    # an explicit specification in the C definition,
    # None otherwise.
    explicit_alignment: Optional[int]

    # A dictionary of `WrappedType` object for this dtype's fields.
    wrapped_fields: Dict[str, "WrappedType"]

    # A dictionary of alignments for this dtype's fields;
    # similarly to `explicit_alignment`, a value is an integer if the alignment
    # has to be set explicitly, None otherwise.
    field_alignments: Dict[str, Optional[int]]

    @classmethod
    def non_struct(cls, dtype: "numpy.dtype[Any]", alignment: int) -> "WrappedType":
        return cls(dtype, alignment, None, {}, {})


def _align(dtype: "numpy.dtype[Any]") -> WrappedType:
    """
    Builds a `WrappedType` object with the alignment information for a dtype,
    aligning it if it is not aligned, and checking the consistency of metadata if it is.
    """

    if len(dtype.shape) > 0:
        wt = _align(dtype.base)
        return WrappedType(
            numpy.dtype((wt.dtype, dtype.shape)),
            wt.alignment,
            explicit_alignment=wt.explicit_alignment,
            wrapped_fields=wt.wrapped_fields,
            field_alignments={},
        )

    if dtype.names is None:
        return WrappedType.non_struct(dtype, dtype.itemsize)

    # Since `.names` is not `None` at this point, we can restrict the type to help the inference
    dtype_fields = cast(Mapping[str, Tuple["numpy.dtype[Any]", int]], dtype.fields)

    wrapped_fields = {name: _align(dtype_fields[name][0]) for name in dtype.names}

    if dtype.isalignedstruct:
        # Find out what alignment has to be set for the field in order for the compiler
        # to place it at the offset specified in the description of `dtype`.
        field_alignments = [wrapped_fields[dtype.names[0]].alignment]
        for i in range(1, len(dtype.names)):
            prev_field_dtype, prev_offset = dtype_fields[dtype.names[i - 1]]
            _, offset = dtype_fields[dtype.names[i]]
            prev_end = prev_offset + prev_field_dtype.itemsize
            field_alignment = _find_minimum_alignment(
                offset, wrapped_fields[dtype.names[i]].alignment, prev_end
            )
            field_alignments.append(field_alignment)

        offsets = [dtype_fields[name][1] for name in dtype.names]
    else:
        # Build offsets for the structure using a procedure
        # similar to the one a compiler would use
        offsets = [0]
        for i in range(1, len(dtype.names)):
            prev_field_dtype, _ = dtype_fields[dtype.names[i - 1]]
            prev_end = offsets[-1] + prev_field_dtype.itemsize
            alignment = wrapped_fields[dtype.names[i]].alignment
            offsets.append(min_blocks(prev_end, alignment) * alignment)

        field_alignments = [wrapped_fields[name].alignment for name in dtype.names]

    # Same principle as above, but for the whole struct:
    # find out what alignment has to be set in order for the compiler
    # to place the next field at some dtype where this struct is a field type
    # at the offset corresponding to this struct's itemsize.

    last_dtype, _ = dtype_fields[dtype.names[-1]]
    last_offset = offsets[-1]
    struct_end = last_offset + last_dtype.itemsize

    # Find the total itemsize.
    # According to the standard, it must be a multiple of the struct alignment.
    base_struct_alignment = _struct_alignment(field_alignments)
    itemsize = min_blocks(struct_end, base_struct_alignment) * base_struct_alignment
    if dtype.isalignedstruct:
        if 2 ** log2(dtype.itemsize) != dtype.itemsize:
            raise ValueError(
                f"Invalid non-default itemsize for dtype {dtype}: "
                f"must be a power of 2 (currently {dtype.itemsize})"
            )

        # Should be already checked by `numpy.dtype` when an aligned struct was created.
        # Checking it just in case the behavior changes.
        assert dtype.itemsize >= itemsize

        aligned_dtype = dtype
        if dtype.itemsize > itemsize:
            struct_alignment = dtype.itemsize
        else:
            struct_alignment = base_struct_alignment
    else:
        # Must be some problems with numpy stubs - the type is too restrictive here.
        aligned_dtype = numpy.dtype(
            dict(
                names=dtype.names,
                formats=[wrapped_fields[name].dtype for name in dtype.names],
                offsets=offsets,
                itemsize=itemsize,
                aligned=True,
            )
        )

        struct_alignment = _find_minimum_alignment(itemsize, base_struct_alignment, struct_end)

    field_alignments_map = {
        dtype.names[i]: field_alignments[i]
        if field_alignments[i] != wrapped_fields[dtype.names[i]].alignment
        else None
        for i in range(len(dtype.names))
    }

    return WrappedType(
        aligned_dtype,
        struct_alignment,
        explicit_alignment=struct_alignment if struct_alignment != base_struct_alignment else None,
        wrapped_fields=wrapped_fields,
        field_alignments=field_alignments_map,
    )


def align(dtype: DTypeLike) -> "numpy.dtype[Any]":
    """
    Returns a new struct dtype with the field offsets changed to the ones a compiler would use
    (without being given any explicit alignment qualifiers).
    Ignores all existing explicit itemsizes and offsets.
    """
    dtype = _normalize_type(dtype)
    wrapped_dtype = _align(dtype)
    return wrapped_dtype.dtype


def _lcm(*nums: int) -> int:
    """
    Returns the least common multiple of ``nums``.
    """
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * nums[1] // gcd(nums[0], nums[1])
    return _lcm(nums[0], _lcm(*nums[1:]))


def _alignment_str(alignment: Optional[int]) -> str:
    if alignment is not None:
        return "ALIGN(" + str(alignment) + ")"
    return ""


def _get_struct_module(dtype: "numpy.dtype[Any]", ignore_alignment: bool = False) -> Module:
    """
    Builds and returns a module with the C type definition for a given ``dtype``,
    possibly using modules for nested structures.
    """

    # `dtype.names` is not `None` at this point, restricting types
    dtype_names = cast(Iterable[str], dtype.names)
    dtype_fields = cast(Mapping[str, Tuple["numpy.dtype[Any]", int]], dtype.fields)

    field_alignments: Dict[str, Optional[int]]
    if ignore_alignment:
        struct_alignment = None
        field_alignments = {name: None for name in dtype_names}
    else:
        wrapped_type = _align(dtype)
        struct_alignment = wrapped_type.explicit_alignment
        field_alignments = {name: wrapped_type.field_alignments[name] for name in dtype_names}

    # The tag (${prefix}_) is not necessary, but it helps to avoid
    # CUDA bug #1409907 (nested struct initialization like
    # "mystruct x = {0, {0, 0}, 0};" fails to compile)
    lines = ["typedef struct ${prefix}_ {"]
    kwds: Dict[str, Union[str, Module]] = {}
    for name in dtype_names:
        elem_dtype, _ = dtype_fields[name]

        base_elem_dtype = elem_dtype.base
        elem_dtype_shape = elem_dtype.shape

        array_suffix = "".join(f"[{d}]" for d in elem_dtype_shape)

        typename_var = "typename_" + name
        field_alignment = field_alignments[name]
        lines.append(
            f"    ${{{typename_var}}} {_alignment_str(field_alignment)} {name}{array_suffix};"
        )

        if base_elem_dtype.names is None:
            kwds[typename_var] = _ctype_builtin(base_elem_dtype)
        else:
            kwds[typename_var] = ctype_struct(base_elem_dtype, ignore_alignment=ignore_alignment)

    lines.append("} " + _alignment_str(struct_alignment) + " ${prefix};")

    return Module.from_string("\n".join(lines), render_globals=kwds)


def ctype_struct(dtype: DTypeLike, ignore_alignment: bool = False) -> Module:
    """
    For a struct type, returns a :py:class:`~grunnur.Module` object
    with the ``typedef`` of a struct corresponding to the given ``dtype``
    (with its name set to the module prefix).

    The structure definition includes the alignment required
    to produce field offsets specified in ``dtype``;
    therefore, ``dtype`` must be either a simple type, or have
    proper offsets and dtypes (the ones that can be reporoduced in C
    using explicit alignment attributes, but without additional padding)
    and the attribute ``isalignedstruct == True``.
    An aligned dtype can be produced either by standard means
    (``aligned`` flag in ``numpy.dtype`` constructor and explicit offsets and itemsizes),
    or created out of an arbitrary dtype with the help of :py:func:`~grunnur.dtypes.align`.

    If ``ignore_alignment`` is True, all of the above is ignored.
    The C structures produced will not have any explicit alignment modifiers.
    As a result, the the field offsets of ``dtype`` may differ from the ones
    chosen by the compiler.

    Modules are cached, and the function returns a single module instance for equal ``dtype``'s.
    Therefore inside a kernel it will be rendered with the same prefix everywhere it is used.
    This results in a behavior characteristic for a structural type system,
    same as for the basic dtype-ctype conversion.

    .. warning::

        As of ``numpy`` 1.8, the ``isalignedstruct`` attribute is not enough to ensure
        a mapping between a dtype and a C struct with only the fields that are present in the dtype.
        Therefore, ``ctype_struct`` will make some additional checks and raise ``ValueError``
        if it is not the case.
    """
    dtype = _normalize_type(dtype)

    if len(dtype.shape) > 0:
        raise ValueError("The data type cannot be an array")
    if dtype.names is None:
        raise ValueError("The data type must be a structure")

    # Note that numpy's ``isalignedstruct`` relies on hidden padding fields,
    # and may not mean that the returned C representation actually corresponds to the
    # ``numpy`` dtype.
    # There will be more checking in ``_align()`` which will fail in that case.
    if not ignore_alignment and not dtype.isalignedstruct:
        raise ValueError("The data type must be an aligned struct")

    return _get_struct_module(dtype, ignore_alignment=ignore_alignment)


def _flatten_dtype(
    dtype: "numpy.dtype[Any]", prefix: List[Union[str, int]] = []
) -> List[Tuple[List[Union[str, int]], numpy.dtype[Any]]]:

    if dtype.names is None:
        return [(prefix, dtype)]

    # `dtype.names` is not `None` at this point, restricting types
    dtype_fields = cast(Mapping[str, Tuple["numpy.dtype[Any]", int]], dtype.fields)

    result: List[Tuple[List[Union[str, int]], numpy.dtype[Any]]] = []
    for name in dtype.names:
        elem_dtype, _ = dtype_fields[name]

        elem_dtype_shape: Tuple[int, ...]
        if len(elem_dtype.shape) == 0:
            base_elem_dtype = elem_dtype
            elem_dtype_shape = tuple()
        else:
            base_elem_dtype = elem_dtype.base
            elem_dtype_shape = elem_dtype.shape

        if len(elem_dtype_shape) == 0:
            result += _flatten_dtype(base_elem_dtype, prefix=prefix + [name])
        else:
            for idxs in itertools.product(*[range(dim) for dim in elem_dtype_shape]):
                result += _flatten_dtype(base_elem_dtype, prefix=prefix + [name] + list(idxs))
    return result


def flatten_dtype(
    dtype: DTypeLike,
) -> List[Tuple[List[Union[str, int]], numpy.dtype[Any]]]:
    """
    Returns a list of tuples ``(path, dtype)`` for each of the basic dtypes in
    a (possibly nested) ``dtype``.
    ``path`` is a list of field names/array indices leading to the corresponding element.
    """
    dtype = _normalize_type(dtype)
    return _flatten_dtype(dtype)


def c_path(path: List[Union[str, int]]) -> str:
    """
    Returns a string corresponding to the ``path`` to a struct element in C.
    The ``path`` is the sequence of field names/array indices returned from
    :py:func:`~grunnur.dtypes.flatten_dtype`.
    """
    res = "".join(
        (("." + elem) if isinstance(elem, str) else ("[" + str(elem) + "]")) for elem in path
    )
    return res[1:]  # drop the first dot


def _extract_field(
    arr: "numpy.ndarray[Any, numpy.dtype[Any]]", path: List[Union[str, int]], array_idxs: List[int]
) -> Union[numpy.generic, "numpy.ndarray[Any, numpy.dtype[Any]]"]:
    """
    A helper function for ``extract_field``.
    Need to collect array indices for dtype sub-array fields since they are attached to the end
    of the full array index.
    """
    if len(path) == 0:
        if len(array_idxs) == 0:
            return arr
        numpy_array_indices: List[Union[slice, int]] = [slice(None, None, None)] * (
            len(arr.shape) - len(array_idxs)
        )
        struct_array_indices: List[Union[slice, int]] = list(array_idxs)
        slices = tuple(numpy_array_indices + struct_array_indices)
        return arr[slices]

    if isinstance(path[0], str):
        return _extract_field(arr[path[0]], path[1:], array_idxs)

    return _extract_field(arr, path[1:], array_idxs + [path[0]])


def extract_field(
    arr: "numpy.ndarray[Any, numpy.dtype[Any]]", path: List[Union[str, int]]
) -> Union[numpy.generic, "numpy.ndarray[Any, numpy.dtype[Any]]"]:
    """
    Extracts an element from an array of struct dtype.
    The ``path`` is the sequence of field names/array indices returned from
    :py:func:`~grunnur.dtypes.flatten_dtype`.
    """
    return _extract_field(arr, path, [])

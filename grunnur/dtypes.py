from __future__ import annotations

import itertools
from math import gcd
import platform
from typing import (
    Callable, Any, Sequence, Optional, Tuple, Dict, Iterable, Union, List, Type, Mapping)
from typing import cast as typing_cast

import numpy

from .utils import bounding_power_of_2, log2, min_blocks
from .modules import Module


_DTYPE_TO_BUILTIN_CTYPE = {
    numpy.dtype('bool'): "bool",
    numpy.dtype('int8'): "char",
    numpy.dtype('uint8'): "unsigned char",
    numpy.dtype('int16'): "short",
    numpy.dtype('uint16'): "unsigned short",
    numpy.dtype('int32'): "int",
    numpy.dtype('uint32'): "unsigned int",
    numpy.dtype('float32'): "float",
    numpy.dtype('float64'): "double",
    numpy.dtype('complex64'): "float2",
    numpy.dtype('complex128'): "double2",
    numpy.dtype('int64'): "long long" if platform.system() == 'Windows' else "long",
    numpy.dtype('uint64'):
        "unsigned " + ("long long" if platform.system() == 'Windows' else "long"),
    }


def ctype_builtin(dtype: numpy.dtype) -> str:
    dtype = normalize_type(dtype)
    if dtype in _DTYPE_TO_BUILTIN_CTYPE:
        return _DTYPE_TO_BUILTIN_CTYPE[dtype]
    raise ValueError(f"{dtype} is not a built-in data type")


def ctype(dtype: numpy.dtype) -> Union[str, Module]:
    """
    Returns an object that can be passed as a global to :py:meth:`~grunnur.Program`
    and used to render a C equivalent of the given ``numpy`` dtype.
    If there is a built-in C equivalent, the object is just a string with the type name;
    otherwise it is a :py:class:`~grunnur.Module` object containing
    the corresponding ``struct`` declaration.

    .. note::

        If ``dtype`` is a struct type, it needs to be aligned
        (see :py:func:`ctype_struct` and :py:func:`align`).

    :param dtype:
    """
    dtype = normalize_type(dtype)
    try:
        return ctype_builtin(dtype)
    except ValueError:
        return ctype_struct(dtype)


def normalize_type(dtype: Union[Type, numpy.dtype]) -> numpy.dtype:
    """
    Numpy's dtype shortcuts (e.g. ``numpy.int32``) are ``type`` objects
    and have slightly different properties from actual ``numpy.dtype`` objects.
    This function converts the former to ``numpy.dtype`` and keeps the latter unchanged.

    :param dtype:
    """
    return numpy.dtype(dtype)


def is_complex(dtype: numpy.dtype) -> bool:
    """
    Returns ``True`` if ``dtype`` is complex.

    :param dtype:
    """
    dtype = normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.complexfloating)


def is_double(dtype: numpy.dtype) -> bool:
    """
    Returns ``True`` if ``dtype`` is double precision floating point.

    :param dtype:
    """
    dtype = normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.float_) or numpy.issubdtype(dtype, numpy.complex_)


def is_integer(dtype: numpy.dtype) -> bool:
    """
    Returns ``True`` if ``dtype`` is an integer.

    :param dtype:
    """
    dtype = normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.integer)


def is_real(dtype: numpy.dtype) -> bool:
    """
    Returns ``True`` if ``dtype`` is a real number (but not complex).

    :param dtype:
    """
    dtype = normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.floating)


def _promote_type(dtype: numpy.dtype) -> numpy.dtype:
    # not all numpy datatypes are supported by GPU, so we may need to promote
    dtype = normalize_type(dtype)
    if dtype.kind == 'i' and dtype.itemsize < 4:
        dtype = numpy.dtype('int32')
    elif dtype.kind == 'u' and dtype.itemsize < 4:
        dtype = numpy.dtype('uint32')
    elif dtype.kind == 'f' and dtype.itemsize < 4:
        dtype = numpy.dtype('float32')
    return dtype


def result_type(*dtypes: numpy.dtype) -> numpy.dtype:
    """
    Wrapper for ``numpy.result_type()``
    which takes into account types supported by GPUs.

    :param dtypes:
    """
    return _promote_type(numpy.result_type(*dtypes))


def min_scalar_type(val, force_signed: bool=False) -> numpy.dtype:
    """
    Wrapper for ``numpy.min_scalar_dtype()``
    which takes into account types supported by GPUs.

    If ``force_signed`` is ``True``, a signed type will be returned even if ``val`` is positive.
    """
    if force_signed and val >= 0:
        # Signed integer range has one extra element on the negative side.
        # So if val=2^31, min_scalar_type(-2^31)=int32, but 2^31 will not fit in it.
        # Therefore we're forcing a larger type by subtracting 1.
        dtype = numpy.min_scalar_type(-val-1)
    else:
        dtype = numpy.min_scalar_type(val)
    return _promote_type(dtype)


def detect_type(val) -> numpy.dtype:
    """
    Returns the data type of ``val``.
    """
    if hasattr(val, 'dtype'):
        return _promote_type(val.dtype)
    return min_scalar_type(val)


def complex_for(dtype: numpy.dtype) -> numpy.dtype:
    """
    Returns complex dtype corresponding to given floating point ``dtype``.

    :param dtype:
    """
    dtype = normalize_type(dtype)
    if dtype == numpy.float32:
        return numpy.dtype('complex64')
    if dtype == numpy.float64:
        return numpy.dtype('complex128')
    raise ValueError(f"{dtype} does not have a corresponding complex type")


def real_for(dtype: numpy.dtype) -> numpy.dtype:
    """
    Returns floating point dtype corresponding to given complex ``dtype``.

    :param dtype:
    """
    dtype = normalize_type(dtype)
    if dtype == numpy.complex64:
        return numpy.dtype('float32')
    if dtype == numpy.complex128:
        return numpy.dtype('float64')
    raise ValueError(f"{dtype} does not have a corresponding real type")


def complex_ctr(dtype: numpy.dtype) -> str:
    """
    Returns name of the constructor for the given ``dtype``.

    :param dtype:
    """
    return 'COMPLEX_CTR(' + ctype_builtin(dtype) + ')'


def cast(dtype: numpy.dtype) -> Callable[[Any], Any]:
    """
    Returns function that takes one argument and casts it to ``dtype``.

    :param dtype:
    """
    def _cast(val) -> numpy.dtype:
        # Numpy cannot handle casts to struct dtypes (#4148),
        # so we're avoiding unnecessary casts.
        if not hasattr(val, 'dtype'):
            # A non-numpy scalar
            return numpy.array([val], dtype)[0]
        if val.dtype != dtype:
            return numpy.cast[dtype](val)
        return val

    return _cast


def _c_constant_arr(val, shape: Sequence[int]) -> str:
    if len(shape) == 0:
        return c_constant(val)
    return "{" + ", ".join(_c_constant_arr(val[i], shape[1:]) for i in range(shape[0])) + "}"


def c_constant(val, dtype: Optional[numpy.dtype]=None) -> str:
    """
    Returns a C-style numerical constant.
    If ``val`` has a struct dtype, the generated constant will have the form ``{ ... }``
    and can be used as an initializer for a variable.

    :param val:
    :param dtype:
    """
    if dtype is None:
        dtype = detect_type(val)
    else:
        dtype = normalize_type(dtype)

    val = cast(dtype)(val)

    if len(val.shape) > 0:
        return _c_constant_arr(val, val.shape)
    if dtype.names is not None:
        return "{" + ", ".join([c_constant(val[name]) for name in dtype.names]) + "}"

    if is_complex(dtype):
        return "COMPLEX_CTR(" + ctype_builtin(dtype) + ")(" + \
            c_constant(val.real) + ", " + c_constant(val.imag) + ")"

    if is_integer(dtype):
        if dtype.itemsize > 4:
            postfix = "L" if numpy.issubdtype(dtype, numpy.signedinteger) else "UL"
        else:
            postfix = ""
        return str(val) + postfix

    return repr(float(val)) + ("f" if dtype.itemsize <= 4 else "")


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
            f"Field offset ({offset}) must be a multiple of the base alignment ({base_alignment}).")

    alignment = base_alignment
    while offset % alignment == 0:
        if offset - prev_end < alignment:
            return alignment

        alignment *= 2

    raise ValueError(
        f"Could not find a suitable alignment for the field at offset {offset}; "
        "consider adding explicit padding.")


class WrappedType:
    """
    Contains an accompanying information for an aligned dtype.
    """

    def __init__(
            self, dtype: numpy.dtype, alignment: int,
            explicit_alignment: Optional[int]=None,
            wrapped_fields: Dict[str, WrappedType]={},
            field_alignments: Dict[str, Optional[int]]={}):
        self.dtype = dtype

        # This type's alignment
        self.alignment = alignment

        # An in integer if the type's alignment requires
        # an explicit specification in the C definition,
        # None otherwise.
        self.explicit_alignment = explicit_alignment

        # A dictionary of `WrappedType` object for this dtype's fields.
        self.wrapped_fields = wrapped_fields

        # A dictionary of alignments for this dtype's fields;
        # similarly to `explicit_alignment`, a value is an integer if the alignment
        # has to be set explicitly, None otherwise.
        self.field_alignments = field_alignments

    def __eq__(self, other):
        return (
            self.dtype == other.dtype
            and self.explicit_alignment == other.explicit_alignment
            and self.alignment == other.alignment
            and self.wrapped_fields == other.wrapped_fields
            and self.field_alignments == other.field_alignments)

    def __repr__(self):
        return (
            f"WrappedType({self.dtype}, {self.alignment}, "
            f"explicit_alignment={self.explicit_alignment}, "
            f"wrapped_fields={self.wrapped_fields}, "
            f"field_alignments={self.field_alignments})")


def _align(dtype: numpy.dtype) -> WrappedType:
    """
    Builds a `WrappedType` object with the alignment information for a dtype,
    aligning it if it is not aligned, and checking the consistency of metadata if it is.
    """

    if len(dtype.shape) > 0:
        wt = _align(dtype.base)
        return WrappedType(
            numpy.dtype((wt.dtype, dtype.shape)), wt.alignment,
            explicit_alignment=wt.explicit_alignment, wrapped_fields=wt.wrapped_fields)

    if dtype.names is None:
        return WrappedType(dtype, dtype.itemsize)

    # Since `.names` is not `None` at this point, we can restrict the type to help the inference
    dtype_fields = typing_cast(Mapping[str, Tuple[numpy.dtype, int]], dtype.fields)

    wrapped_fields = {name: _align(dtype_fields[name][0]) for name in dtype.names}

    if dtype.isalignedstruct:
        # Find out what alignment has to be set for the field in order for the compiler
        # to place it at the offset specified in the description of `dtype`.
        field_alignments = [wrapped_fields[dtype.names[0]].alignment]
        for i in range(1, len(dtype.names)):
            prev_field_dtype, prev_offset = dtype_fields[dtype.names[i-1]]
            _, offset = dtype_fields[dtype.names[i]]
            prev_end = prev_offset + prev_field_dtype.itemsize
            field_alignment = _find_minimum_alignment(
                offset, wrapped_fields[dtype.names[i]].alignment, prev_end)
            field_alignments.append(field_alignment)

        offsets = [dtype_fields[name][1] for name in dtype.names]
    else:
        # Build offsets for the structure using a procedure
        # similar to the one a compiler would use
        offsets = [0]
        for i in range(1, len(dtype.names)):
            prev_field_dtype, _ = dtype_fields[dtype.names[i-1]]
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
        if 2**log2(dtype.itemsize) != dtype.itemsize:
            raise ValueError(
                f"Invalid non-default itemsize for dtype {dtype}: "
                f"must be a power of 2 (currently {dtype.itemsize})")

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
        aligned_dtype = numpy.dtype(dict( # type: ignore
            names=dtype.names,
            formats=[wrapped_fields[name].dtype for name in dtype.names],
            offsets=offsets,
            itemsize=itemsize,
            aligned=True))

        struct_alignment = _find_minimum_alignment(itemsize, base_struct_alignment, struct_end)

    field_alignments_map = {
        dtype.names[i]: field_alignments[i]
            if field_alignments[i] != wrapped_fields[dtype.names[i]].alignment
            else None
        for i in range(len(dtype.names))}

    return WrappedType(
        aligned_dtype, struct_alignment,
        explicit_alignment=struct_alignment if struct_alignment != base_struct_alignment else None,
        wrapped_fields=wrapped_fields,
        field_alignments=field_alignments_map)


def align(dtype: numpy.dtype) -> numpy.dtype:
    """
    Returns a new struct dtype with the field offsets changed to the ones a compiler would use
    (without being given any explicit alignment qualifiers).
    Ignores all existing explicit itemsizes and offsets.

    :param dtype:
    """
    dtype = normalize_type(dtype)
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


def _get_struct_module(dtype: numpy.dtype, ignore_alignment: bool=False) -> Module:
    """
    Builds and returns a module with the C type definition for a given ``dtype``,
    possibly using modules for nested structures.
    """

    # `dtype.names` is not `None` at this point, restricting types
    dtype_names = typing_cast(Iterable[str], dtype.names)
    dtype_fields = typing_cast(Mapping[str, Tuple[numpy.dtype, int]], dtype.fields)

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
            f"    ${{{typename_var}}} {_alignment_str(field_alignment)} {name}{array_suffix};")

        if base_elem_dtype.names is None:
            kwds[typename_var] = ctype_builtin(base_elem_dtype)
        else:
            kwds[typename_var] = ctype_struct(base_elem_dtype, ignore_alignment=ignore_alignment)

    lines.append("} " + _alignment_str(struct_alignment) + " ${prefix};")

    return Module.from_string("\n".join(lines), render_globals=kwds)


def ctype_struct(dtype: Union[Type, numpy.dtype], ignore_alignment: bool=False) -> Module:
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

    :param dtype:
    :param ignore_alignment:

    .. warning::

        As of ``numpy`` 1.8, the ``isalignedstruct`` attribute is not enough to ensure
        a mapping between a dtype and a C struct with only the fields that are present in the dtype.
        Therefore, ``ctype_struct`` will make some additional checks and raise ``ValueError``
        if it is not the case.
    """
    dtype = normalize_type(dtype)

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
        dtype: numpy.dtype, prefix: List[Union[str, int]]=[]) \
        -> List[Tuple[List[Union[str, int]], numpy.dtype]]:

    if dtype.names is None:
        return [(prefix, dtype)]

    # `dtype.names` is not `None` at this point, restricting types
    dtype_fields = typing_cast(Mapping[str, Tuple[numpy.dtype, int]], dtype.fields)

    result: List[Tuple[List[Union[str, int]], numpy.dtype]] = []
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


def flatten_dtype(dtype: numpy.dtype) -> List[Tuple[List[Union[str, int]], numpy.dtype]]:
    """
    Returns a list of tuples ``(path, dtype)`` for each of the basic dtypes in
    a (possibly nested) ``dtype``.
    ``path`` is a list of field names/array indices leading to the corresponding element.

    :param dtype:
    """
    dtype = normalize_type(dtype)
    return _flatten_dtype(dtype)


def c_path(path: List[Union[str, int]]) -> str:
    """
    Returns a string corresponding to the ``path`` to a struct element in C.
    The ``path`` is the sequence of field names/array indices returned from
    :py:func:`~grunnur.dtypes.flatten_dtype`.

    :param path:
    """
    res = "".join(
        (("." + elem) if isinstance(elem, str) else ("[" + str(elem) + "]"))
        for elem in path)
    return res[1:] # drop the first dot


def _extract_field(arr: numpy.ndarray, path: List[Union[str, int]], array_idxs: List[int]):
    """
    A helper function for ``extract_field``.
    Need to collect array indices for dtype sub-array fields since they are attached to the end
    of the full array index.
    """
    if len(path) == 0:
        if len(array_idxs) == 0:
            return arr
        numpy_array_indices: List[Union[slice, int]] = (
            [slice(None, None, None)] * (len(arr.shape) - len(array_idxs)))
        struct_array_indices: List[Union[slice, int]] = list(array_idxs)
        slices = tuple(numpy_array_indices + struct_array_indices)
        return arr[slices]

    if isinstance(path[0], str):
        return _extract_field(arr[path[0]], path[1:], array_idxs)

    return _extract_field(arr, path[1:], array_idxs + [path[0]])


def extract_field(arr: numpy.ndarray, path: List[Union[str, int]]) -> numpy.ndarray:
    """
    Extracts an element from an array of struct dtype.
    The ``path`` is the sequence of field names/array indices returned from
    :py:func:`~grunnur.dtypes.flatten_dtype`.

    :param arr:
    :param path:
    """
    return _extract_field(arr, path, [])

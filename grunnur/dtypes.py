from __future__ import annotations

import functools
import itertools
import platform
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import gcd
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast, overload

import numpy

from ._modules import Module
from ._utils import bounding_power_of_2, log2, min_blocks, prod

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import DTypeLike, NDArray


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


def _ctype_builtin(dtype: numpy.dtype[Any]) -> str:
    if dtype in _DTYPE_TO_BUILTIN_CTYPE:
        return _DTYPE_TO_BUILTIN_CTYPE[dtype]
    raise ValueError(f"{dtype} is not a built-in data type")


def _normalize_type(dtype: DTypeLike) -> numpy.dtype[Any]:
    """
    Numpy's dtype shortcuts (e.g. ``numpy.int32``) are ``type`` objects
    and have slightly different properties from actual ``numpy.dtype`` objects.
    This function converts the former to ``numpy.dtype`` and keeps the latter unchanged.
    """
    return numpy.dtype(dtype)


def is_complex(dtype: DTypeLike) -> bool:
    """Returns ``True`` if ``dtype`` is complex."""
    dtype = _normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.complexfloating)


def is_double(dtype: DTypeLike) -> bool:
    """Returns ``True`` if ``dtype`` is double precision floating point."""
    dtype = _normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.float64) or numpy.issubdtype(dtype, numpy.complex128)


def is_integer(dtype: DTypeLike) -> bool:
    """Returns ``True`` if ``dtype`` is an integer."""
    dtype = _normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.integer)


def is_real(dtype: DTypeLike) -> bool:
    """Returns ``True`` if ``dtype`` is a real number (but not complex)."""
    dtype = _normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.floating)


def _promote_type(dtype: numpy.dtype[Any]) -> numpy.dtype[Any]:
    # not all numpy datatypes are supported by GPU, so we may need to promote
    if issubclass(dtype.type, numpy.signedinteger) and dtype.itemsize < 4:  # noqa: PLR2004
        return numpy.dtype("int32")
    if issubclass(dtype.type, numpy.unsignedinteger) and dtype.itemsize < 4:  # noqa: PLR2004
        return numpy.dtype("uint32")
    if issubclass(dtype.type, numpy.floating) and dtype.itemsize < 4:  # noqa: PLR2004
        return numpy.dtype("float32")
    # Not checking complex dtypes since there is nothing smaller than complex32, which is supported
    return dtype


def result_type(*dtypes: DTypeLike) -> numpy.dtype[Any]:
    """
    Wrapper for :py:func:`numpy.result_type`
    which takes into account types supported by GPUs.
    """
    return _promote_type(numpy.result_type(*dtypes))


def min_scalar_type(val: complex | numpy.number[Any]) -> numpy.dtype[Any]:
    """
    Wrapper for :py:func:`numpy.min_scalar_type`
    which takes into account types supported by GPUs.
    """
    dtype = val.dtype if isinstance(val, numpy.number) else numpy.min_scalar_type(val)
    return _promote_type(dtype)


def complex_for(dtype: DTypeLike) -> numpy.dtype[Any]:
    """Returns complex dtype corresponding to given floating point ``dtype``."""
    dtype = _normalize_type(dtype)
    if dtype == numpy.float32:
        return numpy.dtype("complex64")
    if dtype == numpy.float64:
        return numpy.dtype("complex128")
    raise ValueError(f"{dtype} does not have a corresponding complex type")


def real_for(dtype: DTypeLike) -> numpy.dtype[Any]:
    """Returns floating point dtype corresponding to given complex ``dtype``."""
    dtype = _normalize_type(dtype)
    if dtype == numpy.complex64:
        return numpy.dtype("float32")
    if dtype == numpy.complex128:
        return numpy.dtype("float64")
    raise ValueError(f"{dtype} does not have a corresponding real type")


def complex_ctr(dtype: DTypeLike) -> str:
    """Returns name of the constructor for the given ``dtype``."""
    return "COMPLEX_CTR(" + _ctype_builtin(_normalize_type(dtype)) + ")"


def _c_constant_arr(val: Any, shape: Sequence[int]) -> str:
    if len(shape) == 0:
        return c_constant(val)
    return "{" + ", ".join(_c_constant_arr(val[i], shape[1:]) for i in range(shape[0])) + "}"


def c_constant(
    val: complex | numpy.generic | NDArray[Any],
    dtype: DTypeLike | None = None,
) -> str:
    """
    Returns a C-style numerical constant.
    If ``val`` has a struct dtype, the generated constant will have the form ``{ ... }``
    and can be used as an initializer for a variable.
    """
    if dtype is not None:
        dtype = _promote_type(_normalize_type(dtype))
    elif isinstance(val, int | float | complex):
        dtype = min_scalar_type(val)
    else:
        dtype = _promote_type(val.dtype)

    numpy_val: numpy.generic | NDArray[Any]
    if isinstance(val, numpy.ndarray):
        numpy_val = numpy.asarray(val, dtype)
    else:
        numpy_val = numpy.asarray(val, dtype).flat[0]

    if len(numpy_val.shape) > 0:
        return _c_constant_arr(numpy_val, numpy_val.shape)

    scalar_val: numpy.generic
    scalar_val = numpy_val.flat[0] if isinstance(numpy_val, numpy.ndarray) else numpy_val

    if isinstance(scalar_val, numpy.void) and scalar_val.dtype.names is not None:
        return (
            "{" + ", ".join([c_constant(scalar_val[name]) for name in scalar_val.dtype.names]) + "}"
        )

    if isinstance(scalar_val, numpy.complexfloating):
        return (
            f"COMPLEX_CTR({_ctype_builtin(dtype)})"
            f"({c_constant(scalar_val.real)}, {c_constant(scalar_val.imag)})"
        )

    if isinstance(scalar_val, numpy.integer):
        if dtype.itemsize > 4:  # noqa: PLR2004
            postfix = "L" if numpy.issubdtype(scalar_val.dtype, numpy.signedinteger) else "UL"
        else:
            postfix = ""
        return str(scalar_val) + postfix

    if isinstance(scalar_val, numpy.floating):
        return repr(float(scalar_val)) + ("f" if scalar_val.dtype.itemsize <= 4 else "")  # noqa: PLR2004

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


class _WrappedType(NamedTuple):
    """Contains an accompanying information for an aligned dtype."""

    # The original dtype (note that it can be an array, that is have a non-empty ``shape``)
    dtype: numpy.dtype[Any]

    # The calculated structure alignment.
    # Can be different from ``dtype.alignment`` if it's needed to ensure the chosen ``itemsize``.
    alignment: int

    # A dictionary of `_WrappedType` objects for this dtype's fields.
    wrapped_fields: dict[str, _WrappedType]

    # An in integer if the type's alignment requires
    # an explicit specification in the C definition,
    # None otherwise.
    explicit_alignment: int | None

    # A dictionary of alignments for this dtype's fields;
    # similarly to `explicit_alignment`, a value is an integer if the alignment
    # has to be set explicitly, None otherwise.
    explicit_field_alignments: dict[str, int | None]

    @classmethod
    def _wrap_single(
        cls, dtype: numpy.dtype[Any], wrapped_fields: dict[str, _WrappedType]
    ) -> _WrappedType:
        """
        Attempts to build a `_WrappedType` object with the alignment information
        for the given dtype, aiming to preserve the given offsets and itemsizes.
        Raises a ``ValueError`` if it is not possible.
        """
        base_dtype = dtype.base

        # At this point dtype is guaranteed to be a struct type
        dtype_fields = cast(Mapping[str, tuple[numpy.dtype[Any], int]], base_dtype.fields)
        names = list(dtype_fields)

        # Find out what alignment has to be set for the field in order for the compiler
        # to place it at the offset specified in the description of `dtype`.
        explicit_field_alignments: dict[str, int | None] = {names[0]: None}
        field_alignments = [wrapped_fields[names[0]].alignment]
        for i in range(1, len(names)):
            name = names[i]
            prev_field_dtype, prev_offset = dtype_fields[names[i - 1]]
            _, offset = dtype_fields[name]
            prev_end = prev_offset + prev_field_dtype.itemsize
            inherent_alignment = wrapped_fields[name].alignment

            try:
                field_alignment = _find_minimum_alignment(offset, inherent_alignment, prev_end)
            except ValueError as exc:
                raise ValueError(f"Failed to find alignment for field `{name}`: {exc}") from exc

            explicit_field_alignments[name] = (
                field_alignment if field_alignment != inherent_alignment else None
            )
            field_alignments.append(field_alignment)

        # Same principle as above, but for the whole struct:
        # find out what alignment has to be set in order for the compiler
        # to place the next field at some dtype where this struct is a field type
        # at the offset corresponding to this struct's itemsize.

        last_dtype, last_offset = dtype_fields[names[-1]]
        struct_end = last_offset + last_dtype.itemsize

        # Find the total itemsize.
        # According to the standard, it must be a multiple of the struct alignment.
        base_struct_alignment = _struct_alignment(field_alignments)
        min_itemsize = min_blocks(struct_end, base_struct_alignment) * base_struct_alignment

        # A sanity check; `numpy` won't allow a dtype to have an itemsize smaller than the minimum
        assert base_dtype.itemsize >= min_itemsize, "dtype's itemsize is too small"  # noqa: S101

        if base_dtype.itemsize == min_itemsize:
            explicit_alignment = None
            alignment = base_struct_alignment
        elif base_dtype.itemsize != 2 ** log2(base_dtype.itemsize):
            raise ValueError("An itemsize that requires an explicit alignment must be a power of 2")
        else:
            explicit_alignment = base_dtype.itemsize
            alignment = explicit_alignment

        return cls(
            dtype=dtype,
            alignment=alignment,
            wrapped_fields=wrapped_fields,
            explicit_alignment=explicit_alignment,
            explicit_field_alignments=explicit_field_alignments,
        )

    @classmethod
    def _align_single(
        cls, dtype: numpy.dtype[Any], aligned_fields: dict[str, _WrappedType]
    ) -> _WrappedType:
        """
        Builds a `_WrappedType` object with the alignment information for a dtype,
        aligning it if it is not aligned, and checking the consistency of metadata if it is.
        """
        names = list(aligned_fields)

        # Build offsets for the structure using a procedure
        # similar to the one a compiler would use
        offsets = [0]
        for i in range(1, len(names)):
            prev_end = offsets[-1] + aligned_fields[names[i - 1]].dtype.itemsize
            alignment = aligned_fields[names[i]].alignment
            offsets.append(min_blocks(prev_end, alignment) * alignment)

        # Same principle as above, but for the whole struct:
        # find out the offset of the next struct of the same type,
        # which will be the struct's itemsize.
        # According to the standard, the itemsize must be a multiple of the struct alignment.
        field_alignments = [aligned_fields[name].alignment for name in names]
        struct_end = offsets[-1] + aligned_fields[names[-1]].dtype.itemsize
        base_struct_alignment = _struct_alignment(field_alignments)
        itemsize = min_blocks(struct_end, base_struct_alignment) * base_struct_alignment

        base_dtype = numpy.dtype(
            dict(
                names=list(aligned_fields),
                formats=[aligned.dtype for aligned in aligned_fields.values()],
                offsets=offsets,
                itemsize=itemsize,
                aligned=True,
            )
        )
        aligned_dtype = numpy.dtype((base_dtype, dtype.shape))

        # Now let the other method find out if any explicit alignments need to be set at this level.
        return cls._wrap_single(aligned_dtype, aligned_fields)

    @classmethod
    def wrap(cls, dtype: numpy.dtype[Any]) -> _WrappedType:
        if dtype.base.names is None:
            return cls(
                dtype=dtype,
                alignment=dtype.alignment,
                wrapped_fields={},
                explicit_alignment=None,
                explicit_field_alignments={},
            )

        # Since `.names` is not `None` at this point, we can restrict the type to help the inference
        dtype_fields = cast(Mapping[str, tuple[numpy.dtype[Any], int]], dtype.base.fields)

        wrapped_fields = {
            name: cls.wrap(field_dtype) for name, (field_dtype, _offset) in dtype_fields.items()
        }

        return cls._wrap_single(dtype, wrapped_fields)

    @classmethod
    def align(cls, dtype: numpy.dtype[Any]) -> _WrappedType:
        if dtype.base.names is None:
            return _WrappedType.wrap(dtype)

        # Since `.names` is not `None` at this point, we can restrict the type to help the inference
        dtype_fields = cast(Mapping[str, tuple[numpy.dtype[Any], int]], dtype.base.fields)

        aligned_fields = {
            name: cls.align(field_dtype) for name, (field_dtype, _offset) in dtype_fields.items()
        }

        return cls._align_single(dtype, aligned_fields)


def align(dtype: DTypeLike) -> numpy.dtype[Any]:
    """
    Returns a new struct dtype with the field offsets changed to the ones a compiler would use
    (without being given any explicit alignment qualifiers).
    Ignores all existing explicit itemsizes and offsets.
    """
    dtype = _normalize_type(dtype)
    wrapped_dtype = _WrappedType.align(dtype)
    return wrapped_dtype.dtype


def _lcm(*nums: int) -> int:
    """Returns the least common multiple of ``nums``."""
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:  # noqa: PLR2004
        return nums[0] * nums[1] // gcd(nums[0], nums[1])
    return _lcm(nums[0], _lcm(*nums[1:]))


def _alignment_str(alignment: int | None) -> str:
    if alignment is not None:
        return "ALIGN(" + str(alignment) + ")"
    return ""


def _get_struct_module(wrapped_type: _WrappedType) -> Module:
    """
    Builds and returns a module with the C type definition for a given ``dtype``,
    possibly using modules for nested structures.
    """
    # `dtype.names` is not `None` at this point, restricting types
    dtype_fields = cast(Mapping[str, tuple[numpy.dtype[Any], int]], wrapped_type.dtype.base.fields)

    struct_alignment = wrapped_type.explicit_alignment
    field_alignments = wrapped_type.explicit_field_alignments

    # The tag (${prefix}_) is not necessary, but it helps to avoid
    # CUDA bug #1409907 (nested struct initialization like
    # "mystruct x = {0, {0, 0}, 0};" fails to compile)
    lines = ["typedef struct ${prefix}_ {"]
    kwds: dict[str, str | Module] = {}
    for name, (elem_dtype, _offset) in dtype_fields.items():
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
            kwds[typename_var] = _get_struct_module(wrapped_type.wrapped_fields[name])

    lines.append("} " + _alignment_str(struct_alignment) + " ${prefix};")
    return Module.from_string("\n".join(lines), render_globals=kwds)


# TODO: this has a possibility of a memory leak, albeit it's very unlikely -
# the number of different types will most surely be limited.
# A way to avoid that is to introduce some custom class that encapsulates the dtype
# and the corresponding module (generated once on the first request).
# This way we can even have "newtypes" of sorts for dtypes with the same structure.
@functools.cache
def ctype(dtype: DTypeLike) -> str | Module:
    """
    Returns an object that can be passed as a global to :py:meth:`~grunnur.Program`
    and used to render a C equivalent of the given ``numpy`` dtype.
    If there is a built-in C equivalent, the object is just a string with the type name;
    otherwise it is a :py:class:`~grunnur.Module` object containing
    the corresponding ``struct`` declaration.

    The given ``dtype`` cannot be an array type (that is, its shape must be ``()``).

    Modules are cached, and the function returns a single module instance for equal ``dtype``'s.
    Therefore inside a kernel it will be rendered with the same prefix everywhere it is used.
    This results in a behavior characteristic for a structural type system,
    same as for the basic dtype-ctype conversion.

    .. note::

        If ``dtype`` is a struct type, it needs to be aligned
        (manually or with :py:func:`align`) in order for the field offsets
        on the host and the devices coincide.

    .. warning::

        ``numpy``'s ``isalignedstruct`` attribute, or ``aligned`` parameter
        is neither necessary nor sufficient for the memory layout
        on the host and the device to coincide.
        Use :py:func:`align` if you are not sure how to align your struct type.
    """
    # It doesn't seem like `numpy` is going to fix the alignment behavior.
    # For now the `isalignedstruct` attribute is more of an internal marker,
    # not useful for external users. So we will have to rely on `align()`.
    # See https://github.com/numpy/numpy/issues/4084 for the discussion.
    #
    # Would be nice to have our own marker, but it doesn't seem like `dtype`
    # can be safely inherited from.

    dtype = _normalize_type(dtype)

    if len(dtype.shape) > 0:
        raise ValueError("The data type cannot be an array")

    if dtype.names is None:
        return _ctype_builtin(dtype)

    wrapped_type = _WrappedType.wrap(dtype)
    return _get_struct_module(wrapped_type)


def _flatten_dtype(dtype: numpy.dtype[Any], offset: int, path: list[str | int]) -> list[FieldInfo]:
    if dtype.names is None:
        return [FieldInfo(path=path, dtype=dtype, offset=offset)]

    # `dtype.names` is not `None` at this point, restricting types
    dtype_fields = cast(Mapping[str, tuple[numpy.dtype[Any], int]], dtype.fields)

    result: list[FieldInfo] = []
    for name in dtype.names:
        elem_dtype, elem_offset = dtype_fields[name]

        elem_dtype_shape: tuple[int, ...]
        if len(elem_dtype.shape) == 0:
            base_elem_dtype = elem_dtype
            elem_dtype_shape = tuple()
        else:
            base_elem_dtype = elem_dtype.base
            elem_dtype_shape = elem_dtype.shape

        if len(elem_dtype_shape) == 0:
            result += _flatten_dtype(
                dtype=base_elem_dtype, offset=offset + elem_offset, path=[*path, name]
            )
        else:
            strides = list(
                reversed(numpy.multiply.accumulate([1, *reversed(elem_dtype_shape[1:])]))
            )
            for idxs in itertools.product(*[range(dim) for dim in elem_dtype_shape]):
                flat_idx = sum(idx * stride for idx, stride in zip(idxs, strides, strict=True))
                result += _flatten_dtype(
                    dtype=base_elem_dtype,
                    offset=offset + elem_offset + base_elem_dtype.itemsize * flat_idx,
                    path=[*path, name, *idxs],
                )
    return result


class FieldInfo(NamedTuple):
    """Metadata of a structure field."""

    path: list[str | int]
    """The full path to the field."""

    dtype: numpy.dtype[Any]
    """The field's datatype."""

    offset: int
    """The field's offset from the beginning of the struct."""

    @property
    def c_path(self) -> str:
        """Returns a string corresponding to the ``path`` to a struct element in C."""
        return "".join(
            (("." + elem) if isinstance(elem, str) else ("[" + str(elem) + "]"))
            for elem in self.path
        )


def flatten_dtype(dtype: DTypeLike) -> list[FieldInfo]:
    """
    Returns a list of tuples ``(path, dtype)`` for each of the basic dtypes in
    a (possibly nested) ``dtype``.
    ``path`` is a list of field names/array indices leading to the corresponding element.
    """
    dtype = _normalize_type(dtype)
    return _flatten_dtype(dtype, 0, [])


def _extract_field(
    arr: NDArray[Any], path: list[str | int], array_idxs: list[int]
) -> numpy.generic | NDArray[Any]:
    """
    A helper function for ``extract_field``.
    Need to collect array indices for dtype sub-array fields since they are attached to the end
    of the full array index.
    """
    if len(path) == 0:
        if len(array_idxs) == 0:
            return arr
        numpy_array_indices: list[slice | int] = [slice(None, None, None)] * (
            len(arr.shape) - len(array_idxs)
        )
        struct_array_indices: list[slice | int] = list(array_idxs)
        slices = tuple(numpy_array_indices + struct_array_indices)
        return arr[slices]

    if isinstance(path[0], str):
        return _extract_field(arr[path[0]], path[1:], array_idxs)

    return _extract_field(arr, path[1:], [*array_idxs, path[0]])


def extract_field(arr: NDArray[Any], path: list[str | int]) -> numpy.generic | NDArray[Any]:
    """
    Extracts an element from an array of struct dtype.
    The ``path`` is the sequence of field names/array indices returned from
    :py:func:`~grunnur.dtypes.flatten_dtype`.
    """
    return _extract_field(arr, path, [])

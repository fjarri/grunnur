from typing import Any

import numpy

from grunnur import Array, Context, Program, Queue, dtypes


def check_offsets_on_device(queue: Queue, dtype: numpy.dtype[Any]) -> None:
    flat_dtype = dtypes.flatten_dtype(dtype)

    program = Program(
        [queue.device],
        """
        #define my_offsetof(type, field) ((size_t)(&((type *)0)->field))

        KERNEL void test(GLOBAL_MEM int *dest)
        {
            ${struct} temp;

            %for i, field_info in enumerate(flat_dtype):
            dest[${i}] = (size_t)&(temp${field_info.c_path}) - (size_t)&temp;
            %endfor
            dest[${len(flat_dtype)}] = sizeof(${struct});
        }
        """,
        render_globals=dict(struct=dtypes.ctype(dtype), dtype=dtype, flat_dtype=flat_dtype),
    )

    offsets_dev = Array.empty(queue.device, (len(flat_dtype) + 1,), numpy.int32)
    test = program.kernel.test
    test(queue, [1], None, offsets_dev)
    offsets = offsets_dev.get(queue)
    offsets_int = [int(offset) for offset in offsets]

    for field_info, device_offset in zip(flat_dtype, offsets_int[:-1], strict=True):
        message = (
            f"offset for {field_info.c_path} is different: {field_info.offset} in numpy, "
            f"{device_offset} on device"
        )
        assert field_info.offset == device_offset, message

    assert offsets_int[-1] == dtype.itemsize


def test_offsets_simple(context: Context) -> None:
    queue = Queue(context.device)
    dtype = numpy.dtype(
        dict(
            names=["val1", "pad"],
            formats=[numpy.int32, numpy.int8],
            offsets=[0, 4],
            itemsize=8,
            aligned=True,
        )
    )
    check_offsets_on_device(queue, dtype)


def test_offsets_nested(context: Context) -> None:
    queue = Queue(context.device)
    dtype_nested = numpy.dtype(
        dict(
            names=["val1", "pad"],
            formats=[numpy.int32, numpy.int8],
            offsets=[0, 4],
            itemsize=8,
            aligned=True,
        )
    )
    dtype = numpy.dtype(
        dict(
            names=["val1", "val2", "nested"],
            formats=[numpy.int32, numpy.int16, dtype_nested],
            offsets=[0, 4, 8],
            itemsize=32,
            aligned=True,
        )
    )
    check_offsets_on_device(queue, dtype)


def test_offsets_arrays(context: Context) -> None:
    queue = Queue(context.device)
    dtype = numpy.dtype(
        dict(
            names=["val1", "val2"],
            formats=[(numpy.int8, 3), (numpy.int8, 2)],
            offsets=[0, 8],
            itemsize=16,
            aligned=True,
        )
    )
    check_offsets_on_device(queue, dtype)


def test_offsets_nested_arrays(context: Context) -> None:
    queue = Queue(context.device)
    dtype_nested = numpy.dtype(dict(names=["val1", "pad"], formats=[numpy.int8, numpy.int8]))
    dtype = numpy.dtype(
        dict(
            names=["pad", "struct_arr", "regular_arr"],
            formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))],
        )
    )
    dtype_aligned = dtypes.align(dtype)
    check_offsets_on_device(queue, dtype_aligned)


def test_offsets_custom_itemsize(context: Context) -> None:
    queue = Queue(context.device)
    dtype = numpy.dtype(
        dict(
            names=["x", "y", "z"],
            formats=[numpy.int8, numpy.int16, numpy.int32],
            offsets=[0, 4, 16],
            itemsize=32,
            aligned=True,
        )
    )
    check_offsets_on_device(queue, dtype)


def test_offsets_custom_itemsize_nested(context: Context) -> None:
    queue = Queue(context.device)
    dtype_nested = numpy.dtype(
        dict(
            names=["val1", "pad"],
            formats=[numpy.int32, numpy.int8],
            offsets=[0, 4],
            itemsize=16,
            aligned=True,
        )
    )

    dtype = numpy.dtype(
        dict(
            names=["val1", "val2", "nested"],
            formats=[numpy.int32, numpy.int16, dtype_nested],
            offsets=[0, 4, 16],
            itemsize=64,
            aligned=True,
        )
    )

    check_offsets_on_device(queue, dtype)

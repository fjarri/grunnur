import numpy

from grunnur import dtypes, Program, Queue, Array


def check_struct_fill(context, dtype):
    """
    Fill every field of the given ``dtype`` with its number and check the results.
    This helps detect issues with offsets in the struct.
    """
    struct = dtypes.ctype_struct(dtype)

    program = Program(
        context,
        """
        KERNEL void test(GLOBAL_MEM ${struct} *dest, GLOBAL_MEM int *itemsizes)
        {
            const SIZE_T i = get_global_id(0);
            ${struct} res;

            %for i, field_info in enumerate(dtypes.flatten_dtype(dtype)):
            res.${dtypes.c_path(field_info[0])} = ${i};
            %endfor

            dest[i] = res;
            itemsizes[i] = sizeof(${struct});
        }
        """,
        render_globals=dict(
            struct=struct,
            dtypes=dtypes,
            dtype=dtype))

    test = program.kernel.test
    queue = Queue.on_all_devices(context)

    a_dev = Array.empty(queue, 128, dtype)
    itemsizes_dev = Array.empty(queue, 128, numpy.int32)
    test(queue, 128, None, a_dev, itemsizes_dev)
    a = a_dev.get()
    itemsizes = itemsizes_dev.get()

    for i, field_info in enumerate(dtypes.flatten_dtype(dtype)):
        path, _ = field_info
        assert (dtypes.extract_field(a, path) == i).all()
    assert (itemsizes == dtype.itemsize).all()


def test_struct_offsets(context):
    """
    Test the correctness of alignment for an explicit set of field offsets.
    """

    dtype_nested = numpy.dtype(dict(
        names=['val1', 'pad'],
        formats=[numpy.int32, numpy.int8],
        offsets=[0, 4],
        itemsize=8,
        aligned=True))

    dtype = numpy.dtype(dict(
        names=['val1', 'val2', 'nested'],
        formats=[numpy.int32, numpy.int16, dtype_nested],
        offsets=[0, 4, 8],
        itemsize=32,
        aligned=True))

    check_struct_fill(context, dtype)


def test_struct_offsets_array(context):
    """
    Test the correctness of alignment for an explicit set of field offsets.
    """
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

    check_struct_fill(context, dtype_aligned)


def test_struct_offsets_field_alignments(context):

    dtype = numpy.dtype(dict(
        names=['x', 'y', 'z'],
        formats=[numpy.int8, numpy.int16, numpy.int32],
        offsets=[0, 4, 16],
        itemsize=32))

    dtype_aligned = dtypes.align(dtype)

    check_struct_fill(context, dtype_aligned)

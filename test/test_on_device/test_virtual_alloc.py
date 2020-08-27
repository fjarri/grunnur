from collections import Counter
import weakref

import numpy
import pytest

from grunnur.adapter_base import APIID
from grunnur import Queue, Program, Array, Context, Buffer
import grunnur.dtypes as dtypes
from grunnur.virtual_alloc import extract_dependencies

from ..test_mocked.test_virtual_alloc import allocate_test_set


@pytest.mark.parametrize('pack', [False, True], ids=['no_pack', 'pack'])
def test_contract(context, valloc_cls, pack):

    dtype = numpy.int32

    program = Program(
        context,
        """
        KERNEL void fill(GLOBAL_MEM ${ctype} *dest, ${ctype} val)
        {
            const SIZE_T i = get_global_id(0);
            dest[i] = val;
        }
        """,
        render_globals=dict(ctype=dtypes.ctype(dtype)))
    fill = program.kernel.fill

    queue = Queue.on_all_devices(context)
    virtual_alloc = valloc_cls(queue)

    buffers_metadata, arrays = allocate_test_set(
        virtual_alloc,
        # Bump size to make sure buffer alignment doesn't hide any out-of-bounds access
        lambda allocator, size: Array.empty(queue, size * 100, dtype, allocator=allocator))
    dependencies = {id_: deps for id_, _, deps in buffers_metadata}

    if pack:
        virtual_alloc.pack()

    # Clear all arrays
    for name in sorted(arrays.keys()):
        fill(queue, arrays[name].shape, None, arrays[name], dtype(0))

    for i, name in enumerate(sorted(arrays.keys())):
        val = dtype(i + 1)
        fill(queue, arrays[name].shape, None, arrays[name], val)
        # According to the virtual allocator contract, the allocated buffer
        # will not intersect with the buffers from the specified dependencies.
        # So we're filling the buffer and checking that the dependencies did not change.
        for dep in dependencies[name]:
            assert (arrays[dep].get() != val).all()

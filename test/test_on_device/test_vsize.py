import itertools

import numpy

from grunnur import Queue, Program, Array
from grunnur.vsize import VirtualSizes
from grunnur.utils import prod, min_blocks


class VirtualSizesTest:

    def __init__(self, global_size, local_size, max_num_groups, max_local_sizes):
        self.global_size = global_size
        self.local_size = local_size
        if local_size is not None:
            self.grid_size = tuple(min_blocks(gs, ls) for gs, ls in zip(global_size, local_size))

        self.max_num_groups = max_num_groups
        self.max_local_sizes = max_local_sizes
        self.max_total_local_size = prod(max_local_sizes)

    def __str__(self):
        return "{gs}-{ls}-limited-by-{mng}-{mlss}".format(
            gs=self.global_size, ls=self.local_size,
            mng=self.max_num_groups, mlss=self.max_local_sizes)


_VSTESTS = [
    # This set of parameters ensures that all three skip functions will be nontrivial:
    # - skip_local_threads(), because the local size cannot be trivially fit into max_local_sizes
    #   (that is, one has to construct a flat index from real local ids, and then decompose it
    #   into virtual local ids);
    # - skip_groups(), for the same reason, but applied to groups and max_num_groups;
    # - skip_global_threads(), because global size elements are not multiples
    #   of corresponding local size elements.
    VirtualSizesTest(
        global_size=(51, 201),
        local_size=(5, 10),
        max_num_groups=(16, 16),
        max_local_sizes=(8, 8)),

    # Same thing, but with unspecified local size
    VirtualSizesTest(
        global_size=(51, 201),
        local_size=None,
        max_num_groups=(16, 16),
        max_local_sizes=(8, 8)),
    ]


def pytest_generate_tests(metafunc):
    if 'vstest' in metafunc.fixturenames:
        metafunc.parametrize('vstest', _VSTESTS, ids=[str(x) for x in _VSTESTS])


class ReferenceIds:

    def __init__(self, global_size, local_size):
        self.global_size = global_size
        if local_size is not None:
            self.local_size = local_size
            self.grid_size = tuple(min_blocks(gs, ls) for gs, ls in zip(global_size, local_size))

    def _tile_pattern(self, pattern, axis, full_shape):

        pattern_shape = [x if i == axis else 1 for i, x in enumerate(full_shape)]
        pattern = pattern.reshape(*pattern_shape)

        tile_shape = [x if i != axis else 1 for i, x in enumerate(full_shape)]
        pattern = numpy.tile(pattern, tile_shape)

        return pattern.astype(numpy.int32)

    def predict_local_ids(self, dim):
        global_len = self.global_size[dim]
        local_len = self.local_size[dim]
        repetitions = min_blocks(global_len, local_len)

        pattern = numpy.tile(numpy.arange(local_len), repetitions)[:global_len]
        return self._tile_pattern(pattern, dim, self.global_size)

    def predict_group_ids(self, dim):
        global_len = self.global_size[dim]
        local_len = self.local_size[dim]
        repetitions = min_blocks(global_len, local_len)

        pattern = numpy.repeat(numpy.arange(repetitions), local_len)[:global_len]
        return self._tile_pattern(pattern, dim, self.global_size)

    def predict_global_ids(self, dim):
        global_len = self.global_size[dim]

        pattern = numpy.arange(global_len)
        return self._tile_pattern(pattern, dim, self.global_size)


def test_ids(context, vstest):
    """
    Test that virtual IDs are correct for each thread.
    """
    ref = ReferenceIds(vstest.global_size, vstest.local_size)

    vs = VirtualSizes(
        max_total_local_size=vstest.max_total_local_size,
        max_local_sizes=vstest.max_local_sizes,
        max_num_groups=vstest.max_num_groups,
        local_size_multiple=2,
        virtual_global_size=vstest.global_size,
        virtual_local_size=vstest.local_size)

    program = Program(
        context,
        """
        KERNEL void get_ids(
            GLOBAL_MEM int *local_ids,
            GLOBAL_MEM int *group_ids,
            GLOBAL_MEM int *global_ids,
            int vdim)
        {
            ${static.begin};
            const VSIZE_T i = ${static.global_flat_id}();
            local_ids[i] = ${static.local_id}(vdim);
            group_ids[i] = ${static.group_id}(vdim);
            global_ids[i] = ${static.global_id}(vdim);
        }
        """,
        render_globals=dict(static=vs.vsize_modules))

    get_ids = program.kernel.get_ids

    queue = Queue.on_all_devices(context)
    local_ids = Array.empty(queue, ref.global_size, numpy.int32)
    group_ids = Array.empty(queue, ref.global_size, numpy.int32)
    global_ids = Array.empty(queue, ref.global_size, numpy.int32)

    for vdim in range(len(vstest.global_size)):

        get_ids(
            queue, vs.real_global_size, vs.real_local_size,
            local_ids, group_ids, global_ids, numpy.int32(vdim))

        assert (global_ids.get() == ref.predict_global_ids(vdim)).all()
        if vstest.local_size is not None:
            assert (local_ids.get() == ref.predict_local_ids(vdim)).all()
            assert (group_ids.get() == ref.predict_group_ids(vdim)).all()


def test_sizes(context, vstest):
    """
    Test that virtual sizes are correct.
    """
    ref = ReferenceIds(vstest.global_size, vstest.local_size)

    vs = VirtualSizes(
        max_total_local_size=vstest.max_total_local_size,
        max_local_sizes=vstest.max_local_sizes,
        max_num_groups=vstest.max_num_groups,
        local_size_multiple=2,
        virtual_global_size=vstest.global_size,
        virtual_local_size=vstest.local_size)

    vdims = len(vstest.global_size)

    program = Program(
        context,
        """
        KERNEL void get_sizes(GLOBAL_MEM int *sizes)
        {
            if (${static.global_id}(0) > 0) return;

            for (int i = 0; i < ${vdims}; i++)
            {
                sizes[i] = ${static.local_size}(i);
                sizes[i + ${vdims}] = ${static.num_groups}(i);
                sizes[i + ${vdims * 2}] = ${static.global_size}(i);
            }
            sizes[${vdims * 3}] = ${static.global_flat_size}();
        }
        """,
        render_globals=dict(vdims=vdims, static=vs.vsize_modules))

    get_sizes = program.kernel.get_sizes

    queue = Queue.on_all_devices(context)
    sizes = Array.empty(queue, vdims * 3 + 1, numpy.int32)
    get_sizes(queue, vs.real_global_size, vs.real_local_size, sizes)

    sizes = sizes.get()
    local_sizes = sizes[0:vdims]
    grid_sizes = sizes[vdims:vdims*2]
    global_sizes = sizes[vdims*2:vdims*3]
    flat_size = sizes[vdims*3]

    global_sizes_ref = numpy.array(vstest.global_size)
    assert (global_sizes == global_sizes_ref).all()
    assert flat_size == prod(vstest.global_size)

    if vstest.local_size is not None:
        grid_sizes_ref = numpy.array(vstest.grid_size)
        assert (grid_sizes == grid_sizes_ref).all()
        local_sizes_ref = numpy.array(vstest.local_size)
        assert (local_sizes == local_sizes_ref).all()

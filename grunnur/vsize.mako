<%!
    vdim_inverse = lambda ndim, dim: ndim - dim - 1
%>


<%def name="local_id(prefix)">
FUNCTION VSIZE_T ${prefix}(unsigned int dim)
{
    %for vdim in range(len(virtual_local_size)):
    if (dim == ${vdim_inverse(len(virtual_local_size), vdim)})
    {
        %if virtual_local_size[vdim] == 1:
        ## A shortcut, mostly to make the generated code more readable
        ## (the compiler would probably simplify the full version without any problems).

        return 0;

        %else:

        SIZE_T flat_id =
        %for i, rdim in enumerate(local_groups.real_dims[vdim]):
            get_local_id(${rdim}) * ${local_groups.real_strides[vdim][i]} +
        %endfor
            0;

        ## The modulus operation will not be optimized away by the compiler,
        ## but we can omit it for the major dimension,
        ## knowing that VIRTUAL_SKIP_THREADS will skip redundant threads.
        %if vdim == local_groups.major_vdims[vdim]:
        return (flat_id / ${local_groups.virtual_strides[vdim]});
        %else:
        return (flat_id / ${local_groups.virtual_strides[vdim]}) % ${virtual_local_size[vdim]};
        %endif

        %endif
    }
    %endfor

    return 0;
}
</%def>


<%def name="local_size(prefix)">
FUNCTION VSIZE_T ${prefix}(unsigned int dim)
{
    %for vdim in range(len(virtual_local_size)):
    if (dim == ${vdim_inverse(len(virtual_local_size), vdim)})
    {
        return ${virtual_local_size[vdim]};
    }
    %endfor

    return 1;
}
</%def>


<%def name="group_id(prefix)">
FUNCTION VSIZE_T ${prefix}(unsigned int dim)
{
    %for vdim in range(len(virtual_grid_size)):
    if (dim == ${vdim_inverse(len(virtual_grid_size), vdim)})
    {
        %if virtual_grid_size[vdim] == 1:
        ## A shortcut, mostly to make the generated code more readable
        ## (the compiler would probably simplify the full version without any problems).

        return 0;

        %else:

        SIZE_T flat_id =
        %for i, rdim in enumerate(grid_groups.real_dims[vdim]):
            get_group_id(${rdim}) * ${grid_groups.real_strides[vdim][i]} +
        %endfor
            0;

        ## The modulus operation will not be optimized away by the compiler,
        ## but we can omit it for the major dimension,
        ## knowing that VIRTUAL_SKIP_THREADS will skip redundant threads.
        %if vdim == grid_groups.major_vdims[vdim]:
        return (flat_id / ${grid_groups.virtual_strides[vdim]});
        %else:
        return (flat_id / ${grid_groups.virtual_strides[vdim]}) % ${virtual_grid_size[vdim]};
        %endif

        %endif
    }
    %endfor

    return 0;
}
</%def>


<%def name="num_groups(prefix)">
FUNCTION VSIZE_T ${prefix}(unsigned int dim)
{
    %for vdim in range(len(virtual_grid_size)):
    if (dim == ${vdim_inverse(len(virtual_grid_size), vdim)})
    {
        return ${virtual_grid_size[vdim]};
    }
    %endfor

    return 1;
}
</%def>


<%def name="global_id(prefix)">
FUNCTION VSIZE_T ${prefix}(unsigned int dim)
{
    return ${local_id_mod}(dim) + ${group_id_mod}(dim) * ${local_size_mod}(dim);
}
</%def>


<%def name="global_size(prefix)">
FUNCTION VSIZE_T ${prefix}(unsigned int dim)
{
    %for vdim in range(len(virtual_global_size)):
    if(dim == ${vdim_inverse(len(virtual_global_size), vdim)})
    {
        return ${virtual_global_size[vdim]};
    }
    %endfor

    return 1;
}
</%def>


<%def name="global_flat_id(prefix)">
FUNCTION VSIZE_T ${prefix}()
{
    return
    %for vdim in range(len(virtual_global_size)):
        ${global_id_mod}(${vdim_inverse(len(virtual_global_size), vdim)}) * ${prod(virtual_global_size[:vdim])} +
    %endfor
        0;
}
</%def>


<%def name="global_flat_size(prefix)">
FUNCTION VSIZE_T ${prefix}()
{
    return
    %for vdim in range(len(virtual_global_size)):
        ${global_size_mod}(${vdim_inverse(len(virtual_global_size), vdim)}) *
    %endfor
        1;
}
</%def>


<%def name="skip_local_threads(prefix)">
FUNCTION bool ${prefix}()
{
    %for threshold, strides in local_groups.skip_thresholds:
    {
        VSIZE_T flat_id =
        %for rdim, stride in strides:
            get_local_id(${rdim}) * ${stride} +
        %endfor
            0;

        if (flat_id >= ${threshold})
            return true;
    }
    %endfor

    return false;
}
</%def>


<%def name="skip_groups(prefix)">
FUNCTION bool ${prefix}()
{
    %for threshold, strides in grid_groups.skip_thresholds:
    {
        VSIZE_T flat_id =
        %for rdim, stride in strides:
            get_group_id(${rdim}) * ${stride} +
        %endfor
            0;

        if (flat_id >= ${threshold})
            return true;
    }
    %endfor

    return false;
}
</%def>


<%def name="skip_global_threads(prefix)">
FUNCTION bool ${prefix}()
{
    %for vdim in range(len(virtual_global_size)):
    %if virtual_global_size[vdim] < bounding_global_size[vdim]:
    if (${global_id_mod}(${vdim_inverse(len(virtual_global_size), vdim)}) >= ${virtual_global_size[vdim]})
        return true;
    %endif
    %endfor

    return false;
}
</%def>


<%def name="begin_static_kernel()">
if(${skip_local_threads_mod}() || ${skip_groups_mod}() || ${skip_global_threads_mod}()) return
</%def>

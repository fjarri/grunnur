<%def name="prelude()">

%if fast_math:
#define GRUNNUR_FAST_MATH
%endif

#define GRUNNUR_OPENCL_API

#define LOCAL_BARRIER barrier(CLK_LOCAL_MEM_FENCE)

// 'static' helps to avoid the "no previous prototype for function" warning
#if __OPENCL_VERSION__ >= 120
#define FUNCTION static
#else
#define FUNCTION
#endif

#define KERNEL __kernel
#define GLOBAL_MEM __global
#define GLOBAL_MEM_ARG __global
#define LOCAL_MEM_DECL __local
#define LOCAL_MEM_DYNAMIC __local
#define LOCAL_MEM __local
#define CONSTANT_MEM_DECL __constant
#define CONSTANT_MEM __constant
// INLINE is already defined in Beignet driver
#ifndef INLINE
#define INLINE inline
#endif
#define SIZE_T size_t
#define VSIZE_T size_t

// used to align fields in structures
#define ALIGN(bytes) __attribute__ ((aligned(bytes)))

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif

#define COMPLEX_CTR(T) (T)

</%def>

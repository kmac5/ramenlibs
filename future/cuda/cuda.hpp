//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#ifndef BASE_CUDA_CUDA_HPP
#define BASE_CUDA_CUDA_HPP

#include<base/config.hpp>

#include<base/cuda/driver_exceptions.hpp>

#ifdef BASE_WITH_GL
    #include<base/GL/glew.hpp>
    #include<cudaGL.h>
#endif

namespace base
{
namespace cuda
{

// unload libcuda
BASE_API void unload_libcuda();

// error handling
BASE_API void check_cu_error( CUresult err);

// forwarding funs
BASE_API void cu_init( unsigned int flags = 0);
BASE_API int cu_driver_get_version();

BASE_API int cu_device_get_count();
BASE_API CUdevice cu_device_get( int ordinal);
BASE_API void cu_device_get_name( char *name, int len, CUdevice dev);
BASE_API void cu_device_compute_capability( int *major, int *minor, CUdevice dev);
BASE_API size_t cu_device_total_mem( CUdevice dev);

BASE_API int cu_device_get_attribute( CUdevice_attribute attrib, CUdevice dev);

BASE_API void cu_ctx_create( CUcontext *pctx, unsigned int flags, CUdevice dev);
BASE_API void cu_ctx_destroy( CUcontext ctx);
BASE_API void cu_ctx_push_current( CUcontext ctx);
BASE_API void cu_ctx_pop_current( CUcontext *pctx);
BASE_API void cu_ctx_get_current( CUcontext *pctx);
BASE_API void cu_ctx_attach( CUcontext *pctx, unsigned int flags);
BASE_API void cu_ctx_detach( CUcontext ctx);

#ifdef BASE_WITH_GL

    BASE_API void cuGLCtxCreate( CUcontext *pCtx, unsigned int Flags, CUdevice device );
    BASE_API void cuGraphicsGLRegisterBuffer( CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags);
    BASE_API void cuGraphicsGLRegisterImage( CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags);
    BASE_API void cuGLGetDevices( unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices,
                                    unsigned int cudaDeviceCount, CUGLDeviceList deviceList);

#endif

} // namespace
} // namespace

#endif

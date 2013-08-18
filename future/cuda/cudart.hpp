//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#ifndef BASE_CUDA_CUDART_HPP
#define BASE_CUDA_CUDART_HPP

#include<base/config.hpp>

#include<base/cuda/runtime_exceptions.hpp>

namespace base
{
namespace cuda
{

// error handling
BASE_API void check_cuda_error( cudaError_t err);

// forwarding funs
BASE_API int cuda_get_device_count();
BASE_API void cuda_get_device_properties( struct cudaDeviceProp *prop, int device);
BASE_API void cuda_choose_device( int *device, const struct cudaDeviceProp *prop);
BASE_API void cuda_set_device( int device);

BASE_API void *cuda_malloc( size_t size);
BASE_API void cuda_free( void *ptr);
BASE_API void cuda_memcpy( void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

} // namespace
} // namespace

#endif

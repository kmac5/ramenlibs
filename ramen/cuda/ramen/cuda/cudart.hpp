/*
 Copyright (c) 2013 Esteban Tovagliari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RAMEN_CUDA_CUDART_HPP
#define RAMEN_CUDA_CUDART_HPP

#include<ramen/cuda/exceptions.hpp>

namespace ramen
{
namespace cuda
{

// error handling
RAMEN_CUDA_API void check_cuda_error( cudaError_t err);

// forwarding funs
RAMEN_CUDA_API int cuda_get_device_count();
RAMEN_CUDA_API void cuda_get_device_properties( struct cudaDeviceProp *prop, int device);
RAMEN_CUDA_API void cuda_choose_device( int *device, const struct cudaDeviceProp *prop);
RAMEN_CUDA_API void cuda_set_device( int device);

RAMEN_CUDA_API void *cuda_malloc( size_t size);
RAMEN_CUDA_API void cuda_free( void *ptr);
RAMEN_CUDA_API void cuda_memcpy( void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

} // cuda
} // ramen

#endif

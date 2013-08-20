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

template<class T>
T *cuda_malloc( std::size_t size)
{
    void *ptr = 0;
    check_cuda_error( cudaMalloc( &ptr, size * sizeof( T)));
    return reinterpret_cast<T*>( ptr);
}

template<class T>
void cuda_free( T *ptr)
{
    // assuming this will be used mostly in destructors,
    // don't check the result, and specially, don't throw.
    cudaFree( reinterpret_cast<void*>( ptr));
}

template<class T>
T *cuda_host_alloc( std::size_t size, unsigned int flags = cudaHostAllocDefault)
{
    void *ptr = 0;
    check_cuda_error( cudaHostAlloc( &ptr, size * sizeof( T), flags));
    return reinterpret_cast<T*>( ptr);
}

template<class T>
void cuda_free_host( T *ptr)
{
    // assuming this will be used mostly in destructors,
    // don't check the result, and specially, don't throw.
    cudaFreeHost( reinterpret_cast<void*>( ptr));
}

template<class T>
T *cuda_host_get_device_ptr( T *ptr, unsigned int flags = 0)
{
    void *tmp;
    check_cuda_error( cudaHostGetDevicePointer( &tmp, ptr, flags));
    return reinterpret_cast<T*>( tmp);
}

template<class T>
void cuda_memcpy( T *dst, const T *src, size_t count, enum cudaMemcpyKind kind)
{
    check_cuda_error( cudaMemcpy( reinterpret_cast<void*>( dst),
                                  reinterpret_cast<const void*>( src),
                                  count * sizeof( T), kind));
}


template<class T>
void cuda_memcpy_async( T *dst, const T *src, std::size_t count,
                        enum cudaMemcpyKind kind, cudaStream_t stream)
{
    check_cuda_error( cudaMemcpyAsync( reinterpret_cast<void*>( dst),
                                       reinterpret_cast<const void*>( src),
                                       count * sizeof( T), kind, stream));
}

// events
RAMEN_CUDA_API void cuda_event_create( cudaEvent_t *event);
RAMEN_CUDA_API void cuda_event_destroy( cudaEvent_t event);
RAMEN_CUDA_API void cuda_event_record( cudaEvent_t event, cudaStream_t stream);
RAMEN_CUDA_API void cuda_event_synchronize( cudaEvent_t event);
RAMEN_CUDA_API float cuda_event_elapsed_time( cudaEvent_t start, cudaEvent_t end);

// streams
RAMEN_CUDA_API void cuda_stream_create( cudaStream_t *stream);
RAMEN_CUDA_API void cuda_stream_destroy( cudaStream_t stream);
RAMEN_CUDA_API void cuda_stream_synchronize( cudaStream_t stream);

// memcopy

} // cuda
} // ramen

#endif

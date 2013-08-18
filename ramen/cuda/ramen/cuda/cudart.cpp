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

#include<ramen/cuda/cudart.hpp>

namespace ramen
{
namespace cuda
{

void check_cuda_error( cudaError_t err)
{
    if( err != cudaSuccess)
        throw runtime_error( err);
}

int cuda_get_device_count()
{
    int count;
    check_cuda_error( cudaGetDeviceCount( &count));
    return count;
}

void cuda_get_device_properties( struct cudaDeviceProp *prop, int device)
{
    check_cuda_error( cudaGetDeviceProperties( prop, device));
}

void cuda_choose_device( int *device, const struct cudaDeviceProp *prop)
{
    check_cuda_error( cudaChooseDevice( device, prop));
}

void cuda_set_device( int device)
{
    check_cuda_error( cudaSetDevice( device));
}

void *cuda_malloc( size_t size)
{
    void *ptr = 0;
    check_cuda_error( cudaMalloc( &ptr, size));
    return ptr;
}

void cuda_free( void *ptr)
{
    check_cuda_error( cudaFree( ptr));
}

void cuda_memcpy( void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    check_cuda_error( cudaMemcpy( dst, src, count, kind));
}

} // cuda
} // ramen

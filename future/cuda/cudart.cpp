//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#include<base/cuda/cudart.hpp>

namespace base
{
namespace cuda
{

void check_cuda_error( cudaError_t err)
{
	switch( err)
	{
		case cudaSuccess:
			return;

		case cudaErrorMemoryAllocation:
			throw out_of_memory();

		default:
			throw runtime_error( err);
	}
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

} // namespace
} // namespace

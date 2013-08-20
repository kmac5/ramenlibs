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

void cuda_check_last_error()
{
    check_cuda_error( cudaPeekAtLastError());
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

void cuda_device_synchronize()
{
    check_cuda_error( cudaDeviceSynchronize());
}

void cuda_event_create( cudaEvent_t *event)
{
    assert( event);

    check_cuda_error( cudaEventCreate( event));
}

void cuda_event_destroy( cudaEvent_t event)
{
    // assuming this will be used mostly in destructors,
    // don't check the result, and specially, don't throw.
    cudaEventDestroy( event);
}

void cuda_event_record( cudaEvent_t event, cudaStream_t stream)
{
    check_cuda_error( cudaEventRecord( event, stream));
}

void cuda_event_synchronize( cudaEvent_t event)
{
    check_cuda_error( cudaEventSynchronize( event));
}

float cuda_event_elapsed_time( cudaEvent_t start, cudaEvent_t end)
{
    float ms;
    check_cuda_error( cudaEventElapsedTime( &ms, start, end));
    return ms;
}

void cuda_stream_create( cudaStream_t *stream)
{
    check_cuda_error( cudaStreamCreate( stream));
}

void cuda_stream_destroy( cudaStream_t stream)
{
    // assuming this will be used mostly in destructors,
    // don't check the result, and specially, don't throw.
    cudaStreamDestroy( stream);
}

void cuda_stream_synchronize( cudaStream_t stream)
{
    check_cuda_error( cudaStreamSynchronize( stream));
}

} // cuda
} // ramen

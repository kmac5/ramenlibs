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

#include<iostream>

#include<boost/cstdlib.hpp>
#include<boost/scoped_array.hpp>

#include<ramen/cuda/cudart.hpp>
#include<ramen/cuda/auto_ptr.hpp>
#include<ramen/cuda/event.hpp>

using namespace ramen::cuda;

__global__ void add_vectors( int n, float *a, float *b, float *c)
{
    int tid = blockIdx.x;
    if( tid < n)
        c[tid] = a[tid] + b[tid];
}

int main(int argc, char **argv)
{
    const int vector_size = 10;

    try
    {
        if( cuda_get_device_count() == 0)
        {
            std::cout << "No CUDA devices found. Exiting" << std::endl;
            std::cout << "test failed" << std::endl;
            return boost::exit_failure;
        }

        // allocate and fill an array on the host.
        boost::scoped_array<float> src( new float[vector_size]);
        for( int i = 0; i < vector_size; ++i)
            src[i] = i;

        // we start GPU code here. create a start event.
        event_t start;

        // alloc device mem and copy the host array.
        auto_ptr_t<float,device_ptr_policy> gpu_src1( cuda_malloc<float>( vector_size));
        cuda_memcpy( gpu_src1.get_device_ptr(), src.get(), vector_size, cudaMemcpyHostToDevice);

        // alloc device mem and copy the host array.
        auto_ptr_t<float,device_ptr_policy> gpu_src2( cuda_malloc<float>( vector_size));
        cuda_memcpy( gpu_src2.get_device_ptr(), src.get(), vector_size, cudaMemcpyHostToDevice);

        // allocate a device mem that will contain the result.
        auto_ptr_t<float,device_ptr_policy> gpu_dst( cuda_malloc<float>( vector_size));

        add_vectors<<<vector_size,1>>>( vector_size,
                                        gpu_src1.get_device_ptr(),
                                        gpu_src2.get_device_ptr(),
                                        gpu_dst.get_device_ptr());

        // get back the result on the host.
        boost::scoped_array<float> dst( new float[vector_size]);
        cuda_memcpy( dst.get(), gpu_dst.get_device_ptr(), vector_size, cudaMemcpyDeviceToHost);

        // gpu code ends here. Create stop event and synchronize it.
        event_t stop( true);

        std::cout << "Finished. Elapsed time = " << elapsed_time( start, stop) << " ms" << std::endl;

        // check everything was ok.
        for( int i = 0; i < vector_size; ++i)
        {
            if( dst[i] != ( i + i))
                return boost::exit_failure;
        }

        std::cout << "test ok" << std::endl;
        return boost::exit_success;
    }
    catch( ramen::core::exception& e)
    {
        std::cout << "Exception caught, " << e.what() << std::endl;
        std::cout << "test failed" << std::endl;
        return boost::exit_failure;
    }
}
